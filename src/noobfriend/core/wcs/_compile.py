"""Compile astropy / gwcs model trees into ``noobase`` WCS program specs.

The compiler walks a transform's :class:`~astropy.modeling.CompoundModel`
expression tree *once*, extracts plain ``float`` coefficients, and emits the
flat register-machine spec evaluated by
:class:`noobase.image.wcs.WcsProgram` (see that module for the schema). Two
ideas keep the op set small:

- **Plumbing disappears.** ``Mapping`` and ``Identity`` become register
  wiring at compile time; they cost nothing at runtime.
- **Rotations collapse.** A :class:`~astropy.modeling.rotations.RotationSequence3D`
  is a linear map on the unit sphere's embedding space, so its matrix is
  read off numerically by pushing the three basis vectors through the model
  -- no re-derivation of astropy's Euler conventions (and no convention
  bugs). The same trick turns the whole native-to-celestial rotation of a
  FITS TAN header into one ``rot3`` op.

Where jwst inverts grism trace polynomials on a sampled ``t`` grid
(``sampling=40`` interpolation), the emitted program solves the (at most
quadratic) polynomial exactly; agreement with jwst on that leg is bounded
by *jwst's* sampling error, not ours.

Bounding boxes and units are deliberately out of scope: programs are pure
``float64`` math, matching how noobfriend consumes ``get_transform``
callables (which never apply gwcs bounding boxes).
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from astropy.modeling import CompoundModel, Model
from astropy.modeling import models as astropy_models
from astropy.modeling.polynomial import Polynomial1D, Polynomial2D
from astropy.modeling.rotations import RotationSequence3D
from astropy.wcs import WCS as AstropyWCS
from gwcs import selector as gwcs_selector
from gwcs import spectroscopy as gwcs_spectroscopy
from gwcs.geometry import CartesianToSpherical, SphericalToCartesian

#: A JSON-able ``noobase.image.wcs`` program spec.
Spec = dict[str, Any]

#: Grism dispersion model class names (stdatamodels.jwst.transforms.models)
#: handled by name so importing this module never pulls in stdatamodels.
_GRISM_FORWARD_AXIS: dict[str, str] = {
    "NIRCAMForwardRowGrismDispersion": "row",
    "NIRCAMForwardColumnGrismDispersion": "column",
}
_GRISM_BACKWARD = "NIRCAMBackwardGrismDispersion"


class UnsupportedTransformError(ValueError):
    """A model (or model tree node) the compiler cannot express as ops."""


class _SpecBuilder:
    """Accumulates ops and allocates SSA registers for one program."""

    def __init__(self) -> None:
        self._ops: list[dict[str, Any]] = []
        self._n_regs: int = 0

    def new_regs(self, count: int) -> list[int]:
        regs = list(range(self._n_regs, self._n_regs + count))
        self._n_regs += count
        return regs

    def emit(
        self, op: str, in_regs: list[int], out_regs: list[int], **params: Any
    ) -> None:
        self._ops.append({"op": op, "in": in_regs, "out": out_regs, **params})

    def build(self, inputs: list[int], outputs: list[int]) -> Spec:
        return {
            "n_regs": max(self._n_regs, 1),
            "inputs": inputs,
            "outputs": outputs,
            "ops": self._ops,
        }


def compile_transform(model: Model) -> Spec:
    """Compile an astropy transform (model tree) into a program spec.

    Parameters
    ----------
    model : astropy.modeling.Model
        The transform to compile, e.g. one gwcs pipeline stage's
        ``step.transform`` or its ``.inverse``.

    Returns
    -------
    dict
        A :class:`noobase.image.wcs.WcsProgram` spec whose inputs/outputs
        match ``model.n_inputs`` / ``model.n_outputs`` in order.

    Raises
    ------
    UnsupportedTransformError
        If the tree contains a model class (or compound operator) with no
        op-code equivalent.
    """
    builder = _SpecBuilder()
    in_regs = builder.new_regs(model.n_inputs)
    out_regs = _emit_model(model, in_regs, builder)
    return builder.build(in_regs, out_regs)


def _emit_model(model: Model, ins: list[int], b: _SpecBuilder) -> list[int]:
    """Recursively emit ops for ``model``; returns its output registers."""
    if isinstance(model, CompoundModel):
        if model.op == "|":
            return _emit_model(model.right, _emit_model(model.left, ins, b), b)
        if model.op == "&":
            n_left = model.left.n_inputs
            left = _emit_model(model.left, ins[:n_left], b)
            right = _emit_model(model.right, ins[n_left:], b)
            return left + right
        if model.op in ("*", "+", "/", "-"):
            # Arithmetic against a constant folds to scale / shift; JWST
            # uses ``Identity * Const1D`` (and ``/`` in the inverse) for
            # the WFSS wavelength barycentric velocity correction. For the
            # non-commutative ops the constant must be the right operand.
            candidates = [(model.left, model.right)]
            if model.op in ("*", "+"):
                candidates.append((model.right, model.left))
            for primary, secondary in candidates:
                if not isinstance(secondary, astropy_models.Const1D):
                    continue
                inner = _emit_model(primary, ins, b)
                if len(inner) != 1:
                    break
                out = b.new_regs(1)
                value = float(secondary.amplitude.value)
                if model.op in ("*", "/"):
                    factor = value if model.op == "*" else 1.0 / value
                    b.emit("scale", inner, out, factor=factor)
                else:
                    offset = value if model.op == "+" else -value
                    b.emit("shift", inner, out, offset=offset)
                return out
            # General model (+|-|*|/) model: both operands see the same
            # inputs and must produce one output each (NIRSpec IFU slice
            # chains combine polynomial sub-expressions this way).
            if model.left.n_outputs == 1 and model.right.n_outputs == 1:
                left = _emit_model(model.left, ins, b)
                right = _emit_model(model.right, ins, b)
                out = b.new_regs(1)
                kind = {"+": "add", "-": "sub", "*": "mul", "/": "div"}[model.op]
                b.emit("binary", [left[0], right[0]], out, kind=kind)
                return out
        if model.op == "fix_inputs":
            # ``fix_inputs(model, {name_or_index: value})``: pinned inputs
            # become const registers; the remaining caller inputs slot into
            # the inner model's input order.
            inner_model = model.left
            fixed: dict[Any, Any] = model.right
            remaining = iter(ins)
            full_ins: list[int] = []
            for position, input_name in enumerate(inner_model.inputs):
                if input_name in fixed or position in fixed:
                    value = (
                        fixed[input_name] if input_name in fixed else fixed[position]
                    )
                    reg = b.new_regs(1)
                    b.emit("const", [], reg, value=float(value))
                    full_ins.append(reg[0])
                else:
                    full_ins.append(next(remaining))
            return _emit_model(inner_model, full_ins, b)
        raise UnsupportedTransformError(
            f"Compound operator {model.op!r} is not supported."
        )

    # -- plumbing: pure register wiring, no runtime op ----------------------
    if isinstance(model, astropy_models.Mapping):
        return [ins[i] for i in model.mapping]
    if isinstance(model, astropy_models.Identity):
        return list(ins)

    # -- scalar / polynomial primitives -------------------------------------
    if isinstance(model, astropy_models.Shift):
        out = b.new_regs(1)
        b.emit("shift", ins, out, offset=float(model.offset.value))
        return out
    if isinstance(model, astropy_models.Scale):
        out = b.new_regs(1)
        b.emit("scale", ins, out, factor=float(model.factor.value))
        return out
    if isinstance(model, astropy_models.Const1D):
        out = b.new_regs(1)
        # Const1D reads (and ignores) one input; the op takes none.
        b.emit("const", [], out, value=float(model.amplitude.value))
        return out
    if isinstance(model, Polynomial1D):
        out = b.new_regs(1)
        b.emit("poly1d", ins, out, coeffs=_poly1d_coeffs(model))
        return out
    if isinstance(model, Polynomial2D):
        out = b.new_regs(1)
        degree, coeffs = _poly2d_coeffs(model)
        b.emit("poly2d", ins, out, degree=degree, coeffs=coeffs)
        return out
    if isinstance(model, astropy_models.AffineTransformation2D):
        out = b.new_regs(2)
        b.emit(
            "affine2",
            ins,
            out,
            matrix=np.asarray(model.matrix.value, dtype=float).tolist(),
            translation=np.asarray(model.translation.value, dtype=float).tolist(),
        )
        return out

    # -- spherical geometry --------------------------------------------------
    if isinstance(model, SphericalToCartesian):
        out = b.new_regs(3)
        b.emit("sph2cart", ins, out)
        return out
    if isinstance(model, CartesianToSpherical):
        out = b.new_regs(2)
        b.emit("cart2sph", ins, out, wrap_lon_at=int(model.wrap_lon_at))
        return out
    if isinstance(model, RotationSequence3D):
        out = b.new_regs(3)
        b.emit("rot3", ins, out, matrix=_cartesian_rotation_matrix(model))
        return out
    if isinstance(model, astropy_models.Rotation2D):
        out = b.new_regs(2)
        columns = [model(1.0, 0.0), model(0.0, 1.0)]
        matrix = np.array(columns, dtype=float).T.tolist()
        b.emit("affine2", ins, out, matrix=matrix, translation=[0.0, 0.0])
        return out
    if isinstance(model, astropy_models.Pix2Sky_Gnomonic):
        out = b.new_regs(2)
        b.emit("tan_deproject", ins, out)
        return out
    if isinstance(model, astropy_models.Sky2Pix_Gnomonic):
        out = b.new_regs(2)
        b.emit("tan_project", ins, out)
        return out
    if isinstance(
        model,
        (astropy_models.RotateNative2Celestial, astropy_models.RotateCelestial2Native),
    ):
        # Any lon/lat sphere rotation is an O(3) action on the embedding
        # space: compile as s2c | rot3 | c2s with the matrix read off basis
        # vectors (jwst cube_build / resample output WCS uses these).
        xyz = b.new_regs(3)
        b.emit("sph2cart", ins, xyz)
        rotated = b.new_regs(3)
        b.emit("rot3", xyz, rotated, matrix=_sphere_rotation_matrix(model))
        out = b.new_regs(2)
        b.emit("cart2sph", rotated, out, wrap_lon_at=360)
        return out

    # -- spectroscopy (NIRSpec IFU) ------------------------------------------
    if isinstance(model, gwcs_spectroscopy.WavelengthFromGratingEquation):
        out = b.new_regs(1)
        b.emit("grating_wavelength", ins, out, factor=_grating_factor(model))
        return out
    if isinstance(model, gwcs_spectroscopy.AnglesFromGratingEquation3D):
        out = b.new_regs(3)
        b.emit("grating_angles3d", ins, out, factor=_grating_factor(model))
        return out
    if isinstance(model, astropy_models.Tabular1D):
        out = b.new_regs(1)
        b.emit("tabular1d", ins, out, **_tabular1d_params(model))
        return out
    if isinstance(model, gwcs_selector.RegionsSelector):
        if not np.isnan(model.undefined_transform_value):
            raise UnsupportedTransformError(
                "RegionsSelector with a non-NaN undefined_transform_value "
                "is not supported."
            )
        cases = [
            {"label": int(label), "program": compile_transform(transform)}
            for label, transform in model.selector.items()
        ]
        out = b.new_regs(model.n_outputs)
        b.emit(
            "select",
            list(ins),
            out,
            label=_label_key_spec(model.label_mapper),
            cases=cases,
        )
        return out

    # -- JWST models (dispatched by name; the classes live in
    #    stdatamodels.jwst.transforms.models, which stays unimported here) --
    name = type(model).__name__
    if name == "Unitless2DirCos":
        out = b.new_regs(3)
        b.emit("unitless2dircos", ins, out)
        return out
    if name == "DirCos2Unitless":
        out = b.new_regs(2)
        b.emit("dircos2unitless", ins, out)
        return out
    if name == "Rotation3DToGWA":
        # NOT a linear rotation: each step renormalizes
        # ``z = sqrt(1 - x^2 - y^2)`` (jwst keeps only direction cosines),
        # so the sequence is replicated step by step. The stored angles are
        # degrees; evaluation applies them in radians.
        out = b.new_regs(3)
        steps = [
            {"axis": str(axis), "angle": float(np.deg2rad(angle))}
            for angle, axis in zip(
                np.atleast_1d(model.angles.value), model.axes_order, strict=True
            )
        ]
        b.emit("rot3_gwa", ins, out, steps=steps)
        return out
    if name == "Logical":
        compareto, value = model.compareto, model.value
        if isinstance(compareto, np.ndarray) or isinstance(value, np.ndarray):
            raise UnsupportedTransformError(
                "Logical with array operands is not supported."
            )
        if model.condition not in ("GT", "LT", "EQ", "NE"):
            raise UnsupportedTransformError(
                f"Logical condition {model.condition!r} is not supported."
            )
        out = b.new_regs(1)
        b.emit(
            "logical",
            ins,
            out,
            condition=str(model.condition),
            compareto=float(compareto),
            value=float(value),
        )
        return out
    if name in _GRISM_FORWARD_AXIS:
        axis = _GRISM_FORWARD_AXIS[name]
        alongdisp = model.xmodels if axis == "row" else model.ymodels
        lam = b.new_regs(1)
        b.emit(
            "grism_forward",
            ins,
            lam,
            axis=axis,
            orders=[int(order) for order in model.orders],
            alongdisp=[_tpoly_spec(entry) for entry in alongdisp],
            lmodels=[_tpoly_spec(entry) for entry in model.lmodels],
        )
        # jwst forward evaluate: (x, y, x0, y0, order) -> (x0, y0, lam, order)
        return [ins[2], ins[3], lam[0], ins[4]]
    if name == _GRISM_BACKWARD:
        dispersed = b.new_regs(2)
        b.emit(
            "grism_backward",
            ins,
            dispersed,
            orders=[int(order) for order in model.orders],
            lmodels=[_tpoly_spec(entry) for entry in model.lmodels],
            xmodels=[_tpoly_spec(entry) for entry in model.xmodels],
            ymodels=[_tpoly_spec(entry) for entry in model.ymodels],
        )
        # jwst backward evaluate: (x, y, lam, order) -> (xg, yg, x, y, order)
        return [dispersed[0], dispersed[1], ins[0], ins[1], ins[3]]

    raise UnsupportedTransformError(f"Model class {name!r} has no compiled equivalent.")


def _poly1d_coeffs(model: Polynomial1D) -> list[float]:
    """``[c0, c1, ...]`` of an astropy 1-D polynomial."""
    return [float(getattr(model, f"c{k}").value) for k in range(model.degree + 1)]


def _poly2d_coeffs(model: Polynomial2D) -> tuple[int, list[float]]:
    """Dense row-major ``(degree+1)**2`` coefficient list, row = x power."""
    degree = int(model.degree)
    matrix = np.zeros((degree + 1, degree + 1))
    for name in model.param_names:
        i, j = (int(part) for part in name[1:].split("_"))
        matrix[i, j] = float(getattr(model, name).value)
    return degree, matrix.ravel().tolist()


def _cartesian_rotation_matrix(model: Model) -> list[list[float]]:
    """3x3 matrix of a linear cartesian model, read off basis vectors."""
    columns = [model(*basis) for basis in np.eye(3)]
    return np.array(columns, dtype=float).T.tolist()


def _grating_factor(model: Model) -> float:
    """``groove_density * spectral_order`` of a grating-equation model."""
    return float(model.groove_density.value) * float(model.spectral_order.value)


def _tabular1d_params(model: Model) -> dict[str, Any]:
    """Op parameters of an astropy ``Tabular1D`` (linear, no bounds error)."""
    if model.method != "linear" or model.bounds_error:
        raise UnsupportedTransformError(
            f"Tabular1D with method={model.method!r} / "
            f"bounds_error={model.bounds_error} is not supported."
        )
    points = np.asarray(model.points[0], dtype=float)
    values = np.asarray(model.lookup_table, dtype=float)
    if points.size >= 2 and points[0] > points[-1]:  # descending: flip both
        points, values = points[::-1], values[::-1]
    fill = model.fill_value
    return {
        "points": points.tolist(),
        "values": values.tolist(),
        "fill": float("nan") if fill is None else float(fill),
    }


def _label_key_spec(mapper: Model) -> dict[str, Any]:
    """Label-key spec of a gwcs label mapper (array or quantized dict)."""
    if isinstance(mapper, gwcs_selector.LabelMapperArray):
        if mapper.inputs_mapping is not None:
            raise UnsupportedTransformError(
                "LabelMapperArray with an inputs_mapping is not supported."
            )
        data = np.ascontiguousarray(np.asarray(mapper.mapper), dtype=np.int64)
        if data.ndim != 2:
            raise UnsupportedTransformError(
                f"LabelMapperArray mapper must be 2-D, got {data.ndim}-D."
            )
        return {"kind": "array", "data": data}
    if isinstance(mapper, gwcs_selector.LabelMapperDict):
        inputs_mapping = mapper.inputs_mapping
        if inputs_mapping is None:
            key_input = 0
        else:
            mapping = tuple(inputs_mapping.mapping)
            if len(mapping) != 1:
                raise UnsupportedTransformError(
                    f"LabelMapperDict must select one key input, got {mapping}."
                )
            key_input = int(mapping[0])
        keys: list[float] = []
        labels: list[int] = []
        for key, label_model in mapper.mapper.items():
            keys.append(float(key))
            labels.append(_constant_label(label_model))
        return {
            "kind": "dict",
            "keys": keys,
            "labels": labels,
            "key_input": key_input,
            "atol": float(mapper.atol),
        }
    raise UnsupportedTransformError(
        f"Label mapper {type(mapper).__name__!r} is not supported."
    )


def _constant_label(model: Model) -> int:
    """Extract the constant integer a label model returns (probed twice)."""
    probes = [
        model(*([np.array([value])] * model.n_inputs)) for value in (0.37, -1.234)
    ]
    first, second = (float(np.asarray(p).ravel()[0]) for p in probes)
    if first != second:
        raise UnsupportedTransformError(
            "LabelMapperDict value models must be constant."
        )
    return round(first)


def _tpoly_spec(entry: Any) -> dict[str, Any]:
    """Spec of one jwst grism trace polynomial ("guess form").

    ``entry`` is one per-order element of the model's ``lmodels`` /
    ``xmodels`` / ``ymodels``: either a single 1-input model in ``t`` (or a
    length-1 list of one), or a list of 2-input ``Polynomial2D`` giving each
    ``t`` power's spatial coefficient.
    """
    models = [entry] if isinstance(entry, Model) else list(entry)
    if len(models) == 1 and models[0].n_inputs == 1:
        if not isinstance(models[0], Polynomial1D):
            raise UnsupportedTransformError(
                f"Grism t-model {type(models[0]).__name__!r} is not Polynomial1D."
            )
        return {"kind": "t", "coeffs": _poly1d_coeffs(models[0])}

    degrees: set[int] = set()
    coeffs: list[list[float]] = []
    for m in models:
        if not isinstance(m, Polynomial2D):
            raise UnsupportedTransformError(
                f"Grism spatial coefficient model {type(m).__name__!r} "
                "is not Polynomial2D."
            )
        degree, dense = _poly2d_coeffs(m)
        degrees.add(degree)
        coeffs.append(dense)
    if len(degrees) != 1:
        raise UnsupportedTransformError(
            f"Grism spatial coefficient models mix degrees {sorted(degrees)}."
        )
    return {"kind": "spatial", "degree": degrees.pop(), "coeffs": coeffs}


def concat_specs(specs: Sequence[Spec]) -> Spec:
    """Chain program specs: each one's outputs feed the next one's inputs.

    Registers of each segment are renumbered into one shared space; a
    segment's input registers are aliased to the previous segment's output
    registers (valid because compiled specs never write to their inputs).

    Parameters
    ----------
    specs : sequence of dict
        Program specs in evaluation order. Adjacent arities must match.

    Returns
    -------
    dict
        The combined spec.
    """
    if not specs:
        raise ValueError("concat_specs needs at least one spec.")
    ops: list[dict[str, Any]] = []
    inputs: list[int] | None = None
    previous_out: list[int] | None = None
    total_regs = 0
    for spec in specs:
        alias: dict[int, int] = {}
        if previous_out is not None:
            if len(previous_out) != len(spec["inputs"]):
                raise ValueError(
                    f"Cannot chain: {len(previous_out)} outputs feed "
                    f"{len(spec['inputs'])} inputs."
                )
            alias = dict(zip(spec["inputs"], previous_out))

        def remap(
            reg: int, alias: dict[int, int] = alias, base: int = total_regs
        ) -> int:
            return alias.get(reg, reg + base)

        for op in spec["ops"]:
            renumbered = dict(op)
            renumbered["in"] = [remap(reg) for reg in op["in"]]
            renumbered["out"] = [remap(reg) for reg in op["out"]]
            ops.append(renumbered)
        if inputs is None:
            inputs = [remap(reg) for reg in spec["inputs"]]
        previous_out = [remap(reg) for reg in spec["outputs"]]
        total_regs += spec["n_regs"]
    assert inputs is not None and previous_out is not None
    return {
        "n_regs": max(total_regs, 1),
        "inputs": inputs,
        "outputs": previous_out,
        "ops": ops,
    }


def compile_fits_tan(wcs: AstropyWCS) -> tuple[Spec, Spec]:
    """Compile a plain (distortion-free) FITS TAN WCS into program specs.

    Covers the mosaic-tile / cutout grid headers noobfriend synthesises and
    the jwst resample outputs: 2-axis ``RA---TAN`` / ``DEC--TAN`` with an
    arbitrary linear ``PC``/``CD`` matrix -- no SIP, no lookup-table
    distortion.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        The header WCS. Pixel convention of the emitted programs is
        astropy's 0-based ``pixel_to_world_values``.

    Returns
    -------
    tuple of (dict, dict)
        ``(forward, backward)`` specs: pixel ``(x, y)`` -> ``(ra, dec)``
        degrees, and its exact inverse.

    Raises
    ------
    UnsupportedTransformError
        If the WCS is not a plain 2-axis TAN.
    """
    ctypes = tuple(wcs.wcs.ctype)
    if wcs.naxis != 2 or ctypes != ("RA---TAN", "DEC--TAN"):
        raise UnsupportedTransformError(
            f"Only plain 2-axis ('RA---TAN', 'DEC--TAN') headers are "
            f"supported, got naxis={wcs.naxis}, ctype={ctypes}."
        )
    if wcs.sip is not None or len(wcs.wcs.get_pv()) > 0:
        raise UnsupportedTransformError("SIP / PV distortion terms are not supported.")

    cd = np.asarray(wcs.pixel_scale_matrix, dtype=float)
    crpix0 = np.asarray(wcs.wcs.crpix, dtype=float) - 1.0  # 0-based
    translation = -cd @ crpix0
    cd_inv = np.linalg.inv(cd)
    rotation = _native_to_celestial_matrix(
        float(wcs.wcs.crval[0]), float(wcs.wcs.crval[1]), float(wcs.wcs.lonpole)
    )
    rotation_t = np.array(rotation).T.tolist()

    forward = _SpecBuilder()
    pix = forward.new_regs(2)
    plane = forward.new_regs(2)
    forward.emit(
        "affine2", pix, plane, matrix=cd.tolist(), translation=translation.tolist()
    )
    native = forward.new_regs(2)
    forward.emit("tan_deproject", plane, native)
    xyz = forward.new_regs(3)
    forward.emit("sph2cart", native, xyz)
    rotated = forward.new_regs(3)
    forward.emit("rot3", xyz, rotated, matrix=rotation)
    world = forward.new_regs(2)
    forward.emit("cart2sph", rotated, world, wrap_lon_at=360)
    forward_spec = forward.build(pix, world)

    backward = _SpecBuilder()
    world = backward.new_regs(2)
    xyz = backward.new_regs(3)
    backward.emit("sph2cart", world, xyz)
    rotated = backward.new_regs(3)
    backward.emit("rot3", xyz, rotated, matrix=rotation_t)
    native = backward.new_regs(2)
    backward.emit("cart2sph", rotated, native, wrap_lon_at=360)
    plane = backward.new_regs(2)
    backward.emit("tan_project", native, plane)
    pix = backward.new_regs(2)
    backward.emit(
        "affine2",
        plane,
        pix,
        matrix=cd_inv.tolist(),
        translation=(-cd_inv @ translation).tolist(),
    )
    backward_spec = backward.build(world, pix)

    return forward_spec, backward_spec


def _native_to_celestial_matrix(
    crval1: float, crval2: float, lonpole: float
) -> list[list[float]]:
    """3x3 native-spherical -> celestial rotation of a zenithal projection."""
    return _sphere_rotation_matrix(
        astropy_models.RotateNative2Celestial(crval1, crval2, lonpole)
    )


def _sphere_rotation_matrix(model: Model) -> list[list[float]]:
    """3x3 matrix of a lon/lat sphere-rotation model.

    Read off basis vectors pushed through the model (an O(3) action on the
    embedding space), so the FITS Euler-angle conventions are astropy's
    own -- works for ``RotateNative2Celestial``, ``RotateCelestial2Native``,
    and any other pure spherical rotation.
    """
    columns = []
    for x, y, z in np.eye(3):
        lon = np.degrees(np.arctan2(y, x))
        lat = np.degrees(np.arctan2(z, np.hypot(x, y)))
        out_lon, out_lat = (float(v) for v in model(lon, lat))
        lon_r, lat_r = np.radians(out_lon), np.radians(out_lat)
        columns.append(
            [
                np.cos(lat_r) * np.cos(lon_r),
                np.cos(lat_r) * np.sin(lon_r),
                np.sin(lat_r),
            ]
        )
    return np.array(columns, dtype=float).T.tolist()
