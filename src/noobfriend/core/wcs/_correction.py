"""Tangent-plane WCS alignment corrections, for NoobWCS and gwcs alike.

The stage-3 alignment fit produces one :class:`TangentCorrection` per
exposure (or group): an affine transform in the tangent plane about a
fiducial point. That single parameter set drives every consumer:

- :meth:`noobfriend.core.wcs.NoobWCS.with_tangent_correction` composes it
  into the compiled fast path that feeds the coadd;
- :func:`apply_correction_to_gwcs` composes it into the *original* gwcs
  (the one still sitting in the file), producing a corrected gwcs built
  from **astropy-core models only** -- readable by anyone with stock
  ``gwcs`` + ``asdf-astropy``, no jwst / tweakwcs / noobfriend required.
  This is the interoperability exit (optional "2c" products) and what a
  jwst-engine fallback would consume.

Both consumers compose the correction into the WCS's *final* pipeline
stage rather than inserting a new frame: frame names stay untouched, and
the two representations stay structurally symmetric. The correction chain
is ``world -> tangent plane (about the fiducial) -> affine -> world``;
since the affine is analytically invertible, so is the whole correction.
"""

from dataclasses import dataclass

import numpy as np
from astropy.modeling import Model
from astropy.modeling import models as astropy_models
from gwcs import WCS as GwcsWCS

from ._compile import Spec, _SpecBuilder, _sphere_rotation_matrix

__all__ = ["TangentCorrection", "apply_correction_to_gwcs"]


@dataclass(frozen=True)
class TangentCorrection:
    """An affine alignment correction in a tangent plane.

    The correction maps *observed* world positions to *reference-aligned*
    world positions: project onto the tangent plane about the fiducial,
    apply ``plane' = matrix @ plane + offset``, deproject.

    Parameters
    ----------
    fiducial : tuple of (float, float)
        Tangent-point ``(ra, dec)`` in degrees (typically the output-grid
        fiducial the alignment was fitted in).
    matrix : tuple of tuple of float
        Row-major 2x2 affine matrix (identity for a pure shift).
    offset : tuple of (float, float)
        Tangent-plane translation in **degrees**.
    """

    fiducial: tuple[float, float]
    matrix: tuple[tuple[float, float], tuple[float, float]] = ((1.0, 0.0), (0.0, 1.0))
    offset: tuple[float, float] = (0.0, 0.0)

    def to_models(self) -> Model:
        """Build the correction as a compound of astropy-core models.

        ``(ra, dec) -> (ra', dec')`` in degrees; analytically invertible
        and ASDF-serializable with standard ``asdf-astropy`` tags only.
        """
        ra0, dec0 = self.fiducial
        return (
            astropy_models.RotateCelestial2Native(ra0, dec0, 180.0)
            | astropy_models.Sky2Pix_Gnomonic()
            | astropy_models.AffineTransformation2D(
                matrix=np.asarray(self.matrix, dtype=float),
                translation=np.asarray(self.offset, dtype=float),
            )
            | astropy_models.Pix2Sky_Gnomonic()
            | astropy_models.RotateNative2Celestial(ra0, dec0, 180.0)
        )

    def to_specs(self) -> tuple[Spec, Spec]:
        """Build the correction as ``(forward, backward)`` program specs.

        Both directions are exact (the backward uses the analytically
        inverted affine), so a corrected NoobWCS round-trips as tightly as
        the uncorrected one.
        """
        ra0, dec0 = self.fiducial
        to_native = _sphere_rotation_matrix(
            astropy_models.RotateCelestial2Native(ra0, dec0, 180.0)
        )
        to_celestial = _sphere_rotation_matrix(
            astropy_models.RotateNative2Celestial(ra0, dec0, 180.0)
        )
        matrix = np.asarray(self.matrix, dtype=float)
        offset = np.asarray(self.offset, dtype=float)
        inverse_matrix = np.linalg.inv(matrix)

        def build(affine: np.ndarray, translation: np.ndarray) -> Spec:
            b = _SpecBuilder()
            world = b.new_regs(2)
            xyz = b.new_regs(3)
            b.emit("sph2cart", world, xyz)
            native_xyz = b.new_regs(3)
            b.emit("rot3", xyz, native_xyz, matrix=to_native)
            native = b.new_regs(2)
            b.emit("cart2sph", native_xyz, native, wrap_lon_at=360)
            plane = b.new_regs(2)
            b.emit("tan_project", native, plane)
            corrected = b.new_regs(2)
            b.emit(
                "affine2",
                plane,
                corrected,
                matrix=affine.tolist(),
                translation=translation.tolist(),
            )
            native_out = b.new_regs(2)
            b.emit("tan_deproject", corrected, native_out)
            xyz_out = b.new_regs(3)
            b.emit("sph2cart", native_out, xyz_out)
            cel_xyz = b.new_regs(3)
            b.emit("rot3", xyz_out, cel_xyz, matrix=to_celestial)
            out = b.new_regs(2)
            b.emit("cart2sph", cel_xyz, out, wrap_lon_at=360)
            return b.build(world, out)

        forward = build(matrix, offset)
        backward = build(inverse_matrix, -(inverse_matrix @ offset))
        return forward, backward


def apply_correction_to_gwcs(wcs: GwcsWCS, correction: TangentCorrection) -> GwcsWCS:
    """Compose a tangent-plane correction into a gwcs object.

    Parameters
    ----------
    wcs : gwcs.WCS
        The original WCS (e.g. straight from the file). Not modified.
    correction : TangentCorrection
        The alignment correction to apply on the world side.

    Returns
    -------
    gwcs.WCS
        A new WCS whose final pipeline transform is
        ``old_transform | correction``. Frame names and count are
        unchanged, and the added models are astropy-core only, so the
        result serializes to ASDF with standard tags.

    Notes
    -----
    The last pipeline transform must expose ``(ra, dec)`` degrees as its
    *first two* outputs (true for all JWST imaging and WFSS pipelines,
    where any extra outputs -- wavelength, order -- pass through
    untouched). Extra outputs are routed around the correction with
    ``Mapping`` / ``Identity`` plumbing.
    """
    steps = list(wcs.pipeline)
    frame, transform = steps[-2].frame, steps[-2].transform
    n_extra = transform.n_outputs - 2
    correction_model = correction.to_models()
    if n_extra > 0:
        correction_model = correction_model & astropy_models.Identity(n_extra)
    return GwcsWCS(
        [
            *[(step.frame, step.transform) for step in steps[:-2]],
            (frame, transform | correction_model),
            (steps[-1].frame, None),
        ]
    )
