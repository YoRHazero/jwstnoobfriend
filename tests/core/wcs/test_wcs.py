"""Tests for :mod:`noobfriend.core.wcs` (gwcs -> noobase program compiler).

Everything runs on synthetic astropy / stdatamodels models -- the compiler
is exercised against the *source* models' own evaluations, so no FITS files
or gwcs reference data are needed.
"""

import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest
from astropy.modeling import models as astropy_models
from astropy.modeling.polynomial import Polynomial1D, Polynomial2D
from astropy.modeling.rotations import RotationSequence3D
from astropy.wcs import WCS as AstropyWCS
from gwcs import WCS as GwcsWCS
from gwcs import coordinate_frames as cf
from gwcs.geometry import CartesianToSpherical, SphericalToCartesian

from noobfriend.core.wcs import (
    NoobWCS,
    TangentCorrection,
    UnsupportedTransformError,
    apply_correction_to_gwcs,
    compile_transform,
    from_fits_wcs,
    from_gwcs,
)
from noobase.image.wcs import WcsProgram


def _random_poly2d(degree: int, seed: int) -> Polynomial2D:
    """Build a distortion-like random polynomial.

    Coefficients decay with order so outputs stay O(10) over a
    2048-pixel domain (like real SIAF distortion solutions).
    """
    rng = np.random.default_rng(seed)
    poly = Polynomial2D(degree)
    for name in poly.param_names:
        i, j = (int(part) for part in name[1:].split("_"))
        setattr(poly, name, rng.normal() * 10.0 ** (-3.5 * (i + j)))
    return poly


def _distortion_like_transform() -> astropy_models.Mapping:
    """Build a detector->v2v3-shaped chain: shifts, input fan-out, 2-D polys."""
    shifts = astropy_models.Shift(-1024.5) & astropy_models.Shift(-1024.5)
    duplicate = astropy_models.Mapping((0, 1, 0, 1))
    polys = _random_poly2d(5, seed=1) & _random_poly2d(5, seed=2)
    return (
        shifts
        | duplicate
        | polys
        | (astropy_models.Shift(0.03) & astropy_models.Scale(1.001))
    )


def test_compiled_distortion_chain_matches_astropy() -> None:
    model = _distortion_like_transform()
    program = WcsProgram(compile_transform(model))

    rng = np.random.default_rng(0)
    x = rng.uniform(0, 2048, 500)
    y = rng.uniform(0, 2048, 500)
    want = model(x, y)
    got = program(x, y)
    np.testing.assert_allclose(got[0], want[0], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(got[1], want[1], rtol=1e-13, atol=1e-13)


def _sky_rotation_transform() -> astropy_models.Mapping:
    """Build a v2v3->world-shaped chain: arcsec scale, sphere rotation."""
    scales = astropy_models.Scale(1 / 3600) & astropy_models.Scale(1 / 3600)
    rotation = RotationSequence3D(
        [-0.02, 0.13, 230.7, 62.25, -189.2], axes_order="zyxyz"
    )
    return (
        scales
        | SphericalToCartesian(wrap_lon_at=180)
        | rotation
        | CartesianToSpherical(wrap_lon_at=360)
    )


def test_compiled_sky_rotation_matches_astropy() -> None:
    model = _sky_rotation_transform()
    program = WcsProgram(compile_transform(model))

    rng = np.random.default_rng(1)
    v2 = rng.uniform(-400, 400, 500)  # arcsec
    v3 = rng.uniform(-400, 400, 500)
    want = model(v2, v3)
    got = program(v2, v3)
    np.testing.assert_allclose(got[0], want[0], rtol=0, atol=1e-11)
    np.testing.assert_allclose(got[1], want[1], rtol=0, atol=1e-11)


def _synthetic_gwcs() -> GwcsWCS:
    detector = cf.Frame2D(name="detector", axes_order=(0, 1))
    v2v3 = cf.Frame2D(name="v2v3", axes_order=(0, 1), unit=(u.arcsec, u.arcsec))
    world = cf.CelestialFrame(name="world", reference_frame=coord.ICRS())
    det_to_v2v3 = (astropy_models.Shift(-1024.0) & astropy_models.Shift(-1024.0)) | (
        astropy_models.Scale(0.031) & astropy_models.Scale(0.031)
    )
    return GwcsWCS(
        [
            (detector, det_to_v2v3),
            (v2v3, _sky_rotation_transform()),
            (world, None),
        ]
    )


def test_from_gwcs_matches_source_both_directions() -> None:
    wcs = _synthetic_gwcs()
    noob = from_gwcs(wcs)
    assert noob.available_frames == ["detector", "v2v3", "world"]

    rng = np.random.default_rng(2)
    x = rng.uniform(0, 2048, 300)
    y = rng.uniform(0, 2048, 300)

    want_ra, want_dec = wcs.get_transform("detector", "world")(x, y)
    got_ra, got_dec = noob.get_transform("detector", "world")(x, y)
    np.testing.assert_allclose(got_ra, want_ra, atol=1e-11)
    np.testing.assert_allclose(got_dec, want_dec, atol=1e-11)

    back_x, back_y = noob.get_transform("world", "detector")(got_ra, got_dec)
    np.testing.assert_allclose(back_x, x, atol=1e-8)
    np.testing.assert_allclose(back_y, y, atol=1e-8)

    # Scalar fast path returns plain floats.
    ra, dec = noob.get_transform("detector", "world")(1024.0, 1024.0)
    assert isinstance(ra, float) and isinstance(dec, float)


def test_spec_round_trip_rebuilds_identical_transforms() -> None:
    noob = from_gwcs(_synthetic_gwcs())
    rebuilt = NoobWCS.from_spec(noob.to_spec())

    x, y = 123.4, 987.6
    assert rebuilt.get_transform("detector", "world")(x, y) == noob.get_transform(
        "detector", "world"
    )(x, y)
    assert rebuilt.available_frames == noob.available_frames


def test_pickle_round_trip_drops_program_cache() -> None:
    import pickle

    noob = from_gwcs(_synthetic_gwcs())
    want = noob.get_transform("detector", "world")(123.4, 987.6)

    # The program cache holds Rust objects; pickling must survive it.
    rebuilt = pickle.loads(pickle.dumps(noob))
    assert rebuilt.available_frames == noob.available_frames
    assert rebuilt.get_transform("detector", "world")(123.4, 987.6) == want


def test_get_transform_accepts_gwcs_style_call_forms() -> None:
    noob = from_gwcs(_synthetic_gwcs())
    transform = noob.get_transform("detector", "world")
    want_ra, want_dec = transform(np.array([100.0, 200.0]), np.array([50.0, 60.0]))

    # Mixed array/scalar broadcasts; int arrays, lists, and int scalars coerce.
    mixed_ra, mixed_dec = transform(np.array([100.0, 200.0]), 50.0)
    assert mixed_ra.shape == (2,)
    np.testing.assert_allclose(mixed_ra[0], want_ra[0])
    int_ra, _ = transform(np.array([100, 200]), np.array([50, 60]))
    np.testing.assert_allclose(int_ra, want_ra)
    list_ra, _ = transform([100.0, 200.0], [50.0, 60.0])
    np.testing.assert_allclose(list_ra, want_ra)
    scalar_out = transform(100, 50)
    assert isinstance(scalar_out[0], float)
    np.testing.assert_allclose(scalar_out[0], want_ra[0])

    # Direct call evaluates the full forward pipeline, like gwcs.
    called_ra, _ = noob(np.array([100.0, 200.0]), np.array([50.0, 60.0]))
    np.testing.assert_allclose(called_ra, want_ra)


def _grism_pair():
    """Synthetic NIRCam row-dispersion forward/backward model pair."""
    from stdatamodels.jwst.transforms.models import (
        NIRCAMBackwardGrismDispersion,
        NIRCAMForwardRowGrismDispersion,
    )

    def spatial(c0: float, cx: float, cy: float) -> Polynomial2D:
        poly = Polynomial2D(2)
        poly.c0_0 = c0
        poly.c1_0 = cx
        poly.c0_1 = cy
        return poly

    lmodels = [
        [spatial(3.9, 1e-6, -2e-6), spatial(1.1, 2e-6, 1e-6), spatial(0.02, 0.0, 0.0)]
    ]
    xmodels = [[Polynomial1D(1, c0=-1200.0, c1=3000.0)]]
    ymodels = [[spatial(0.05, 1e-7, 0.0), spatial(0.4, 0.0, 0.0)]]
    forward = NIRCAMForwardRowGrismDispersion(
        [1], lmodels=lmodels, xmodels=xmodels, ymodels=ymodels
    )
    backward = NIRCAMBackwardGrismDispersion(
        [1], lmodels=lmodels, xmodels=xmodels, ymodels=ymodels
    )
    return forward, backward


def test_grism_forward_matches_stdatamodels_exactly() -> None:
    forward, _ = _grism_pair()
    program = WcsProgram(compile_transform(forward))

    rng = np.random.default_rng(3)
    x0 = rng.uniform(100, 1900, 200)
    y0 = rng.uniform(100, 1900, 200)
    x = x0 + rng.uniform(-1150, 1750, 200)  # t in ~[0.02, 0.98]
    order = np.ones_like(x0)

    want = forward.evaluate(x, y0, x0, y0, np.ones(1))
    got = program(x, y0, x0, y0, order)
    np.testing.assert_array_equal(got[0], want[0])  # x0 pass-through
    np.testing.assert_array_equal(got[1], want[1])  # y0 pass-through
    np.testing.assert_allclose(got[2], want[2], rtol=0, atol=1e-12)  # wavelength


def test_grism_backward_round_trips_through_exact_forward() -> None:
    forward, backward = _grism_pair()
    forward_program = WcsProgram(compile_transform(forward))
    backward_program = WcsProgram(compile_transform(backward))

    rng = np.random.default_rng(4)
    x0 = rng.uniform(100, 1900, 200)
    y0 = rng.uniform(100, 1900, 200)
    wavelength = rng.uniform(3.96, 4.95, 200)
    order = np.ones_like(x0)

    xg, yg, x0_out, y0_out, order_out = backward_program(x0, y0, wavelength, order)
    np.testing.assert_array_equal(x0_out, x0)
    np.testing.assert_array_equal(order_out, order)

    # The forward wavelength solution is exact (linear alongdisp model), so
    # it certifies our exact quadratic inversion independent of jwst's
    # sampled-grid approximation.
    _, _, lam, _ = forward_program(xg, yg, x0, y0, order)
    np.testing.assert_allclose(lam, wavelength, rtol=0, atol=1e-10)

    # And jwst's own (sampling=40 interpolated) backward agrees to within
    # its sampling error. jwst broadcasts 1-D inputs outer-product style
    # (wavelength x position); the element-wise result is the diagonal.
    want_xg = np.diagonal(backward.evaluate(x0, y0, wavelength, np.ones(1))[0])
    np.testing.assert_allclose(xg, want_xg, rtol=0, atol=0.1)


def test_from_fits_wcs_matches_astropy_tan() -> None:
    wcs = AstropyWCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [53.16, -27.78]
    wcs.wcs.crpix = [1024.5, 1024.5]
    wcs.wcs.cdelt = [-0.05 / 3600, 0.05 / 3600]
    theta = np.deg2rad(37.0)
    wcs.wcs.pc = [
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ]

    noob = from_fits_wcs(wcs)
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 2048, 400)
    y = rng.uniform(0, 2048, 400)

    want_ra, want_dec = wcs.pixel_to_world_values(x, y)
    got_ra, got_dec = noob.get_transform("detector", "world")(x, y)
    np.testing.assert_allclose(got_ra, want_ra, rtol=0, atol=1e-10)
    np.testing.assert_allclose(got_dec, want_dec, rtol=0, atol=1e-10)

    back_x, back_y = noob.get_transform("world", "detector")(
        np.asarray(want_ra), np.asarray(want_dec)
    )
    np.testing.assert_allclose(back_x, x, rtol=0, atol=1e-8)
    np.testing.assert_allclose(back_y, y, rtol=0, atol=1e-8)


def test_const1d_arithmetic_folds_to_scale_and_shift() -> None:
    # The JWST WFSS wavelength barycentric velocity correction pattern:
    # ``Identity * Const1D`` forward, ``Identity / Const1D`` in the inverse.
    correction = astropy_models.Const1D(1.0000052)
    x = np.linspace(3.9, 5.0, 64)
    for model in (
        astropy_models.Identity(1) * correction,
        astropy_models.Identity(1) / correction,
        astropy_models.Identity(1) + correction,
        astropy_models.Identity(1) - correction,
    ):
        program = WcsProgram(compile_transform(model))
        np.testing.assert_allclose(program(x)[0], model(x), rtol=1e-15)


def test_unsupported_model_raises() -> None:
    with pytest.raises(UnsupportedTransformError, match="Gaussian1D"):
        compile_transform(astropy_models.Gaussian1D())


def test_unknown_frame_and_missing_inverse_raise() -> None:
    noob = from_gwcs(_synthetic_gwcs())
    with pytest.raises(ValueError, match="Unknown frame"):
        noob.get_transform("detector", "warp")
    with pytest.raises(ValueError, match="both"):
        noob.get_transform("world", "world")


def test_rotation3d_to_gwa_matches_stdatamodels() -> None:
    from stdatamodels.jwst.transforms.models import Rotation3DToGWA

    model = Rotation3DToGWA(np.array([10.0, -5.0, 3.0]), axes_order="xyz")
    program = WcsProgram(compile_transform(model))
    rng = np.random.default_rng(6)
    x = rng.uniform(-0.3, 0.3, 100)
    y = rng.uniform(-0.3, 0.3, 100)
    z = np.sqrt(1 - x**2 - y**2)
    want = model(x, y, z)
    got = program(x, y, z)
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, w, rtol=0, atol=1e-14)


def test_grating_equation_pair_round_trips() -> None:
    from gwcs.spectroscopy import (
        AnglesFromGratingEquation3D,
        WavelengthFromGratingEquation,
    )

    groove, order = 150292.7, -1
    to_angles = AnglesFromGratingEquation3D(groove, order)
    to_lam = WavelengthFromGratingEquation(groove, order)
    prog_angles = WcsProgram(compile_transform(to_angles))
    prog_lam = WcsProgram(compile_transform(to_lam))

    rng = np.random.default_rng(7)
    lam = rng.uniform(1.5e-6, 3.0e-6, 100)  # meters, NIRSpec internal
    alpha_in = rng.uniform(-0.4, 0.4, 100)
    beta_in = rng.uniform(-0.4, 0.4, 100)
    want = to_angles(lam, alpha_in, beta_in)
    got = prog_angles(lam, alpha_in, beta_in)
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, w, rtol=0, atol=1e-14)
    # The wavelength equation matches its gwcs model (the two equations are
    # not naive mutual inverses -- the real chain reflects in between).
    want_lam = to_lam(alpha_in, beta_in)
    np.testing.assert_allclose(prog_lam(alpha_in, beta_in)[0], want_lam, rtol=1e-14)


def test_tabular_logical_and_fix_inputs_compile() -> None:
    from stdatamodels.jwst.transforms.models import Logical

    table = astropy_models.Tabular1D(
        points=np.linspace(0.0, 1.0, 50),
        lookup_table=np.linspace(0.0, 1.0, 50) ** 2,
        bounds_error=False,
        fill_value=np.nan,
    )
    program = WcsProgram(compile_transform(table))
    x = np.array([0.25, 0.777, 1.0, -0.1, 1.1])
    want = table(x)
    np.testing.assert_allclose(program(x)[0][:3], want[:3], rtol=0, atol=1e-15)
    assert np.isnan(program(x)[0][3:]).all()

    clamp = Logical("GT", 0.55, np.nan)
    program = WcsProgram(compile_transform(clamp))
    got = program(np.array([0.5, 0.6, np.nan]))[0]
    assert got[0] == 0.5 and np.isnan(got[1]) and np.isnan(got[2])

    # General model arithmetic (both operands see the same input).
    arith = astropy_models.Polynomial1D(2, c0=1.0, c1=2.0, c2=-0.5) * (
        astropy_models.Polynomial1D(1, c0=0.5, c1=1.0)
        + astropy_models.Polynomial1D(1, c0=-1.0, c1=3.0)
    )
    program = WcsProgram(compile_transform(arith))
    xs = np.linspace(-2, 2, 11)
    np.testing.assert_allclose(program(xs)[0], arith(xs), rtol=1e-13)

    # fix_inputs pins one input of a 3-input chain, IFU-chain style.
    from astropy.modeling import fix_inputs

    inner = (
        astropy_models.Shift(1.0)
        & astropy_models.Shift(2.0)
        & astropy_models.Shift(3.0)
    ) | astropy_models.Mapping((0, 2))
    fixed = fix_inputs(inner, {"x1": 4.0})
    program = WcsProgram(compile_transform(fixed))
    ys = np.linspace(0, 5, 11)
    want = fixed(xs, ys)
    got = program(xs, ys)
    np.testing.assert_allclose(got[0], want[0], rtol=1e-14)
    np.testing.assert_allclose(got[1], want[1], rtol=1e-14)


def test_regions_selector_matches_gwcs() -> None:
    from gwcs.selector import LabelMapperArray, RegionsSelector

    labels = np.zeros((16, 16), dtype=np.int64)
    labels[:, 3:8] = 1
    labels[:, 9:14] = 2
    region1 = (astropy_models.Shift(10.0) & astropy_models.Shift(0.0)) | (
        astropy_models.Identity(2) | astropy_models.Mapping((0, 1, 0))
    )
    region2 = (astropy_models.Scale(2.0) & astropy_models.Scale(3.0)) | (
        astropy_models.Identity(2) | astropy_models.Mapping((0, 1, 1))
    )
    selector = RegionsSelector(
        inputs=("x", "y"),
        outputs=("a", "b", "c"),
        selector={1: region1, 2: region2},
        label_mapper=LabelMapperArray(labels),
        undefined_transform_value=np.nan,
    )
    program = WcsProgram(compile_transform(selector))

    yy, xx = np.mgrid[0:16, 0:16].astype(float)
    want = selector(xx.ravel(), yy.ravel())
    got = program(xx.ravel(), yy.ravel())
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, np.asarray(w), rtol=0, atol=1e-14, equal_nan=True)


def test_cube_build_style_projection_chain() -> None:
    # jwst cube_build s3d output WCS: TAN deprojection + sphere rotation
    # as astropy leaf models (not the gwcs geometry chain).
    model = astropy_models.Pix2Sky_Gnomonic() | astropy_models.RotateNative2Celestial(
        150.1, 2.2, 180.0
    )
    program = WcsProgram(compile_transform(model))
    rng = np.random.default_rng(8)
    x = rng.uniform(-0.02, 0.02, 300)  # tangent-plane degrees
    y = rng.uniform(-0.02, 0.02, 300)
    want = model(x, y)
    got = program(x, y)
    np.testing.assert_allclose(got[0], want[0], rtol=0, atol=1e-11)
    np.testing.assert_allclose(got[1], want[1], rtol=0, atol=1e-11)

    inverse = model.inverse  # RotateCelestial2Native | Sky2Pix_Gnomonic
    inv_program = WcsProgram(compile_transform(inverse))
    back = inv_program(np.asarray(want[0]), np.asarray(want[1]))
    np.testing.assert_allclose(back[0], x, rtol=0, atol=1e-11)
    np.testing.assert_allclose(back[1], y, rtol=0, atol=1e-11)


def test_tan_project_nan_matches_gwcs_on_far_hemisphere() -> None:
    # The gnomonic projection is undefined > 90 deg from the tangent point.
    # gwcs (wcslib) returns NaN there; the compiled op must too, rather than
    # a finite wrong-sign value, so garbage never propagates silently.
    model = astropy_models.Sky2Pix_Gnomonic()
    program = WcsProgram(compile_transform(model))
    lon = np.array([0.0, 30.0, 90.0, 179.0])
    lat = np.array([80.0, 10.0, -1.0, -45.0])  # last two on the far side
    want = model(lon, lat)
    got = program(lon, lat)
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, np.asarray(w), rtol=0, atol=1e-11, equal_nan=True)
    assert np.isnan(got[0][2:]).all()


def _sample_correction() -> TangentCorrection:
    theta = np.deg2rad(0.01)  # 36 arcsec rotation: large enough to matter
    return TangentCorrection(
        fiducial=(150.1, 2.2),
        matrix=(
            (1.0001 * np.cos(theta), -np.sin(theta)),
            (np.sin(theta), 1.0001 * np.cos(theta)),
        ),
        offset=(2e-5, -1.5e-5),  # ~0.07 arcsec shifts
    )


def test_tangent_correction_gwcs_and_noobwcs_agree() -> None:
    correction = _sample_correction()
    wcs = _synthetic_gwcs()
    corrected_gwcs = apply_correction_to_gwcs(wcs, correction)
    corrected_noob = from_gwcs(wcs).with_tangent_correction(correction)

    rng = np.random.default_rng(9)
    x = rng.uniform(0, 2048, 300)
    y = rng.uniform(0, 2048, 300)
    want = corrected_gwcs.get_transform("detector", "world")(x, y)
    got = corrected_noob.get_transform("detector", "world")(x, y)
    np.testing.assert_allclose(got[0], want[0], rtol=0, atol=1e-11)
    np.testing.assert_allclose(got[1], want[1], rtol=0, atol=1e-11)

    # exact backward: corrected world coordinates return to the pixels
    back = corrected_noob.get_transform("world", "detector")(
        np.asarray(got[0]), np.asarray(got[1])
    )
    np.testing.assert_allclose(back[0], x, rtol=0, atol=1e-8)
    np.testing.assert_allclose(back[1], y, rtol=0, atol=1e-8)

    # and the correction actually moved things
    plain = wcs.get_transform("detector", "world")(x, y)
    assert np.max(np.abs(np.asarray(want[0]) - np.asarray(plain[0]))) > 1e-6


def test_identity_correction_is_a_noop() -> None:
    wcs = _synthetic_gwcs()
    noob = from_gwcs(wcs)
    corrected = noob.with_tangent_correction(TangentCorrection(fiducial=(150.1, 2.2)))
    x, y = 512.3, 1400.7
    got = corrected.get_transform("detector", "world")(x, y)
    want = noob.get_transform("detector", "world")(x, y)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-11)


def test_corrected_gwcs_serializes_with_standard_tags(tmp_path) -> None:
    # The interoperability contract: a corrected gwcs must round-trip
    # through ASDF using astropy/gwcs standard tags only.
    import asdf

    corrected = apply_correction_to_gwcs(_synthetic_gwcs(), _sample_correction())
    path = tmp_path / "corrected_wcs.asdf"
    asdf.AsdfFile({"wcs": corrected}).write_to(path)
    with asdf.open(path) as af:
        reloaded = af["wcs"]
        x, y = np.array([100.0, 1800.0]), np.array([900.0, 40.0])
        np.testing.assert_allclose(
            reloaded.get_transform("detector", "world")(x, y),
            corrected.get_transform("detector", "world")(x, y),
            rtol=0,
            atol=1e-13,
        )
