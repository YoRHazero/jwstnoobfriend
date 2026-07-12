"""Focused validation of shared spectrum-fit visualization adapters."""

from __future__ import annotations

import numpy as np
import pytest

from noobfriend.inference.spectrum import NoobLine, NoobSpectrum
from noobfriend.inference.spectrum.visualization.fit import _resolve_2d_input
from noobfriend.inference.spectrum.visualization.prediction import dense_wavelength
from noobfriend.inference.spectrum.visualization.style import resolve_component_colors


def test_dense_wavelength_subdivides_nonuniform_intervals() -> None:
    dense = dense_wavelength(np.array([1.0, 3.0, 7.0]), 4)

    assert dense.size == 9
    assert dense.tolist() == pytest.approx(
        [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0]
    )
    assert dense.flags.writeable is False


@pytest.mark.parametrize("value", [0, -1])
def test_dense_wavelength_rejects_nonpositive_oversampling(value: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        dense_wavelength(np.array([1.0, 2.0]), value)


def test_retained_2d_source_is_used_automatically() -> None:
    wavelength = np.linspace(1.0, 2.0, 5)
    source = NoobSpectrum.from_2d(
        np.ones((4, 5)),
        np.full((4, 5), 0.2),
        obs=wavelength,
        collapse_window=(1, 3),
    )

    resolved = _resolve_2d_input(
        source,
        flux_2d=None,
        spatial=None,
        dispersion=None,
        spatial_window=None,
    )

    assert resolved is not None
    assert resolved.flux.shape == (4, 5)
    assert resolved.spatial_window == pytest.approx((0.5, 2.5))

    custom_spatial = np.array([-3.0, -1.0, 2.0, 6.0])
    custom = _resolve_2d_input(
        source,
        flux_2d=None,
        spatial=custom_spatial,
        dispersion=None,
        spatial_window=None,
    )
    assert custom is not None
    assert custom.spatial_window == pytest.approx((-2.0, 4.0))


def test_external_2d_axis_is_inferred_and_square_input_is_ambiguous() -> None:
    spectrum = NoobSpectrum(
        np.ones(5),
        np.full(5, 0.2),
        obs=np.linspace(1.0, 2.0, 5),
    )

    resolved = _resolve_2d_input(
        spectrum,
        flux_2d=np.ones((5, 3)),
        spatial=None,
        dispersion=None,
        spatial_window=None,
    )
    assert resolved is not None
    assert resolved.flux.shape == (3, 5)

    with pytest.raises(ValueError, match="cannot infer"):
        _resolve_2d_input(
            spectrum,
            flux_2d=np.ones((5, 5)),
            spatial=None,
            dispersion=None,
            spatial_window=None,
        )


def test_mle_result_plot_automatically_uses_retained_2d_source() -> None:
    wavelength = np.linspace(4990.0, 5010.0, 21)
    source = NoobSpectrum.from_2d(
        np.ones((3, wavelength.size)),
        np.full((3, wavelength.size), 0.2),
        obs=wavelength,
        collapse_window=(0, 3),
    )
    line = (
        NoobLine("line", obs=5000.0)
        .flux(override=0.0)
        .fwhm(override=180.0)
        .center(delta_v_kms=0.0)
    )

    result = source.prepare([line], continuum_order=0).mle()
    figure = result.plot(size=500)

    assert len(figure.axes) == 3

    import matplotlib.pyplot as plt

    plt.close(figure)


def test_component_color_overrides_use_public_component_ids() -> None:
    colors = resolve_component_colors(("Ha.narrow", "Ha.broad"), {"Ha.broad": "pink"})

    assert colors["Ha.narrow"] == "#4C78A8"
    assert colors["Ha.broad"] == "pink"
    with pytest.raises(KeyError, match="valid components"):
        resolve_component_colors(("Ha.narrow",), {"missing": "red"})
