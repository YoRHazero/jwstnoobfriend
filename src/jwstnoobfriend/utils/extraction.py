import numpy as np
from typing import Literal
import gwcs
from scipy.ndimage import map_coordinates

## Grism Extraction Utilities
def wave_grid_centers_to_edges(
    centers: np.ndarray,
) -> np.ndarray:
    """
    Convert an array of wavelength grid centers to edges.
    
    Parameters
    ----------
    centers : np.ndarray
        Array of wavelength grid centers.
    """
    if len(centers) < 2:
        raise ValueError("At least two centers are required to compute edges.")
    mid = 0.5 * (centers[1:] + centers[:-1])
    first = centers[0] - 0.5 * (centers[1] - centers[0])
    last = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    edges = np.concatenate(([first], mid, [last]))
    return edges

def auto_build_grid(
    wavelengths_list: list[np.ndarray],
    strategy: Literal['median', 'mean'] = 'median',
) -> np.ndarray:
    """
    Automatically build a common wavelength grid from a list of wavelength arrays.
    
    Parameters
    ----------
    wavelengths_list : list[np.ndarray]
        List of wavelength arrays from different spectra.
    strategy : Literal['median', 'mean'], optional
        Strategy to use for determining the common grid spacing, by default 'median'.
    """
    
    deltas = []
    for wave_grid in wavelengths_list:
        if len(wave_grid) < 2:
            continue
        delta = np.diff(wave_grid)
        deltas.append(delta)
    
    deltas = np.concatenate(deltas)
    if len(deltas) == 0:
        raise ValueError("Not enough data to determine grid spacing.")
    delta_lambda = np.median(deltas) if strategy == 'median' else np.mean(deltas)

    lambda_min = min(np.min(wave_grid) for wave_grid in wavelengths_list)
    lambda_max = max(np.max(wave_grid) for wave_grid in wavelengths_list)

    num_points = int(np.floor((lambda_max - lambda_min) / delta_lambda)) + 1

    common_grid = lambda_min + np.arange(num_points) * delta_lambda 

    return common_grid

def flux_conserved_resample(
    spec_2d: np.ndarray,
    err_2d: np.ndarray,
    wavelengths_original: np.ndarray,
    wavelengths_resample: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample 2D spectra onto a new wavelength grid while conserving flux.
    
    Parameters
    ----------
    spec_2d : np.ndarray
        2D array of spectra with shape (ny, nx).
    err_2d : np.ndarray
        2D array of errors with shape (ny, nx).
    wavelengths_original : np.ndarray
        Original wavelength grid.
    wavelengths_resample : np.ndarray
        New wavelength grid to resample onto.
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Resampled spectra and errors with shape (ny, nout).
    """
    ny, nin = spec_2d.shape
    nout = len(wavelengths_resample)
    
    edges_in = wave_grid_centers_to_edges(wavelengths_original)
    edges_out = wave_grid_centers_to_edges(wavelengths_resample)
    
    edges_in_low = edges_in[:-1][:, np.newaxis]
    edges_in_high = edges_in[1:][:, np.newaxis]
    edges_out_low = edges_out[:-1][np.newaxis, :]
    edges_out_high = edges_out[1:][np.newaxis, :]
    
    edges_low = np.maximum(edges_in_low, edges_out_low)
    edges_high = np.minimum(edges_in_high, edges_out_high)
    overlap = np.clip(edges_high - edges_low, 0, None)
    
    delta_lambda_out = (edges_out[1:] - edges_out[:-1])
    constant = overlap / delta_lambda_out[np.newaxis, :]

    good = np.isfinite(spec_2d) & np.isfinite(err_2d) & (err_2d > 0)

    spec_filled = np.where(good, spec_2d, 0.0)
    err_filled = np.where(good, err_2d, 0)
    
    out_spec = spec_filled @ constant
    out_var = (err_filled**2) @ (constant * constant)
    out_err = np.sqrt(out_var)
    
    contribution = good.astype(float) @ constant
    mask_no_contrib = (contribution <= 0)
    out_spec[mask_no_contrib] = np.nan
    out_err[mask_no_contrib] = np.nan
    
    return out_spec, out_err

def combine_2d(
    spec_2d_list: list[np.ndarray],
    err_2d_list: list[np.ndarray],
    method: Literal['mean', 'median'] = 'mean',
    sigma_clip: float | None = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Combine multiple 2D spectra using mean or median with optional sigma clipping.
    
    Parameters
    ----------
    spec_2d_list : list[np.ndarray]
        List of 2D spectra arrays to combine.
    err_2d_list : list[np.ndarray]
        List of 2D error arrays corresponding to the spectra.
    method : Literal['mean', 'median'], optional
        Method to use for combining, by default 'mean'.
    sigma_clip : float | None, optional
        Sigma clipping threshold for method of 'median', by default 3.0, if None, no clipping is applied.
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Combined spectra and errors.
    """
    
    if method == 'mean':
        spec_stack = np.stack(spec_2d_list, axis=0)
        err_stack = np.stack(err_2d_list, axis=0)
        
        weights = np.where(np.isfinite(err_stack) & (err_stack > 0), 1.0 / err_stack**2, 0.0)
        numerator = np.nansum(spec_stack * weights, axis=0)
        denominator = np.nansum(weights, axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            mean_spec = np.where(denominator > 0, numerator / denominator, np.nan)
        if sigma_clip is not None and sigma_clip > 0:
            
            var_weighted = np.nansum(weights * (spec_stack - mean_spec[np.newaxis, :, :])**2, axis=0) / np.clip(denominator, 1e-30, None)
            std_weighted = np.sqrt(var_weighted)
            
            mask = np.abs(spec_stack - mean_spec[np.newaxis, :, :]) > (sigma_clip * std_weighted[np.newaxis, :, :])
            weights = np.where(mask, 0.0, weights)
            numerator = np.nansum(spec_stack * weights, axis=0)
            denominator = np.nansum(weights, axis=0)
            mean_spec = np.where(denominator > 0, numerator / denominator, np.nan)
        combined_err = np.where(denominator > 0, np.sqrt(1.0 / denominator), np.nan)
        return mean_spec, combined_err
    
    elif method == 'median':
        spec_stack = np.stack(spec_2d_list, axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            median_spec = np.nanmedian(spec_stack, axis=0)
            mad = np.nanmedian(np.abs(spec_stack - median_spec[np.newaxis, :, :]), axis=0)
        N_eff = np.sum(np.isfinite(spec_stack), axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            combined_err = 1.4826 * mad / np.sqrt(N_eff)
        
        all_nan = (N_eff == 0)
        combined_err[all_nan] = np.nan
        median_spec[all_nan] = np.nan
        return median_spec, combined_err
    
def resample_and_combine_spectra_2d(
    spec_2d_list: list[np.ndarray],
    err_2d_list: list[np.ndarray],
    wavelengths_list: list[np.ndarray],
    resample_grid: np.ndarray | None = None,
    grid_strategy: Literal['median', 'mean'] = 'median',
    combine_method: Literal['mean', 'median'] = 'median',
) -> dict[str, np.ndarray]:
    """
    Resample and combine multiple 2D spectra onto a common wavelength grid.
    
    Parameters
    ----------
    spec_2d_list : list[np.ndarray]
        List of 2D spectra arrays to resample and combine.
    err_2d_list : list[np.ndarray]
        List of 2D error arrays corresponding to the spectra.
    wavelengths_list : list[np.ndarray]
        List of wavelength arrays corresponding to the spectra.
    resample_grid : np.ndarray | None, optional
        Wavelength grid to resample onto, by default None, which triggers automatic grid creation.
    grid_strategy : Literal['median', 'mean'], optional
        Strategy for automatic grid creation, by default 'median'.
    combine_method : Literal['mean', 'median'], optional
        Method to use for combining, by default 'mean'.
        
    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing 'wavelength', 'spectrum_2d', and 'error_2d'.
    """
    
    if not (len(spec_2d_list) == len(err_2d_list) == len(wavelengths_list)):
        raise ValueError("Input lists must have the same length.")
    
    if resample_grid is None:
        resample_grid = auto_build_grid(wavelengths_list, strategy=grid_strategy)
    else:
        resample_grid = np.sort(resample_grid)
    
    nout = len(resample_grid)
    resampled_specs = []
    resampled_errs = []
    ny_ref = None
    
    for spec_2d, err_2d, wave_grid in zip(spec_2d_list, err_2d_list, wavelengths_list):
        if spec_2d.shape != err_2d.shape:
            raise ValueError("Spectra and error arrays must have the same shape.")
        if ny_ref is None:
            ny_ref = spec_2d.shape[0]
        elif spec_2d.shape[0] != ny_ref:
            raise ValueError("All spectra must have the same number of spatial rows.")
        
        resampled_spec, resampled_err = flux_conserved_resample(
            spec_2d, err_2d, wave_grid, resample_grid
        )
        resampled_specs.append(resampled_spec)
        resampled_errs.append(resampled_err)
        
    if combine_method == 'mean':
        combined_spec, combined_err = combine_2d(
            resampled_specs, resampled_errs, method='mean'
        )
    elif combine_method == 'median':
        combined_spec, combined_err = combine_2d(
            resampled_specs, resampled_errs, method='median'
        )
        
    return {
        'wavelength': resample_grid,
        'spectrum_2d': combined_spec,
        'error_2d': combined_err
    }
    
## Clear Image Extraction Utilities
def reproject_image(
    wcs_ref: gwcs.WCS,
    wcs_to_extract: gwcs.WCS,
    data_to_extract: np.ndarray,
    shape_out: tuple[int, int],
    center: tuple[int, int],
    ref_type: Literal['GRISM', 'CLEAR'] = 'GRISM',
    extract_type: Literal['GRISM', 'CLEAR'] = 'CLEAR',
    order: int = 3,
) -> np.ndarray:
    """
    Reproject an image from one WCS frame to another using WCS transformations. The bounding_box of wcs_ref is assumed to be None!
    
    Parameters
    ----------
    wcs_ref : gwcs.WCS
        WCS of the reference frame.
    wcs_to_extract : gwcs.WCS
        WCS of the clear image frame.
    data_to_extract : np.ndarray
        2D array of data to extract from.
    shape_out : tuple[int, int]
        Shape of the output extracted image. shape[0] is ny, shape[1] is nx.
    center : tuple[int, int]
        Center position (y | axis0, x | axis1) in the wcs_ref frame to extract around.
    ref_type : Literal['GRISM', 'CLEAR'], optional
        Type of the reference WCS, by default 'GRISM'.
    extract_type : Literal['GRISM', 'CLEAR'], optional
        Type of the WCS to extract to, by default 'CLEAR'.
    order : int, optional
        Interpolation order for map_coordinates, by default 3.
        
    Returns
    -------
    np.ndarray
        Extracted data, with shape shape_out, centered at (shape_out[1]//2, shape_out[0]//2) of the wcs_ref frame.
    """
    if ref_type == 'GRISM':
        w2d = wcs_ref.fix_inputs({"x": 0, "y": 0, "order": 1})
    elif ref_type == 'CLEAR':
        w2d = wcs_ref
    else:
        raise ValueError("ref_type must be either 'GRISM' or 'CLEAR'.")    
    ny_out, nx_out = shape_out
    y_indices, x_indices = np.mgrid[0:ny_out, 0:nx_out]
    center_y, center_x = center
    y0 = y_indices + (center_y - ny_out // 2)
    x0 = x_indices + (center_x - nx_out // 2)

    if ref_type == 'GRISM':
        lon, lat, *_ = w2d(x0, y0)
    elif ref_type == 'CLEAR':
        lon, lat = w2d(x0, y0)
    
    if extract_type == 'CLEAR':
        xin, yin = wcs_to_extract.get_transform('world', 'detector')(lon, lat)
    elif extract_type == 'GRISM':
        xin, yin, _, _ = wcs_to_extract.get_transform('world', 'detector')(lon, lat, 2, 1)
    else:
        raise ValueError("extract_type must be either 'GRISM' or 'CLEAR'.")
    mask = np.isfinite(data_to_extract).astype(float)
    data_filled = np.nan_to_num(data_to_extract, nan=0.0)
    weight = map_coordinates(
        mask,
        [yin, xin],
        mode = "constant",
        cval = 0.0,
        order = 1,
    )
    value = map_coordinates(
        data_filled,
        [yin, xin],
        mode = "constant",
        cval = 0.0,
        order = order,
        prefilter= (order > 1),
    )
    with np.errstate(invalid='ignore', divide='ignore'):
        extracted_image = np.where(weight > 0, value / weight, np.nan)
    return extracted_image

def reproject_by_coordinate(
    lon: np.ndarray,
    lat: np.ndarray,
    wcs_to_extract: gwcs.WCS,
    data_to_extract: np.ndarray,
    extract_type: Literal['GRISM', 'CLEAR'] = 'CLEAR',
    order: int = 1,
) -> np.ndarray:
    """
    Reproject data values at given world coordinates using WCS transformations.
    
    Parameters
    ----------
    lon : np.ndarray
        Array of RA of each pixel of the output image. Should be the 2D arrays of the transform result of np.mgrid.
    lat : np.ndarray
        Array of Dec of each pixel of the output image. Should be the 2D arrays of the transform result of np.mgrid.
    wcs_to_extract : gwcs.WCS
        WCS of the frame to extract to.
    data_to_extract : np.ndarray
        2D array of data to extract from.
    extract_type : Literal['GRISM', 'CLEAR'], optional
        Type of the WCS to extract to, by default 'CLEAR'.
    order : int, optional
        Interpolation order for map_coordinates, by default 1.
        
    Returns
    -------
    np.ndarray
        Reprojected data values at the specified world coordinates.
    """
    if extract_type == 'CLEAR':
        xin, yin = wcs_to_extract.get_transform('world', 'detector')(lon, lat)
    elif extract_type == 'GRISM':
        xin, yin, _, _ = wcs_to_extract.get_transform('world', 'detector')(lon, lat, 2, 1)
    else:
        raise ValueError("extract_type must be either 'GRISM' or 'CLEAR'.")
    
    mask = np.isfinite(data_to_extract).astype(float)
    data_filled = np.nan_to_num(data_to_extract, nan=0.0)
    weight = map_coordinates(
        mask,
        [yin, xin],
        mode = "constant",
        cval = 0.0,
        order = 1,
    )
    value = map_coordinates(
        data_filled,
        [yin, xin],
        mode = "constant",
        cval = 0.0,
        order = order,
        prefilter= (order > 1),
    )
    with np.errstate(invalid='ignore', divide='ignore'):
        reprojected_values = np.where(weight > 0, value / weight, np.nan)
    return reprojected_values