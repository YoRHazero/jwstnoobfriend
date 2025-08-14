import cv2
import numpy as np
from scipy import stats
from scipy.interpolate import SmoothBivariateSpline
from typing import Literal

__all__ = ['mad_clipped_stats', 'gaussian_smoothing', 'segmentation_mask']

def mad_clipped_stats(data: np.ndarray,
                      mask: np.ndarray | None, 
                      sigma: float = 3.0,
                      ) -> tuple[float, float, float, float]:
    """
    Calculate the median, standard deviation, and median absolute deviation (MAD) of the data,
    clipping outliers based on the MAD method.
    
    Parameters
    ----------
    data : np.ndarray
        The input data array.
    mask : np.ndarray | None
        A boolean mask indicating which elements to ignore (True for invalid data).
        If None, all NaN values in the data will be masked.
    sigma : float, optional
        The number of standard deviations to use for clipping outliers. Default is 3.0
    
    Returns
    -------
    tuple[float, float, float, float]
        A tuple containing the mean, standard deviation, median, and MAD of the clipped data.
    """
    if mask is None:
        mask = np.isnan(data)
        
    valid_data = data[~mask]
    median_original = np.median(valid_data)
    mad_original = stats.median_abs_deviation(valid_data, scale='normal')

    mask_to_clip = np.abs(valid_data - median_original) > sigma * mad_original
    data_clipped = valid_data[~mask_to_clip]
    return np.mean(data_clipped), np.std(data_clipped), np.median(data_clipped), stats.median_abs_deviation(data_clipped, scale='normal')


def gaussian_smoothing(data: np.ndarray, 
                      mask_invalid: np.ndarray | None = None, 
                      kernel_radius_x: int = 15, 
                      kernel_radius_y: int | None = None,
                      sigma_x: float | None = None,
                      sigma_y: float | None = None,
                      fill_outer_nan: Literal['nan', 'zero', 'mean', 'median', 'nearest'] = 'nan'
                      ):
    """
    Gaussian smoothing of the input data array, with masking the NaN values.
    
    Parameters
    ----------
    data : np.ndarray
        The input data array with NaN values to be filled.
    mask_invalid : np.ndarray
        A boolean mask indicating which elements are invalid (True for invalid data).
    kernel_radius_x : int, optional
        The radius of the Gaussian kernel to use for interpolation in the x, axis 1, and width direction. Default is 15.
    kernel_radius_y : int, optional
        The radius of the Gaussian kernel to use for interpolation in the y, axis 0, and height direction. Default is 15.
    sigma_x : float, optional
        The standard deviation of the Gaussian kernel in the x, axis 1, and width direction. If None, it will be set to half of the kernel radius.
    sigma_y : float, optional
        The standard deviation of the Gaussian kernel in the y, axis 0, and height direction. If None, it will be set to half of the kernel radius.
    fill_outer_nan : Literal['nan', 'zero', 'mean', 'median', 'nearest'], optional
        How to fill the outer NaN values after interpolation. Options are:
        - 'nan': Keep outer NaN values as is.
        - 'zero': Fill outer NaN values with zero.
        - 'mean': Fill outer NaN values with the mean of the data.
        - 'median': Fill outer NaN values with the median of the data.
        - 'nearest': Fill outer NaN values with the nearest valid value (not implemented in this version).
    
    Returns
    -------
    np.ndarray
        The data array with NaN values filled using Gaussian interpolation.
    """
    data = data.copy()
    if mask_invalid is None:
        mask_valid = ~np.isnan(data)
    else:
        if not np.array_equal(mask_invalid & np.isnan(data), mask_invalid):
            raise ValueError("Custom mask_invalid must mask NaN values in data.")
        mask_valid = ~mask_invalid
    if kernel_radius_y is None:
        kernel_radius_y = kernel_radius_x
    kernel_size_x = 2 * kernel_radius_x + 1
    kernel_size_y = 2 * kernel_radius_y + 1
    if sigma_x is None:
        sigma_x = kernel_radius_x / 2
    if sigma_y is None:
        sigma_y = kernel_radius_y / 2
        

    # Exclude the outer shell of the data which is all NaN
    mask_nonnan = ~np.isnan(data)
    valid_rows = np.any(mask_nonnan, axis=1)
    valid_cols = np.any(mask_nonnan, axis=0)

    row_indices = np.where(valid_rows)[0]
    col_indices = np.where(valid_cols)[0]

    row_start, row_end = row_indices[0], row_indices[-1] + 1
    col_start, col_end = col_indices[0], col_indices[-1] + 1
    central_data = data[row_start:row_end, col_start:col_end]
    central_mask_valid = mask_valid[row_start:row_end, col_start:col_end]
    
    # Use cv2 to apply Gaussian blur
    data_f32 = central_data.astype(np.float32)
    mask_valid_f32 = (central_mask_valid).astype(np.float32)
    # Fill NaN values with zero for the Gaussian blur of cv2
    data_filled = np.where(central_mask_valid, data_f32, 0.0)
    # Apply Gaussian blur to the data and the mask
    data_blurred = cv2.GaussianBlur(data_filled, (kernel_size_x, kernel_size_y), sigma_x)
    weights_blurred = cv2.GaussianBlur(mask_valid_f32, (kernel_size_x, kernel_size_y), sigma_x)

    data_smoothed = np.where(weights_blurred > 0, data_blurred / weights_blurred, np.nanmedian(data))

    #data_filled = np.where(central_mask_valid, data_f32, data_smoothed)
    data[row_start:row_end, col_start:col_end] = data_smoothed

    # Deal with outer NaN values
    match fill_outer_nan:
        case 'nan':
            pass  # Keep outer NaN values as is
        case 'zero':
            data[np.isnan(data)] = 0.0
        case 'mean':
            mean_value = np.nanmean(data)
            data[np.isnan(data)] = mean_value
        case 'median':
            median_value = np.nanmedian(data)
            data[np.isnan(data)] = median_value
        case 'nearest':
            # Not implemented in this version
            pass
    return data

def segmentation_mask(data: np.ndarray, 
                      factor: float = 2, 
                      min_pixels_connected: int = 10,
                      kernel_radius: int = 4, 
                      sigma: float | None = None) -> np.ndarray: # data.shape, bool
    """
    Create a segmentation mask for the input data based on the median absolute deviation (MAD) method
    and Gaussian smoothing.
    
    Parameters
    ----------
    data : np.ndarray
        The input data array.
    factor : float, optional
        The factor to multiply the MAD for thresholding. Default is 1.5.
    min_pixels_connected : int, optional
        The minimum number of connected pixels to consider a segment valid. Default is 10.
    kernel_radius : int, optional
        The radius of the Gaussian kernel to use for smoothing the segmentation mask. Default is 4.
    sigma : float | None, optional
        The standard deviation of the Gaussian kernel. If None, it will be set to half of
        the kernel radius.
    
    Returns
    -------
    np.ndarray
        A boolean mask indicating the segmented regions in the data, where True indicates a segment and False indicates background.
    """

    mean, std, median, mad = mad_clipped_stats(data, mask=np.isnan(data))
    threshold = factor * mad
    data_med_subed = data - median
    
    segmentation = (data_med_subed > threshold)
    
    num_labels, labels = cv2.connectedComponents(segmentation.astype(np.uint8), connectivity=8)
    unique_labels, counts = np.unique(labels, return_counts=True)
    large_labels = unique_labels[(counts >= min_pixels_connected) & (unique_labels != 0)]
    segmentation = np.isin(labels, large_labels).astype(bool)
    
    kernel_size = 2 * kernel_radius + 1
    if sigma is None:
        sigma = kernel_radius / 2
    blurred_segmentation = cv2.GaussianBlur(segmentation.astype(np.float32), (kernel_size, kernel_size), sigma)
    
    return (blurred_segmentation > 0.05).astype(bool)

def background_model(
    data: np.ndarray, 
    mask: np.ndarray | None = None, 
    fraction_non_masked: float = 0.5,
    block_size: int = 128,
    block_xsize: int | None = None,
    block_ysize: int | None = None,
    **kwargs
) -> np.ndarray:
    """
    Create a background model for the input data by calculating the median of non-masked values
    in blocks of the specified size.
    
    Parameters
    ----------
    data : np.ndarray
        The input data array.
    mask : np.ndarray | None
        A boolean mask indicating which elements to ignore (True for invalid data).
        If None, all NaN values in the data will be masked.
    fraction_non_masked : float, optional
        The fraction of non-masked pixels required to consider a block valid. Default is 0.5.
    block_size : int, optional
        The size of the blocks to process. Default is 128.
    block_xsize : int | None, optional
        The x size of the blocks. If None, it will be set to block_size.
    block_ysize : int | None, optional
        The y size of the blocks. If None, it will be set to block_size.
    **kwargs
        Additional keyword arguments to pass to the `SmoothBivariateSpline` constructor.
    
    Returns
    -------
    np.ndarray
        The background model as a 2D array.
    """
    
    if mask is None:
        mask = np.zeros_like(data, dtype=bool)
        
    if block_xsize is None:
        block_xsize = block_size
    if block_ysize is None:
        block_ysize = block_size
    
    height, width = data.shape
    if width % block_xsize != 0 or height % block_ysize != 0:
        raise ValueError("Data dimensions must be divisible by block_xsize and block_ysize.")
    n_block_x = width // block_xsize
    n_block_y = height // block_ysize
    median_grid = np.full((n_block_y, n_block_x), np.nan)
    for bx in range(n_block_x):
        for by in range(n_block_y):
            block_data = data[by * block_ysize:(by + 1) * block_ysize, bx * block_xsize:(bx + 1) * block_xsize]
            block_mask = mask[by * block_ysize:(by + 1) * block_ysize, bx * block_xsize:(bx + 1) * block_xsize]
            if np.sum(block_mask) > fraction_non_masked * block_size ** 2:
                continue
            median_grid[by, bx] = np.nanmedian(block_data[~block_mask])
    
    x_coords = np.arange(n_block_x // 2, width, block_xsize)
    y_coords = np.arange(n_block_y // 2, height, block_ysize)
    
    xx, yy = np.meshgrid(x_coords, y_coords)
    
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    z_flat = median_grid.ravel()

    x_valid = x_flat[~np.isnan(z_flat)]
    y_valid = y_flat[~np.isnan(z_flat)]
    z_valid = z_flat[~np.isnan(z_flat)]

    spline = SmoothBivariateSpline(x_valid, y_valid, z_valid, **kwargs)
    x_full, y_full = np.meshgrid(np.arange(width), np.arange(height))
    background_model = spline.ev(x_full.ravel(), y_full.ravel()).reshape(height, width)

    return background_model