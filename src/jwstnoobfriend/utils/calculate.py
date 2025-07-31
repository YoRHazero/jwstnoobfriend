import cv2
import numpy as np
from scipy import stats
from typing import Literal

__all__ = ['mad_clipped_stats', 'gaussian_fill_nan', 'segmentation_mask']

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


def gaussian_fill_nan(data: np.ndarray, 
                      mask_valid: np.ndarray | None = None, 
                      kernel_radius_x: int = 15, 
                      kernel_radius_y: int | None = None,
                      sigma_x: float | None = None,
                      sigma_y: float | None = None,
                      fill_outer_nan: Literal['nan', 'zero', 'mean', 'median', 'nearest'] = 'nan'
                      ):
    """
    Fill NaN values in the data using Gaussian interpolation.
    
    Parameters
    ----------
    data : np.ndarray
        The input data array with NaN values to be filled.
    mask_valid : np.ndarray
        A boolean mask indicating which elements are valid (True for valid data).
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
    if mask_valid is None:
        mask_valid = ~np.isnan(data)
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

    data_filled = np.where(central_mask_valid, data_f32, 0.0)

    data_blurred = cv2.GaussianBlur(data_filled, (kernel_size_x, kernel_size_y), sigma_x)
    weights_blurred = cv2.GaussianBlur(mask_valid_f32, (kernel_size_x, kernel_size_y), sigma_x)

    interpolate = np.where(weights_blurred > 0, data_blurred / weights_blurred, np.nanmedian(data))

    data_filled = np.where(central_mask_valid, data_f32, interpolate)
    data[row_start:row_end, col_start:col_end] = data_filled

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
        A boolean mask indicating the segmented regions in the data.
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
