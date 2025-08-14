import numpy as np
from jwstnoobfriend.utils.calculate import mad_clipped_stats, gaussian_smoothing, segmentation_mask
from jwstnoobfriend.navigation import JwstInfo

def nircam_1f_noise(
    info: JwstInfo,
    stage: str = '2a',
    seg_mask: np.ndarray | None = None,
    bin_size: int = 1,
):
    """
    Calculate the 1F noise for NIRCam data.
    
    Parameters
    ----------
    info : JwstInfo
        The JwstInfo object containing the data and metadata.
    seg_mask : np.ndarray | None, optional
        A segmentation mask to apply to the data. If None, a mask will be created.
    bin_size : int, optional
        The bin size for the data. Default is 1, meaning no binning. According to our observations, 
        binning is not necessary for 1F noise calculation.
    """
    
    # Create a segmentation mask if not provided
    data = info[stage].data.copy()
    if data.shape[0] % bin_size != 0 or data.shape[1] % bin_size != 0:
        raise ValueError(f"Data shape {data.shape} is not divisible by bin size {bin_size}.")
    
    # Apply Gaussian fill to NaN values
    match info.pupil:
        case "CLEAR":
            data_conv = gaussian_smoothing(data, np.isnan(data))
        case "GRISMR":
            data_conv = gaussian_smoothing(data, np.isnan(data), kernel_radius_y=1)
        case "GRISMC":
            data_conv = gaussian_smoothing(data, np.isnan(data), kernel_radius_x=1)
        case _:
            raise ValueError(f"Unsupported pupil type: {info.pupil} for NIRCam 1F noise calculation.")
    if seg_mask is None:
        seg_mask = segmentation_mask(data_conv)

    # Apply the segmentation mask (source) and dq mask (bad pixels)
    dq = info[stage].dq
    mask = (dq != 0) | seg_mask
    data[mask] = np.nan
    
    median = np.nanmedian(data)
    
    ## when binsize > 1, the logic needs to be refined
    
    match info.pupil:
        # For CLEAR image, we construct a correction image based on row and column medians
        case "CLEAR":
            # We don't subtract the background.
            collapsed_rows = np.nanmedian(data - median, axis=1)
            collapsed_cols = np.nanmedian(data - median, axis=0)
            
            collapsed_cols_binned = [np.nanmedian(collapsed_cols[idx:idx+bin_size])
                                    for idx in np.arange(0, len(collapsed_cols), bin_size)]
            collapsed_rows_binned = [np.nanmedian(collapsed_rows[idx:idx+bin_size])
                                   for idx in np.arange(0, len(collapsed_rows), bin_size)]
            correction_image = np.tile(np.repeat(collapsed_cols_binned, bin_size), (data.shape[0], 1)) + \
                            np.swapaxes(np.tile(collapsed_rows, (data.shape[1], 1)), 0, 1) 
        # For GRISMR image, we construct a correction image only based on column medians
        case "GRISMR":
            collapsed_cols = np.nanmedian(data - median, axis=0)
            collapsed_cols_binned = [np.nanmedian(collapsed_cols[idx:idx+bin_size])
                                    for idx in np.arange(0, len(collapsed_cols), bin_size)]
            correction_image = np.tile(np.repeat(collapsed_cols_binned, bin_size), (data.shape[0], 1))
        # For GRISMC image, we construct a correction image only based on row medians
        case "GRISMC":
            collapsed_rows = np.nanmedian(data - median, axis=1)
            collapsed_rows_binned = [np.nanmedian(collapsed_rows[idx:idx+bin_size])
                                   for idx in np.arange(0, len(collapsed_rows), bin_size)]
            correction_image = np.tile(np.repeat(collapsed_rows_binned, bin_size), (data.shape[1], 1)).T
        case _:
            raise ValueError(f"Unsupported pupil type: {info.pupil} for NIRCam 1F noise calculation.")
            
    return correction_image