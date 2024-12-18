from .box import *
from .manager import *


from photutils.detection import DAOStarFinder
from photutils.segmentation import detect_sources, detect_threshold
from jwst.outlier_detection.utils import gwcs_blot
from astropy.stats import sigma_clipped_stats, sigma_clip
from astropy.convolution import convolve, Gaussian2DKernel
def custom_1f_noise(datamodel, bin_size = 4):
    """
    Construct 1/f noise correction image.
    
    Parameters
    ------------
    datamodel: jwst.datamodels.CubeModel
        The input datamodel.
        
    bin_size: int
        The bin size for collapsing the 1/f noise.
        
    Returns
    ------------
    numpy.ndarray
        The 1/f noise correction image.
        
    Examples
    ------------
    Subtracts the 1/f noise before running the Image2Pipeline:
    >>> rate_model = dm.open(rate_path)
    >>> rate_model.data =  rate_model.data - custom_1f_noise(datamodel)
    >>> cal_model = Image2Pipeline.call(rate_model, save_results = True, output_dir = cal_folder,
    >>>                                 steps = {
    >>>                                         'resample': {'skip': True},
    >>>                                         }
    >>>                                 )[0]
    Not work well for WFSS data.
    """
    data = datamodel.data
    dq = datamodel.dq
    data_masked = np.ma.masked_array(data = data, mask = (dq != 0))

    ############# Construct source segmentation #############
    mean, median, stddev = sigma_clipped_stats(data_masked)
    threshold = 1 * stddev
    data_med_subed = data_masked - median
    data_conv = convolve(data_med_subed, Gaussian2DKernel(4))
    segmap_orig = detect_sources(data_conv, threshold, npixels = 9).data.astype(int)
    segmap_orig[segmap_orig!=0]= 1
    segmap = convolve(segmap_orig, Gaussian2DKernel(4))
    segmap[segmap<0.05] = 0
    segmap[segmap>=0.05] = 1
    
    ############# Construct 1/f noise #############
    data_bkg = np.copy(data)
    data_bkg[(dq!=0) | (segmap!=0)] = np.nan
    clipped = sigma_clip(data_bkg, sigma =3)
    data_bkg[clipped.mask == True] = np.nan
    clipped_median = np.nanmedian(data_bkg)
    collapsed_rows = np.nanmedian(data_bkg - clipped_median, axis = 1)
    collapsed_cols = np.nanmedian(data_bkg - clipped_median, axis = 0) 
    collapsed_cols_binned = [np.nanmedian(collapsed_cols[idx:idx+bin_size]) 
                                for idx in np.arange(0, len(collapsed_cols), bin_size)]
    correction_image = np.tile(np.repeat(collapsed_cols_binned, bin_size), (2048, 1)) + \
                        np.swapaxes(np.tile(collapsed_rows, (2048, 1)), 0, 1)
    return correction_image