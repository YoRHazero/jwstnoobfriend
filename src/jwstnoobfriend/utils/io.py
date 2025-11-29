from astropy.io import fits
import stdatamodels.asdf_in_fits as aif
from pydantic import validate_call, FilePath
from gwcs import WCS as gwcs_WCS
import numpy as np
import warnings
from asdf.exceptions import AsdfPackageVersionWarning
warnings.filterwarnings("ignore", category=AsdfPackageVersionWarning)

@validate_call
def direct_read_gwcs(filepath: FilePath) -> gwcs_WCS:
    """Directly reads the GWCS from a FITS file without caching."""
    try:
        with aif.open(filepath) as af:
            wcs = af.tree['meta']['wcs']
            return wcs
    except Exception as e:
        raise ValueError(f"Failed to read WCS for {filepath}: {e}") from e
    
@validate_call
def direct_read_data(filepath: FilePath) -> np.ndarray:
    """Directly reads the data from a FITS file without caching."""
    try:
        with fits.open(filepath) as hdul:
            data = hdul['SCI'].data
            return data
    except Exception as e:
        raise ValueError(f"Failed to read data for {filepath}: {e}") from e
    
@validate_call
def direct_read_err(filepath: FilePath) -> np.ndarray:
    """Directly reads the error array from a FITS file without caching."""
    try:
        with fits.open(filepath) as hdul:
            err = hdul['ERR'].data
            return err
    except Exception as e:
        raise ValueError(f"Failed to read error array for {filepath}: {e}") from e
    
@validate_call
def direct_read_dq(filepath: FilePath) -> np.ndarray:
    """Directly reads the data quality array from a FITS file without caching."""
    try:
        with fits.open(filepath) as hdul:
            dq = hdul['DQ'].data
            return dq
    except Exception as e:
        raise ValueError(f"Failed to read data quality array for {filepath}: {e}") from e