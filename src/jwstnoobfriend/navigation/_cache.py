from pyexpat import model
from cachetools import LRUCache
from typing import Any
from gwcs.wcs import WCS
from jwst import datamodels as dm
from pydantic import validate_call, FilePath
import aiofiles
from astropy.io import fits
import io

_wcs_cache: LRUCache[str, WCS] = LRUCache(maxsize=2000)
_datamodel_cache: LRUCache[str, Any] = LRUCache(maxsize=8)

@validate_call
def _open_and_cache_datamodel(filepath: FilePath):
    """Caches the datamodel for a given file path."""
    if filepath in _datamodel_cache:
        return _datamodel_cache[filepath]
    
    try:
        model = dm.open(filepath)
        _datamodel_cache[filepath] = model
        return model
    except Exception as e:
        raise ValueError(f"Failed to open datamodel for {filepath}: {e}") from e

@validate_call
async def _open_and_cache_datamodel_async(filepath: FilePath) -> Any:
    """Asynchronously caches the datamodel for a given file path."""
    if filepath in _datamodel_cache:
        return _datamodel_cache[filepath]
    
    try:
        async with aiofiles.open(filepath, mode='rb') as f:
            content = await f.read()
            hdul = fits.open(io.BytesIO(content))
            model = dm.open(hdul)
            _datamodel_cache[filepath] = model
            return model
    except Exception as e:
        raise ValueError(f"Failed to open datamodel asynchronously for {filepath}: {e}") from e

@validate_call
def _open_and_cache_wcs(filepath: FilePath) -> WCS:
    """Caches the WCS for a given file path."""
    if filepath in _wcs_cache:
        return _wcs_cache[filepath]
    
    if filepath in _datamodel_cache:
        model = _datamodel_cache[filepath]
        return model.meta.wcs
    
    try:
        model = dm.open(filepath)
        _datamodel_cache[filepath] = model
        _wcs_cache[filepath] = model.meta.wcs
        return model.meta.wcs
    except Exception as e:
        raise ValueError(f"Failed to open WCS for {filepath}: {e}") from e
    
@validate_call
async def _open_and_cache_wcs_async(filepath: FilePath) -> WCS:
    """Asynchronously caches the WCS for a given file path."""
    if filepath in _wcs_cache:
        return _wcs_cache[filepath]
    
    if filepath in _datamodel_cache:
        model = _datamodel_cache[filepath]
        return model.meta.wcs
    
    try:
        model = await _open_and_cache_datamodel_async(filepath)
        _wcs_cache[filepath] = model.meta.wcs
        return model.meta.wcs
    except Exception as e:
        raise ValueError(f"Failed to open WCS asynchronously for {filepath}: {e}") from e
  
def _clear_datamodel_cache():
    """Clears the datamodel cache."""
    _datamodel_cache.clear()

def _clear_wcs_cache():
    """Clears the WCS cache."""
    _wcs_cache.clear()