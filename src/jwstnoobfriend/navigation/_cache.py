from pyexpat import model
from cachetools import LRUCache
from typing import Any
from gwcs.wcs import WCS
from jwst import datamodels as dm
from pydantic import validate_call, FilePath

_wcs_cache: LRUCache[str, WCS] = LRUCache(maxsize=2000)
_datamodel_cache: LRUCache[str, Any] = LRUCache(maxsize=8)

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
    
def _clear_datamodel_cache():
    """Clears the datamodel cache."""
    _datamodel_cache.clear()

def _clear_wcs_cache():
    """Clears the WCS cache."""
    _wcs_cache.clear()