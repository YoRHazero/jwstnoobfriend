from fileinput import filename
from numpy import cov
from pydantic import BaseModel, Field, field_validator, FilePath, validate_call
from typing import Any, ClassVar, Annotated
from cachetools import LRUCache
from gwcs.wcs import WCS
import re
from jwst import datamodels as dm
from jwstnoobfriend.utils.log import getLogger
from jwstnoobfriend.navigation.footprint import FootPrint

logger = getLogger(__name__)

class JwstCover(BaseModel):
    wcs_cache: ClassVar[LRUCache[FilePath, WCS]] = LRUCache(maxsize=2000)
    """ Cache for WCS objects, keyed by basename. """
    
    datamodel_cache: ClassVar[LRUCache[FilePath, Any]] = LRUCache(maxsize=4)
    """ Cache for datamodels, keyed by filepath. """
    
    filepath: Annotated[FilePath, Field(
        description="Path to the file.",
    )]
    """Path to the file. will be resolved to an absolute path."""
    
    @field_validator('filepath', mode='after')
    @classmethod
    def resolve_filepath(cls, value: FilePath) -> FilePath:
        """Resolve the filepath to an absolute path."""
        return value.resolve()
    
    footprint: Annotated[FootPrint | None, Field(
        description="Footprint of the file.",
    )] = None
    """Footprint of the file, can be None if not available."""
    
    @property
    def datamodel(self) -> Any:
        """Get the datamodel for this file."""
        if self.filepath not in self.datamodel_cache:
            try:
                datamodel = dm.open(self.filepath)
                self.datamodel_cache[self.filepath] = datamodel
            except Exception as e:
                logger.error(f"Error loading datamodel for {self.filepath}: {e}")
                raise e
        return self.datamodel_cache[self.filepath]
    
    @property
    def wcs(self) -> WCS:
        """Get the WCS object for this file."""
        if self.filepath not in self.wcs_cache:
            try:
                datamodel = self.datamodel
                if not datamodel.meta.wcs:
                    logger.warning(f"WCS is None for {self.filepath}.")
                    return None
                self.wcs_cache[self.filepath] = datamodel.meta.wcs
            except Exception as e:
                logger.error(f"Error loading WCS for {self.filepath}: {e}")
                raise e
        return self.wcs_cache[self.filepath]
    
    @property
    def meta(self) -> Any:
        """Get the metadata for this file."""
        try:
            datamodel = self.datamodel
            return datamodel.meta
        except Exception as e:
            logger.error(f"Error loading metadata for {self.filepath}: {e}")
            raise e
    
    @property
    def data(self) -> Any:
        """Get the data for this file."""
        try:
            datamodel = self.datamodel
            return datamodel.data
        except Exception as e:
            logger.error(f"Error loading data for {self.filepath}: {e}")
            raise e
        
    @property
    def err(self) -> Any:
        """Get the error data for this file."""
        try:
            datamodel = self.datamodel
            return datamodel.err
        except Exception as e:
            logger.error(f"Error loading error data for {self.filepath}: {e}")
            raise e
    
    @property
    def dq(self) -> Any:
        """Get the data quality data for this file."""
        try:
            datamodel = self.datamodel
            return datamodel.dq
        except Exception as e:
            logger.error(f"Error loading data quality data for {self.filepath}: {e}")
            raise e
        
    @classmethod
    def clear_wcs_cache(cls) -> None:
        """Clear the WCS cache."""
        cls.wcs_cache.clear()
        logger.info("WCS cache cleared.")
    
    @classmethod
    def clear_datamodel_cache(cls) -> None:
        """Clear the datamodel cache."""
        cls.datamodel_cache.clear()
        logger.info("Datamodel cache cleared.")
    
    @classmethod
    def new(cls, filepath: FilePath, with_wcs: bool = True) -> 'JwstCover':
        """
        Create a new JwstCover instance.
        
        Parameters
        ----------
        filepath : FilePath
            Path to the file.
        with_wcs : bool, optional
            whether this file has a WCS object assigned, by default True. Note: if with_wcs is True,
            but the file does not have a WCS object assigned, the footprint will be None.
        """
        footprint = FootPrint.from_file(filepath) if with_wcs else None
        if with_wcs and footprint is None:
            logger.warning(f"Footprint could not be created for {filepath}. WCS may not be assigned.")
        return cls(
            filepath=filepath,
            footprint=footprint
        )
        

class JwstInfo(BaseModel):
    """
    Base class for JWST file information.
    
    Example
    -------
    For a new basename:
    >>> from jwstnoobfriend.navigation import JwstInfo
    >>> info = JwstInfo(
    ...     basename="jw00001001001_00001_00001_nrca1_uncal.fits",
    ...     filter="F115W",
    ...     detector="NRCA1",
    ...     pupil="CLEAR",
    ...     filepaths={
    ...         "1b": "/path/to/1b/jw00001001001_00001_00001_nrca1_uncal.fits",
    ...         "2a": "/path/to/2a/jw00001001001_00001_00001_nrca1_rated.fits"
    ...     },
    ...     footprint=None
    ... )
    >>> # In this case, the filepaths dictionary does not contain '2b' or '2c', which
    >>> # are expected to have WCS objects assigned. Hence, using wcs or footprint is 
    >>> # not appropriate.
    
    For an existing basename:
    >>> from jwstnoobfriend.navigation import JwstInfo
    >>> info: JwstInfo = JwstInfo(...)
    >>> info.add_filepath("2b", "/path/to/2b/jw00001001001_00001_00001_nrca1_rated.fits")
    """
    
    basename_pattern: ClassVar[str] = r"jw\d{5}\d{3}\d{3}_\d{5}_\d{5}_[^_]+"
    """ The basename pattern for JWST files, used for validation. """

    suffix_without_wcs: ClassVar[list[str]] = ['_uncal', '_rate']
    """ Suffixes that do not have WCS objects assigned. """

    basename: Annotated[str, Field(
        description="JWST nameing convention is jw<ppppp><ooo><vvv>_<gg><s><aa>_<eeeee>_<detector>_<filetype>.fits \
                (ref: https://jwst-pipeline.readthedocs.io/en/latest/jwst/data_products/file_naming.html) \
                Here the basename is jw<ppppp><ooo><vvv>_<gg><s><aa>_<eeeee>_<detector>",
        pattern = r"jw\d{5}\d{3}\d{3}_\d{5}_\d{5}_[^_]+"
    )]
    """jw\<ppppp\>\<ooo\>\<vvv\>_\<gg\>\<s\>\<aa\>_\<eeeee\>_\<detector\>"""
    
    @field_validator('basename', mode = 'before')
    @classmethod
    def extract_basename(cls, value: str) -> str:
        """Extracts the basename from the full filename."""
        return re.match(cls.basename_pattern, value).group()
    
    filter: Annotated[str, Field(
        description="Filter of this file, e.g. F090W, F115W. It is required to be uppercase and start with 'F'.",
        pattern = r"^F[A-Z0-9]+$"
    )]
    """Filter of this file, e.g. F090W, F115W."""
    
    detector: Annotated[str, Field(
        description="Detector of this file, e.g. NRCA1, NRCBLONG, etc. It is required to be uppercase.",
        pattern = r"^[A-Z0-9]+$"
    )]
    """Detector of this file, e.g. NRCA1, NRCBLONG."""
    
    pupil: Annotated[str, Field(
        description="Pupil of this file, e.g. CLEAR, GRISMR, etc. It is required to be uppercase.",
        pattern = r"^[A-Z0-9]+$"
    )]
    """Pupil of this file, e.g. CLEAR, GRISMR."""
    
    cover_dict: Annotated[dict[str, JwstCover], Field(
        description="Dictionary of JwstCover objects, keyed by calibration level (e.g. '1b', '2a', '2b', '2c').",
        default_factory=dict
    )]
    
    @classmethod
    @validate_call
    def new_from_filepath(cls, *, filepath: FilePath, stage: str, force_with_wcs: bool = False) -> 'JwstInfo':
        """
        Create a new JwstInfo instance from a file path. Note that whether the file has
        a WCS object assigned is determined by the suffix of the file name.
        
        Parameters
        ----------
        filepath : FilePath
            Path to the file.
            
        stage : str
            Calibration stage of the file, e.g. '1b', '2a', '2b', '2c'.
            
        force_with_wcs : bool, optional
            If True, the file is assumed to have a WCS object assigned regardless of its suffix.
            
            
        Returns
        -------
        JwstInfo
            A new instance of JwstInfo with the file information.
        """
        
        filename = filepath.name
        with_wcs = any(suffix in filename for suffix in cls.suffix_without_wcs)
        if force_with_wcs:
            with_wcs = True
        jwst_cover = JwstCover.new(filepath, with_wcs=with_wcs)
        instrument_info = jwst_cover.meta.instrument
        return cls(
            basename=filename,
            filter=instrument_info.filter,
            detector=instrument_info.detector,
            pupil=instrument_info.pupil,
            cover_dict={stage: jwst_cover}
        )
        
    def add_from_filepath(self, filepath: FilePath, stage: str, force_with_wcs: bool = False) -> None:
        """
        Add a new JwstCover to the cover_dict from a file path.
        
        Parameters
        ----------
        filepath : FilePath
            Path to the file.
            
        stage : str
            Calibration stage of the file, e.g. '1b', '2a', '2b', '2c'.
            
        force_with_wcs : bool, optional
            If True, the file is assumed to have a WCS object assigned regardless of its suffix.
        """
        filename = filepath.name
        with_wcs = any(suffix in filename for suffix in self.suffix_without_wcs)
        if force_with_wcs:
            with_wcs = True
        jwst_cover = JwstCover.new(filepath, with_wcs=with_wcs)
        instrument_info = jwst_cover.meta.instrument
        # Check the instrument info matches the existing one
        if (self.filter != instrument_info.filter or
            self.detector != instrument_info.detector or
            self.pupil != instrument_info.pupil):
            raise ValueError("Instrument information does not match existing JwstInfo.")
        # Add the new cover to the cover_dict
        if stage in self.cover_dict:
            logger.warning(f"Stage {stage} already exists in cover_dict. Overwriting the existing cover.")
        self.cover_dict[stage] = jwst_cover
        return self
    
        