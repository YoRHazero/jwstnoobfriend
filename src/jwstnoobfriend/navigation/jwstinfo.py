from pydantic import BaseModel, Field, field_validator, validate_call, FilePath
from pathlib import Path
from typing import Any, ClassVar, Annotated
from gwcs.wcs import WCS
import re
from jwstnoobfriend.utils.log import getLogger
from jwstnoobfriend.navigation.footprint import FootPrint
from jwstnoobfriend.navigation._cache import (
    _open_and_cache_datamodel,
    _open_and_cache_wcs,
)

logger = getLogger(__name__)


class JwstCover(BaseModel):
    """
    A class similar to the 'cover' of JWST data, which gives the footprint and the path to the file.
    It also provides access to the datamodel, WCS, metadata, data, err, and dq.

    Attributes
    ----------
    filepath : FilePath
        Path to the file. Will be resolved to an absolute path.
    footprint : FootPrint | None
        Footprint of the file. Can be None if not available.
    datamodel : Any
        The datamodel for this file, loaded from the file.
    wcs : WCS
        The WCS object for this file, loaded from the file.
    meta : Any
        Metadata for this file, extracted from the datamodel.
    data : Any
        Data for this file, extracted from the datamodel.
    err : Any
        Error data for this file, extracted from the datamodel.
    dq : Any
        Data quality data for this file, extracted from the datamodel.

    Methods
    -------
    new(filepath: FilePath, with_wcs: bool = True) -> 'JwstCover':
        Create a new JwstCover instance.

    resolve_filepath(value: FilePath) -> FilePath:
        Resolve the filepath to an absolute path.

    Example
    -------
    >>> from jwstnoobfriend.navigation.jwstinfo import JwstCover
    Directly creating an instance:
    >>> cover = JwstCover(
    ...     filepath='path/to/jw01895002001_02010_00001_nrca1_cal.fits',
    ...     footprint=FootPrint.new([(0, 0), (1, 0), (1, 1), (0, 1)])
    ... )
    Creating an instance with a filepath:
    >>> cover = JwstCover.new(filepath='path/to/jw01895002001_02010_00001_nrca1_cal.fits', with_wcs=True)
    Provide a correct with_wcs parameter will receive a clear log message
    """

    filepath: Annotated[
        FilePath,
        Field(
            description="Path to the file.",
        ),
    ]
    """Path to the file. will be resolved to an absolute path."""

    @field_validator("filepath", mode="after")
    @classmethod
    def resolve_filepath(cls, value: FilePath) -> FilePath:
        """Resolve the filepath to an absolute path."""
        return value.resolve()

    footprint: Annotated[
        FootPrint | None,
        Field(
            description="Footprint of the file.",
        ),
    ] = None
    """Footprint of the file, can be None if not available."""

    @property
    def datamodel(self) -> Any:
        """Get the datamodel for this file."""
        datamodel = _open_and_cache_datamodel(self.filepath)
        return datamodel

    @property
    def wcs(self) -> WCS:
        """Get the WCS object for this file."""
        wcs = _open_and_cache_wcs(self.filepath)
        return wcs

    @property
    def meta(self) -> Any:
        """Get the metadata for this file."""
        datamodel = self.datamodel
        return datamodel.meta

    @property
    def data(self) -> Any:
        """Get the data for this file."""
        datamodel = self.datamodel
        return datamodel.data

    @property
    def err(self) -> Any:
        """Get the error data for this file."""
        datamodel = self.datamodel
        return datamodel.err

    @property
    def dq(self) -> Any:
        """Get the data quality data for this file."""
        datamodel = self.datamodel
        return datamodel.dq

    @classmethod
    def new(cls, filepath: Path, with_wcs: bool = True) -> "JwstCover":
        """
        Create a new JwstCover instance.

        Parameters
        ----------
        filepath : Path
            Path to the file.
        with_wcs : bool, optional
            whether this file has a WCS object assigned, by default True. Note: if with_wcs is True,
            but the file does not have a WCS object assigned, the footprint will be None.
        """
        filepath = Path(filepath)
        footprint = FootPrint.new(filepath) if with_wcs else None
        if with_wcs and footprint is None:
            logger.warning(
                f"Footprint could not be created for {filepath}. WCS may not be assigned."
            )
        return cls(filepath=filepath, footprint=footprint)

    @classmethod
    async def _new_async(cls, filepath: Path, with_wcs: bool = True) -> "JwstCover":
        """
        Create a new JwstCover instance asynchronously, this requires to be executed in an async context.

        Parameters
        ----------
        filepath : Path
            Path to the file.
        with_wcs : bool, optional
            whether this file has a WCS object assigned, by default True. Note: if with_wcs is True,
            but the file does not have a WCS object assigned, the footprint will be None.
        """
        filepath = Path(filepath)
        footprint = await FootPrint._new_async(filepath) if with_wcs else None
        if with_wcs and footprint is None:
            logger.warning(
                f"Footprint could not be created for {filepath}. WCS may not be assigned."
            )
        return cls(filepath=filepath, footprint=footprint)


class JwstInfo(BaseModel):
    """
    Information about a JWST file, including its basename, filter, detector, pupil, and associated covers.

    Attributes
    ----------
    basename : str
        The basename of the JWST file, following the naming convention jw\<ppppp\>\<ooo\>\<vvv\>_\<gg\>\<s\>\<aa\>_\<eeeee\>_\<detector\>.
    filter : str
        The filter of the JWST file, e.g. F090W, F115W.
    detector : str
        The detector of the JWST file, e.g. NRCA1, NRCBLONG.
    pupil : str
        The pupil of the JWST file, e.g. CLEAR, GRISMR.
    cover_dict : dict[str, JwstCover]
        A dictionary of JwstCover objects, keyed by calibration level (e.g. '1b', '2a', '2b', '2c').

    Methods
    -------
    new(filepath: FilePath, stage: str, force_with_wcs: bool = False) -> 'JwstInfo':
        Create a new JwstInfo instance from a file path. The stage parameter indicates the calibration stage of the file,
        and force_with_wcs determines whether the file is assumed to have a WCS object assigned regardless of its suffix.

    update(filepath: FilePath, stage: str, force_with_wcs: bool = False) -> None:
        Add a new JwstCover to the cover_dict from a file path. The stage parameter indicates the calibration stage of the file,
        and force_with_wcs determines whether the file is assumed to have a WCS object assigned regardless of its suffix.

    Example
    -------
    >>> from jwstnoobfriend.navigation.jwstinfo import JwstInfo
    Directly creating an instance:
    >>> info = JwstInfo(
    ...     basename='jw01895002001_02010_00001_nrca1_cal.fits', # The basename will automatically be converted to 'jw01895002001_02010_00001_nrca1'
    ...     filter='F090W',
    ...     detector='NRCA1',
    ...     pupil='CLEAR',
    ...     cover_dict={
    ...         '2b': JwstCover.new(filepath='path/to/jw01895002001_02010_00001_nrca1_cal.fits', with_wcs=True),
    ...         },
    ...     }

    Creating an instance from a file path:

    >>> info = JwstInfo.new(filepath='path/to/jw01895002001_02010_00001_nrca1_cal.fits', stage='2b', force_with_wcs=True)

    This will automatically extract the basename, filter, detector, and pupil from the file name and metadata.
    The cover_dict will contain the JwstCover object for the specified stage.
    Add a new cover to the existing JwstInfo instance:

    >>> info.update(filepath='path/to/jw01895002001_02010_00002_nrca1_cal.fits', stage='2c', force_with_wcs=True)

    This will add a new JwstCover to the cover_dict for the '2c' stage.
    Note that the filter, detector, and pupil must match the existing JwstInfo instance, otherwise a ValueError will be raised.

    """

    basename_pattern: ClassVar[str] = r"jw\d{5}\d{3}\d{3}_\d{5}_\d{5}_[^_]+"
    """ The basename pattern for JWST files, used for validation. """

    suffix_without_wcs: ClassVar[list[str]] = ["_uncal", "_rate"]
    """ Suffixes that do not have WCS objects assigned. """

    basename: Annotated[
        str,
        Field(
            description="JWST nameing convention is jw<ppppp><ooo><vvv>_<gg><s><aa>_<eeeee>_<detector>_<filetype>.fits \
                (ref: https://jwst-pipeline.readthedocs.io/en/latest/jwst/data_products/file_naming.html) \
                Here the basename is jw<ppppp><ooo><vvv>_<gg><s><aa>_<eeeee>_<detector>",
            pattern=r"jw\d{5}\d{3}\d{3}_\d{5}_\d{5}_[^_]+",
        ),
    ]
    """jw\<ppppp\>\<ooo\>\<vvv\>_\<gg\>\<s\>\<aa\>_\<eeeee\>_\<detector\>"""

    @field_validator("basename", mode="before")
    @classmethod
    def extract_basename(cls, value: str) -> str:
        """Extracts the basename from the full filename."""
        return re.match(cls.basename_pattern, value).group()

    filter: Annotated[
        str,
        Field(
            description="Filter of this file, e.g. F090W, F115W. It is required to be uppercase and start with 'F'.",
            pattern=r"^F[A-Z0-9]+$",
        ),
    ]
    """Filter of this file, e.g. F090W, F115W."""

    detector: Annotated[
        str,
        Field(
            description="Detector of this file, e.g. NRCA1, NRCBLONG, etc. It is required to be uppercase.",
            pattern=r"^[A-Z0-9]+$",
        ),
    ]
    """Detector of this file, e.g. NRCA1, NRCBLONG."""

    pupil: Annotated[
        str,
        Field(
            description="Pupil of this file, e.g. CLEAR, GRISMR, etc. It is required to be uppercase.",
            pattern=r"^[A-Z0-9]+$",
        ),
    ]
    """Pupil of this file, e.g. CLEAR, GRISMR."""

    cover_dict: Annotated[
        dict[str, JwstCover],
        Field(
            description="Dictionary of JwstCover objects, keyed by calibration level (e.g. '1b', '2a', '2b', '2c').",
            default_factory=dict,
        ),
    ]

    @classmethod
    def new(
        cls, filepath: FilePath | str, stage: str, force_with_wcs: bool = False
    ) -> "JwstInfo":
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
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} does not exist")
        filename = filepath.name
        with_wcs = all(suffix not in filename for suffix in cls.suffix_without_wcs)
        if force_with_wcs:
            with_wcs = True
        jwst_cover = JwstCover.new(filepath, with_wcs=with_wcs)
        instrument_info = jwst_cover.meta.instrument
        return cls(
            basename=filename,
            filter=instrument_info.filter,
            detector=instrument_info.detector,
            pupil=instrument_info.pupil,
            cover_dict={stage: jwst_cover},
        )

    @classmethod
    async def _new_async(
        cls, *, filepath: FilePath | str, stage: str, force_with_wcs: bool = False
    ) -> "JwstInfo":
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} does not exist")
        filename = filepath.name
        with_wcs = all(suffix not in filename for suffix in cls.suffix_without_wcs)
        if force_with_wcs:
            with_wcs = True
        jwst_cover = await JwstCover._new_async(filepath, with_wcs=with_wcs)
        instrument_info = jwst_cover.meta.instrument
        return cls(
            basename=filename,
            filter=instrument_info.filter,
            detector=instrument_info.detector,
            pupil=instrument_info.pupil,
            cover_dict={stage: jwst_cover},
        )

    @validate_call
    def update(
        self, *, filepath: FilePath, stage: str, force_with_wcs: bool = False
    ) -> None:
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
        with_wcs = all(suffix not in filename for suffix in self.suffix_without_wcs)
        if force_with_wcs:
            with_wcs = True
        jwst_cover = JwstCover.new(filepath, with_wcs=with_wcs)
        instrument_info = jwst_cover.meta.instrument
        # Check the instrument info matches the existing one
        if (
            self.filter != instrument_info.filter
            or self.detector != instrument_info.detector
            or self.pupil != instrument_info.pupil
        ):
            raise ValueError("Instrument information does not match existing JwstInfo.")
        # Add the new cover to the cover_dict
        if stage in self.cover_dict:
            logger.warning(
                f"Stage {stage} already exists in cover_dict. Overwriting the existing cover."
            )
        self.cover_dict[stage] = jwst_cover
        return self

    def merge(self, other: "JwstInfo") -> "JwstInfo":
        """
        Merge another JwstInfo instance into this one.

        Parameters
        ----------
        other : JwstInfo
            The other JwstInfo instance to merge.

        Returns
        -------
        JwstInfo
            A new JwstInfo instance with the merged information.
        """
        if self.basename != other.basename:
            raise ValueError("Basenames do not match. Cannot merge.")

        merged_cover_dict = {**self.cover_dict, **other.cover_dict}
        self.cover_dict = merged_cover_dict
        return self
