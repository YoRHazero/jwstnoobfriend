from pydantic import BaseModel, Field, field_validator, validate_call, FilePath
from pathlib import Path
from typing import Any, ClassVar, Annotated, Self, overload, Literal
from gwcs.wcs import WCS
import re
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from jwstnoobfriend.utils.log import getLogger
from jwstnoobfriend.utils.display import plotly_figure_and_mask
from jwstnoobfriend.navigation.footprint import FootPrint
from jwstnoobfriend.navigation._cache import (
    _open_and_cache_datamodel,
    _open_and_cache_wcs,
)
from jwstnoobfriend.utils.calculate import mad_clipped_stats, gaussian_smoothing, segmentation_mask, background_model

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

    Example
    -------
    ```python
    from jwstnoobfriend.navigation.jwstinfo import JwstCover
    Directly creating an instance:
    cover = JwstCover(
        filepath='path/to/jw01895002001_02010_00001_nrca1_cal.fits',
        footprint=FootPrint.new([(0, 0), (1, 0), (1, 1), (0, 1)])
    )
    Creating an instance with a filepath:
    cover = JwstCover.new(filepath='path/to/jw01895002001_02010_00001_nrca1_cal.fits', with_wcs=True)
    ```
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
    
    def data_filled(self, 
                    mask_to_fill: np.ndarray | None = None,
                    data_for_fill: np.ndarray | None = None,
                    method: Literal['gaussian', 'background'] = 'background',
                    fill_outer_nan: Literal['nan', 'zero', 'mean', 'median', 'nearest'] = 'nan'
                    ) -> np.ndarray:
        """
        Replace the masked values in the data with the provided data.
        
        Parameters
        ----------
        mask_to_fill : np.ndarray | None, optional
            A boolean mask indicating which values to be replaced. If None, it will be set to the NaN values of the data.
        data_for_fill : np.ndarray | None, optional
            The data to fill the masked values with. If None, it will use the result of `self.gaussian_blur()` or `self.background()`.
        method : Literal['gaussian', 'background'], optional
            The method to use for filling the masked values. Options are 'gaussian' for Gaussian smoothing or 'background' for background modeling. Default is 'background'.
        fill_outer_nan : Literal['nan', 'zero', 'mean', 'median', 'nearest'], optional
            The method to use for filling the outer NaN values. Options are 'nan' to keep them as is, 'zero' to replace with 0, 'mean' to replace with the mean, 'median' to replace with the median, and 'nearest' to replace with the nearest valid value. Default is 'nan'.
            
        Returns
        -------
        np.ndarray
            The data with the masked values replaced.
            
        Notes
        -----
        The `method` only works when `data_for_fill` is None. Then it will apply the corresponding method 
        with default arguments. If custom arguments are needed, please provide `data_for_fill` directly.
        """
        data = self.data.copy()
        # Exclude the outer shell of the data which is all NaN
        mask_nonnan = ~np.isnan(data)
        valid_rows = np.any(mask_nonnan, axis=1)
        valid_cols = np.any(mask_nonnan, axis=0)

        row_indices = np.where(valid_rows)[0]
        col_indices = np.where(valid_cols)[0]

        row_start, row_end = row_indices[0], row_indices[-1] + 1
        col_start, col_end = col_indices[0], col_indices[-1] + 1
        central_data = data[row_start:row_end, col_start:col_end]
        if mask_to_fill is None:
            mask_to_fill = np.isnan(data)
        if data_for_fill is None:
            match method:
                case 'gaussian':
                    data_for_fill = self.gaussian_blur()
                case 'background':
                    data_for_fill = self.background()
        central_data_for_fill = data_for_fill[row_start:row_end, col_start:col_end]
        central_mask_to_fill = mask_to_fill[row_start:row_end, col_start:col_end]
        central_data[central_mask_to_fill] = central_data_for_fill[central_mask_to_fill]
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

    def background(self,
                   **kwargs: Any) -> np.ndarray:
        """
        Model the background of the data. See also `jwstnoobfriend.utils.calculate.background_model`.
        
        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments to pass to the `background_model` function.
            If `mask` is not provided in `kwargs`, it will use the segmentation mask created by `self.segmentation()` with default arguments.
            
        Returns
        -------
        np.ndarray
            The background model of the data.
        """
        
        if kwargs.get('mask', None) is None:
            kwargs['mask'] = self.segmentation()

        return background_model(self.data, **kwargs)

    def segmentation(self,
                     data: np.ndarray | None = None,
                     factor: float = 2,
                     min_pixels_connected: int = 10,
                     kernel_radius: int = 4,
                     sigma: float | None = None) -> np.ndarray:
        """
        Create a segmentation mask for the data using the MAD method. See also `jwstnoobfriend.utils.calculate.segmentation_mask`.

        Parameters
        ----------
        data : np.ndarray | None, optional
            The data to create the segmentation mask for. If None, it will use the data from the instance.
        factor : float, optional
            The factor by which to multiply the MAD for the segmentation threshold. Default is 2.
        min_pixels_connected : int, optional
            The minimum number of connected pixels to consider a segment valid. Default is 10.
        kernel_radius : int, optional
            The radius of the kernel to use for the segmentation. Default is 4.
        sigma : float | None, optional
            The standard deviation of the Gaussian kernel to use for the segmentation. If None, it will be set to half of `kernel_radius`. Default is None.
            
        Returns
        -------
        np.ndarray
            A boolean mask indicating the segments in the data, where True indicates a segment and False indicates no segment.
        """
        if data is None:
            data = self.data
        if data.ndim != 2:
            raise ValueError("Segmentation mask can only be created for 2D data.")
        return segmentation_mask(data, factor=factor,
                                 min_pixels_connected=min_pixels_connected,
                                 kernel_radius=kernel_radius, sigma=sigma)
    
    def gaussian_blur(self, 
                        mask_invalid: np.ndarray | None = None,
                        kernel_radius_x: int = 15,
                        kernel_radius_y: int | None = None,
                        sigma_x: float | None = None,
                        sigma_y: float | None = None,
                        fill_outer_nan: Literal['nan', 'zero', 'mean', 'median', 'nearest'] = 'nan'
                        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply Gaussian smoothing to the data. See also `jwstnoobfriend.utils.calculate.gaussian_smoothing`.

        Parameters
        ----------
        mask_invalid : np.ndarray | None, optional
            A boolean mask indicating invalid data points. If None, it will be set to the NaN values of the data.
        kernel_radius_x : int, optional
            The radius of the Gaussian kernel in the x direction. Default is 15.
        kernel_radius_y : int | None, optional
            The radius of the Gaussian kernel in the y direction. If None, it will be set
            to the same value as `kernel_radius_x`. Default is None.
        sigma_x : float | None, optional
            The standard deviation of the Gaussian kernel in the x direction. If None, it will be
            set to half of `kernel_radius_x`. Default is None.
        sigma_y : float | None, optional
            The standard deviation of the Gaussian kernel in the y direction. If None, it will be
            set to half of `kernel_radius_y` if `kernel_radius_y` is not None, otherwise it will be set to the same value as `sigma_x`. Default is None.
        fill_outer_nan : Literal['nan', 'zero', 'mean', 'median', 'nearest'], optional
            How to handle outer NaN values after smoothing. Options are:
            - 'nan': Keep outer NaN values as is.
            - 'zero': Fill outer NaN values with zero.
            - 'mean': Fill outer NaN values with the mean of the data.
            - 'median': Fill outer NaN values with the median of the data.
            - 'nearest': Fill outer NaN values with the nearest valid value (not implemented in
            this version).
            
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing the smoothed data and the smoothed error data.
        """
        if self.data.ndim != 2:
            raise ValueError("Gaussian smoothing can only be applied to 2D data.")
        if self.err.ndim != 2:
            raise ValueError("Gaussian smoothing can only be applied to 2D error data.")
        if mask_invalid is None:
            mask_invalid = np.isnan(self.data)
        data_smoothed = gaussian_smoothing(
            data=self.data,
            mask_invalid=mask_invalid,
            kernel_radius_x=kernel_radius_x,
            kernel_radius_y=kernel_radius_y,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            fill_outer_nan=fill_outer_nan
        )
        err_smoothed = gaussian_smoothing(
            data=self.err,
            mask_invalid=mask_invalid,
            kernel_radius_x=kernel_radius_x,
            kernel_radius_y=kernel_radius_y,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            fill_outer_nan=fill_outer_nan
        )
        return data_smoothed, err_smoothed

    @overload
    def plotly_imshow(self, 
                        fig_height: int | None = None,
                        fig_width: int | None = None,
                        facet_col_wrap: int = 2,
                        pmin: float = 1.,
                        pmax: float = 99.,
                        color_map: str = 'gray',
                        *,
                        return_figure: Literal[True]) -> go.Figure: ...
    @overload
    def plotly_imshow(self,
                        fig_height: int | None = None,
                        fig_width: int | None = None,
                        facet_col_wrap: int = 2,
                        pmin: float = 1.,
                        pmax: float = 99.,
                        color_map: str = 'gray',
                        *, 
                        return_figure: Literal[False]) -> None: ...
    
    @overload
    def plotly_imshow(self,
                        fig_height: int | None = None,
                        fig_width: int | None = None,
                        facet_col_wrap: int = 2,
                        pmin: float = 1.,
                        pmax: float = 99.,
                        color_map: str = 'gray') -> None: ...

    @validate_call
    def plotly_imshow(self,
                        fig_height: int| None = None,
                        fig_width: int | None = None,
                        facet_col_wrap: int = 2,
                        pmin: float = 1.,
                        pmax: float = 99.,
                        color_map: str = 'gray',
                        return_figure: bool = False
                        ) -> go.Figure | None:
        """Plot the data using Plotly in a notebook.
        
        Parameters
        ----------
        fig_height : int, optional
            Height of the figure in pixels, by default None. If None, it will be calculated based on the data shape.
            
        fig_width : int, optional
            Width of the figure in pixels, by default None. If None, it will be calculated based on the data shape.
            
        facet_col_wrap : int, optional
            Number of columns to wrap the facets, by default 2.
            
        pmin : float, optional
            Minimum percentile for the color scale, by default 1.0.
            
        pmax : float, optional
            Maximum percentile for the color scale, by default 99.0.
            
        color_map : str, optional
            Color map to use for the plot, by default 'gray'.
            
        return_figure : bool, optional
            If True, return the figure object instead of showing it, by default False.
        """
        data = self.data
        shape = data.shape
        zmin, zmax = np.nanpercentile(data, [pmin, pmax])
        match len(shape):
            case 2:
                fig = px.imshow(
                    data,
                    zmin=zmin,
                    zmax=zmax,
                    color_continuous_scale=color_map,
                    binary_string=True,
                )
            case 3:
                if fig_height is None:
                    fig_height = np.ceil(shape[0] / facet_col_wrap).astype(int) * 500
                if fig_width is None:
                    fig_width = facet_col_wrap * 500
                fig = px.imshow(
                    data,
                    facet_col=0,
                    facet_col_wrap=facet_col_wrap,
                    zmin=zmin,
                    zmax=zmax,
                    color_continuous_scale=color_map,
                    binary_string=True,
                )
            case 4:
                if shape[0] == 1:
                    facet_col_wrap = 1
                if fig_height is None:
                    fig_height = np.ceil(shape[0] / facet_col_wrap).astype(int) * 500 + 50
                if fig_width is None:
                    fig_width = facet_col_wrap * 500
                fig = px.imshow(
                    data,
                    facet_col=0,
                    facet_col_wrap=facet_col_wrap,
                    animation_frame=1,
                    zmin=zmin,
                    zmax=zmax,
                    binary_string=True,
                    color_continuous_scale=color_map,
                )
            case _:
                raise ValueError(
                    f"Unsupported data shape {shape}. Only 2D, 3D, and 4D data are supported."
                )
        fig.update_layout(
            height=fig_height,
            width=fig_width,
            newshape=dict(
                line_color='red',
                line_width=2,
            )
        )
        if return_figure:
            return fig
        else:
            fig.show(config={'modeBarButtonsToAdd':['drawrect',
                                                    'eraseshape']})

    @classmethod
    def new(cls, filepath: Path, with_wcs: bool = True) -> Self:
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
    async def _new_async(cls, filepath: Path, with_wcs: bool = True) -> Self:
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
    ```python
    from jwstnoobfriend.navigation.jwstinfo import JwstInfo
    Directly creating an instance:
    info = JwstInfo(
        basename='jw01895002001_02010_00001_nrca1_cal.fits', # The basename will automatically be converted to 'jw01895002001_02010_00001_nrca1'
        filter='F090W',
        detector='NRCA1',
        pupil='CLEAR',
        cover_dict={
            '2b': JwstCover.new(filepath='path/to/jw01895002001_02010_00001_nrca1_cal.fits', with_wcs=True),
        }
    }
    ```
    Creating an instance from a file path:
    ```python
    info = JwstInfo.new(filepath='path/to/jw01895002001_02010_00001_nrca1_cal.fits', stage='2b', force_with_wcs=True)
    ```
    This will automatically extract the basename, filter, detector, and pupil from the file name and metadata.
    The cover_dict will contain the JwstCover object for the specified stage.
    Add a new cover to the existing JwstInfo instance:
    ```python
    info.update(filepath='path/to/jw01895002001_02010_00002_nrca1_cal.fits', stage='2c', force_with_wcs=True)
    ```

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
                Here the basename is jw<ppppp><ooo><vvv>_<gg><s><aa>_<eeeee>_<detector>. This will be checked before coadding.",
        ),
    ]
    """jw\<ppppp\>\<ooo\>\<vvv\>_\<gg\>\<s\>\<aa\>_\<eeeee\>_\<detector\>"""

    @field_validator("basename", mode="before")
    @classmethod
    def extract_basename(cls, value: str) -> str:
        """Extracts the basename from the full filename."""
        basename_match = re.match(cls.basename_pattern, value)
        if basename_match:
            return basename_match.group()
        else:
            return value

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
    
    @property
    def filesetname(self) -> str:
        """
        Get the fileset name from the basename.
        
        Returns
        -------
        str
            The fileset name, which is the first part of the basename.
        """
        file_set_regex = r"jw\d{5}\d{3}\d{3}_\d{5}_\d{5}"
        match = re.match(file_set_regex, self.basename)
        if match:
            return match.group(0)
        else:
            raise ValueError(f"Basename {self.basename} does not match the expected pattern.")

    @classmethod
    def new(
        cls, filepath: FilePath | str, stage: str, force_with_wcs: bool = False
    ) -> Self:
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
    ) -> Self:
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

    def merge(self, other: Self) -> Self:
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
    
    def is_same_pointing(self, 
                         other: Self,
                         stage_with_wcs: str = '2b',
                         overlap_percent: float = 0.6,
                         same_instrument: bool = True,
                         ) -> bool:
        """
        Check if the two JwstInfo instances have the same pointing based on their footprints.
        
        Parameters
        ----------
        other : JwstInfo
            The other JwstInfo instance to compare with.
        stage_with_wcs : str, optional
            The stage to use for the WCS comparison, by default '2b'.
        overlap_percent : float, optional
            The minimum overlap percentage required to consider the pointings the same, by default 0.8, maximum is 1.0.
        same_instrument : bool, optional
            If True, also check if the filter, detector, and pupil match between the two JwstInfo instances.
            If False, only check the overlap of the footprints, by default True.
        
        Returns
        -------
        bool
            True if the two JwstInfo instances are different dithers of the same pointing, False otherwise.
        """
        
        if stage_with_wcs not in self.cover_dict or stage_with_wcs not in other.cover_dict:
            raise ValueError(f"Stage '{stage_with_wcs}' not found in cover_dict.")
        if self.cover_dict[stage_with_wcs].footprint is None or other.cover_dict[stage_with_wcs].footprint is None:
            raise ValueError(f"Footprint for stage '{stage_with_wcs}' is not available.")
        self_fp = self[stage_with_wcs].footprint
        other_fp = other[stage_with_wcs].footprint
        overlap_area = self_fp.polygon.intersection(other_fp.polygon).area
        self_area = self_fp.polygon.area
        other_area = other_fp.polygon.area
        if overlap_area / self_area >= overlap_percent or overlap_area / other_area >= overlap_percent:
            # If same_instrument is True, check if the filter, detector, and pupil match
            if same_instrument:
                return (
                    self.filter == other.filter and
                    self.detector == other.detector and
                    self.pupil == other.pupil
                )
            # If same_instrument is False, we only check the overlap
            else:
                return True
        else:
            return False
        
    
    def plotly_imshow(self,
                        stages: list[str] | None = None,
                        stage_types: list[Literal['data', 'mask']] | None = None,
                        data: list[np.ndarray] | None = None,
                        mask: list[np.ndarray] | None = None,
                        pmin: float = 1.0,
                        pmax: float = 99.0,
                        zmin: float | None = None,
                        zmax: float | None = None,
                        cmap: str = 'gray',
                        binary_mode: bool = True,
                        height: int = 600,
                        width: int = 600,
                        align_mode: Literal['blink', 'wrap'] = 'blink',
                        subtitles: list[str] | None = None
                        ) -> go.Figure:
        """
        Create a Plotly figure with the data and masks from the specified stages or provided data and mask arrays.
        
        Parameters
        ----------
        stages : list[str] | None
            A list of stage names to extract data and masks from the JwstCover objects. If None, no stages are used. Make sure the stages are valid keys in the cover_dict.
        stage_types : list[Literal['data', 'mask']] | None
            A list of stage types corresponding to the stages. If None, it defaults to 'data' for all stages.
            The length must match the length of `stages` if `stages` is provided.
        data : list[np.ndarray] | None
            A list of numpy arrays representing the data to be displayed. If None, it will be extracted from the stages.
        mask : list[np.ndarray] | None
            A list of numpy arrays representing the masks to be displayed.
        pmin : float, optional
            The minimum percentile for the color scale. Default is 1.0. If `zmin` is provided, this is ignored.
        pmax : float, optional
            The maximum percentile for the color scale. Default is 99.0. If `zmax` is provided, this is ignored.
        zmin : float | None, optional
            The minimum value for the color scale. If None, it is calculated from the data.
        zmax : float | None, optional
            The maximum value for the color scale. If None, it is calculated from the data.
        cmap : str, optional
            The color map to use for the figure. Default is 'gray'.
        binary_mode : bool, optional
            Whether to treat the data as binary strings, this will have better performance. Default is True.
        height : int, optional
            The height of the figure in pixels. Default is 600.
        width : int, optional
            The width of the figure in pixels. Default is 600.
        align_mode : Literal['blink', 'wrap'], optional
            The alignment mode for the figure. Options are 'blink' for animation frame alignment or 'wrap' for facet column wrapping. Default is 'animate'.
        subtitles : list[str] | None
            A list of subtitles for each data and mask array. If None, default subtitles are generated.

        Returns
        -------
        go.Figure
            A Plotly figure object containing the data and masks visualized with the specified parameters.
        """
        if stages is None and data is None and mask is None:
            raise ValueError("At least one of 'stages', 'data', or 'mask' must be provided.")
        if data is None:
            data = []
        if mask is None:
            mask = []
        if stages is None:
            stages = []
        if stages:
            if stage_types is None:
                stage_types = ['data'] * len(stages)
            elif len(stage_types) != len(stages):
                raise ValueError("Length of 'stage_types' must match length of 'stages'.")
            data_stages = []
            for stage, stage_type in zip(stages, stage_types):
                if stage_type == 'data':
                    data_stages.append(self[stage].data)
                elif stage_type == 'mask':
                    mask.append(self[stage].data)
        
            data = data_stages + data

        if zmin is None:
            zmin = np.nanpercentile(np.concatenate(data), pmin)
        if zmax is None:
            zmax = np.nanpercentile(np.concatenate(data), pmax)
        fig = plotly_figure_and_mask(
            data=data,
            mask=mask,
            pmin=pmin,
            pmax=pmax,
            zmin=zmin,
            zmax=zmax,
            cmap=cmap,
            binary_mode=binary_mode,
            height=height,
            width=width,
            align_mode=align_mode,
            subtitles=subtitles
        )

        return fig

    def plotly_add_footprint(self,
                         fig: go.Figure,
                         stage: str,
                         show_more: bool = True,
                         attrs_for_hover: list[str] | None = None,
                         fig_mode: Literal['sky', 'cartesian'] = 'sky',
                         **kwargs) -> go.Figure:
        """
        Add the footprint of a specific stage to a Plotly figure.
        
        Parameters
        ----------
        fig : go.Figure
            The Plotly figure to which the footprint will be added.
        stage : str
            Calibration stage of the file with wcs assigned, e.g. '2b', '2c'.
        show_more : bool, optional
            If True, additional hover information will be added to the footprint trace. Default is True, 
            which will add the basename and attributes of filter and pupil to the hover template.
        attrs_for_hover : list[str] | None, optional
            A list of attributes to include in the hover template for the footprint trace. All the attributes
            in the list will be added to "fp_customdata" for the FootPrint.add_trace_in_sky Method.
            If None and show_more is True,
            it defaults to ['filter', 'pupil']. If show_more is False, this parameter is ignored.
        fig_mode : Literal['sky', 'cartesian'], optional
            The mode in which to add the footprint trace. 'sky' for sky coordinates, 'cartesian' for Cartesian coordinates.
            Default is 'sky'.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the footprint trace addition method. Check the FootPrint class, add_trace_in_sky or add_trace_in_cartesian 
            for more details on the available parameters.
        """
        if stage not in self.cover_dict:
            raise ValueError(f"Stage '{stage}' not found in cover_dict.")
        cover = self.cover_dict[stage]
        fp = cover.footprint
        if fp is None:
            raise ValueError(f"Footprint for stage '{stage}' is None.")
        if show_more:
            default_fp_hovertemplate = f"{self.basename}<br>"
            kwargs.setdefault('fp_hovertemplate', default_fp_hovertemplate)
            if attrs_for_hover is None:
                attrs_for_hover = ['filter', 'pupil']
        
        if attrs_for_hover is not None:
            fp_customdata = kwargs.get('fp_customdata', {})
            for attr in attrs_for_hover:
                if not hasattr(self, attr):
                    raise ValueError(f"Attribute '{attr}' not found in JwstInfo. Check the attrs_for_hover contains valid attributes.")
                fp_customdata[attr] = getattr(self, attr)
            kwargs['fp_customdata'] = fp_customdata

        match fig_mode:
            case 'sky':
                fig = fp.add_trace_in_sky(fig, **kwargs)
            case 'cartesian':
                fig = fp.add_trace_in_cartesian(fig, **kwargs)
        return fig

    def __getitem__(self, stage: str) -> JwstCover:
        """
        Get the JwstCover for a specific stage.

        Parameters
        ----------
        stage : str
            Calibration stage of the file, e.g. '1b', '2a', '2b', '2c'.

        Returns
        -------
        JwstCover
            The JwstCover object for the specified stage.
        """
        return self.cover_dict[stage]
