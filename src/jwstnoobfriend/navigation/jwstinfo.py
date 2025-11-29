from pydantic import BaseModel, Field, field_validator, validate_call, FilePath
from pathlib import Path
from typing import Any, ClassVar, Annotated, Self, overload, Literal
from gwcs.wcs import WCS
import re
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pandas import DataFrame
from jwstnoobfriend.utils.log import getLogger
from jwstnoobfriend.utils.display import plotly_figure_and_mask
from jwstnoobfriend.navigation.footprint import FootPrint
from jwstnoobfriend.navigation._cache import (
    _open_and_cache_datamodel,
)
from jwstnoobfriend.utils.io import direct_read_data, direct_read_err, direct_read_dq, direct_read_gwcs
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
        wcs = direct_read_gwcs(self.filepath)
        return wcs

    @property
    def meta(self) -> Any:
        """Get the metadata for this file."""
        datamodel = self.datamodel
        return datamodel.meta

    @property
    def data(self) -> Any:
        """Get the data for this file."""
        data = direct_read_data(self.filepath)
        return data

    @property
    def err(self) -> Any:
        """Get the error data for this file."""
        err = direct_read_err(self.filepath)
        return err

    @property
    def dq(self) -> Any:
        """Get the data quality data for this file."""
        dq = direct_read_dq(self.filepath)
        return dq
    
    def read_catalog(
        self,
        catalog: DataFrame,
        ra_key: str = 'ra',
        dec_key: str = 'dec',
        id_key: str = 'id',
    ):
        """
        Read a catalog and extract objects within the footprint, and add their pixel coordinates 
        as 'pix_x' and 'pix_y' columns based on the WCS of this cover.

        Parameters
        ----------
        catalog : DataFrame
            The catalog to read from.
        ra_key : str, optional
            The key for the right ascension column. Default is 'ra'.
        dec_key : str, optional
            The key for the declination column. Default is 'dec'.
        id_key : str, optional
            The key for the ID column. Default is 'id'.

        Returns
        -------
        DataFrame
            A DataFrame containing the objects within the footprint.
        """
        if ra_key not in catalog.columns:
            raise ValueError(f"Catalog must contain a '{ra_key}' column.")
        if dec_key not in catalog.columns:
            raise ValueError(f"Catalog must contain a '{dec_key}' column.")
        if id_key not in catalog.columns:
            raise ValueError(f"Catalog must contain a '{id_key}' column.")

        footprint = self.footprint
        if footprint is None:
            raise ValueError("WCS is not available in this stage, please use a product with wcs assigned.")
        result_catalog = footprint.read_catalog(
            catalog, ra_key=ra_key, dec_key=dec_key, id_key=id_key
        )
        if result_catalog is None:
            logger.warning(
                "No points from the catalog are within the footprint."
            )
            return None
        wcs = self.wcs
        sky2pix = wcs.get_transform('world', 'detector')
        try:
            pix_x, pix_y = sky2pix(
                result_catalog[ra_key],
                result_catalog[dec_key]
            )
        except:
            pix_x, pix_y, _, _ = sky2pix(
                result_catalog[ra_key],
                result_catalog[dec_key],
                [0] * len(result_catalog),
                [1] * len(result_catalog)
            )
        result_catalog['pix_x'] = pix_x
        result_catalog['pix_y'] = pix_y
        return result_catalog

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
                        catalog: DataFrame | None = None,
                        id_key: str = 'id',
                        cat_marker_dict: dict | None = None,
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
                        catalog: DataFrame | None = None,
                        id_key: str = 'id',
                        cat_marker_dict: dict | None = None,
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
                        catalog: DataFrame | None = None,
                        id_key: str = 'id',
                        cat_marker_dict: dict | None = None,
                        facet_col_wrap: int = 2,
                        pmin: float = 1.,
                        pmax: float = 99.,
                        color_map: str = 'gray') -> None: ...

    def plotly_imshow(self,
                        fig_height: int| None = None,
                        fig_width: int | None = None,
                        catalog: DataFrame | None = None,
                        id_key: str = 'id',
                        cat_marker_dict: dict | None = None,
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
        
        catalog : DataFrame | None, optional
            Catalog data to overlay on the image, by default None.
            
        id_key : str, optional
            The key for the ID column in the catalog, by default 'id'.

        cat_marker_dict : dict | None, optional
            Dictionary passed to plotly.express.scatter, by default None.

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
        if catalog is not None:
            if 'pix_x' not in catalog.columns or 'pix_y' not in catalog.columns:
                loaded_catalog = self.read_catalog(catalog)
            else:
                loaded_catalog = catalog
            if cat_marker_dict is None:
                cat_marker_dict = {}
            scatter_traces = px.scatter(
                loaded_catalog,
                x='pix_x',
                y='pix_y',
                custom_data=[id_key],
                **cat_marker_dict
            )
            for trace in scatter_traces.data:
                trace.hovertemplate = (
                    f"{id_key}: %{{customdata[0]}}<br>"
                    f"pix_x: %{{x}}<br>"
                    f"pix_y: %{{y}}<br>"
                    "<extra></extra>"
                )
            fig.add_traces(scatter_traces.data)            
        
        if return_figure:
            return fig
        else:
            fig.show(config={'modeBarButtonsToAdd':['drawrect',
                                                    'eraseshape'],
                             'scrollZoom': True})

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
        The basename of the JWST file, following the naming convention ``jw<ppppp><ooo><vvv>_<gg><s><aa>_<eeeee>_<detector>``.
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
    """``jw<ppppp><ooo><vvv>_<gg><s><aa>_<eeeee>_<detector>``"""

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
                         other: Self | FootPrint,
                         stage_with_wcs: str = '2b',
                         overlap_fraction: float = 0.6,
                         same_instrument: bool = True,
                         ) -> bool:
        """
        Check if the two JwstInfo instances have the same pointing based on their footprints.
        
        Parameters
        ----------
        other : JwstInfo | FootPrint
            The other JwstInfo instance to compare with, or a FootPrint instance.
        stage_with_wcs : str, optional
            The stage to use for the WCS comparison, by default '2b'.
        overlap_fraction : float, optional
            The minimum overlap fraction required to consider the pointings the same, by default 0.8, maximum is 1.0.
        same_instrument : bool, optional
            If True, also check if the filter, detector, and pupil match between the two JwstInfo instances.
            If False, only check the overlap of the footprints, by default True. Only applicable when `other` is a JwstInfo instance.
        
        Returns
        -------
        bool
            True if the overlapped area is greater than at least `overlap_fraction` of either footprint area
        """
        if isinstance(other, FootPrint):
            other_fp = other
        else:
            if stage_with_wcs not in self.cover_dict or stage_with_wcs not in other.cover_dict:
                raise ValueError(f"Stage '{stage_with_wcs}' not found in cover_dict.")
            if self.cover_dict[stage_with_wcs].footprint is None or other.cover_dict[stage_with_wcs].footprint is None:
                raise ValueError(f"Footprint for stage '{stage_with_wcs}' is not available.")
            other_fp = other[stage_with_wcs].footprint
        self_fp = self[stage_with_wcs].footprint
        overlap_area = self_fp.polygon.intersection(other_fp.polygon).area
        self_area = self_fp.polygon.area
        other_area = other_fp.polygon.area
        if overlap_area / self_area >= overlap_fraction or overlap_area / other_area >= overlap_fraction:
            if isinstance(other, FootPrint):
                return True
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

    def read_catalog(
        self,
        catalog: DataFrame,
        ra_key: str = 'ra',
        dec_key: str = 'dec',
        id_key: str = 'id',
        stage_with_wcs: str = '2b'
    ):
        return self[stage_with_wcs].read_catalog(
            catalog=catalog,
            ra_key=ra_key,
            dec_key=dec_key,
            id_key=id_key
        )
    
    def extract(
        self,
        ra: float,
        dec: float,
        stage_with_wcs: str = '2b',
        aperture_size: int = 40,
        wave_end_short: float = 3.8,
        wave_end_long: float = 5.0
    ) -> dict[str, np.ndarray]:
        """
        Extract the 2D spectrum for a source at the given RA and Dec from a grism observation.
        
        Parameters
        ----------
        ra : float
            Right Ascension of the source in degrees.
        dec : float
            Declination of the source in degrees.
        stage_with_wcs : str, optional
            The stage to use for the WCS, by default '2b'.
        aperture_size : int, optional
            The size of the aperture in pixels, by default 40.
        wave_end_short : float, optional
            The short wavelength end in microns, by default 3.8.
        wave_end_long : float, optional
            The long wavelength end in microns, by default 5.0.
        
        Returns
        -------
        dict[str, np.ndarray]
            A dictionary containing:
            - 'wavelength': 1D array of wavelengths in microns.
            - 'spectrum_2d': 2D array of the extracted spectrum.
            - 'error_2d': 2D array of the errors associated with the extracted spectrum.
            - 'world_short': [RA, Dec] of the point at the short wavelength end.
            - 'world_long': [RA, Dec] of the point at the long wavelength end.
        """
        
        wcs = self[stage_with_wcs].wcs
        data = self[stage_with_wcs].data
        err = self[stage_with_wcs].err
        
        if self.pupil == 'GRISMR':
            world_to_grism = wcs.get_transform('world', 'grism_detector') # get the transform from world coordinates (ra, dec, wavelength, order) to grism detector coordinates (x_trace, y_trace, x_source, y_source, order)
            grism_to_detector = wcs.get_transform('grism_detector', 'detector') # get the transform from grism detector coordinates to detector coordinates (x_source, y_source, wavelength, order)
            detector_to_grism = wcs.get_transform('detector', 'grism_detector') # get the transform from detector coordinates to grism detector coordinates
            detector_to_world = wcs.get_transform('detector', 'world') # get the transform from detector coordinates to world coordinates
            
            x_s, y_s, x_source, y_source, _ = world_to_grism(
                ra, dec, wave_end_short, 1
            )
            x_l, y_l, x_source, y_source, _ = world_to_grism(
                ra, dec, wave_end_long, 1
            )
            ra_s, dec_s, _, _ = detector_to_world(
                x_s, y_s, wave_end_short, 1
            )
            ra_l, dec_l, _, _ = detector_to_world(
                x_l, y_l, wave_end_long, 1
            )
            Height, Width = data.shape
            
            covered = (
                (0<=x_s<Width and 0<=y_s<Height) or
                (0<=x_l<Width and 0<=y_l<Height)
            )
            if not covered:
                return {
                    'wavelength': np.array([]),
                    'spectrum_2d': np.array([[]]),
                    'error_2d': np.array([[]]),
                    'world_short': np.array([ra_s, dec_s]),
                    'world_long': np.array([ra_l, dec_l]),
                }
            
            clamp_start = max(0, int(np.floor(min(x_s, x_l))))
            clamp_end = min(Width-1, int(np.ceil(max(x_s, x_l))))
            x_pixels = np.arange(clamp_start, clamp_end+1, 1)
            
            # Pixel coordinates in grism detector frame -> wavelengths
            _, _, arr_wave, _ = grism_to_detector(
                x_pixels, y_source, x_source, y_source, 1
            ) # here the second argument y_source is not used, just a placeholder
            
            # Wavelengths -> (x_trace, y_trace) in grism detector frame
            x_trace, y_trace, _, _, _ = detector_to_grism(
                x_source, y_source, arr_wave, 1
            )
            spec_ny = aperture_size
            spec_nx = len(x_pixels)
            
            spec_2d = np.full((spec_ny, spec_nx), np.nan)
            err_2d = np.full((spec_ny, spec_nx), np.nan)
            
            # Extract the spectrum
            for i, (xt, yt) in enumerate(zip(x_trace, y_trace)):
                x_col = int(np.round(xt))
                y0 = int(np.round(yt)) - spec_ny // 2
                y1 = y0 + spec_ny
                trim_start = max(0, -y0)
                trim_end = max(0, y1 - Height)
                y0 = max(0, y0)
                y1 = min(Height, y1)
                spec_2d[trim_start:spec_ny - trim_end, i] = data[y0:y1, x_col]
                err_2d[trim_start:spec_ny - trim_end, i] = err[y0:y1, x_col]
            
            
            order_idx = np.argsort(arr_wave)
            return {
                'wavelength': arr_wave[order_idx],
                'spectrum_2d': spec_2d[:, order_idx],
                'error_2d': err_2d[:, order_idx],
                'world_short': np.array([ra_s, dec_s]),
                'world_long': np.array([ra_l, dec_l]),
            }
            
    def plotly_imshow(self,
                        stages: list[str] | None = None,
                        stage_types: list[Literal['data', 'mask']] | None = None,
                        data: list[np.ndarray] | None = None,
                        mask: list[np.ndarray] | None = None,
                        stage_wcs: str | None = None,
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
            
        wcs_transform = None
        if stage_wcs is not None:
            wcs = self[stage_wcs].wcs
            if wcs is None:
                raise ValueError(f"WCS for stage '{stage_wcs}' is None.")
            
            if self.pupil in ['CLEAR']:
                wcs_transform = wcs.get_transform('detector', 'world')
            if self.pupil in ['GRISMR', 'GRISMC']:
                wcs_transform_temp = wcs.get_transform('detector', 'world')
                wcs_transform = lambda x, y: wcs_transform_temp(x, y, np.full_like(x, 0), np.full_like(x, 1))

        fig = plotly_figure_and_mask(
            data=data,
            mask=mask,
            wcs_transform=wcs_transform,
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
