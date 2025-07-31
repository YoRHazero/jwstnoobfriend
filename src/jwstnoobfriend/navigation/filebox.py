from typing import Annotated, Any, Literal, Self, Callable
import functools
import re
import os
import asyncer
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from functools import partial
from rich.table import Table
from collections import Counter
import random
from pathlib import Path
from pydantic import BaseModel, FilePath, field_validator, model_validator, computed_field, Field, validate_call, DirectoryPath
from dash import Dash, html, dcc
import plotly.colors as pc
import plotly.graph_objects as go
import numpy as np

from jwstnoobfriend.navigation.jwstinfo import JwstInfo, JwstCover
from jwstnoobfriend.navigation.footprint import FootPrint
from jwstnoobfriend.utils.log import getLogger
from jwstnoobfriend.utils.environment import load_environment
from jwstnoobfriend.utils.display import track, console, plotly_sky_figure

logger = getLogger(__name__)

class FileBox(BaseModel):
    infos: Annotated[dict[str,JwstInfo], Field(
        description="List of JwstInfo objects containing information about each file.",
        default_factory=dict
    )]
    
    proposal_ids: Annotated[list[str], Field(
        description="List of proposal IDs in the FileBox.",
        default_factory=list
    )]
    
    @field_validator('proposal_ids', mode='after')
    def check_unique_proposal_ids(cls, proposal_ids: list[str]) -> list[str]:
        """Ensure proposal IDs are unique."""
        unique_ids = sorted(list(set(proposal_ids)))
        if len(unique_ids) != len(proposal_ids):
            logger.warning("Duplicate proposal IDs found, keeping only unique ones.")
        return unique_ids

    @property
    def info_list(self) -> list[JwstInfo]:
        """Returns a list of JwstInfo objects"""
        return list(self.infos.values())
    
    @property
    def basenames(self) -> list[str]:
        """Returns a list of basenames of the JwstInfo objects"""
        return list(self.infos.keys())

    @property
    def filters(self) -> dict[str, str]:
        """Returns a dictionary of filters for each file, keyed by the basename"""
        return {info.basename: info.filter for info in self.info_list if info.filter is not None}

    @property
    def pupils(self) -> dict[str, str]:
        """Returns a dictionary of pupils for each file, keyed by the basename"""
        return {info.basename: info.pupil for info in self.info_list if info.pupil is not None}

    @property
    def detectors(self) -> dict[str, str]:
        """Returns a dictionary of detectors for each file, keyed by the basename"""
        return {info.basename: info.detector for info in self.info_list if info.detector is not None}

    @property
    def stages(self) -> list[str]:
        """Returns a list of stages available in the JwstInfo objects"""
        if not self.infos:
            return []
        # Get the stages from the first JwstInfo object
        sample_info: JwstInfo = list(self.infos.values())[0]
        return list(sample_info.cover_dict.keys())
    
    @property
    def filesetnames(self) -> list[str]:
        """Returns a list of fileset names in the FileBox."""
        return [info.filesetname for info in self.info_list]

    @property
    def cover_dicts(self) -> dict[str, JwstCover]:
        """Returns a dictionary of JwstCover object dict for each file, keyed by the basename."""
        return {info.basename: info.cover_dict for info in self.info_list}

    def footprints(self, stage:str | None = None) -> dict[str, FootPrint]:
        """Returns a dictionary of footprints for each file, keyed by the basename.
        If stage is provided, it will be used to filter the footprints.
        If stage is None, it will use the first available stage from the first JwstInfo.
        
        Note
        ----
        If the stage does not exist in all JwstInfo objects, it will be ignored for those objects.
        """
        ## get a key where the footprint in that JwstCover is not None
        if stage is None:
            sample_info: JwstInfo = self.infos[0] if self.infos else None
            for key, cover in sample_info.cover_dict.items():
                if cover.footprint is not None:
                    stage = key
                    break
        return {info.basename: info.cover_dict[stage].footprint for info in self.infos if stage in info.cover_dict}        
    
    ## Methods for manipulating the FileBox
    @validate_call
    def update(self, *, infos: dict[str, JwstInfo] | list[JwstInfo] | JwstInfo) -> Self:
        """
        Update the FileBox with new JwstInfo objects.
        If a JwstInfo with the same basename already exists, it will merge the new information with the existing one.
        
        Parameters
        ----------
        infos : dict[str, JwstInfo] | list[JwstInfo] | JwstInfo
            The JwstInfo objects to add or update in the FileBox. If a dictionary is provided, the keys should be the basenames of the JwstInfo objects.
        
        Returns
        -------
        FileBox
            The updated FileBox with the new JwstInfo objects.
        """
        if isinstance(infos, JwstInfo):
            infos = {infos.basename: infos}
        elif isinstance(infos, list):
            infos = {info.basename: info for info in infos}
        
        ## extract proposal IDs from the basenames of the JwstInfo objects
        for info in infos.values():
            proposal_id_match = re.search(r'jw(\d{5})', info.basename)
            if proposal_id_match:
                proposal_id = proposal_id_match.group(1)
                if proposal_id not in self.proposal_ids:
                    self.proposal_ids.append(proposal_id)
        ## If the Basename already exists in the infos, merge the new JwstInfo with the existing one.
        for key, info in infos.items():
            if key in self.infos:
                self.infos[key].merge(info)
            else:
                self.infos[key] = info
        return self
    
    @validate_call
    def update_from_file(self, *, filepath: FilePath, stage: str, force_with_wcs: bool = False) -> Self:
        """Add a new JwstInfo to the infos from a file path. If the file already exists, it will merge the new information with the existing one.
        
        Parameters
        ----------
        filepath : FilePath
            The path to the file to be added.
        stage : str
            The stage of the JwstCover to be added.
        force_with_wcs : bool, optional
            If True, the file is assumed to have a WCS object assigned regardless of its suffix.
        """
        info = JwstInfo.new(filepath=filepath, stage=stage, force_with_wcs=force_with_wcs)
        ## extract proposal ID from the basename of the file
        self.update(infos=info)
        return self
    
    @validate_call
    def update_from_folder(self, *, folder_path: DirectoryPath, stage: str, wildcard: str = '*.fits', force_with_wcs: bool = False) -> Self:
        """
        Updates the FileBox with JwstInfo objects from a folder containing files.
        It will create a JwstInfo for each file that matches the wildcard.
        
        Parameters
        ----------
        folder_path : DirectoryPath
            The path to the folder containing the files.
        stage : str
            The stage of the JwstCover to be added for each file.
        wildcard : str, optional
            The wildcard pattern to match files in the folder. Default is '*.fits'.
        force_with_wcs : bool, optional
            If True, the files are assumed to have a WCS object assigned regardless of their suffix.
        
        Returns
        -------
        FileBox
            The updated FileBox with the new JwstInfo objects.
        """
        new_box = self.__class__.init_from_folder(
            stage=stage,
            folder_path=folder_path,
            wildcard=wildcard,
            force_with_wcs=force_with_wcs,
            method='parallel')
        self.merge(new_box)
        return self
    
    @classmethod
    async def _infos_from_folder_async(cls,
                                      *,
                                      folder_path: DirectoryPath,
                                      stage: str,
                                      wildcard: str = '*.fits',
                                      force_with_wcs: bool = False) -> list[JwstInfo]:
        infos = []
        tasks = []
        async with asyncer.create_task_group() as task_group:
            for filepath in folder_path.glob(wildcard):
                if filepath.is_file():
                    task = task_group.soonify(
                        JwstInfo._new_async
                    )(filepath=filepath, stage=stage, force_with_wcs=force_with_wcs)
                    tasks.append(task)
        for task in tasks:
            infos.append(task.value)
        return infos
            
    @classmethod
    @validate_call
    def init_from_folder(cls, *, stage: str, folder_path: DirectoryPath | None = None,
                         wildcard='*.fits', force_with_wcs: bool = False, method: Literal['async', 'parallel', 'loop'] = 'parallel') -> Self:
        """
        Initializes the FileBox from a folder containing files. It will create a JwstInfo for each file that matches the wildcard.
        
        Parameters
        ----------
        stage : str
            The stage of the JwstCover to be added for each file.
        folder_path : DirectoryPath | None, optional
            The path to the folder containing the files. If None, it will use the environment variable
            STAGE_{stage.upper()}_PATH to find the folder path.
            If the environment variable is not set, it will raise a ValueError.
        wildcard : str, optional
            The wildcard pattern to match files in the folder. Default is '*.fits'.
        force_with_wcs : bool, optional
            If True, the files are assumed to have a WCS object assigned regardless of their suffix
        method : Literal['async', 'parallel', 'loop'], optional
            The method to use for loading files. 'async' will load files asynchronously, 'parallel' will use parallel processing.
            'loop' will load files in a loop, should be used for small numbers of files.
            Default is 'parallel'.
        """
        ## Get the folder path from the environment variable
        self = cls(infos={}, proposal_ids=[])
        if folder_path is None:
            load_environment()
            folder_path = os.getenv(f"STAGE_{stage.upper()}_PATH", None)
            if folder_path is None:
                raise ValueError(f"Folder path for stage '{stage}' is not set in the environment variables. Please provide a valid folder path manually.")
            else:
                folder_path = DirectoryPath(folder_path)
        ## IO operation to load files
        match method:
            case 'async':
                infos = asyncer.runnify(self._infos_from_folder_async)(folder_path=folder_path, stage=stage, wildcard=wildcard, force_with_wcs=force_with_wcs)
                self.update(infos=infos)
            case 'parallel':
                valid_files = [f for f in folder_path.glob(wildcard) if f.is_file()]
                # make closure to convert JwstInfo.new to a partial function
                new_info_func = partial(JwstInfo.new, stage=stage, force_with_wcs=force_with_wcs)
                # Use ProcessPoolExecutor to parallelize the creation of JwstInfo objects
                with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
                    infos = list(executor.map(
                        new_info_func,
                        valid_files
                    ))
                self.update(infos=infos)
            case 'loop':
                for filepath in folder_path.glob(wildcard):
                    if filepath.is_file():
                        self.update_from_file(filepath=filepath, stage=stage, force_with_wcs=force_with_wcs)
        return self

    def merge(self, other: Self) -> Self:
        """
        Merges another FileBox into this one. If a JwstInfo with the same basename exists, it will merge the information.
        Else, it will add the new JwstInfo to the infos dictionary.
        
        Parameters
        ----------
        other : FileBox
            The FileBox to merge into this one.
            
        Returns
        -------
        FileBox
            The updated FileBox with merged information.
            
        Side Effects
        -------
        Modifies the current FileBox instance by merging the JwstInfo objects from the other FileBox.
        """
        ## Merge JwstInfo objects, if they have the same basename, merge their cover_dicts 
        for key, info in other.infos.items():
            if key in self.infos:
                self.infos[key].merge(info)
            else:
                self.infos[key] = info
                logger.warning(f"Adding new JwstInfo with basename {key} to FileBox.")
                
        ## Merge proposal IDs
        for proposal_id in other.proposal_ids:
            if proposal_id not in self.proposal_ids:
                self.proposal_ids.append(proposal_id)
        return self
    
    @validate_call
    def select(self, condition: dict[str, list[Any]] | None = None, predicate: Callable[[JwstInfo], bool] | None = None) -> Self:
        """
        Selects JwstInfo objects from the FileBox based on specified conditions.
        
        Parameters
        ----------
        condition : dict[str, list[Any]] | None
            A dictionary where keys are attribute names of JwstInfo objects and values are 
            lists of acceptable values to match. The selection uses AND logic between different 
            attributes (all conditions must be satisfied) and OR logic within each attribute's 
            value list (any value in the list can match).
            
            Supported attribute keys include:
            - 'filter': Filter names (e.g., ['F200W', 'F115W'])
            - 'detector': Detector names (e.g., ['NRCA1', 'NRCA2'])
            - 'pupil': Pupil names (e.g., ['CLEAR', 'GRISMR'])
            - 'basename': File basenames
            - Any other valid JwstInfo attribute
            
        predicate : Callable[[JwstInfo], bool] | None
            A predicate function that takes a JwstInfo object and returns True if it 
            should be selected. If both condition and predicate are provided, the 
            result will contain the union of both selections.

        Returns
        -------
        Self
            A new FileBox containing only the JwstInfo objects that match the criteria.
            
        Raises
        ------
        ValueError
            If neither condition nor predicate is provided, or if condition contains 
            invalid attribute keys that don't exist in JwstInfo objects.
            
        Examples
        --------
        ```python
        # Select files with specific filters and detectors
        selected_box = filebox.select(condition={
            'filter': ['F200W', 'F115W'],
            'detector': ['NRCA1']
        })
        
        # Select files using a predicate function
        selected_box = filebox.select(predicate=lambda info: 'F200W' in info.filter)
        ```
        Notes
        -----
        - The condition dictionary uses AND logic between different keys and OR logic 
        within each key's value list
        - For example: {'filter': ['F200W', 'F115W'], 'detector': ['NRCA1']} 
        means "(filter is F200W OR F115W) AND (detector is NRCA1)"
        - All attribute values are compared using Python's `in` operator with exact matching
        - Use predicate functions for more complex selection logic like partial string 
        matching or numerical comparisons
        """
        selected_infos = {}
        if condition is None and predicate is None:
            raise ValueError("At least one of 'condition' or 'predicate' must be provided.")
        if condition:
            try:
                for info in self.info_list:
                    matched = True
                    for key, values in condition.items():
                        if getattr(info, key) not in values:
                            matched = False
                            break
                    if matched:
                        selected_infos[info.basename] = info
            except AttributeError as e:
                raise ValueError(f"Invalid attribute in condition: {e}")
            
        if predicate:
            for info in self.info_list:
                if predicate(info):
                    selected_infos[info.basename] = info
        return FileBox(infos=selected_infos)

    @validate_call
    def save(self, filepath: Path | None = None, force_overwrite: bool = False) -> None:
        """
        Saves the FileBox to a file.
        If filepath is None, it will use the environment variable FILE_BOX_PATH or default to 'noobox.json' in the current directory.
        """
        if filepath is None:
            filepath = os.getenv('FILE_BOX_PATH', 'noobox.json')
            filepath = Path(filepath)
        if filepath.exists() and not force_overwrite:
            old_box = self.load(filepath=filepath)
            if len(old_box) > len(self):
                raise ValueError(f"FileBox at {filepath} already exists and has more files ({len(old_box)}) than the current FileBox ({len(self)}). Use force_overwrite=True to overwrite if this is desired or save in a different file.")
        with open(filepath, 'w') as f:
            f.write(self.model_dump_json(indent=4))
    
    @classmethod
    @validate_call
    def load(cls, filepath: FilePath | None = None) -> Self:
        """Loads a FileBox from a file."""
        if filepath is None:
            filepath = os.getenv('FILE_BOX_PATH', 'noobox.json')
            filepath = FilePath(filepath)
        with open(filepath, 'r') as f:
            data = f.read()
        return cls.model_validate_json(data)
    
    
    ## Special methods for accessing JwstInfo objects

    def __getitem__(self, key: str | int) -> JwstInfo:
        """Returns the JwstInfo object for the given key or index."""
        if isinstance(key, int):
            return self.info_list[key]
        else:
            key = JwstInfo.extract_basename(key)  # Automatically extract basename if needed
        return self.infos[key]
    
    def __len__(self) -> int:
        """Returns the number of JwstInfo objects in the FileBox."""
        return len(self.infos)
    
    def __contains__(self, key: str) -> bool:
        """Checks if the given key exists in the FileBox."""
        return key in self.infos
    
    def __iter__(self):
        """Returns an iterator over the JwstInfo objects in the FileBox with their basenames as keys."""
        return iter(self.infos.items())
    
    ## Methods for grouping, and filtering
    def summary(self) -> None:
        combinations = []
        for info in self.info_list:
            combinations.append((info.pupil, info.filter, info.detector))        
        combo_counts = Counter(combinations)
        
        nested_data = {}
        for (pupil, filter_, detector), count in combo_counts.items():
            if pupil not in nested_data:
                nested_data[pupil] = {}
            if filter_ not in nested_data[pupil]:
                nested_data[pupil][filter_] = {}
            nested_data[pupil][filter_][detector] = count
        
        main_table = Table(title=f"FileBox Summary ({len(self)} files)",)
        main_table.add_column("Pupil", style="bold cyan")
        main_table.add_column("Filter       Detector      Count", style="white")

        for pupil in sorted(nested_data.keys()):
            filters = nested_data[pupil]
            filter_table = Table(show_header=False, box=None, padding=(0, 1))
            filter_table.add_column("Filter", style="bold magenta", width=12)
            filter_table.add_column("Detectors", style="white")
            
            for filter_ in sorted(filters.keys()):
                detectors = filters[filter_]
                detector_table = Table(show_header=False, box=None, padding=(0, 0))
                detector_table.add_column("Detector", style="bold green", width=10)
                detector_table.add_column("Count", style="bold yellow", justify="right", width=6)
                
                for detector in sorted(detectors.keys()):
                    count = detectors[detector]
                    detector_table.add_row(detector, str(count))
                
                filter_table.add_row(filter_, detector_table)
            
            main_table.add_row(pupil, filter_table)
        console.print(main_table)
    
    @validate_call
    def random_sample(self, size: Annotated[int, Field(gt=0)] = 10, attrs_in_sample: list[str] | None = None) -> Self:
        """
        Returns a new FileBox containing a random sample of JwstInfo objects from the current FileBox.
        
        Parameters
        ----------
        size : int, optional
            The number of JwstInfo objects to include in the sample. Default is 10.
            
        Returns
        -------
        FileBox
            A new FileBox containing a random sample of JwstInfo objects.
        """
        if size > len(self):
            size = len(self)
        
        if attrs_in_sample is None:
            attrs_in_sample = ['pupil']   
            
        sampled_infos = random.sample(self.info_list, size)        
        
        sampled_types = set(tuple(getattr(info, attr) for attr in attrs_in_sample) for info in sampled_infos)
        total_types = set(tuple(getattr(info, attr) for attr in attrs_in_sample) for info in self.info_list)
        

        if len(total_types) > size:
            diff = len(total_types) - size
            logger.warning(f"Total unique combinations ({len(total_types)}) exceed sample size ({size}). The example cannot cover all types.")
        else:
            diff = 0
        
        attempt = 0
        max_attempts = 1000
        while attempt < max_attempts:
            attempt += 1
            if (len(total_types) - len(sampled_types)) <= diff:
                break
            sampled_infos = random.sample(self.info_list, size)
            sampled_types = set(tuple(getattr(info, attr) for attr in attrs_in_sample) 
                                    for info in sampled_infos)
        return self.__class__(infos={info.basename: info for info in sampled_infos})
        
    def example(self, target: JwstInfo | None = None) -> Self:
        if target is None:
            target = random.choice(self.info_list)
        
        match_regex = r"jw\d{5}\d{3}\d{3}_\d{5}"
        matched_index = re.match(match_regex, target.basename)
        if not matched_index:
            raise ValueError(f"'{target.basename}' does not follow the naming convention for JWST files.")

        # Extract the relevant parts from the matched index
        jw_index = matched_index.group(0)
        return self.__class__(infos={info.basename: info 
                                     for info in self.info_list 
                                     if jw_index in info.basename})
        
        
    ## Methods for visualization
    @classmethod
    def sky_figure(cls,
                projection_type: str = "orthographic",
                showlatgrid: bool = True,
                showlongrid: bool = True,
                lataxis_dtick: int = 90,
                lonaxis_dtick: int = 90,
                gridcolor: str = "gray",
                griddash: str = "dash",) -> go.Figure:
        """
        Creates a Plotly figure representing the sky projection with specified parameters.
        
        Parameters
        ----------
        projection_type : str, optional
            The type of sky projection to use. Default is "orthographic".
        showlatgrid : bool, optional
            Whether to show latitude grid lines. Default is True.
        showlongrid : bool, optional
            Whether to show longitude grid lines. Default is True.
        lataxis_dtick : int, optional
            The tick interval for latitude axis. Default is 90.
        lonaxis_dtick : int, optional
            The tick interval for longitude axis. Default is 90.
        gridcolor : str, optional
            The color of the grid lines. Default is "gray".
        griddash : str, optional
            The dash style of the grid lines. Default is "dash".
            
        Returns
        -------
        go.Figure
            A Plotly figure object representing the sky projection with the specified parameters.
        """
        return plotly_sky_figure(
            projection_type=projection_type,
            showlatgrid=showlatgrid,
            showlongrid=showlongrid,
            lataxis_dtick=lataxis_dtick,
            lonaxis_dtick=lonaxis_dtick,
            gridcolor=gridcolor,
            griddash=griddash)
        
    def show_footprints(
        self,
        fig: go.Figure | None = None,
        stage: str = '2b',
        show_more: bool = True,
        fig_mode: Literal['sky', 'cartesian'] = 'sky',
        color_by: list[str] | None = None,
        color_list: list[str] | None = None,
        **kwargs
    ) -> go.Figure:
        if fig is None:
            match fig_mode:
                case 'sky':
                    fig = self.sky_figure()
                case 'cartesian':
                    fig = go.Figure()
                    fig.update_layout(
                        dragmode='pan',
                        xaxis=dict(
                            fixedrange=False,  
                        ),
                        yaxis=dict(
                            fixedrange=False, 
                        )
                    )
        
        if color_by:
            for attr in color_by:
                if not hasattr(self[0], attr):
                    raise ValueError(f"Attribute '{attr}' not found in FileBox. Available attributes: {list(self.__dict__.keys())}")
            if kwargs.pop('color', None) is not None:
                raise ValueError("Cannot use 'color' in kwargs when 'color_by' is specified. Use 'color_by' to specify the attribute for coloring.")        
            if color_list is None:
                color_list = pc.qualitative.Plotly + pc.qualitative.Set1 + pc.qualitative.Set2 + pc.qualitative.Set3
            
            color_map = {}
            unique_combinations = set()
            for info in self.info_list:
                combo = tuple(getattr(info, attr) for attr in color_by)
                unique_combinations.add(combo)
            
            if len(unique_combinations) > len(color_list):
                raise ValueError(f"Too many unique combinations ({len(unique_combinations)}) for the provided color list. Please provide a longer color list or reduce the number of unique combinations.")
            
            for i, combo in enumerate(unique_combinations):
                color_map[combo] = color_list[i]
                
        for info in self.info_list:
            if stage not in info.cover_dict:
                logger.warning(f"Stage '{stage}' not found in JwstInfo {info.basename}. Skipping.")
                continue
            
            if color_by:
                kwargs['color'] = color_map[tuple(getattr(info, attr) for attr in color_by)]
            
            info.plotly_add_footprint(
                fig=fig,
                stage=stage,
                show_more=show_more,
                fig_mode=fig_mode,
                **kwargs
            )
        
        return fig