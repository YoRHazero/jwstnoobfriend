from typing import Annotated, Any, Literal
import re
import os
import asyncer
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from functools import partial
from pydantic import BaseModel, FilePath, field_validator, model_validator, computed_field, Field, validate_call, DirectoryPath
from jwstnoobfriend.navigation.jwstinfo import JwstInfo, JwstCover
from jwstnoobfriend.navigation.footprint import FootPrint
from jwstnoobfriend.utils.log import getLogger
from jwstnoobfriend.utils.environment import load_environment
from jwstnoobfriend.utils.display import track

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
    def update(self, *, infos: dict[str, JwstInfo] | list[JwstInfo] | JwstInfo):
        """
        Update the FileBox with new JwstInfo objects.
        If a JwstInfo with the same basename already exists, it will merge the new information with the existing one.
        
        Parameters
        ----------
        infos : dict[str, JwstInfo] | list[JwstInfo] | JwstInfo
            The JwstInfo objects to add or update in the FileBox. If a dictionary is provided, the keys should be the basenames of the JwstInfo objects.
        
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
    
    @validate_call
    def update_from_file(self, *, filepath: FilePath, stage: str, force_with_wcs: bool = False) -> None:
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
                    task = task_group.create_task(
                        JwstInfo._new_async(filepath=filepath, stage=stage, force_with_wcs=force_with_wcs)
                    )
                    tasks.append(task)
        for task in tasks:
            infos.append(task.value)
        return infos
            

    @validate_call
    def init_from_folder(self, *, stage: str, folder_path: DirectoryPath | None = None,
                         wildcard='*.fits', force_with_wcs: bool = False, method: Literal['normal', 'async', 'parallel'] = 'normal') -> None:
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
        method : Literal['normal', 'async', 'parallel'], optional
            The method to use for loading files. 'normal' will load files sequentially, 'async' will load files asynchronously,
            and 'parallel' will load files with multiprocessing.
        """
        ## Get the folder path from the environment variable
        if folder_path is None:
            load_environment()
            folder_path = os.getenv(f"STAGE_{stage.upper()}_PATH", None)
            if folder_path is None:
                raise ValueError(f"Folder path for stage '{stage}' is not set in the environment variables. Please provide a valid folder path manually.")
            else:
                folder_path = DirectoryPath(folder_path)
        ## IO operation to load files
        if method == 'normal':
            for filepath in track(list(folder_path.glob(wildcard))):
                if filepath.is_file():
                    self.update_from_file(filepath=filepath, stage=stage, force_with_wcs=force_with_wcs)
        elif method == 'async':
            infos = asyncer.runnify(self._infos_from_folder_async)(folder_path=folder_path, stage=stage, wildcard=wildcard, force_with_wcs=force_with_wcs)
            self.update(infos=infos)
        elif method == 'parallel':
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

    def merge(self, other: 'FileBox') -> 'FileBox':
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
                
        ## Merge proposal IDs
        for proposal_id in other.proposal_ids:
            if proposal_id not in self.proposal_ids:
                self.proposal_ids.append(proposal_id)
        return self
    
    def select(self, condition: dict[str: Any]) -> 'FileBox':
        """
        Selects JwstInfo objects from the FileBox based on a condition.
        
        Parameters
        ----------
        condition : dict[str, Any]
            A dictionary where keys are attributes of JwstInfo and values are the values to match.
        
        Returns
        -------
        FileBox
            A new FileBox containing only the JwstInfo objects that match the condition.
        """
        selected_infos = {}
        for key, value in condition.items():
            if not hasattr(JwstInfo, key):
                raise ValueError(f"Invalid condition key: {key}")
            for info in self.info_list:
                if getattr(info, key) == value:
                    selected_infos[info.basename] = info
        return FileBox(infos=selected_infos)

    def save(self, filepath: FilePath | None = None) -> None:
        """
        Saves the FileBox to a file.
        If filepath is None, it will use the environment variable FILE_BOX_PATH or default to 'noobox.json' in the current directory.
        """
        if filepath is None:
            filepath = os.getenv('FILE_BOX_PATH', 'noobox.json')
            filepath = FilePath(filepath)
        with open(filepath, 'w') as f:
            f.write(self.model_dump_json(indent=4))
    
    @classmethod
    def load(cls, filepath: FilePath) -> 'FileBox':
        """Loads a FileBox from a file."""
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
    
    ## Methods for visualization, grouping, and filtering
    
    