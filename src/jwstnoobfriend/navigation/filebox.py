from typing import Annotated
from pydantic import BaseModel, FilePath, field_validator, model_validator, computed_field, Field, validate_call
from jwstnoobfriend.navigation.jwstinfo import JwstInfo, JwstCover
from jwstnoobfriend.navigation.footprint import FootPrint

class FileBox(BaseModel):
    infos: Annotated[dict[str,JwstInfo], Field(
        description="List of JwstInfo objects containing information about each file.",
        default_factory=dict
    )]
    
    @property
    def info_list(self) -> list[JwstInfo]:
        """Returns a list of JwstInfo objects"""
        return list(self.infos.values())

    @property
    def filters(self) -> dict[str, str]:
        """Returns a dictionary of filters for each file, keyed by the basename"""
        return {info.basename: info.filter for info in self.infos.values() if info.filter is not None}

    @property
    def pupils(self) -> dict[str, str]:
        """Returns a dictionary of pupils for each file, keyed by the basename"""
        return {info.basename: info.pupil for info in self.infos.values() if info.pupil is not None}

    @property
    def detectors(self) -> dict[str, str]:
        """Returns a dictionary of detectors for each file, keyed by the basename"""
        return {info.basename: info.detector for info in self.infos.values() if info.detector is not None}

    @property
    def stages(self) -> list[str]:
        """Returns a list of stages available in the JwstInfo objects"""
        if not self.infos:
            return []
        # Get the stages from the first JwstInfo object
        sample_info: JwstInfo = list(self.infos.values())[0]
        return list(sample_info.cover_dict.keys())

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
        if info.basename in self.infos:
            self.infos[info.basename].merge(info)
        else:
            self.infos[info.basename] = info
            
    def save(self, filepath: FilePath) -> None:
        """Saves the FileBox to a file."""
        with open(filepath, 'w') as f:
            f.write(self.model_dump_json(indent=4))
    
    @classmethod
    def load(cls, filepath: FilePath) -> 'FileBox':
        """Loads a FileBox from a file."""
        with open(filepath, 'r') as f:
            data = f.read()
        return cls.model_validate_json(data)        