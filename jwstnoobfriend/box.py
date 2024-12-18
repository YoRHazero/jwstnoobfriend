"""This module provides classes to manage the information of JWST data, including
footprints, file paths, and wcs, etc.
"""

from ._logging import *
import logging
from dataclasses import dataclass, field, fields, make_dataclass
from typing import Any, ClassVar
import sys
import re
logger = logging.getLogger(__name__)
logger = set_logging_level(logger, logging.INFO)
logger = set_logging_file(logger)

@dataclass
class JwstInfo:
    """
    Dataclass to save jwst information.
    
    Parameters
    ----------
    basename : str
        The basename of the file.
    wcs : Any, optional
        The wcs information. The default is None.
    footprint : Any, optional
        The footprint information. The default is None.
    
    Attributes
    ----------
    basic_containers : dict
        The basic containers for the dataclass. Now supports 3 levels of containers. 
        0: {}, 1: {'detector': str, 'pupil':str, 'filter':str, 'filepaths': dict}, 
        2: {'detector': str, 'pupil':str, 'filter':str, 'filepaths': dict, 'other': dict}
    basic_repr : list
        The basic representation of the containers.
    
    
    Notes
    ---------
    
    The following attributes have supported methods:
    
    detectors : str
        The detector name.
    pupil : str
        The pupil name. 
    filter : str
        The filter name.
    filepaths : dict
        The filepaths.
    other : dict
        Potential container for other information.
    
     
    """
    basename: str
    wcs: Any = field(default=None, repr=False)
    footprint: Any = field(default=None, repr=False)
    basic_containers: ClassVar[dict] = {
        0: {},
        1: {'detector': str, 'pupil':str, 'filter':str, 'filepaths': dict, },
        2: {'detector': str, 'pupil':str, 'filter':str, 'filepaths': dict, 'other': dict}
                                        }
    basic_repr: ClassVar[list] = [
        [],
        [True, True, True, False],
        [True, True, True, False, False]
                                ]
    
    @classmethod
    def _register(cls, containers):
        """Register new container to the JwstInfo class variable

        Parameters
        ----------
        containers : int | dict
            the index of default container or a dict of new containers. If a list is provided, the type of each 
            attribute will be set to typing.Any.

        Returns
        -------
        cls_index (int)
            Return the index of the input container

        Raises
        ------
        TypeError
            Type of input containers is not list | dict | int.
        """
        match containers:
            case existed if isinstance(existed, int) and existed in cls.basic_containers.keys():
                cls_index = existed
            case [*attributes]:
                logger.info('Only attribute names are provided, the type will default to typing.Any')
                cls_index = max(cls.basic_containers.keys()) + 1
                cls.basic_containers[cls_index] = {attr:Any for attr in attributes}
            case {**new_container}:
                cls_index = max(cls.basic_containers.keys()) + 1
                cls.basic_containers[cls_index] = new_container
            case others:
                raise TypeError(f"Unsupported type: {type(others)}, please provided dict (recommended) or at least list")
        
        return cls_index
    
    @classmethod
    def with_info(cls, containers = 1, defaults = None, representation = None):
        """Custom dataclass to save jwst information.

        Parameters
        ----------
        containers : int or dict or iterable, optional
            The type of containers to be created. The default is 1.
        defaults : Any, optional
            The default values for the containers. The default is None.
        representation : list, optional
            The representation of the containers. The default is None.
            
        
        Returns
        -------
        new_class : dataclass
            The new dataclass created with the containers.
            
        Raises
        ------
        ValueError
            If the containers index or type is not supported.
        """
        
        match containers:
            case index if isinstance(index, int) and index <= max(cls.basic_containers.keys()):
                cls_index = cls._register(index)
                containers = cls.basic_containers[index]
                if representation is None:
                    representation = cls.basic_repr[index]  
            
            case iterable if hasattr(iterable, '__iter__'):
                cls_index = cls._register(iterable)
                if representation is None:
                    representation = [True] * len(iterable)
                pass
            
            case _:
                raise ValueError(f'Unsupported containers index or type!')
        
        if (not hasattr(defaults, '__iter__')) or isinstance(defaults, str):
            defaults_list = [defaults] * len(containers)
            
        elif len(defaults) != len(containers):
            raise ValueError(f"The length of default values ({len(defaults)=}) does not match that of containers ({len(containers)=})")
        
        else:
            defaults_list = defaults
        
        new_fields = [(f.name, f.type, field(default=f.default, repr=f.repr)) for f in fields(cls)] +\
            [(key, val, field(default = defaults_list[i], repr=representation[i])) for i, (key, val) in enumerate(containers.items())]
        
        new_class = make_dataclass(cls_name=f'Custom{cls.__name__}{cls_index}', fields= new_fields, bases=(cls,))
        module = sys.modules[__name__]
        logger.debug(f'New class {new_class.__name__} registered in module {module.__name__}')
        setattr(module, new_class.__name__, new_class)
        return new_class
    
    @classmethod
    def all_attributes(cls):
        """Return the list of all the attributes of the class"""
        return [f.name for f in fields(cls)]
    
    
############################################################################################################
############################################################################################################    
############################################################################################################
    

from shapely.geometry import Polygon, Point, linestring
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from astropy import units as u
class FootPrint:
    """
    Create a FootPrint object from a list of 4 SkyCoord objects.
    
    Parameters
    ----------
    footprint_coords (list): A list of 4 SkyCoord objects representing the polygon vertices.

    
    Attributes
    ----------
    footprint_coords (list): 
        A list of 4 SkyCoord objects representing the polygon vertices.
    footprint_xys (list): 
        A list of (x, y) tuples representing the polygon vertices.
    center_xy (tuple): 
        A tuple of (x, y) representing the center of the footprint.
    center_sky (SkyCoord): 
        A SkyCoord object representing the center of the footprint.
    footprint (Polygon): 
        A shapely Polygon object representing the footprint.
        
    Raises
    ------
    ValueError: If the number of vertices is not 4.
    
     
    """
    def __init__(self, footprint_coords):
        """
        Create a FootPrint object from a list of 4 SkyCoord objects.
        
        Parameters
        ----------
        footprint_coords (list): A list of 4 SkyCoord objects representing the polygon vertices.
             
        Raises
        ------
        ValueError: If the number of vertices is not 4.
    
        """
        self.footprint_coords = footprint_coords
        self.footprint_xys = [(endpoint.ra.deg, endpoint.dec.deg) for endpoint in footprint_coords]
        if len(self.footprint_xys) != 4:
            raise ValueError('Footprint must have 4 vertices.')
        self._sort_points_and_find_center()
        self.footprint = Polygon(self.footprint_xys)
        
    def _sort_points_and_find_center(self):
        """
        Sort the 4 points in the footprint to make convex polygon and find the center.
        """
        line1 = linestring.LineString(self.footprint_xys[0::2])
        line2 = linestring.LineString(self.footprint_xys[1::2])
        if line1.intersects(line2):
            center = line1.intersection(line2)
            self.center_xy = (center.x, center.y)
            self.center_sky = SkyCoord(*self.center_xy, unit='deg')
        else:
            self.footprint_coords[1], self.footprint_coords[2] = self.footprint_coords[2], self.footprint_coords[1]
            self.footprint_xys[1], self.footprint_xys[2] = self.footprint_xys[2], self.footprint_xys[1]
            line1 = linestring.LineString(self.footprint_xys[0::2])
            line2 = linestring.LineString(self.footprint_xys[1::2])
            center = line1.intersection(line2)
            self.center_xy = (center.x, center.y)
            self.center_sky = SkyCoord(*self.center_xy, unit='deg')
    
    def _is_point_in(self, point_coord):
        """
        Check if a point is inside the footprint.
        
        Parameters
        ----------
        point_coord (SkyCoord): The point to check.        
        """
        point_xy = (point_coord.ra.deg, point_coord.dec.deg)
        point = Point(point_xy)
        return self.footprint.contains(point)
    def are_points_in(self, point_coords):
        """
        Check if a list of points are inside the footprint.
        
        Parameters
        ----------
        point_coords (list): A list of SkyCoord objects representing the points, or a single SkyCoord object.
        
        Returns
        -------
        list: A list of bools indicating whether each point is inside the footprint.
        """
        if isinstance(point_coords, SkyCoord):
            return self._is_point_in(point_coords)
        else:
            return [self._is_point_in(point_coord) for point_coord in point_coords]
    
    def is_overlap(self, other):
        """
        Check if the footprint of this object overlaps with another FootPrint object.
        
        Parameters
        ----------
        other (FootPrint): Another FootPrint object.
        """
        return self.footprint.intersects(other.footprint)
    
    def plot_footprint(self, ax=None, color='forestgreen', if_fill=False, **kwargs):
        """
        Plot the footprint on a given axis.
        
        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            The axis to plot the footprint. The default is None.
        color : str, optional
            The color of the footprint. The default is 'forestgreen'.
        if_fill : bool, optional
            If fill the footprint. The default is False.
        kwargs : dict
            Additional keyword arguments to pass to the plot
        
        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
            The axis with the footprint plotted.
        """
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            ax.set_xlabel('RA')
            ax.set_ylabel('Dec')
            ax.set_title('Footprint')
        
        plot_kwargs = {'color': color, 'ls': '--'}
        plot_kwargs.update(kwargs)
        
        ax.plot(*self.footprint.exterior.xy, **plot_kwargs)
        
        if if_fill:
            ax.fill(*self.footprint.exterior.xy, color=color, alpha=0.2)
        return ax
    
    def distance(self, other, d_unit = u.arcmin):
        """
        Calculate the distance between the centers of two footprints.
        
        Parameters
        ----------
        other : FootPrint
            The other FootPrint object.
        d_unit : astropy.units.Unit, optional
            The unit of the distance. The default is u.arcmin.
        
        Returns
        -------
        astropy.units.Quantity
            The distance between the centers of the two footprints.
        """
        return self.center_sky.separation(other.center_sky).to(d_unit)
    
    def __repr__(self):
        return f'FootPrint({self.footprint_coords})'


############################################################################################################
############################################################################################################
############################################################################################################


import os
import jwst.datamodels as dm
import pickle
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
class FileBox:
    """Collection of basic information using JwstInfo

    Parameters
    ----------
    container : int, optional
        See JwstInfo, by default 0. Can be 0, 1, 2 and a dict of new containers.
        For the attributes in basic_containers, related methods could be find in this class.
    defaults : Any, optional
        See JwstInfo, by default None
    representation : list of bool, optional
        See JwstInfo, by default None
    Pattern : regexp, optional
        See set_regex and jwst naming convention.
    
    Attributes
    ----------
    basename_pattern : regexp
        The regular expression to obtain the basename.
    all_files : list
        A list of CustomJwstInfo objects. Main storage of the class.
    attributes : list
        A list of all the attributes of CustomJwstInfro used in the class.
    custom_methods_dict : dict
        A dictionary of custom methods to retrieve attributes.
    _auto_generated : bool
        Trace whether auto generation is used.
    boxes : int, class attribute
        The number of FileBox instances.
    logger : logging.Logger, class attribute
        The logger for the class.
    if_logger_set : bool, class attribute
        If the logger is set.
        
    See Also
    --------
    
    JwstInfo : The basic dataclass to store jwst information.

    Examples
    --------
    
    >>> from jwstnoobfriend.box import FileBox
    >>> import jwst.datamodels as dm
    >>> fb = FileBox(container = 2) # most recommended type of container
    >>> example_datamodel = dm.open(cal_file_path)
    >>> fb.auto_generate_function(example_datamodel)
    Then the file box is ready to extract information from the datamodels.
    >>> fb.from_filename('jw00001d00001_00001_00001_nrcb1_cal.fits') # add a new item, we recommend to use a file with wcs information
    Or use the following code to manage your file during reduction. Now parallel processing is not supported.
    >>> cal_model = Image2Pipeline.call(some_rate_path)[0]
    >>> fb.from_datamodels(cal_model)
    >>> fb.save_to_file('filebox.pkl')
    When you do the post-analysis, you can use the following code in another script.
    >>> fb = FileBox(container = 2).load_from_file('filebox.pkl')
    """
    boxes = 0
    logger = logging.getLogger('FileBox')  # Set a unique logger name
    if_logger_set = False
    
    def __init__(self, container = 0, defaults = None, representation = None, pattern = r"jw\d{5}\d{3}\d{3}_\d{5}_\d{5}_[^_]+"):
        """Collection of basic information using JwstInfo

        Parameters
        ----------
        container : int, optional
            See JwstInfo, by default 0. Can be 0, 1, 2 and a dict of new containers.
            For the attributes in basic_containers, related methods could be find in this class.
        defaults : Any, optional
            See JwstInfo, by default None
        representation : list of bool, optional
            See JwstInfo, by default None
        Pattern : regexp, optional
            See set_regex and jwst naming convention.
        
        Attributes
        ----------
        basename_pattern : regexp
            The regular expression to obtain the basename.
        all_files : list
            A list of CustomJwstInfo objects. Main storage of the class.
        attributes : list
            A list of all the attributes of CustomJwstInfro used in the class.
        custom_methods_dict : dict
            A dictionary of custom methods to retrieve attributes.
        _auto_generated : bool
            Trace whether auto generation is used.

        Examples
        --------
        >>> from jwstnoobfriend.box import FileBox
        >>> import jwst.datamodels as dm
        >>> fb = FileBox(container = 2) # most recommended type of container
        >>> example_datamodel = dm.open(cal_file_path)
        >>> fb.auto_generate_function(example_datamodel)
        Then the file box is ready to extract information from the datamodels.
        >>> fb.from_filename('jw00001d00001_00001_00001_nrcb1_cal.fits') # add a new item, we recommend to use a file with wcs information
        Or use the following code to manage your file during reduction. Now parallel processing is not supported.
        >>> cal_model = Image2Pipeline.call(some_rate_path)[0]
        >>> fb.from_datamodels(cal_model)
        >>> fb.save_to_file('filebox.pkl')
        When you do the post-analysis, you can use the following code in another script.
        >>> fb = FileBox(container = 2).load_from_file('filebox.pkl')
        """
        FileBox.boxes += 1
        if not FileBox.if_logger_set:
            FileBox.set_logger()
        self.CustomJwstInfo = JwstInfo.with_info(containers=container, defaults=defaults, representation=representation)
        self.attributes = self.CustomJwstInfo.all_attributes()
        self.all_files = []
        FileBox.logger.info(f'Class attributes: {self.attributes}')
        self.custom_methods_dict = {attr: None for attr in self.attributes}
        self._auto_generated = False # trace whether auto generation is used
        self.set_regex(pattern)
        
    def __getitem__(self, index):
        return self.all_files[index]
        
    def __len__(self):
        return len(self.all_files)
    
    def __repr__(self):
        return f'FileBox ({len(self.all_files)} files)'

    def __iter__(self):
        return iter(self.all_files)
    # basic set and get methods
    
    @classmethod
    def set_logger(cls, log_level = logging.INFO, log_file = None, propagate = False):
        """Set the logger for the class

        Parameters
        ----------
        log_level : int, optional
            The logging level, by default logging.INFO
        log_file : str, optional
            The logging file, by default None
        """
        FileBox.logger.handlers.clear()  # Clear existing handlers
        FileBox.logger = set_logging_level(FileBox.logger, log_level)
        FileBox.logger = set_logging_file(FileBox.logger, log_file)
        FileBox.logger.debug(f"logger has {len(FileBox.logger.handlers)} handlers")
        FileBox.logger.propagate = propagate  # Set propagate to False
        FileBox.if_logger_set = True
    
    def set_regex(self, pattern = r"jw\d{5}\d{3}\d{3}_\d{5}_\d{5}_[^_]+" ):
        """Set the regular expression to obtain the basename

        Parameters
        ----------
        pattern : regexp, optional
            the regular expression, by default see jwst file naming convention

        Returns
        -------
        self : FileBox
            return the instance of the class
        """
        self.basename_pattern = pattern
        return self
    
    def set_custom_method(self, attr, function):
        """Set the function to retrieve attributes

        Parameters
        ----------
        attr : str
            Attributes of CustomJwstInfo
        function : function or bool
            The function to retrieve value from datamodel, the passed function should only 
            receive one parameter: datamodels. If set to False, related attribute retrieval will be skipped.

        Returns
        -------
        self : FileBox 
            return the instance of the class
        """
        self.custom_methods_dict[attr] = function
        return self
    
    def set_file_attribute(self, name, val, index = -1, if_update = True):
        """set the file attribute

        Parameters
        ----------
        name : str
            the attribute name
        val : Any
            the value of the attribute
        index : int, optional
            the index of the file, by default -1
        if_update : bool, optional
            if update or replace the attribute, by default True 
        """
        if (type(getattr(self.all_files[index], name)) is dict) and if_update:
            setattr(self.all_files[index], name, getattr(self.all_files[index], name) | val)
        else:
            setattr(self.all_files[index], name, val)


    def eval_file_attribute(self, name, val, func, index = -1):
        """Use a function to set file attribute

        Parameters
        ----------
        name : str
            attribute to be changed
        val : Any
            New value to update
        func : function
            function for updating the attribute, the function must have shape of func(getattr(self.all_files[index], name), val),
            that is function takes two parameters, the first is the old value, and the second is the new value
        index : int, optional
            The index of file to be updated, by default -1
        """
        setattr(self.all_files[index], name, func(getattr(self.all_files[index], name), val))
    
    def get_all_items_attribute(self, attr):
        """Retrieve the list of a chosen attribute of all files

        Parameters
        ----------
        attr : str
            The attribute name

        Returns
        -------
        list
            List of value for the given attribute name.
        """
        return [getattr(ji, attr) for ji in self.all_files]
    
    def summary(self, attr_name:list = []):
        """
        Print a summary of the box.
        
        Parameters
        ----------
        attr_name : list, optional
            The list of attributes to summarize, by default [], which means all attributes.
        """
        if not attr_name:
            attr_name = [attr for attr in self.attributes 
                         if attr != 'basename' 
                         and self.CustomJwstInfo.__dataclass_fields__[attr].repr == True]
        
        for attr in attr_name:
            if attr not in self.attributes:
                FileBox.logger.warning(f'{attr} is not in attributes, ignored')
                attr_name.remove(attr)
        
        console = Console()
        for attr in attr_name:
            all_attr = self.get_all_items_attribute(attr)
            unique_val = sorted(list(set(all_attr)))

            # Format attribute name in the title
            title = f"FileBox has [bold red]{len(unique_val)}[/bold red] {attr}"

            # Create a table for each attribute
            table = Table(title=title)
            table.add_column(f"{attr}", style="cyan", justify="left", min_width=15)
            table.add_column("Count", style="green", justify="right", min_width=15,)

            for val in unique_val:
                count = len(self.match({attr: val}))
                table.add_row("","")
                table.add_row(str(val), str(count))

            # Display the table
            console.print(table)

    
    def match(self, dict2match = {}, dict2match_func = {}, if_return_file = False):
        """Match the files with the given attributes. You can provide values or functions to match.
        The function should take one parameter: the value of the attribute in the file and return a boolean value.
        If for some attributes, both values and functions are provided, the function will be used.
        If the value is a string, the regular expression is supported. Note that the matching mode is 'fullmatch'.
        
        Parameters
        ----------
        dict2match : dict, optional
            The dictionary of attributes to match, by default {}
        dict2match_func : dict, optional
            The dictionary of attributes to match with functions, by default {}
        if_return_file : bool, optional
            If return the file object, by default False
            
            
        Returns
        -------
        list or list of CustomJwstInfo
            The list of matched files or the list of matched CustomJwstInfo objects
            
        Examples
        --------
        First use summary to check your potential choices
        
        >>> fb.summary(['detector', 'filter'])
        
        Then use match to get the files you want
        
        >>> fb.match({'detector':'NRCB1', 'filter':'F444W'}, if_return_file = True)
        
        Or you can return the index for further processing
        
        >>> index_b = fb.match({'detector':'NRCBLONG', 'filter':'F444W'})
        >>> index_a = fb.match({'detector':'NRCALONG', 'filter':'F444W'})
        >>> index_ab = index_a + index_b
        """
        
        for key in dict2match.keys():
            if key not in self.attributes:
                FileBox.logger.warning(f'{key} is not in attributes, ignored')
                dict2match.pop(key)
                
        for key in dict2match_func.keys():
            if key not in self.attributes:
                FileBox.logger.warning(f'{key} is not in attributes, ignored')
                dict2match_func.pop(key)
                
        if not dict2match and not dict2match_func:
            FileBox.logger.warning('No attributes to match, return all files')
            return self.all_files
        
        # pop the attributes of dict2match_func from dict2match
        for key in dict2match_func.keys():
            if key in dict2match.keys():
                dict2match.pop(key)
                FileBox.logger.warning(f'{key} is in both dict2match and dict2match_func, dict2match is popped')
        
        
        
        matched_index = [i for i in range(len(self.all_files))]
        for key, val in dict2match.items():
            all_attr = self.get_all_items_attribute(key)
            if isinstance(val, str):
                matched_index = [i for i, attr in enumerate(all_attr) if re.fullmatch(val, attr) and i in matched_index] # for string, we support regular expression
            else: 
                matched_index = [i for i, attr in enumerate(all_attr) if attr == val and i in matched_index]
        
        for key, func in dict2match_func.items():
            all_attr = self.get_all_items_attribute(key)
            matched_index = [i for i, attr in enumerate(all_attr) if func(attr) and i in matched_index]
        
        if if_return_file:
            return [self.all_files[i] for i in matched_index]
        else:
            return matched_index  
    
    def from_filebox(self, other, indices = None, dict2match = {}, dict2match_func = {}):
        """Add files from another FileBox
        
        Parameters
        ----------
        other : FileBox
            The other FileBox to add files from.
        indices : list, optional
            The indices of the files to add, by default None.
        dict2match : dict, optional
            The dictionary of attributes to match, by default {}.
        dict2match_func : dict, optional
            The dictionary of attributes to match with functions, by default {}.
            The function should take one parameter: the value of the attribute in the file and return a boolean value.
            
            
        Returns
        -------
        self : FileBox
            The instance of the class
        """
        self.attributes = other.attributes
        self.CustomJwstInfo = other.CustomJwstInfo
        self.custom_methods_dict = other.custom_methods_dict
        self._auto_generated = other._auto_generated    
        
        if indices is None:
            indices = other.match(dict2match, dict2match_func)
        
        self.all_files += [other.all_files[i] for i in indices]
        return self
      
        
        
                
    
    # advanced methods
    def auto_generate_function(self, datamodel):
        """Try to guess the retrieval function for each attribute.

        Parameters
        ----------
        datamodel : jwst.datamodel
            A example jwst.datamodel which will be passed to FileBox

        Returns
        -------
        self : FileBox
            return the instance of the class
            
        """
        self._auto_generated = True
        def wcs_func(x):
            return x.meta.wcs
        def basename_func(x):
            filename = x.meta.filename
            return re.match(self.basename_pattern, filename).group()
        def footprint_func(x):
            wcs = x.meta.wcs
            data = x.data
            shape = data.shape
            footprint_pixels = [[0,0],[0,shape[1]-1],[shape[0]-1,shape[1]-1],[shape[0]-1,0]]
            try:
                test_transform = wcs.pixel_to_world(*footprint_pixels[0])
                footprint_sky = [wcs.pixel_to_world(*footprint_pixel) for footprint_pixel in footprint_pixels]
            except:
                footprint_sky = [wcs.pixel_to_world(0,0, *footprint_pixel,1)[0] for footprint_pixel in footprint_pixels]
            return FootPrint(footprint_sky)
        
        for attr, func in self.custom_methods_dict.items():
            ############# Try to find same attributes in meta #############
            if func is not None:
                FileBox.logger.info(f'{attr} has a function before auto generation, please check the function if it is not expected.')
            else:
                if attr == "basename":
                    self.set_custom_method(attr, basename_func)
                                           
                                           #lambda x: re.match(self.basename_pattern, 
                                            #                        os.path.basename(x.meta.filename)).group())
                elif attr == "wcs":
                    try: 
                        wcs = datamodel.meta.wcs
                        self.set_custom_method(attr, wcs_func)
                        self.set_custom_method('footprint', footprint_func)
                        FileBox.logger.info('wcs exists in provided datamodel, retrieval function for wcs and footprint built, make sure following datamodel also have wcs')
                    except:
                        FileBox.logger.warning(f"Cannot find wcs information in datamodels, check {datamodel.meta.filename}")
                        self.set_custom_method(attr, False) 
                        self.set_custom_method('footprint', False)
                
                elif attr == "footprint":
                    pass
                else:
                    try:
                        method_string, _ = datamodel.search_schema(attr)[0]
                        func = eval(f'lambda x: x.{method_string}')
                        FileBox.logger.info(f'Visiting {method_string} to retreive {attr}')
                    except:
                        func = False
                        FileBox.logger.warning(f'Try to retreive {attr} but failed')
                    
                    self.set_custom_method(attr, func)
        
        return self

    # add new item
    def from_datamodels(self, datamodel):
        """
        Add a new item of CustomJwstInfo to the box.
        
        Parameters
        ----------
        datamodel : jwst.datamodel
            The datamodel to be added.
        
        Returns
        -------
        new_item : CustomJwstInfo
            The new item added to the box.
        """
        info_dict = {}
        if not self._auto_generated:
            self.auto_generate_function(datamodel)
            FileBox.logger.info('auto_generate_function is called')
        
        
        for attr, func in self.custom_methods_dict.items():
            ############# Try to find same attributes in meta #############
            if func is False:
                pass
            elif func is None:
                if len(self.all_files) == 0:
                    FileBox.logger.warning(f'{attr} do not have a corresponding retrieval function, will be ignored!')
                else:
                    pass
            else:
                info_dict[attr] = func(datamodel)
                
        new_item = self.CustomJwstInfo(**info_dict)
        
        self.all_files.append(new_item)
        return new_item

    def from_filename(self, file_name, if_only_basename = False, pattern = None, pd_level = 'rate'):
        """
        Add a new item of CustomJwstInfo to the box from the file name.
        
        Parameters
        ----------
        file_name : str
            The file name to be added.
        if_only_basename : bool, optional
            If only the basename is added. The default is False.
        pattern : str, optional
            The pattern to extract the basename. The default is None.
        pd_level : str, optional
            The level of the file. The default is 'rate'.
        
        Returns
        -------
        new_item : CustomJwstInfo
            The new item added to the box.
        """
        if if_only_basename:
            if pattern is None:
                pattern = self.basename_pattern
            basename = re.match(pattern, os.path.basename(file_name)).group()
            new_item = self.CustomJwstInfo(basename = basename)
            self.all_files.append(new_item)
            self.add_file_path(pd_level = pd_level,pd_path=file_name)
            return new_item
        
        else:
            datamodel = dm.open(file_name)
            return self.from_datamodels(datamodel)

    def search_index_from_filename(self, file_name):
        """
        Search the index of the file in the box.
        
        Parameters
        ----------
        file_name : str
            The file name to search.
        
        Returns
        -------
        i : int
            The index of the file in the box
        """
        match = re.match(self.basename_pattern, os.path.basename(file_name))
        if match is None:
            FileBox.logger.warning(f'{file_name} does not match the basename pattern')
            return None
        basename = match.group()
        for i, file in enumerate(self.all_files):
            if file.basename == basename:
                return i
    
    def update_from_datamodels(self, datamodel):
        """
        Update the information in the box from a datamodel.
        
        Parameters
        ----------
        datamodel : jwst.datamodel
            The datamodel to update.
        """
        i = self.search_index_from_filename(datamodel.meta.filename)
        if i is not None:
            self.all_files[i] = self.from_datamodels(datamodel)
            
    # methods for given attributes
    def add_file_path(self, pd_level, pd_path, default_index = -1, path_key = 'filepaths', if_check_exist = False):
        """
        Add a file path to the file.
        
        Parameters
        ----------
        pd_level : str
            The level of the file.
        pd_path : str
            The path of the file.
        default_index : int, optional
            The default index of the file. The default is -1.
        path_key : str, optional
            The attribute name of the saving the path. The default is 'filepaths'.
        if_check_exist : bool, optional
            If check the existence of the path. The default is False.
            
        Raises
        ------  
        ValueError
            If the path_key is not in the attributes.
        """
        if if_check_exist and not os.path.exists(pd_path):
            raise ValueError(f'{pd_path} does not exist')
        
        if path_key not in self.attributes:
            raise ValueError(f'{path_key} is not in the attributes')

        if default_index == -1:
            index = self.search_index_from_filename(pd_path)
            if index is not None:
                self.set_file_attribute(path_key, {pd_level: pd_path}, index = index)
            else:
                self.set_file_attribute(path_key, {pd_level: pd_path}, index = default_index)
                FileBox.logger.warning(f'Cannot find {pd_path} in the box, update the default index {default_index}')
        else:
            self.set_file_attribute(path_key, {pd_level: pd_path}, index = default_index)
            
    def add_other(self, term, value, default_index = -1, filename = None, other_key = 'other'):
        """
        Add other information to the file.
        
        Parameters
        ----------
        term : str
            The term to add.
        value : Any
            The value to add. The {term: value} will be added to the file.other.
        default_index : int, optional
            The default index of the file. The default is -1.
        filename : str, optional
            The filename of the file, if provided, the index will be searched. The default is None.
        other_key : str, optional
            The attribute name of the saving the other information. The default is 'other'.
            
        Raises
        ------
        ValueError
            If the other_key is not in the attributes.
            
        Examples
        --------
        >>> fb = FileBox(container = 2)
        >>> fb.from_filename('jw00001d00001_00001_00001_nrcb1_rate.fits')
        >>> fb.add_other('test', 1)
        >>> fb.all_files[0].other
        {'test': 1}
        """
        if other_key not in self.attributes:
            raise ValueError(f'{other_key} is not in the attributes')
        else:
            if filename is not None:
                index = self.search_index_from_filename(filename)
            else:
                index = default_index
            self.set_file_attribute(other_key, {term: value}, index = index)

################################################################################################
########################## save and load methods ############################################
################################################################################################
    def save_to_file(self, file_name):
        """
        Save the box to a file.
        
        Parameters
        ----------
        file_name : str
            The file name to save the box.
        """
        with open(file_name, 'wb') as f:
            pickle.dump(self.all_files, f)
        FileBox.logger.info(f'File saved to {file_name}, including {len(self.all_files)} files')
    
    def load_from_file(self, file_name):# behavior undefined if the containers is user defined
        """
        Load the box from a file.
        
        Parameters
        ----------
        file_name : str
            The file name to load the box.
        """
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                self.all_files = pickle.load(f)
            if len(self.all_files) > 0:
                class_name = self.all_files[0].__class__.__name__
                self.CustomJwstInfo = getattr(sys.modules[__name__], class_name)
            return self

from glob import glob
class FileCollection:
    """
    A collection of files with specified suffix in the parent folder.
    
    Parameters
    ----------
    parent_folder : str, optional
        The parent folder of the files. The default is './'.
    
    suffix : str, optional
        The suffix of the files. The default is '.fits'.
        
    Attributes
    ----------
    all_files : list
        A list of all the files with specified suffix in the parent folder.
    subset : list
        The subset of the files.
     
    """
    
    def __init__(self, parent_folder='./', suffix='.fits'):
        """
        Initialize the FileCollection object.
        
        Parameters
        ----------
        parent_folder : str, optional
            The parent folder of the files. The default is './'.
            
        suffix : str, optional
            The suffix of the files. The default is '.fits'.

        """
        self.all_files = sorted(glob(os.path.join(parent_folder, f'*{suffix}')))    
    def get_subset(self, keywords):
        """
        Get a subset of the files based on the keywords.
        
        Parameters
        ----------
        keywords : str or list
            The keywords to search for.
            
        Returns
        -------
        subset : list
            The subset of the files.
        """ 
        if (not isinstance(keywords, str)) and (hasattr(keywords, '__iter__')):
            self.subset = self.all_files
            for keyword in keywords:
                if isinstance(keyword, str):
                    self.subset = [file for file in self.subset if re.search(f'{keyword}', file)]
                else:
                    raise TypeError(f'{keyword} is not string')
                
        elif isinstance(keywords, str):
            self.subset = [file for file in self.all_files if re.search(f'{keywords}', file)]
        else:
            raise TypeError('Type of keywords not supported')
        
        return self.subset
