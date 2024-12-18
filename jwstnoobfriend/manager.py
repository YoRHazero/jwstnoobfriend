"""This module contains the functions to manage the information container provided in the box module.
"""
from .box import *
import logging
import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.neighbors import BallTree
from astropy import units as u

default_colors = ['royalblue', 'forestgreen', 'darkorange', 'darkviolet', 'brown', 
                  'darkcyan', 'darkred', 'darkblue', 'darkgreen', 'olive', 
                  'aliceblue', 'teal', 'steelblue', 'darkslategray', 'darkgoldenrod',]


def output_folder(dir):
    """
    Create a folder if it does not exist.
    
    Parameters:
    dir (str): The directory of the folder.
    
    Returns:
    str: The directory of the folder.
    """
    if dir[-1]!='/':
        dir +='/'
    if os.path.isdir(dir):
        print(f'"{dir}" exist') 
    else:
        os.mkdir(dir)
        print(f'"{dir}" created')
    return dir  


from functools import wraps
import inspect
from rich.progress import Progress
def progress_bar(which_list = None, refresh = 10, pb_string = "Processing ..."):
    """
    Add a progress bar to a function with for loop.
    
    Parameters
    ------------
        which_list: the for-loop list to generate progress bar. It can be a variable name in the function or a list will be passed to the function
        Defaults to None, where the only iterable parameter of function will be considered as the for-loop list. Notice that if function contains 
        two iterable parameters, the behavior of None is not defined!
        
        refresh: int. The value of refresh_per_second in rich.progress. defaults to 10.
        
        pb_string: str. The string of progress bar.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal which_list
            sig = inspect.signature(func)
            func_params = sig.bind_partial(*args, **kwargs)
            func_params.apply_defaults()
            if which_list is None:
                which_list = [p_key for p_key, p_val in func_params.arguments.items() if hasattr(p_val, "__iter__")][0]
                this_list = func_params.arguments[which_list]
            elif isinstance(which_list, str):
                this_list = func_params.arguments[which_list]
            elif hasattr(which_list, "__iter__"):
                frame = inspect.currentframe()
                local_vars = frame.f_locals
                which_list = [name for name, val in local_vars.items() if val is which_list][0]
                this_list = func_params.arguments[which_list]
            else:
                raise TypeError(f'Unsupported type: {type(which_list)}')
            
            with Progress(refresh_per_second=refresh) as progress:
                task = progress.add_task(pb_string, total = len(this_list))
                
                def inner_wrapper(iterable):
                    for item in iterable:
                        yield item
                        progress.update(task, advance = 1)
                        
                func_params.arguments[which_list] = inner_wrapper(this_list)
                result = func(*func_params.args, **func_params.kwargs)
                progress.refresh()
                return result
                
        return wrapper
    return decorator


def group_footprints_by_distance(footprints, max_distance = 5 , d_unit = u.arcmin):
    """
    Group footprints by distance.
    
    Parameters
    ------------
    footprints: list
        list of Footprint objects.
    
    max_distance: float
        The maximum distance to consider two footprints as neighbors.
    
    d_unit: str or astropy.units.Unit
        The unit of the distance. Default is 'arcmin'.
    
    Returns
    ------------
    list
        List of groups of indices of footprints.
    """
    if isinstance(d_unit, str):
        d_unit = u.Unit(d_unit)
    elif not isinstance(d_unit, u.Unit):
        raise ValueError('d_unit must be a string or astropy.units.Unit object.')
    
    center_coords = [footprint.center_sky for footprint in footprints]
    
    center_radian = np.array([[coord.ra.radian, coord.dec.radian] for coord in center_coords])
    tree = BallTree(center_radian, metric='haversine')
    max_distance_radian = max_distance * d_unit.to(u.radian)
    neighbors = tree.query_radius(center_radian, r=max_distance_radian)
    visited = set()
    groups = []
    
    def depth_first_search(node, group):
        """
        Depth-first search to find all connected components.
        
        Parameters:
        node (int): The index of the current node.
        group (list): A list of group indices.
        
        """
        if node in visited:
            return
        visited.add(node)
        group.append(node)
        for neighbor in neighbors[node]:
            depth_first_search(neighbor, group)
            
    for i in range(len(footprints)):
        if i in visited:
            continue
        group = []
        depth_first_search(i, group)
        groups.append(group)
        
    sorted_groups = [sorted(group) for group in groups]
    print(f'Found {len(sorted_groups)} groups.')
    return sorted_groups


def show_footprints(footprints: list, axes=None, colors = None, max_distance = 5, d_unit = u.arcmin):
    """
    Show all footprints in the filebox.
    
    Parameters
    ------------
    footprints: list of Footprint
        The footprints to show.
        
    axes: list of matplotlib.axes.Axes
        The axes to plot the footprints. If None, new axes will be created.
        
    colors: list or iterable (same as list) of str
        The colors of the footprints. If None, default colors will be used.
        
    max_distance: float
        The maximum distance to consider two footprints as neighbors.
        
    d_unit: str or astropy.units.Unit
        The unit of the distance. Default is 'arcmin'.
        
    Returns
    ------------
    fig: matplotlib.figure.Figure
        The figure object.
        
    axes: list of matplotlib.axes.Axes  
        The axes object.
    """ 
    # set up some beutiful default colors

    
    groups = group_footprints_by_distance(footprints, max_distance = max_distance, d_unit = d_unit)
    group_num = len(groups)
    
    if colors is None:
        colors = [default_colors[0]] * len(footprints)
    elif len(colors) == len(footprints):
        pass
    else:
        if len(footprints) % len(colors) == 0:
            colors = colors * (len(footprints) // len(colors))
        else:
            raise ValueError('The length of colors must be equal to the length of footprints or a factor of it.')
        
    
    if axes is None:
        fig, axes = plt.subplots(1, group_num, figsize = (6*group_num, 6))
        if not hasattr(axes, '__iter__'):
            axes = [axes]
    elif len(axes) != group_num:
        raise ValueError('The number of axes must be equal to the number of groups.')
    else:
        fig = axes[0].figure
        
    for i, group in enumerate(groups):
        for j in group:
            footprints[j].plot_footprint(axes[i], color = colors[j])
            
    return fig, axes

from itertools import product
def show_footprints_in_box(filebox: FileBox, axes=None, colors_backup = None , max_distance = 5, d_unit = u.arcmin, attrs_for_colors = None):
    """
    Show all footprints in the filebox.
    
    Parameters
    ------------
    filebox: FileBox
        The filebox to show.
        
    axes: list of matplotlib.axes.Axes
        The axes to plot the footprints. If None, new axes will be created.
        
    colors_backup: list or iterable (same as list) of str
        The colors of the footprints. If None, default colors will be used.
        
    max_distance: float
        The maximum distance to consider two footprints as neighbors.
    
    d_unit: str or astropy.units.Unit
        The unit of the distance. Default is 'arcmin'.
        
    attrs_for_colors: str or list of str
        The attributes to determine the colors of footprints. If None, all footprints will have the same color.
        Recommended to use only when the number of unique values of the attributes is small, e.g., filter, pupil, etc.
        
    Returns
    ------------
    fig: matplotlib.figure.Figure
        The figure object.
        
    axes: list of matplotlib.axes.Axes
        The axes object.
    """
    all_footprints = filebox.get_all_items_attribute('footprint')
    if colors_backup is None:
        colors_backup = default_colors
    elif isinstance(colors_backup, str):
        colors_backup = [colors_backup]
        
    if attrs_for_colors is not None:
        if isinstance(attrs_for_colors, str):
            attrs_for_colors = [attrs_for_colors]
        
        val_types_of_attrs = [] # unique values of each attribute
        
        for attr in attrs_for_colors:
            val_of_attr = filebox.get_all_items_attribute(attr)
            val_unique = np.unique(val_of_attr)
            val_types_of_attrs.append(val_unique)
        
        comb_of_attrs = list(product(*val_types_of_attrs))    
        
        if len(colors_backup) < len(comb_of_attrs):
            raise ValueError('The number of colors is not enough for the number of unique values of the attributes.')
        
        comb2num_dict = {comb: i for i, comb in enumerate(comb_of_attrs)}
        
        colors = [comb2num_dict[tuple([getattr(file, attr) for attr in attrs_for_colors])] for file in filebox.all_files]
        
        colors = [colors_backup[i] for i in colors]
        
    else:
        colors = colors_backup
    return show_footprints(all_footprints, axes = axes, colors = colors, max_distance = max_distance, d_unit = d_unit)
        

def find_close_footprints(target_footprint, all_footprints, max_distance = 5, d_unit = u.arcmin, return_type = 'index', ax_to_plot_search = None):
    
    """
    Find close footprints for a target footprint.
    
    Parameters
    ------------
    target_footprint: Footprint
        The target footprint.
        
    all_footprints: list of Footprint
        The footprints to search.
        
    max_distance: float
        The maximum distance to consider two footprints as neighbors.
        
    d_unit: str or astropy.units.Unit
        The unit of the distance. Default is 'arcmin'.
        
    return_type: str
        The type of return value. 'index' or 'footprint'.
        
    ax_to_plot_search: matplotlib.axes.Axes
        The axes to plot the search area. If None, no plot will be made.
        
        
    Returns
    ------------
    list
        The indices of close footprints.
    """
    
    if isinstance(d_unit, str):
        try:
            d_unit = u.Unit(d_unit)
        except:
            raise ValueError('Invalid unit.')
    elif not isinstance(d_unit, u.Unit):
        raise ValueError('d_unit must be a string or astropy.units.Unit object.')
    
    neighbors = [i for i, footprint in enumerate(all_footprints) if target_footprint.distance(footprint, d_unit = d_unit).value < max_distance]
    
    if ax_to_plot_search is not None:
        # plot a circle of max_distance, centered at the target_footprint
        target_footprint.plot_footprint(ax_to_plot_search, color = 'darkred', ls='-', lw = 2.2)
        
        for i in neighbors:
            all_footprints[i].plot_footprint(ax_to_plot_search, color = 'steelblue', alpha = 0.5)
        
        search_circle = plt.Circle((target_footprint.center_sky.ra.deg, target_footprint.center_sky.dec.deg), 
                                   max_distance * d_unit.to(u.deg), fill = True, alpha = 0.2, edgecolor = 'black', facecolor = 'gray')
        ax_to_plot_search.add_artist(search_circle)
    
    if return_type == 'index':
        return sorted(neighbors)
    elif return_type == 'footprint':
        return [all_footprints[i] for i in sorted(neighbors)]
    
    