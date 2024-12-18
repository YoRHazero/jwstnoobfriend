"""
JWST Noob Friend

This package provides tools to manage and process data from the James Webb Space Telescope (JWST). 
It includes functionalities for handling file collections, managing footprints, and reducing noise in data.

Modules:

- box: Contains classes and functions for handling JWST data information.

- manager: Provides utilities for grouping footprints and managing output folders.

- reduction: Includes functions for reducing noise in JWST data.

Import:

    >>> from jwstnoobfriend import *

Example usage:
    
    >>> filebox = FileBox()
    >>> footprints = filebox.get_all_items_attribute('footprint')
    >>> grouped_footprints = group_footprints_by_distance(footprints)
    >>> show_all_footprints(filebox)
"""

from .box import *
from .manager import *
from .reduction import *

# ...existing code...
