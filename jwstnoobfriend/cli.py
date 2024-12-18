"""This module provides a command line interface for generating a notebook with packages imported and folders created.
"""
# to do:
# 1. read system variables
from ._logging import *
import nbformat

pakages_2b_imported = """# system packages
import os
import sys
from glob import glob
import logging
os.environ['CRDS_PATH'] = '/media/zch/zchao/crds_cache/'
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

# jwst packages
import crds
import jwst
from jdaviz import Imviz
print('Using JWST Pipeline v{}'.format(jwst.__version__))
from jwst import datamodels as dm
from jwst.datamodels import dqflags

# common packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from tqdm.notebook import tqdm

# astropy packages
from astropy.io import fits
import astropy.units as u
from astropy import constants as astro_c
from astropy.visualization import ZScaleInterval, ImageNormalize
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats, sigma_clip
from astropy.convolution import convolve, Gaussian2DKernel

# Pipeline packages
from jwst.pipeline import Image2Pipeline, Image3Pipeline, Spec2Pipeline, Spec3Pipeline
# Stage2 packages
from jwst.background import BackgroundStep
from jwst.assign_wcs import AssignWcsStep
from jwst.flatfield import FlatFieldStep
from jwst.pipeline import Image2Pipeline

# jwstnoobfriend packages
from jwstnoobfriend.box import *
from jwstnoobfriend.manager import *
from jwstnoobfriend.reduction import *
"""

folder_creator = """
work_folder = '/media/zch/zchao/PID1895/'
print(f'Reducing files saved in {work_folder}')
stage2_folder = output_folder(os.path.join(work_folder, 'stage2'))
print(f'Reducing rate files saved in {stage2_folder}')
rate_folder = output_folder(os.path.join(stage2_folder, 'rate'))
bwf_folder = output_folder(os.path.join(stage2_folder, 'bkg_wcs_ff'))
cal_folder = output_folder(os.path.join(stage2_folder, 'cal'))
rate_collection = FileCollection(parent_folder=rate_folder,suffix='rate.fits')
rate_files = rate_collection.all_files
assist_folder = output_folder(os.path.join(work_folder, 'assist'))
bkg_folder = output_folder(os.path.join(assist_folder, 'bkg'))
"""

import argparse

def main():
    """Generate a notebook with packages imported and folders created.
    """
    
    parser = argparse.ArgumentParser(description='Notebook generator for lazy man.')
    parser.add_argument('nb_name', type=str, help='The name of the notebook to be generated.')
    parser.add_argument("--no-import", action = 'store_false', dest='import_flag', help="Do not import packages.")
    parser.add_argument("--no-folder", action = 'store_false', dest='folder_flag', help="Do not create folders.")
    args = parser.parse_args()
    nb_name = args.nb_name
    if not nb_name.endswith('.ipynb'):
        nb_name = nb_name + '.ipynb'
    import_flag = args.import_flag
    folder_flag = args.folder_flag
    nb = nbformat.v4.new_notebook()
    if import_flag:
        nb.cells.append(nbformat.v4.new_code_cell(pakages_2b_imported))
    if folder_flag:
        nb.cells.append(nbformat.v4.new_code_cell(folder_creator))
    with open(nb_name, 'w') as f:
        nbformat.write(nb, f)
    print(f'Notebook {nb_name} generated.')

