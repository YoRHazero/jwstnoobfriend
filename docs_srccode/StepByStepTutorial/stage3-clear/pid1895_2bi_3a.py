import os
import numpy as np
import tomllib
import logging
from pathlib import Path
import time

from jwstnoobfriend.navigation import FileBox
from jwstnoobfriend.utils.environment import load_environment
from jwstnoobfriend.utils.display import console
load_environment()
import jwst
import crds
console.print(f"JWST version: {jwst.__version__}")


# Get potential loggers
tweakwcs_logger = logging.getLogger('tweakwcs')
crds_logger = logging.getLogger('CRDS')
stpipe_logger = logging.getLogger('stpipe')
stcal_logger = logging.getLogger('stcal')
jwst_logger = logging.getLogger('jwst')

# Remove handlers
for handler in tweakwcs_logger.handlers[:]:
    tweakwcs_logger.removeHandler(handler)
for handler in crds_logger.handlers[:]:
    crds_logger.removeHandler(handler)
for handler in stpipe_logger.handlers[:]:
    stpipe_logger.removeHandler(handler)
for handler in jwst_logger.handlers[:]:
    jwst_logger.removeHandler(handler)
for handler in stcal_logger.handlers[:]:
    stcal_logger.removeHandler(handler)
    
tweakwcs_logger.propagate = False
crds_logger.propagate = False
stpipe_logger.propagate = False
jwst_logger.propagate = False
stcal_logger.propagate = False

# In each grouped FileBox, we will combine them to create association file
clear_box = FileBox.load(os.environ['DATA_ROOT_PATH'] + '/noobox_clear.json')  # type: ignore
grouped_box_list = clear_box.group_by_pointing()
from collections import defaultdict
groups_dict = defaultdict(list)
for box in grouped_box_list:
    filesetnames_key = tuple(sorted(box.filesetnames)+[box[0].detector[:4]])
    groups_dict[filesetnames_key].append(box)
    
groups_for_reduction = list(groups_dict.values())
for group in groups_for_reduction:
    for j in range(1, len(group)):
        group[0].merge(group[j])
groups = [g[0] for g in groups_for_reduction]

with open("pipeline_setup_3a.toml", "rb") as f:
    config = tomllib.load(f)

stage_3a_path = Path(os.environ['STAGE_3A_PATH'])

from jwst.associations import asn_from_list
from jwst.associations.lib.rules_level3 import DMS_Level3_Base
from jwst.pipeline import Image3Pipeline

import pyvo as vo
import pandas as pd
from astropy.table import Table
from pathlib import Path
refcat = Table()
svc = vo.dal.TAPService("https://mast.stsci.edu/vo-tap/api/v0.1/candels")

cat = svc.search("""
SELECT * FROM dbo.candels_master_view
WHERE field='GOODS-N' OR field='GOODS-S'
""").to_table() #type: ignore
refcat['RA'] = cat['RA']
refcat['DEC'] = cat['DEC']
refcat.write('candels_master_view_ecsv.ecsv', format='ascii.ecsv', overwrite=True)

def reduce(
    group: FileBox
):
    time1 = time.time()
    
    filesetnames = group.filesetnames
    obs_visit_all = [f.split('_')[0] for f in filesetnames]
    obs_visit_unique = list(set(obs_visit_all))
    if len(obs_visit_unique) != 1:
        print(f"Multiple observation visits found {filesetnames}")
    obs_visit = obs_visit_unique[0]
    mid_seq_all = [f.split('_')[1] for f in filesetnames]
    mid_seq_unique = sorted(list(set(mid_seq_all)))
    product_name = obs_visit + '_' + '+'.join(mid_seq_unique) + '_' + group[0].filter + '_' + group[0].detector[:4]

    # Temporary step to skip existing file:
    #if (stage_3a_path / (product_name+'_i2d.fits')).exists():
    #    console.print(f"Skipping existing product {product_name}")
    #    return

    log_filename = f"./log/3a/{product_name}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    for logger in [tweakwcs_logger, crds_logger, stpipe_logger, jwst_logger, stcal_logger]:
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
    
    # Reduce:
    try:
        association = asn_from_list.asn_from_list(
            [info['2bi'].filepath.__str__() for _, info in group],
            rule=DMS_Level3_Base,
            product_name=product_name
        )
        association['asn_type'] = 'image3'
        association['program'] = group[0]['2b'].meta.observation.program_number
        _, asn_serialized = association.dump(format="json")
        asn_path = Path("./asn") / (product_name + "_asn.json")
        with open(asn_path, "w") as asn_file:
            asn_file.write(asn_serialized)
        time2 = time.time()
        i2d_result = Image3Pipeline.call(
            str(asn_path),
            **config
        )
        time3 = time.time()
        # Show time in mins
        console.print(f"Finished reducing {product_name} in {(time2-time1)/60.:.1f} mins (setup) + {(time3-time2)/60.:.1f} mins (running)")
    except Exception as e:
        with open("log/3a_errors.log", "a") as ef:
            ef.write(f"Error processing group {filesetnames}: {e}\n")
    finally:
        for logger in [tweakwcs_logger, crds_logger, stpipe_logger, jwst_logger, stcal_logger]:
            logger.removeHandler(file_handler)
    return True


from jwstnoobfriend.utils.display import track

def main():
    for group in track(groups):
        reduce(group)

if __name__ == "__main__":
    main()
