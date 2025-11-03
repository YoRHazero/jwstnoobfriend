from jwstnoobfriend.navigation import FileBox
from jwstnoobfriend.utils.environment import load_environment # To load environment variables
from jwstnoobfriend.utils.display import console # Make sure you are use the same console instance.
import os # To access environment variables

## In case you need to do parallel processing, but it is not necessary because the bottleneck is not CPU-bound tasks.
from concurrent.futures import ProcessPoolExecutor

import logging

## To mute noisy logging messages from the JWST pipeline


CRDS_logger = logging.getLogger('CRDS')

for handler in CRDS_logger.handlers[:]:
    CRDS_logger.removeHandler(handler)

CRDS_logger.propagate = False


stpipe_logger = logging.getLogger('stpipe')

for handler in stpipe_logger.handlers[:]:
    stpipe_logger.removeHandler(handler)

stpipe_logger.propagate = False


jwst_logger = logging.getLogger('jwst')

for handler in jwst_logger.handlers[:]:
    jwst_logger.removeHandler(handler)

jwst_logger.propagate = False

### The logger name may change in the future JWST pipeline versions.
### You can always check the logger names and mute them accordingly.


load_environment()
filebox = FileBox.load()

## Load toml configuration file
import tomllib
with open("/media/disk1/zchao/wfss/notebook/reduction_with_noob/pid1895_all_2a.toml", "rb") as f:
    config = tomllib.load(f)

## Reduction process
from jwst.pipeline import Detector1Pipeline

def reduce_info(info):
    
    log_filename = f"./.log/1b_2a/{info.basename}.log"
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    
    

    try:
        ## Run the pipeline with the config loaded from the toml file
        result = Detector1Pipeline.call(info['1b'].filepath, **config)

        logging.getLogger('processing').addHandler(file_handler)
        logging.getLogger('processing').info(f"Successfully processed {info.basename}")
    except Exception as e:
        logging.getLogger('processing').addHandler(file_handler)
        logging.getLogger('processing').error(f"Error processing {info.basename}: {str(e)}")
        raise
    finally:
        
        file_handler.close()
    return True 

def main():
    

    console.print("[bold green]Starting reduction process...[/bold green]")
    console.print(f"Using 1 processes")
    console.print(f"Configuration loaded from: /media/disk1/zchao/wfss/notebook/reduction_with_noob/pid1895_all_2a.toml")

    with ProcessPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(reduce_info, info) for info in filebox.info_list]
        
        successful = 0
        failed = 0

        for i, future in enumerate(futures):
            try:
                future.result()
                successful += 1
                console.print(f"[green] ({i+1}/{len(filebox.info_list)}) Successfully processed: {filebox.info_list[i].basename}[/green]")
            except Exception as e:
                failed += 1
                console.print(f"[red] ({i+1}/{len(filebox.info_list)}) Error processing {filebox.info_list[i].basename}: {str(e)}[/red]")

    console.print(f"[bold green]Reduction process completed![/bold green]")
    console.print(f"[bold green]Successful: {successful}[/bold green]")
    console.print(f"[bold red]Failed: {failed}[/bold red]")

if __name__ == "__main__":
    main()