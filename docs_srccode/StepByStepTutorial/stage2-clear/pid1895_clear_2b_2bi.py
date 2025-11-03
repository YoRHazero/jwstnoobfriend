from jwstnoobfriend.navigation import FileBox, JwstInfo
from jwstnoobfriend.utils.environment import load_environment # To load environment variables
from jwstnoobfriend.utils.display import console # Make sure you are use the same console instance.
import os # To access environment variables

## In case you need to do parallel processing, but it is not necessary because the bottleneck is not CPU-bound tasks.
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

load_environment()
clear_box = FileBox.load(suffix = "clear")
## Reduction process
## This is a custom reduction step for subtracting the sky background.
def reduce_info(info: JwstInfo):
    try:
        datamodel = info['2b'].datamodel
        datamodel.data = datamodel.data - info['2b'].background # Subtract the background
        output_dir = Path(os.getenv('STAGE_2BI_PATH', './stage_2bi'))
        if not output_dir.exists():
            raise FileNotFoundError(f"Output directory {output_dir} does not exist.")
        datamodel.write(output_dir / info['2b'].filepath.name, overwrite=True)
    
    except Exception as e:
        console.print(f"[red]Error processing {info.basename}: {str(e)}[/red]")
        raise
    return True

def main():
    with ProcessPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(reduce_info, info) for info in clear_box.info_list]
        successful = 0
        failed = 0
        
        for i, future in enumerate(futures):
            try:
                result = future.result()
                successful += 1
                console.print(f"[green]Successfully processed {clear_box.info_list[i].basename}[/green]")
            except Exception as e:
                failed += 1
                console.print(f"[red]Failed to process {clear_box.info_list[i].basename}: {str(e)}[/red]")

        console.print(f"[yellow]Summary: {successful} succeeded, {failed} failed[/yellow]")
if __name__ == "__main__":
    main()