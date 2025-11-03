# Noobscript: your noobfriend for JWST data reduction script

We use [`toml`](https://toml.io/en/) format to write configuration parameters for the pipeline. The `toml` file can be loaded as `dict`, and the parameters are stored in a more arranged way, making it easier to manage and modify them.

In this package, we provide a CLI called **noobscript** to help generate the `toml` configuration files and `python` scripts for different stages of the JWST pipeline.

- If the output products are in stage2, you can use the `noobscript stage2` command to generate the configuration file and script.
- If the product is 3a, you can use the subcommand `noobscript stage3`.

In the following section, you will find how to use these commands :nerd:.

## Prepare your noobox

Before the reduction, we need to initialize our noobox. It can be simply done by running:
```python
from jwstnoobfriend.navigation import FileBox

filebox = FileBox.init_from_folder()
filebox.save()
```

??? note "No Parameters needed?"
    The `init_from_folder` method will automatically fill in the parameters based on the environment variables. You can also provide parameters manually if needed. 


## Stage 1: From uncalibrated to rate
In this reduction step, the pipeline will fit the slope of uncalibrated files, and obtain the rate files. In our example, `rate` files and `rateint` files are essentially the same. Additionally, we don't have to distinguish grism and clear image files.

First, use `noobscript` as follows:

<!-- termynal -->
```console
$ uv run noobscript stage2 2a -p pid1895
```

Here, `2a` is the command argument standing for output stage, which is needed to load corresponding template; `-p` stands for prefix of the saved script and configuration file name. (1) 
{ .annotate } 

1.  You can always check the parameter meaning by typing `uv run noobscript stage2 --help`.

Then two files will be generated in your current folder if you don't specify the output directory:
```toml title="pid1895_all_2a.toml"
--8<-- "docs_srccode/StepByStepTutorial/stage1/pid1895_all_2a.toml"
```

Here I give several setup based on my own experience, you can change them as needed.

```python title="pid1895_1b_2a.py" 
--8<-- "docs_srccode/StepByStepTutorial/stage1/pid1895_all_1b_2a.py"
```

Here I tried to add some comments to help you understand the code. You can modify the code as needed.