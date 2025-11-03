# Set Up Your Environment Variables

A good practice of setting up your environment is to create a `.env` file in the root of your project directory instead of hardcoding sensitive information directly into your code or exporting them in your global shell environment. 

In this package, I provide the `noobenv` CLI application to help you build and manage your `.env` file. The application is built using [Typer](https://typer.tiangolo.com/), so don't forget to use '`--help`' to check the help message! :nerd:

## Check the help message
<!-- termynal -->
```console
$ uv run noobenv --help
 Usage: noobenv [OPTIONS] COMMAND [ARGS]...                                                                                                                   
                                                                                                                                                              
 Commands for managing the .env file                                                                                                                          
                                                                                                                                                              
                                                                                                                                                              
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.                                                                                    │
│ --show-completion             Show completion for the current shell, to copy it or customize the installation.                                             │
│ --help                        Show this message and exit.                                                                                                  │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ init     Initialize the .env file                                                                                                                          │
│ append   Append folder information to the .env file                                                                                                        │
│ check    Check whether the file paths in the .env file exist. The existing paths will be printed in green, and the non-existing paths will be printed in   │
│          red.                                                                                                                                              │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

As you can see, there are three commands available: `init`, `append`, and `check`.

By using them one by one, we will finish setting up our environment variables 😎.

## First step: Create a `.env` file

The simplest way to create a `.env` file is to directly use the `init` command without any arguments.

<!-- termynal -->
```console
$ noobenv init
The .env file has been written to /media/disk1/zchao/wfss/notebook/reduction/.env
```
???+ note "Don't forget to use `--help`!"
    You can use `noobenv init --help` to check the help message of each subcommand (here `init`) too!

Now you can see a new file named `.env` in your project folder. You can open it and see the content is like this:
```dotenv title=".env"
CRDS_SERVER_URL=https://crds.stsci.edu
# Path to the CRDS cache directory
CRDS_PATH=

# Root path for data storage
DATA_ROOT_PATH= 
# Stage to start reduction, e.g., '1b'
START_STAGE= 
```
You can fill in the values of these environment variables according to your needs :nerd:. 

Before you proceed to the next step, don't leave any of these variables empty, as they are essential for the workflow :thinking:.

The values of these variables can be set by options in order that you can include them in your shell script. But now let's just fill them in manually.

## Second step: Append concrete paths to the `.env` file
The next step is to append the concrete paths of each steps to the `.env` file. You can use the `append` command to do this. And I highly recommend you to run it with the '`-n`' option, which will automatically generate the paths for you based on the "[stages](https://jwst-docs.stsci.edu/jwst-science-calibration-pipeline/stages-of-jwst-data-processing#gsc.tab=0)" provided.

<!-- termynal -->
```console
$ uv run noobenv append -n 2a 2bi 2c 3a
The .env file has been updated with the new stage paths.
```
- Note that if you don't use the '`-a`' option, every time you run the `append` command, it will overwrite the previously provided stages in the `.env` file. Again, see `noobenv append --help` or directly try it once you are confused with the behavior of the command :nerd:.
- The first character of the stage name must be a number. So if you want to save any middle products, name the stage like `2bi` or `2bii`. The stage name is case-insensitive, but I recommend you to use lowercase letters for consistency 🧐.
- Don't be confused by the term `FILE_BOX_PATH`. This points to the path which will be the default path for the `FileBox.load()`. It is just for convenience. You can still load data from any other paths or rewrite this term based on what analysis you are doing.

??? warning "Be careful if you are using terminal of VSCode🚨"
    If you are using the terminal of VSCode, you may be able to directly use the `noobenv` command without the `uv run` prefix:
    ```console
    $ noobenv append -n 2a 2bi 2c 3a
    ```
    However, when you create a new terminal in VSCode, it will automatically load the environment variables from the `.env` file, which will prevent us from updating the environment variables from the `.env` file.

    This is because out of the consideration of safety, we set the `override` option of the `load_dotenv` function to `False`. This means that the environment variables loaded from the `.env` file will not override the existing environment variables in the shell.

    If you don't want to see this behavior, the simplest way is to open a new terminal in VSCode after you make any changes to the environment variables in the `.env` file. 

## Third step: Check and prepare the folders

Now you can check whether the paths in the `.env` file exist by using the `check` command. This will help you to ensure that there are no typos in the paths.

<!-- termynal -->
```console
$ uv run noobenv check

```

In your terminal, the existing folders will be printed in green, and the non-existing folders will be printed in red.

Once you have checked the paths, you can create the folders by running the following command:
<!-- termynal -->
```console
$ uv run noobenv check -m
```

# Final Summary
Now your `.env` file should look like this:

```dotenv title=".env"
--8<-- "docs_srccode/StepByStepTutorial/EnvironmentVariables/.env"
```