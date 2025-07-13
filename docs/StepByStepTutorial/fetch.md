# Download JWST Data
---
In this part, we will download JWST data from the Multi-Mission Archive at Space Telescope (MAST). If you already have the data you want, you can skip this part.

In this section, we will try to download all the data of a specific proposal id with a specific product level. As an example, we choose the FRESCO program (proposal id: 01895) and the product level '1b', which is the uncalibrated data.

## Visit MAST JWST Portal
The most straightforward way to download JWST data is to visit the [MAST JWST Portal](https://mast.stsci.edu/search/ui/#/jwst) and search for the data you want. 

## Use noobfetch
`jwstnoobfriend` provides a command line tool `noobfetch` to download JWST data, which is built by [**Typer**](https://typer.tiangolo.com/).

As a first step, try to see the help message of `noobfetch` by running `noobfetch --help` in your terminal. You should see something like this:

<!-- termynal -->
```console
$ uv run noobfetch --help
                                                                                                                                                                         
 Usage: noobfetch [OPTIONS] COMMAND [ARGS]...                                                                                                                            
                                                                                                                                                                         
 Check and retrieve JWST data from MAST.                                                                                                                                 
                                                                                                                                                                         
                                                                                                                                                                         
╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.                                                                                               │
│ --show-completion             Show completion for the current shell, to copy it or customize the installation.                                                        │
│ --help                        Show this message and exit.                                                                                                             │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ check      Old version of the check command. In this version, the retrieval is done by astroquery. And for the speed, we assume every dataset of the same instrument  │
│            has the same suffix, which may not be true in some cases. Use retrieve command instead.                                                                    │
│ retrieve   Check the JWST data of given proposal id from MAST.                                                                                                        │
│ download   Download the JWST data of given products list.                                                                                                             │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

As the help message suggests, basically you only need to run the subcommands `retrieve` and `download` to check and download the JWST data.




## Get the product list
First we need to prepare a product list. You can run:

<!-- termynal -->
```console
$ uv run noobfetch retrieve 01895 -l 1b
```
- The proposal id should be 5 digits long, so if you want to retrieve data from a proposal id like `1895`, you should use `01895` instead.
- The `-l` option specifies the [product level](https://jwst-pipeline.readthedocs.io/en/latest/jwst/data_products/stages.html), e.g., `1b`.
- The `--help` option is always available, so you can check the help message to get more flexible options.

??? note "Why it is faster to use `noobfetch`?"
    The `noobfetch` uses asynchronous requests to retrieve the data. 
    
    I don't know why if we sent the FileSetIDs to the MAST API, it will take a long time to get the response. So I use the asynchronous requests and only send one filesetid in one request, which speeds up the retrieval process significantly.

After the command is executed, you should find the product list in a file named `products.json`. You can open it and do some filtering before downloading the data. But in this tutorial, we will just download all the products in the FRESCO program.

## Download the data
If you followed the [Environment Variables](environment.md) part, you can directly run:
<!-- termynal -->
```console
$ uv run noobfetch download products.json
```
The files will be downloaded to the path specified by the `STAGE_1B_PATH` in the `.env` file if you set the `START_STAGE` to `1b`.

Otherwise, you need to specify the folder to save the downloaded data by using the `-o` option. See the help message by running `noobfetch download --help` for more details.

??? tip "Use `tmux` or `screen` to run the command in the background"
    If you are running the command on a remote server, you can use `tmux` or `screen` to run the command in the background. This way, you can close the terminal without interrupting the download process 😋. This method can be applied when you are executing any long-running command in the terminal.

## Use astroquery
You can also use the `astroquery` package.

```python title="Download data with astroquery" hl_lines="10"
from astroquery.mast.missions import MastMissionsClass

mission = MastMissionsClass(mission="jwst")
proposal_id = "01895"  # FRESCO program
product_level = "1b"  # Uncalibrated data
program_query = mission.query_criteria( # type: ignore
    program=proposal_id,
    productLevel=product_level
)
product_list = mission.get_product_list(program_query)
```

But the highlighted line seems to take a super long time to execute :cry:.
You can refer the [astroquery documentation](https://astroquery.readthedocs.io/en/latest/mast/mast_obsquery.html#downloading-data) to learn how to download the data with `astroquery`.