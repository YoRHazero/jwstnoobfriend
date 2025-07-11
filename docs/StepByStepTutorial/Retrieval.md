# Download JWST Data
---
In this part, we will download JWST data from the Multi-Mission Archive at Space Telescope (MAST). If you already have the data you want, you can skip this part.

In this section, we will try to download all the data of a specific proposal id with a specific product level. As an example, we choose the FRESCO program (proposal id: 01895) and the product level '1b', which is the uncalibrated data.

## Visit MAST JWST Portal
The most straightforward way to download JWST data is to visit the [MAST JWST Portal](https://mast.stsci.edu/search/ui/#/jwst) and search for the data you want. 

## Use noobfetch
`jwstnoobfriend` provides a command line tool `noobfetch` to download JWST data, which is built by [**Typer**](https://typer.tiangolo.com/).

As a first step, try to see the help message of `noobfetch` by running `noobfetch --help` in your terminal. You should see something like this:

```shell
 noobfetch --help
                                                                                                                                                                         
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



## Use astroquery
You can also use the `astroquery` package.

```python title="Download data with astroquery" hl_lines="4-9"
from astroquery.mast.missions import MastMissionsClass

mission = MastMissionsClass(mission="jwst")
proposal_id = "01895"  # FRESCO program
product_level = "1b"  # Uncalibrated data
program_query = mission.query_criteria( # type: ignore
    program=proposal_id,
    productLevel=product_level
)
```

