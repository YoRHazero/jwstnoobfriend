# How to install it
---
First, open a terminal window and start your shell. I am using [fish](https://fishshell.com/). (1) 
{ .annotate }

1.  **Fish** 🐟 is a morden shell with smooth experience of auto completion, highly recommend you to have a try 😉！

=== "uv"
    ```shell title="fish"
    uv add jwstnoobfriend
    ```
=== "pip"
    ```shell title="fish"
    pip install jwstnoobfriend
    ```

???+ note "use [uv](https://docs.astral.sh/uv/) for package management"
    Wait🤔, what is [**uv**](https://docs.astral.sh/uv/) here? Personally, I think it is a python version of [**npm**](https://www.npmjs.com/) (javascript/typescript) or [**cargo**](https://doc.rust-lang.org/cargo/) (rust). 
    
    Different from [**anaconda**](https://www.anaconda.com/), you will not have a virtual environment globaly, but each workspace (folder) will have its own virtual environment. This is super benifitial when you plan to output the dependencies of your project. Also, you don't have to maintain a lengthy list of anaconda virtual environment to avoid potential pacakge version conflict.


The following steps are a brief tutorial on how to use uv. Safely skip it if you are already familiar with uv or you want to use old-school `pip`.

## Install uv
Please refer to the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/#installation-methods) for more details. Here I just list the necessary steps so that you don't have to jump around.

=== "macOS and Linux"
    ```shell title="fish"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
=== "Windows"
    ```powershell title="powershell"
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

Note that I don't have experience with Windows System, so if you encounter any issues, please refer to the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/#windows) for troubleshooting. And for the following steps, I will only show the commands for macOS and Linux.

## Initialize a working folder
After you have installed uv, you can create a new working folder and initialize it with uv.

<!-- termynal -->
```
$ uv init project_name
Initialized project `project_name` at `/Users/zero/project_name`
$ cd project_name
```
This will create a new folder named `project_name` and a virtual environment inside it.

??? note "tldr"
    To get the help of uv, of course you can use `uv --help`, or `uv init --help` to get the help of subcommands. But I found uv and its subcommands have `tldr` documentation. So use `tldr uv` or `tldr uv init` will give you a concise summary of the command. But ensure you have installed [tldr](https://tldr.sh/) first.

## Use notebook in vscode with uv
For uv, `uv add package_name` command is almost the same as `pip install package_name`, but I would like to suggest to use `uv add --dev package_name` to install the packages like `jupyter`, `ipykernel`, etc. 

??? tip "why and when to use `--dev`?"

    This is because these packages are only used by you for development, and for people who try to reproduce your work, they don't need to use these packages. A simple way to distinguish whether you should use `--dev` or not is to check whether you have to import the package in your code. 

    For example, I think for most people, they never import `ipykernel` in their code, so you should use `uv add --dev ipykernel` to install it. But for `numpy`, you should use `uv add numpy` to install it, because you will import it in your code.
<!-- termynal -->
```
$ uv add --dev jupyter ipykernel
---> 100%
Installed 2 packages in 0.1s
+ jupyter
+ ipykernel
```

Make sure you have installed the necessary vscode extentions for jupyter notebook (search jupyter in the extension marketplace).

Then create a file with the suffix `.ipynb`, and open it with vscode. You will see a button to select the kernel.

![select_kernel](../assets/images/StepByStepTutorial/select_kernel.png)

If your vscode workspace is the project folder you just created, you will directly see a recommended kernel and choose it. If not, select the file `project_name/.venv/bin/python` as the kernel.

## Install packages
Then you can install the packages you need!

<!-- termynal -->
```
$ uv add jwstnoobfriend
---> 100%
Installed 1 package in 0.1s
jwstnoobfriend
```
Note that you can start to run some code in the notebook and then try to use `uv add` to install a new package. Then import it in the notebook. Try the same thing with the kernel to be an anaconda environment (by using `pip install`).

You will find that you don't have to restart the kernel to use the newly installed package in uv, but not in anaconda. Another reason to try uv 😊!

???+ tip "remove a package"
    If you want to remove a package from your dependencies, you can use `uv remove package_name` command. 

## A brief summary :fire:

* `uv init` to initialize a new project folder with a virtual environment. 
* `uv add package_name` to install a package in the virtual environment. use `--dev` to install a package only for development.
* `uv remove package_name` to remove a package from the virtual environment.

    
