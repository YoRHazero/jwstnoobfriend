# Welcome to the documentation of NOOB project

**NOOB** (abbreviation for *Not Only Observational Bundle*) project consists of 3 key components. You can find the documentation [here](https://yorhazero.github.io/jwstnoobfriend/).

* **NOOBFRIEND** Stands for *Not Only Observational Bundle For Reduction, Inference, Extraction & Navigation of Data*: a python library built mainly on top of [jwst](https://jwst-pipeline.readthedocs.io/en/latest/), [Pydantic](https://docs.pydantic.dev/latest/), [plotly](https://plotly.com/python/), and etc. It provides several CLI applications to help with data reduction, analysis, management, and visualization. (1) 
{ .annotate }

    1.  Of course also for convenience of building noobackend and noobrowser. :nerd:

* **NOOBackend** Stands for *Not Only Organized Backend*: a backend project built with [FastAPI](https://fastapi.tiangolo.com/), [SQLite](https://sqlite.org/), [SQLModel](https://sqlmodel.tiangolo.com/), and Noobfriend as well.

* **NOOBroswer** Stands for *Not Only Observational Broswer*: a frontend application built by [React](https://react.dev/), [Tanstack Query](https://tanstack.com/query/latest), [PixiJS](https://pixijs.com/), [Chakra UI](https://chakra-ui.com/), and etc.

---

## Main Purpose

The initial purpose is to develop

* a convenient library to help manage the complex process of data reduction and analysis

* a well-arranged tool for visual inspection

for JWST/NIRCam WFSS and Image data.

Maybe in the future I will extend it to include NIRSpec and IFU. (1)
{ .annotate }

1.  :smile_cat: Possibly and hopefully.

---

## Who may want to use it :thinking:

Of course everybody is welcome to use this project 🎉.

But you may want to try this project, if:

1. You are freshman of observational astronomy and not familiar with python. Then here is a noob friend (noob modifies friend, i.e., I am not an expert) for you. :nerd:

2. You want to paly with WFSS data, but get tired of reading mountains of documents.

??? note "3rd reason"
    You cannot tolerate running program with python<3.9! :angry:

---

## How to start

If you are busy, just go through the key concept.

Or you can read the step-by-step tutorial, which will provide a process that begins from 0.5. (1) 
{ .annotate }

1.  It will try to be beginner-friendly, but I still assume you have some common sense of computer, such as how to search on the website, how to open the terminal, etc. :cry:

---
## Installation?
Check the [first part of tutorial](StepByStepTutorial/installation.md)!