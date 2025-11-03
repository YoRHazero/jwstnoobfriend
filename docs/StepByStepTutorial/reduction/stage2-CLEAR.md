# Why I need FileBox?
You may want to know why this package provides a class called `FileBox` to help manage files. I will show some convenience of using `FileBox` in the following sections.
## Stage 2 for CLEAR Images: from rate to cal
Based on the [official workflow](https://jwst-pipeline.readthedocs.io/en/latest/jwst/associations/technote_sdp_workflow.html), we need to proceed the CLEAR image reduction first, and then use the results to do the Grism reduction.

In this step, we will use the `noobscript` command again.

<!-- termynal -->
```console
$ uv run noobscript stage 2b -p pid1895
```

Take a look at the generated `toml` file:

```toml title="pid1895_all_2b.toml"
--8<-- "docs_srccode/StepByStepTutorial/stage2-clear/pid1895_clear_2b.toml"
```

As always, you can modify the parameters as needed. Here I just skip the `resample` step, because later we will combine the dithered images.

Here is the generated script:

```python title="pid1895_2a_2b.py"
--8<-- "docs_srccode/StepByStepTutorial/stage2-clear/pid1895_clear_2a_2b.py"
```

Additionally, we provide the script `pid1895_2bi_3a.py` for sky subtraction if you run:

<!-- termynal -->
```console
$ uv run noobscript stage 2bi -p pid1895
```

??? note "source code of pid1895_2b_2bi"
    ```python title="pid1895_2b_2bi.py"
    --8<-- "docs_srccode/StepByStepTutorial/stage2-clear/pid1895_clear_2b_2bi.py"
    ```

After the `Image2Pipeline` step, you can interactively check the footprints of your reduced images by using the following code:

```python title="Check footprints of reduced images"
from jwstnoobfriend.navigation import FileBox
file_box = FileBox.load(suffix='clear')
file_box.show_footprints()
```

Here, we show **FRESCO** footprints as an example:

<iframe src="/assets/images/StepByStepTutorial/footprints_example.html" width="100%" height="600px" frameborder="0"></iframe>