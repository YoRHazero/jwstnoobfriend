# Combination is not easy
Combining dithered images is not as straightforward as running a single command. This is because JWST's dithered observations often involve complex patterns and varying observational conditions that need to be carefully accounted for during the combination process.

One of the main challenges is aligning the images accurately. Usually this is done by using Gaia catalog. However, in many cases, the field stars are too sparse to get a good alignment. In such cases, you may need to use some galaxy catalog for alignment. Here I choose to use **CANDELS** catalog for alignment but unfortunately, this cannot be easily generalized to other fields.

## Stage 3 for CLEAR Images: from cal to i2d
In this step, I will directly show you the script

```python title="pid1895_2bi_3a.py"
--8<-- "docs_srccode/StepByStepTutorial/stage3-clear/pid1895_2bi_3a.py"
```

Unfortunately, the automatically generated catalogs in `pyvo` do not have good quality for alignment. Therefore, you have to manually find a good catalog for alignment by yourself.