# GRISM image is also image

Basically, the reduction of GRISM image is similar to CLEAR image. The main difference is that we skip the `photom` step in Image2Pipeline.

## Stage 2 for GRISM Images: from rate to cal

Everything is the same as CLEAR image [stage2](stage2-CLEAR.md), except that we need to add one line in the `toml` file to skip the photom step.

```toml title="pid1895_all_2b_grism.toml"
# The setup for reduction stage 2a->2b

# whether save results
save_results = true

output_dir = "/media/disk1/zchao/icrrhome/zchao/noobfriend/PID_1895/stage2/2b"


[steps.resample]
skip = true
[steps.photom]
skip = true
```

Well Done! Now all the files are production ready! Move on with our **NooBackend** and **NooBrowser** to explore the data :D