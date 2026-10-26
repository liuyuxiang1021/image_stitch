# Models in this directory

`FSRCNN_x4.pb` is the pretrained FSRCNN x4 model from
[Saafke/FSRCNN_Tensorflow](https://github.com/Saafke/FSRCNN_Tensorflow), used by
OpenCV's `dnn_superres` interface. The upstream project identifies its released
models as trained on T91 and fine-tuned on General100. It is distributed under
Apache License 2.0; see [`FSRCNN_LICENSE.txt`](FSRCNN_LICENSE.txt).

The three-region pipeline's default x6 output first resizes the Global image by
1.5 with Lanczos interpolation, then applies this FSRCNN x4 model. The combined
output is x6 in dimensions; this repository does not include a native FSRCNN x6
checkpoint.

An [official LIIF checkpoint](https://github.com/yinboc/liif) was also tried at
x6 on the sample ROI. Its demo accepts a requested output resolution, but the
upstream [DIV2K training configuration](https://github.com/yinboc/liif/blob/main/configs/train-div2k/train_edsr-baseline-liif.yaml)
uses `scale_max: 4`; the x6 result showed no clear visual gain on this sample.
The project therefore uses the bundled, reproducible two-stage fallback by
default.

The RAFT checkpoint is intentionally not included here. Follow
[`scripts/setup_raft.sh`](../scripts/setup_raft.sh) when recomputing optical flow.
