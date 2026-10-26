# Models in this directory

`FSRCNN_x4.pb` is the pretrained FSRCNN x4 model from
[Saafke/FSRCNN_Tensorflow](https://github.com/Saafke/FSRCNN_Tensorflow), used by
OpenCV's `dnn_superres` interface. The upstream project identifies its released
models as trained on T91 and fine-tuned on General100. It is distributed under
Apache License 2.0; see [`FSRCNN_LICENSE.txt`](FSRCNN_LICENSE.txt).

The RAFT checkpoint is intentionally not included here. Follow
[`scripts/setup_raft.sh`](../scripts/setup_raft.sh) when recomputing optical flow.
