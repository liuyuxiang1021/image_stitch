# 视频静态背景提取

[返回项目首页](../README.md)

## 功能与算法流程

输入固定机位视频，输出一张**灰度背景图**和一张辅助的静态连通掩码。实现整理自原工程 `MotionMask/script/gmm_method_final_script/` 的 `preprocess.py`、`main_gmm.py` 和 `config.py`，代码位于 [`src/video_background_removal/`](../src/video_background_removal/)。

1. **建立时序图像块。** 将每帧转为灰度，默认按 4 × 4、步长 3 划分重叠图像块。同一空间位置在不同帧的图像块组成 16 维时序样本。边缘额外补块，确保像素都被覆盖。
2. **估计背景块。** 对每个位置独立拟合 10 分量、对角协方差的 GMM，取**混合权重最大**的分量均值作为背景块；拟合失败才回退到时间均值。这里不是逐像素拟合，也不是逐像素取峰。
3. **重建背景图。** 同一像素可能属于多个背景块，将它们的预测值平均，得到 `background_model_gmm.png`。
4. **生成辅助掩码。** 从图像中心的背景块开始，检查上下左右相邻块的重叠像素；其均方根误差（RMSE）小于默认阈值 4 时，将相邻块加入连通区域。把连通块覆盖的像素写入 `static_region_mask.png`。4 × 4、步长 3 时，相邻块只有 1 px 宽的重叠带。

第 4 步**不参与第 2、3 步的背景估计**。掩码描述的是背景块之间能否通过重叠误差连通，不是行人检测或可靠的运动掩膜。即使掩码覆盖率很高，背景图仍可能有人影残留；裁剪还会改变中心起点与边缘图像块，因此裁剪掩码和全图掩码也不一定相同。

## 原始 4K Global 样例

商街 `20220806141041707` 的 Global 原视频为 3840 × 2160。下图展示同一原始画布：左图是参与采样的 Frame 5997，中央是 500 帧估计出的灰度背景，右图是辅助连通掩码。以 Frame 4001 为起点，每隔 4 帧取 1 帧，共覆盖到 Frame 5997；详细参数和来源见[样例清单](../examples/video_background_removal/20220806141041707_cam100/full_global_review/manifest.json)。

| 原始 Global · Frame 5997 | GMM 背景估计 | 辅助静态连通掩码 |
| --- | --- | --- |
| <img src="../examples/video_background_removal/20220806141041707_cam100/full_global_review/input/global_frame_5997.jpg" width="300" alt="商街原始 4K Global 帧"> | <img src="../examples/video_background_removal/20220806141041707_cam100/full_global_review/output/background_model_gmm.png" width="300" alt="商街 4K GMM 背景，局部仍有行人残影"> | <img src="../examples/video_background_removal/20220806141041707_cam100/full_global_review/output/static_region_mask.png" width="300" alt="商街 4K 静态连通掩码"> |

这次全幅运行得到 921,600 个图像块，掩码覆盖 98.47% 的像素，但背景图在商店门前和远处行人密集处仍有明显残影。这说明连通掩码不能当作去人的质量分数。仓库保存上述输入帧和输出图供查看；约 880 MB 的原视频未纳入仓库，因此这组全幅结果不能只靠这些静帧重新计算。

## 可运行的小视频样例

[`20220806141041707_cam100/`](../examples/video_background_removal/20220806141041707_cam100/) 另附真实 Global 视频的 1000 帧、160 × 120 局部，`input/video.mp4` 可直接运行；`input/frame_6000.jpg` 是预览。来源和裁剪坐标见 [`sample.json`](../examples/video_background_removal/20220806141041707_cam100/sample.json)。其 `output/` 保存背景图、连通掩码、叠加预览 `final_result.png`、掩码内背景 `final_result_masked.png`、各块所选 GMM 参数 `patch_models.npz` 和 `summary.json`。叠加预览与掩码图只是诊断输出，实际估计的背景是 `background_model_gmm.png`。

## 运行

```bash
python -m pip install -e ".[background]"
video-background-removal \
  examples/video_background_removal/20220806141041707_cam100/input/video.mp4 \
  --output-dir examples/video_background_removal/20220806141041707_cam100/output \
  --jobs 4
```

默认处理输入视频的全部帧。可用 `--start-frame`、`--max-frames`、`--frame-step` 控制取帧，用 `--patch-size`、`--stride`、`--components` 和 `--static-threshold` 调整算法。大视频可通过 `--work-dir` 指定有足够空间的临时数据目录。短片段中行人可能长期遮挡背景，导致残影；样例使用 1000 帧而非短片段。

若已有原始 4K 视频，可用下面的参数复现上面的全幅结果；`--output-dir` 请指向有足够空间的位置。原视频的具体来源及帧号见样例清单。

```bash
video-background-removal /path/to/20220806141041707/main/100.mp4 \
  --output-dir /path/to/output \
  --start-frame 4001 --frame-step 4 --max-frames 500 \
  --patch-size 4 --stride 3 --components 10 --jobs 4 \
  --work-dir /path/to/temporary_storage
```

## 样例运行时间

同一段 160 × 120 商街视频、2120 个重叠图像块、10 个高斯分量，使用 4 个并行任务，在 Intel i9-10900X、Python 3.10、scikit-learn 1.7.2 环境中实测如下。时间包含视频解码、拟合和写出；它只对应这个小尺寸样例，不代表 4K 视频的耗时。

| 处理帧数 | 取帧方式 | 墙钟时间 |
| ---: | --- | ---: |
| 100 | 前 100 帧 | 15.84 秒 |
| 500 | 前 500 帧 | 25.60 秒 |
| 500 | 全片每隔 2 帧取 1 帧（`--frame-step 2`） | 32.68 秒 |
| 1000 | 全部 1000 帧 | 88.54 秒 |

原始 4K Global 的 500 帧、每隔 4 帧取样实验使用相同的 4 个并行任务，实测 **7398.03 秒（约 2 小时 03 分）**，临时图像块文件约 6.9 GiB。两组耗时都包含视频解码、GMM 拟合和写出，不能直接按像素数推算其他帧数的耗时。

小样例前 100 帧仍留有明显残影。同一全片隔帧采样所得背景与其 1000 帧版本的 SSIM 为 0.9818，只表示两张算法输出相近，不是背景真值评分。增大图像块并同步增大步长可以减少待拟合块数，但也会改变背景细节和连通掩码；相关参数对比仍在进行。
