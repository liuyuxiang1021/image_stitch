# image_stitch

来自原工程 `image_stitching` 的跨视角图像拼接项目。仓库按算法分为三块；目前前两块有代码、真实输入和可复现输出，视频背景去除留待后续整理。

| 模块 | 用途 | 可运行样例 |
| --- | --- | --- |
| [`image_stitch`](docs/image_stitch.md) | OmniGlue 点匹配 + LineTR 线匹配，估计/精修单应性并羽化拼接 | 运动场、展厅、校园道路各一组图像 |
| [`three_region_stitch`](docs/three_region_stitch.md) | 双视角视频运动掩膜 + 点线运动区单应性 + RAFT 静态区 + Global 背景合图 | 商街人群、车流、运动场人群各 200 帧双视角短视频 |
| `video_background_removal` | 视频背景去除 | 后续加入 |

## 1. 纯点线单应性拼接

输入是 Local 和 Global 两张图像，可选相机标定初值 `H_init`。输出 `H_final`、投影和羽化融合图。三个不同场景的输入、初值局部放大、点线匹配及最终输出见[纯拼接样例说明](docs/image_stitch.md)。

![运动场最终拼接定位及局部放大](examples/image_stitch/20201024084037322_cam11/output/final_alignment.jpg)

## 2. 运动感知三分区域拼接

输入是同一时刻的 Local/Global 图像、`H_init`，以及两个视角各自的短视频。视频分别生成运动掩膜；运动区的点线匹配估计 `H_final`。最终画面在 Global 坐标系中分为：**红色**，Local 与 Global 都判为运动且 Local 投影有效，用单应性投影；**蓝色**，其余有效重叠区且 RAFT 前后向检查通过，用 RAFT 对齐；**绿色**，由 Global 保留。仅在区域交界附近羽化 2 px。

下面三组样例的 Local/Global 输入、200 帧 ROI 视频、缓存的掩膜/RAFT 中间结果、最终图和同范围对照均在仓库中。视频是真实相机帧，不是合成运动。详见[三分区域算法、数据与复现](docs/three_region_stitch.md)。

### 商街人群 `20220806141041707_cam11`

| Local 最后一帧 | Global 最后一帧 |
| --- | --- |
| ![商街 Local](examples/three_region_stitch/20220806141041707_cam11/input/local.jpg) | ![商街 Global](examples/three_region_stitch/20220806141041707_cam11/input/global.jpg) |

![商街三分区域叠加于 Global](examples/three_region_stitch/20220806141041707_cam11/output/three_region/region_overlay_on_global.jpg)

![商街同范围纯拼接和三分区域对照](examples/three_region_stitch/20220806141041707_cam11/output/three_region/comparison_focus.jpg)

![商街静态共同区域误差热图](examples/three_region_stitch/20220806141041707_cam11/output/benchmark/static_error_heatmaps.jpg)

### 城市道路 `20210304162543165_cam11`

| Local 最后一帧 | Global 最后一帧 |
| --- | --- |
| ![道路 Local](examples/three_region_stitch/20210304162543165_cam11/input/local.jpg) | ![道路 Global](examples/three_region_stitch/20210304162543165_cam11/input/global.jpg) |

![道路三分区域叠加于 Global](examples/three_region_stitch/20210304162543165_cam11/output/three_region/region_overlay_on_global.jpg)

![道路同范围纯拼接和三分区域对照](examples/three_region_stitch/20210304162543165_cam11/output/three_region/comparison_focus.jpg)

![道路静态共同区域误差热图](examples/three_region_stitch/20210304162543165_cam11/output/benchmark/static_error_heatmaps.jpg)

### 运动场人群 `20220601174105159_cam11`

| Local 最后一帧 | Global 最后一帧 |
| --- | --- |
| ![运动场 Local](examples/three_region_stitch/20220601174105159_cam11/input/local.jpg) | ![运动场 Global](examples/three_region_stitch/20220601174105159_cam11/input/global.jpg) |

![运动场三分区域叠加于 Global](examples/three_region_stitch/20220601174105159_cam11/output/three_region/region_overlay_on_global.jpg)

![运动场同范围纯拼接和三分区域对照](examples/three_region_stitch/20220601174105159_cam11/output/three_region/comparison_focus.jpg)

![运动场静态共同区域误差热图](examples/three_region_stitch/20220601174105159_cam11/output/benchmark/static_error_heatmaps.jpg)

## 快速运行

在 Python 3.10–3.12 环境中安装基础包，然后先运行不需下载模型的缓存复现。仓库内缓存的是上述真实视频运行得到的运动掩膜、点线精修矩阵和 RAFT 对齐结果；程序会重新完成三区归属和融合。

```bash
git clone --recursive https://github.com/liuyuxiang1021/image_stitch.git
cd image_stitch
python -m pip install -e .
three-region-stitch --example examples/three_region_stitch/20220806141041707_cam11
three-region-stitch --example examples/three_region_stitch/20210304162543165_cam11
three-region-stitch --example examples/three_region_stitch/20220601174105159_cam11
```

输出在每个样例的 `output/three_region/`：`final_result.jpg` 是无标注最终融合图，`region_overlay_on_global.jpg` 是区域归属，`comparison_focus.jpg` 是同一 ROI 的纯拼接/三分区域并排图，`summary.json` 记录矩阵、运行模式与像素数。可用 `--motion-mode recompute` 从随附视频重新生成两个掩膜；完整点线匹配和 RAFT 重算需要模型安装，命令见[三分区域文档](docs/three_region_stitch.md)。纯拼接运行命令见[纯拼接文档](docs/image_stitch.md)。

## 同场景小规模评测

用相同的最后一帧和 `H_init` 运行纯 OmniGlue + LineTR 单应性拼接、OpenCV SIFT + RANSAC 单应性，以及运动区单应性 + RAFT 静态区。只在三种方法共同有效、且双视角均判为静态的像素上，计算对 Global 帧的 PSNR/SSIM；这衡量**静态重叠区对齐**，不等于整图视觉质量，也不是公开数据集的官方排名。

| 场景 | SIFT + RANSAC | 纯点线 H | 三分区域的 RAFT 静态区 |
| --- | --- | --- | --- |
| 商街人群 | 16.54 dB / 0.528 | 16.75 dB / 0.550 | **18.24 dB / 0.670** |
| 城市道路 | 19.60 dB / 0.664 | 17.75 dB / 0.511 | **20.67 dB / 0.715** |
| 运动场人群 | 18.11 dB / 0.547 | 17.85 dB / 0.512 | **18.50 dB / 0.551** |

表内数值由[`scripts/evaluate_three_region.py`](scripts/evaluate_three_region.py)生成；逐场景像素数、匹配数、SIFT 输出和精确数值见各样例的 `output/benchmark/`。道路场景中 SIFT 静态区指标高于纯点线 H，结果照实保留。运动目标没有无遮挡的真值，不能从表格推断其质量；请同时查看并排图和区域图。

## 仓库结构

```text
src/image_stitch/                # 纯点线单应性与羽化融合
src/three_region_stitch/         # 双运动掩膜、运动区 H、RAFT、三区合图
examples/image_stitch/           # 三组静态图像输入与输出
examples/three_region_stitch/    # 三组双视角视频输入与输出
docs/                            # 算法、数据、评测及运行细节
third_party/                     # 官方 OmniGlue、LineTR、RAFT Git submodules
scripts/                         # 模型准备、样例导出、评测
```

`video_background_removal` 目前没有实现或样例，后续再加入。LineTR 的许可证限制为学术或非营利机构的非商业研究用途，使用前请阅读 [`third_party/LineTR/LICENSE`](third_party/LineTR/LICENSE)。
