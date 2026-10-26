# image_stitch

来自原工程 `image_stitching` 的跨视角图像拼接项目。仓库按算法分为三块；目前前两块有代码、真实输入和可复现输出，视频背景去除留待后续整理。

| 模块 | 用途 | 可运行样例 |
| --- | --- | --- |
| [`image_stitch`](docs/image_stitch.md) | OmniGlue 点匹配 + LineTR 线匹配，估计/精修单应性并羽化拼接 | 运动场、展厅、校园道路各一组图像 |
| [`three_region_stitch`](docs/three_region_stitch.md) | 双视角视频运动掩膜 + 点线运动区单应性 + RAFT 静态区 + Global 背景合图 | 商街人群双视角 200 帧短视频 |
| `video_background_removal` | 视频背景去除 | 后续加入 |

## 1. 纯点线单应性拼接

输入是 Local 和 Global 两张图像，可选相机标定初值 `H_init`。输出 `H_final`、投影和羽化融合图。三个不同场景的输入、初值局部放大、点线匹配及最终输出见[纯拼接样例说明](docs/image_stitch.md)。

![运动场最终拼接定位及局部放大](examples/image_stitch/20201024084037322_cam11/output/final_alignment.jpg)

## 2. 运动感知三分区域拼接

### 算法流程

算法主要处理不同景深的前景与背景因视角变化产生的视差错位，典型情况是树木、路灯杆等近景结构和远处背景难以被同一个单应性同时对齐。Local/Global 短视频得到的运动掩膜提供区域划分线索；两侧共同运动区域、其他有效 Local 区域和不可靠区域分别走不同的融合路径。输入包括同步的 Local/Global 图像、相机标定得到的初始单应矩阵 `H_init`，以及两路相机对应的短视频。

1. **建立几何初值**：用 `H_init` 把 Local 映射到 Global 坐标系，作为两视角对齐的起点。
2. **提取运动线索并精修单应性**：分别从 Local、Global 视频生成 `M_local_init` 和 `M_global`。它们记录一段时间内出现过变化的位置。点线匹配只保留在两侧运动掩膜中都成立的对应，据此估计增量 `H_delta`，并得到 `H_final = H_delta · H_init`。
3. **把掩膜和有效范围放到同一坐标系**：`M_local_init` 位于 `H_init` 初始投影坐标；再用 `H_delta` 将它更新到 `H_final` 坐标，得到 `M_local_final`。同时用 `H_final` 投影 Local 的全有效图，得到 `V_local_final`。现在 `M_local_final`、`M_global` 和 `V_local_final` 都在 Global 画布上，可以逐像素确定归属。
4. **生成互斥的三区归属图**：

   | 区域 | 像素条件 | 该区域使用的图像 |
   | --- | --- | --- |
   | 红色：运动区单应性 | `M_local_final ∩ M_global ∩ V_local_final`。两视角都判为运动，而且 Local 投影有效。 | 用 `H_final` 投影的 Local |
   | 蓝色：RAFT 区 | `V_local_final` 中不属于红区、并通过 RAFT 前后向一致性检查的像素。默认也包含只有一侧判运动的像素。 | 用 RAFT 对齐后的 Local |
   | 绿色：Global 背景区 | 不属于红区或有效蓝区的所有像素，包括 Local 无有效投影、RAFT 检查失败的位置。 | 用 FSRCNN x4 超分后的 Global |

   默认划分可写为 `R_H = M_local_final ∩ M_global ∩ V_local_final`，`R_RAFT = (V_local_final − R_H) ∩ V_RAFT`，`R_Global = Ω_Global − (R_H ∪ R_RAFT)`。`V_RAFT` 是 RAFT 前后向误差小于 3 px 且取样有效的位置，`Ω_Global` 是整张 Global 画布。因此三块互不重叠，并覆盖整张画布。

   **为什么用 motion mask：**运动掩膜不是前景/背景语义标签。它标出视频中随时间发生变化的位置，用来约束共同运动区域里的点线匹配，并决定哪些像素走单应性分支。树木、路灯杆等静止近景不一定进入运动掩膜；它们与远景之间的局部视差由蓝色 RAFT 区处理，RAFT 不可靠时则回退到绿色 Global 超分背景。也就是说，motion mask 提供分区线索，前景/背景视差由不同区域的对齐与取景策略共同处理。
5. **合成高分辨率结果**：先对 Global 背景执行 FSRCNN x4，再将红色单应性区域和蓝色 RAFT 区域的 Local 内容合入对应位置。仅在三区边界附近羽化，默认半径为原画布 2 px。

当前展示这一组商街样例。Local/Global 输入、200 帧 ROI 视频、缓存的掩膜/RAFT 中间结果、最终图和同范围对照均在仓库中。视频是真实相机帧，不是合成运动。详见[三分区域算法、数据与复现](docs/three_region_stitch.md)。

### 商街人群 `20220806141041707_cam11`

| Local 最后一帧 | Global 最后一帧 |
| --- | --- |
| ![商街 Local](examples/three_region_stitch/20220806141041707_cam11/input/local.jpg) | ![商街 Global](examples/three_region_stitch/20220806141041707_cam11/input/global.jpg) |

![商街三区叠加于 Global 超分背景](examples/three_region_stitch/20220806141041707_cam11/output/three_region/region_overlay_on_global_fsrcnn_x4.jpg)

![商街同范围纯拼接和三分区域对照](examples/three_region_stitch/20220806141041707_cam11/output/three_region/comparison_focus.jpg)

![商街 Global 超分背景上的最终三分区域融合结果](examples/three_region_stitch/20220806141041707_cam11/output/three_region/final_result_global_fsrcnn_x4.jpg)

![商街静态共同区域误差热图](examples/three_region_stitch/20220806141041707_cam11/output/benchmark/static_error_heatmaps.jpg)

## 快速运行

在 Python 3.10–3.12 环境中安装基础包，然后运行缓存复现。仓库内附 FSRCNN x4 权重，并缓存真实视频运行得到的运动掩膜、点线精修矩阵和 RAFT 对齐结果；程序会重新完成三区归属、Global 背景超分和融合。

```bash
git clone --recursive https://github.com/liuyuxiang1021/image_stitch.git
cd image_stitch
python -m pip install -e .
three-region-stitch --example examples/three_region_stitch/20220806141041707_cam11
```

输出在该样例的 `output/three_region/`：`final_result.jpg` 是原画布分辨率融合结果，`final_result_global_fsrcnn_x4.jpg` 是 Global 背景经 FSRCNN x4 后再合入 Local 区域的高分辨率结果，`region_overlay_on_global_fsrcnn_x4.jpg` 展示同一高分辨率背景上的区域归属，`comparison_focus.jpg` 是原画布分辨率下同一 ROI 的纯拼接/三分区域对照，`summary.json` 记录矩阵、运行模式与像素数。可用 `--sr-mode none` 跳过超分；可用 `--motion-mode recompute` 从随附视频重新生成两个掩膜。完整点线匹配和 RAFT 重算需要模型安装，命令见[三分区域文档](docs/three_region_stitch.md)。纯拼接运行命令见[纯拼接文档](docs/image_stitch.md)。

## 同场景小规模评测

用相同的最后一帧和 `H_init` 运行纯 OmniGlue + LineTR 单应性拼接、OpenCV SIFT + RANSAC 单应性，以及运动区单应性 + RAFT 静态区。只在三种方法共同有效、且双视角均判为静态的像素上，计算对 Global 帧的 PSNR/SSIM；这衡量**静态重叠区对齐**，不等于整图视觉质量，也不是公开数据集的官方排名。

| 场景 | SIFT + RANSAC | 纯点线 H | 三分区域的 RAFT 静态区 |
| --- | --- | --- | --- |
| 商街人群 | 16.54 dB / 0.528 | 16.75 dB / 0.550 | **18.24 dB / 0.670** |

表内数值由[`scripts/evaluate_three_region.py`](scripts/evaluate_three_region.py)生成；像素数、匹配数、SIFT 输出和精确数值见该样例的 `output/benchmark/`。运动目标没有无遮挡的真值，不能从这些静态区指标推断其融合质量；请同时查看并排图和区域图。

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
