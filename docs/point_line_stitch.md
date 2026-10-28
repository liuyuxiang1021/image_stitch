# 纯点线单应性拼接

[返回项目首页](../README.md)

## 功能

- [OmniGlue](https://github.com/google-research/omniglue) 跨视角点匹配；
- [LineTR](https://github.com/yosungho/LineTR) 的 LSD 线段检测、描述与匹配；
- RANSAC 点匹配初值与线约束后备初值；
- 基于重投影距离和方向角的线匹配过滤；
- 点重投影误差 + 采样点到线误差的 8 自由度联合优化；
- Huber 鲁棒权重和点线匹配置信度权重；
- 直接估计 `H`，或从已有 `H_init` 估计 `H_delta` 并生成 `H_final`；
- 自动画布、有效掩膜和有限宽度的 local 边缘羽化融合；
- 根据 Local/Global 有效重叠区估计 HHM 色彩映射，默认校正 Local 后再融合；
- 保存矩阵、匹配数量、RMSE 以及点线匹配可视化。

算法和坐标变换的详细说明见 [algorithm.md](algorithm.md)。

## 算法流程

1. 使用可选的标定初值 `H_init` 将 Local 投影到 Global，限定后续匹配区域；未提供初值时，在两张输入图像上直接估计单应性。
2. 在有效重叠区分别进行 OmniGlue 点匹配和 LineTR 线匹配，并按置信度、几何距离与方向过滤不可靠匹配。
3. 由点匹配建立稳健初值，再联合优化点重投影误差与线采样距离，得到 `H_final`；有初值时满足 `H_final = H_delta @ H_init`。
4. 根据 Local/Global 有效重叠像素估计一套 HHM 映射，应用于 Local；将 Local 投影到 Global 画布，仅在投影边缘 6 px 范围内羽化融合。

以下五组真实视角均包含 Local/Global 输入、初值定位、点线匹配可视化和最终结果。点线图每组最多展示 40 个点、12 条线，以便辨认；单应性估计仍使用全部过滤后的匹配。

## 输入和输出

五组样例都采用初值精修模式。每个样例目录的输入含义如下：

| 文件 | 含义 |
| --- | --- |
| `input/local.jpg` | 源图：局部相机拍摄的高分辨率、较窄视角图像 |
| `input/global.jpg` | 目标/参考图：全局相机拍摄并已去畸变的宽视角图像 |
| `input/H_init.npz` | 原项目标定得到的粗单应性初值，文件内矩阵键为 `H` |

五份 `H_init.npz` 均来自原项目的标定和地面参数。原项目 `to_yy` 目录中的部分同场景 `H_*.npz` 保存的是已经优化的 `H_final`，没有作为初值使用。

坐标方向始终是从 local 到 global：

```text
p_global ~ H_final @ p_local
H_final = H_delta @ H_init
```

程序先用 `H_init` 将 local 投到 global 坐标系，在有效重叠区域执行 OmniGlue 点匹配和 LineTR 线匹配，再联合优化增量 `H_delta`。随后按投影后的 Local/Global 重叠像素估计 HHM 色彩映射，先校正 Local，再进行边缘羽化融合。默认映射系数为 1，可用 `--no-color-correction` 关闭。每个样例的输出目录包含：

| 文件 | 含义 |
| --- | --- |
| `output/panorama.jpg` | 原分辨率 Global 画布上的拼接诊断图 |
| `output/local_roi_x6.jpg` | Local 投影附近的近 4K 拼接图；直接投影原始 Local |
| `output/global_roi_lanczos_x6.jpg` | 与上一张同范围、经 Lanczos 放大 6 倍的 Global 背景 |
| `output/initial_warp.jpg` | 只用 `H_init` 投影 local 的原始全尺寸结果；投影区域外为黑色 |
| `output/initial_overview.jpg`、`initial_zoom.jpg` | 初值投影的 Global 定位图与 Local/Global 局部叠加图 |
| `output/final_overview.jpg`、`final_zoom.jpg` | 最终画布定位图与羽化融合局部图 |
| `output/point_line_matches.jpg` | 投影 ROI 中过滤后匹配的抽样展示；优化仍使用全部过滤后匹配 |
| `output/metadata.json` | 输入路径、初值和最终投影 ROI、`H_init`、`H_delta`、`H_final`、画布、匹配数量、RMSE、迭代次数及 HHM 采样统计 |

这些样例中 Local 的视野位于 Global 的一小块区域。`panorama.jpg` 保留原来的 3840 × 2160 Global 画布供定位；主要展示的 `local_roi_x6.jpg` 只输出 Local 投影附近的近 4K 区域，Global 背景插值放大，Local 仍使用原始像素投影。命令行运行时还会保存同范围的 `global_roi_lanczos_x6.jpg`。三分区域中的纯单应性对照复用同一投影、HHM 校色和羽化函数，但使用 FSRCNN 生成的 Global 背景。

可用已保存的 `H_final` 重建近 4K 图，无需重新运行匹配模型：

```bash
python scripts/render_point_line_stitch_roi.py --example examples/point_line_stitch/20201024084037322_cam11
```

`initial_overview.jpg` 的青色轮廓表示 local 经 `H_init` 投影后的实际边界，黄色矩形表示带 32 px 留白的可视化 ROI；`initial_zoom.jpg` 将这个 ROI 放大，并按照 local 55%、global 45% 叠加。局部图使用 2 px 青色虚线标记 local 的真实边界，以减少对对齐细节的遮挡。

`final_overview.jpg` 和 `final_zoom.jpg` 基于 `H_final` 和实际羽化融合结果生成：前者是带定位框的完整最终画布，后者是黄色框对应的最终融合像素，并使用相同的 2 px 青色虚线标记 local 边界。原始无标注大图仍保存在 `panorama.jpg`，五个样例的尺寸均为 `3840 × 2160`。

默认根据重叠区估计 HHM 颜色映射并校正 Local，再将 Local/Global 的归属掩膜以高斯权重归一化融合。羽化半径为 6 px，颜色在投影边界两侧的窄带内平滑过渡。可通过 `--feather-width` 调整半径；设为 `0` 表示不羽化、直接覆盖。色彩映射系数和 HHM 稀疏直方图阈值可分别通过 `--color-correction-coeff`、`--color-correction-threshold` 调整。

## 五组真实视角样例

### 1. 运动场：`20201024084037322_cam11`

不同相机视角下的室外运动场。最终保留 329/335 个点匹配和 68/74 个线匹配；点 RMSE 为 1.577 px，线 RMSE 为 1.831 px。相对于相机标定得到的 `H_init`，`H_final` 在 local 全图网格上的平均修正约为 54 px。

| local 输入 | global 输入 |
| --- | --- |
| ![运动场 local](../examples/point_line_stitch/20201024084037322_cam11/input/local.jpg) | ![运动场 global](../examples/point_line_stitch/20201024084037322_cam11/input/global.jpg) |

#### 初值定位及局部放大

| 初值全画布定位 | 初值局部放大 |
| --- | --- |
| ![运动场 initial alignment 全画布](../examples/point_line_stitch/20201024084037322_cam11/output/initial_overview.jpg) | ![运动场 initial alignment 局部](../examples/point_line_stitch/20201024084037322_cam11/output/initial_zoom.jpg) |

#### 最终融合定位及局部放大

| 最终融合全画布定位 | 最终融合局部放大 |
| --- | --- |
| ![运动场 final alignment 全画布](../examples/point_line_stitch/20201024084037322_cam11/output/final_overview.jpg) | ![运动场 final alignment 局部](../examples/point_line_stitch/20201024084037322_cam11/output/final_zoom.jpg) |

#### 点线匹配

![运动场点线匹配](../examples/point_line_stitch/20201024084037322_cam11/output/point_line_matches.jpg)

#### Local 区域近 4K 拼接结果

两图为同一 Local 投影区域，尺寸均为 4080 × 2388。

| Global 局部背景 | 纯点线单应性拼接 |
| --- | --- |
| ![运动场 Global 局部背景](../examples/point_line_stitch/20201024084037322_cam11/output/global_roi_lanczos_x6.jpg) | ![运动场近 4K 拼接](../examples/point_line_stitch/20201024084037322_cam11/output/local_roi_x6.jpg) |

### 2. 室内展厅：`20210313151132256_cam22`

不同相机视角下的室内展厅。最终保留 631/632 个点匹配和 119/121 个线匹配；点 RMSE 为 1.292 px，线 RMSE 为 1.118 px。相对于相机标定得到的 `H_init`，`H_final` 在 local 全图网格上的平均修正约为 21 px。

| local 输入 | global 输入 |
| --- | --- |
| ![展厅 local](../examples/point_line_stitch/20210313151132256_cam22/input/local.jpg) | ![展厅 global](../examples/point_line_stitch/20210313151132256_cam22/input/global.jpg) |

#### 初值定位及局部放大

| 初值全画布定位 | 初值局部放大 |
| --- | --- |
| ![展厅 initial alignment 全画布](../examples/point_line_stitch/20210313151132256_cam22/output/initial_overview.jpg) | ![展厅 initial alignment 局部](../examples/point_line_stitch/20210313151132256_cam22/output/initial_zoom.jpg) |

#### 最终融合定位及局部放大

| 最终融合全画布定位 | 最终融合局部放大 |
| --- | --- |
| ![展厅 final alignment 全画布](../examples/point_line_stitch/20210313151132256_cam22/output/final_overview.jpg) | ![展厅 final alignment 局部](../examples/point_line_stitch/20210313151132256_cam22/output/final_zoom.jpg) |

#### 点线匹配

![展厅点线匹配](../examples/point_line_stitch/20210313151132256_cam22/output/point_line_matches.jpg)

#### Local 区域近 4K 拼接结果

两图为同一 Local 投影区域，尺寸均为 3858 × 2298。

| Global 局部背景 | 纯点线单应性拼接 |
| --- | --- |
| ![展厅 Global 局部背景](../examples/point_line_stitch/20210313151132256_cam22/output/global_roi_lanczos_x6.jpg) | ![展厅近 4K 拼接](../examples/point_line_stitch/20210313151132256_cam22/output/local_roi_x6.jpg) |

### 3. 校园道路：`20220817171240976_cam35`

不同相机视角下的室外校园道路。最终保留 388/434 个点匹配和 79/79 个线匹配；点 RMSE 为 1.663 px，线 RMSE 为 1.400 px。相对于相机标定得到的 `H_init`，`H_final` 在 local 全图网格上的平均修正约为 72 px。

| local 输入 | global 输入 |
| --- | --- |
| ![道路 local](../examples/point_line_stitch/20220817171240976_cam35/input/local.jpg) | ![道路 global](../examples/point_line_stitch/20220817171240976_cam35/input/global.jpg) |

#### 初值定位及局部放大

| 初值全画布定位 | 初值局部放大 |
| --- | --- |
| ![道路 initial alignment 全画布](../examples/point_line_stitch/20220817171240976_cam35/output/initial_overview.jpg) | ![道路 initial alignment 局部](../examples/point_line_stitch/20220817171240976_cam35/output/initial_zoom.jpg) |

#### 最终融合定位及局部放大

| 最终融合全画布定位 | 最终融合局部放大 |
| --- | --- |
| ![道路 final alignment 全画布](../examples/point_line_stitch/20220817171240976_cam35/output/final_overview.jpg) | ![道路 final alignment 局部](../examples/point_line_stitch/20220817171240976_cam35/output/final_zoom.jpg) |

#### 点线匹配

![道路点线匹配](../examples/point_line_stitch/20220817171240976_cam35/output/point_line_matches.jpg)

#### Local 区域近 4K 拼接结果

两图为同一 Local 投影区域，尺寸均为 4104 × 2388。

| Global 局部背景 | 纯点线单应性拼接 |
| --- | --- |
| ![道路 Global 局部背景](../examples/point_line_stitch/20220817171240976_cam35/output/global_roi_lanczos_x6.jpg) | ![道路近 4K 拼接](../examples/point_line_stitch/20220817171240976_cam35/output/local_roi_x6.jpg) |

### 4. 运动场另一视角：`20201025083303988_cam22`

Frame 6000。点线匹配精修后保留 98 个点、92 条线；局部输出为 3828 × 2286。

| Local 输入 | Global 输入 |
| --- | --- |
| ![运动场 cam22 Local](../examples/point_line_stitch/20201025083303988_cam22/input/local.jpg) | ![运动场 cam22 Global](../examples/point_line_stitch/20201025083303988_cam22/input/global.jpg) |

#### 初值定位及局部放大

| 初值全画布定位 | 初值局部放大 |
| --- | --- |
| ![运动场 cam22 初值定位 全画布](../examples/point_line_stitch/20201025083303988_cam22/output/initial_overview.jpg) | ![运动场 cam22 初值定位 局部](../examples/point_line_stitch/20201025083303988_cam22/output/initial_zoom.jpg) |

#### 最终融合定位及局部放大

| 最终融合全画布定位 | 最终融合局部放大 |
| --- | --- |
| ![运动场 cam22 最终融合定位 全画布](../examples/point_line_stitch/20201025083303988_cam22/output/final_overview.jpg) | ![运动场 cam22 最终融合定位 局部](../examples/point_line_stitch/20201025083303988_cam22/output/final_zoom.jpg) |

#### 点线匹配

![运动场 cam22 点线匹配](../examples/point_line_stitch/20201025083303988_cam22/output/point_line_matches.jpg)

#### Local 区域近 4K 拼接结果

两图取自同一 Local 投影区域，尺寸均为 3828 × 2286。

| Global 局部背景 | 纯点线单应性拼接 |
| --- | --- |
| ![运动场 cam22 Global 局部背景](../examples/point_line_stitch/20201025083303988_cam22/output/global_roi_lanczos_x6.jpg) | ![运动场 cam22 近 4K 拼接](../examples/point_line_stitch/20201025083303988_cam22/output/local_roi_x6.jpg) |

### 5. 商街人群：`20220806141041707_cam32`

Frame 6000。点线匹配精修后保留 517 个点、67 条线；局部输出为 3852 × 2304。

| Local 输入 | Global 输入 |
| --- | --- |
| ![商街 cam32 Local](../examples/point_line_stitch/20220806141041707_cam32/input/local.jpg) | ![商街 cam32 Global](../examples/point_line_stitch/20220806141041707_cam32/input/global.jpg) |

#### 初值定位及局部放大

| 初值全画布定位 | 初值局部放大 |
| --- | --- |
| ![商街 cam32 初值定位 全画布](../examples/point_line_stitch/20220806141041707_cam32/output/initial_overview.jpg) | ![商街 cam32 初值定位 局部](../examples/point_line_stitch/20220806141041707_cam32/output/initial_zoom.jpg) |

#### 最终融合定位及局部放大

| 最终融合全画布定位 | 最终融合局部放大 |
| --- | --- |
| ![商街 cam32 最终融合定位 全画布](../examples/point_line_stitch/20220806141041707_cam32/output/final_overview.jpg) | ![商街 cam32 最终融合定位 局部](../examples/point_line_stitch/20220806141041707_cam32/output/final_zoom.jpg) |

#### 点线匹配

![商街 cam32 点线匹配](../examples/point_line_stitch/20220806141041707_cam32/output/point_line_matches.jpg)

#### Local 区域近 4K 拼接结果

两图取自同一 Local 投影区域，尺寸均为 3852 × 2304。

| Global 局部背景 | 纯点线单应性拼接 |
| --- | --- |
| ![商街 cam32 Global 局部背景](../examples/point_line_stitch/20220806141041707_cam32/output/global_roi_lanczos_x6.jpg) | ![商街 cam32 近 4K 拼接](../examples/point_line_stitch/20220806141041707_cam32/output/local_roi_x6.jpg) |

## 安装

项目支持 Python 3.10–3.12。OmniGlue 模型约 400 MB，首次安装需要下载。

```bash
git clone --recursive https://github.com/liuyuxiang1021/image_stitch.git
cd image_stitch
conda env create -f environment.yml
conda activate image_stitch
bash scripts/setup_models.sh
```

已有环境也可以直接运行：

```bash
git submodule update --init --recursive
bash scripts/setup_models.sh
```

`setup_models.sh` 安装本项目及 OmniGlue，并使用 OmniGlue 官方说明中的地址下载 SuperPoint、DINOv2 和 OmniGlue 权重。LineTR 及其官方权重由 Git submodule 固定版本提供。默认自动使用 CUDA；没有可用 GPU 时使用 CPU，但模型推理会明显更慢。

## 运行真实样例

以运动场样例为例，在仓库根目录运行：

```bash
example=examples/point_line_stitch/20201024084037322_cam11

point-line-stitch \
  "$example/input/local.jpg" \
  "$example/input/global.jpg" \
  --initial-h "$example/input/H_init.npz" \
  --output "$example/output/panorama.jpg" \
  --artifacts-dir "$example/output"
```

将 `example` 改为下面任一目录即可复现另外四组：

```text
examples/point_line_stitch/20210313151132256_cam22
examples/point_line_stitch/20220817171240976_cam35
examples/point_line_stitch/20201025083303988_cam22
examples/point_line_stitch/20220806141041707_cam32
```

运行会覆盖对应目录内已提交的输出。不同 GPU、CUDA/cuDNN 或依赖版本可能造成末位数值差异。

## 直接估计模式

对于两幅视角和尺度差异不大、重叠区域足够的图像，也可以不提供初值，直接通过点线匹配估计单应性：

```bash
point-line-stitch source.jpg destination.jpg \
  --output panorama.jpg \
  --artifacts-dir output
```

五组仓库样例的 Local/Global 尺度和视角差异较大，所以使用原项目的 `H_init` 做重叠区域定位，再进行完整的点线匹配精修。

`H_init` 支持以下格式：3×3 `.npy`；含 `H_init`、`homography` 或 `H` 键的 `.npz`；JSON 形式的 3×3 数组或含上述字段的对象。

## 常用参数

```text
--confidence 0.1                OmniGlue 置信度阈值
--max-match-width 1200          匹配阶段的最大图像宽度，0 表示不缩放
--point-weight 0.5              点残差权重
--line-weight 0.5               线残差权重
--ransac-threshold 4.0          点 RANSAC 阈值（匹配尺度像素）
--line-distance-threshold 12.0  线中点到目标直线阈值
--line-angle-threshold 15.0     线方向差阈值（度）
--feather-width 6               Local 边缘羽化宽度（原 Global 画布像素），0 表示直接覆盖
--feather-power 1.0             羽化曲线指数
--no-color-correction           关闭默认开启的 HHM 色差矫正
--color-correction-coeff 1.0    HHM 映射系数，0 到 1
--color-correction-threshold 0.05 HHM 稀疏直方图阈值
--local-roi-output PATH         指定 Local 附近近 4K 拼接图的保存路径
--local-roi-scale 6             局部输出倍率；Global 使用 Lanczos 插值
--local-roi-margin 24           Local 投影外的 Global 留白（原画布像素）
--device auto                   auto / cpu / cuda
--gpu 0                         同时提供给 TensorFlow 与 PyTorch 的物理 GPU
```

OmniGlue 同时使用 TensorFlow 和 PyTorch。CLI 默认只暴露第 0 张 GPU，并开启 TensorFlow 显存按需增长，以避免两个框架抢占显存；可通过 `--gpu` 改用其他卡。

## Python API

```python
import cv2 as cv
from point_line_stitch import (
    LineTRLineMatcher,
    OmniGluePointMatcher,
    compose_local_roi,
    load_homography,
    stitch_pair,
)

local = cv.imread("local.jpg")
global_image = cv.imread("global.jpg")
H_init = load_homography("H_init.npz")

points = OmniGluePointMatcher(
    "models",
    confidence_threshold=0.1,
    omniglue_root="third_party/omniglue",
)
lines = LineTRLineMatcher("third_party/LineTR", device="auto")
result = stitch_pair(
    local,
    global_image,
    points,
    lines,
    initial_homography=H_init,
)

cv.imwrite("panorama.jpg", result.panorama)
roi = compose_local_roi(local, global_image, result.estimation.homography)
cv.imwrite("local_roi_x6.jpg", roi.image)
print(result.estimation.homography)
```

## 目录结构

```text
examples/point_line_stitch/
├── 20201024084037322_cam11/
├── 20201025083303988_cam22/
├── 20210313151132256_cam22/
├── 20220806141041707_cam32/
└── 20220817171240976_cam35/
    ├── input/
    │   ├── local.jpg
    │   ├── global.jpg
    │   └── H_init.npz
    └── output/
        ├── panorama.jpg
        ├── local_roi_x6.jpg
        ├── global_roi_lanczos_x6.jpg
        ├── initial_warp.jpg
        ├── initial_overview.jpg / initial_zoom.jpg
        ├── final_overview.jpg / final_zoom.jpg
        ├── point_line_matches.jpg
        └── metadata.json
```

五组样例的主要输入与输出文件相同；两组新样例还在根目录保存 `sample.json`，记录原视频帧号和输入来源。

## 测试

不下载模型也能运行几何单元测试：

```bash
python -m unittest discover -s tests -v
```

单元测试使用确定性的合成点线约束，覆盖离群点过滤、点线联合优化、无有效线匹配时的回退、初值精修、画布计算和融合；仓库中的五组展示样例由真实图像和 OmniGlue + LineTR 流程产生。

## 第三方项目

- OmniGlue：Apache-2.0，代码和模型安装方式以其官方仓库为准；
- LineTR：随 Git submodule 指向官方仓库；其许可证限定为学术或非营利机构的非商业研究用途，使用前请阅读并确认接受 `third_party/LineTR/LICENSE`。

如在论文或研究中使用相关匹配模型，请引用 OmniGlue 与 LineTR 的原论文。
