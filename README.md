# image_stitch

从 `image_stitching` 项目中独立整理出的完整点线单应性图像拼接流程：

```text
OmniGlue 点匹配 ─┐
                 ├─ 稳健初值 ─ 匹配过滤 ─ 点线联合优化 ─ 单应投影 ─ 羽化融合
LineTR 线匹配 ───┘
```

本仓库不包含深度估计、RAFT 光流、运动区域三分区或视频同步，但保留了单应性估计本身的完整点线匹配能力。

![拼接输出](examples/output/panorama.jpg)

## 功能

- [OmniGlue](https://github.com/google-research/omniglue) 进行跨视角点匹配；
- [LineTR](https://github.com/yosungho/LineTR) 进行 LSD 线段检测、描述与匹配；
- RANSAC 点匹配初值与线约束后备初值；
- 基于重投影距离和角度的线匹配过滤；
- 点重投影误差 + 采样点到线误差的 8 自由度联合优化；
- Huber 鲁棒权重、点线匹配置信度权重；
- 直接估计 `H`，或从已有 `H_init` 估计 `H_delta` 并生成 `H_final`；
- 自动画布、有效掩膜和距离羽化融合；
- 保存矩阵、匹配数量、RMSE 以及点线匹配可视化。

算法和坐标变换的详细说明见 [docs/algorithm.md](docs/algorithm.md)。

## 安装

项目使用 Python 3.10–3.12。OmniGlue 模型约 400 MB，首次安装需要下载。

```bash
git clone --recursive https://github.com/liuyuxiang1021/image_stitch.git
cd image_stitch
conda env create -f environment.yml
conda activate image_stitch
bash scripts/setup_models.sh
```

已有环境也可以：

```bash
python -m pip install -e '.[full]'
git submodule update --init --recursive
python -m pip install --no-deps -e third_party/omniglue
bash scripts/setup_models.sh
```

`setup_models.sh` 使用 OmniGlue 官方说明中的地址下载 SuperPoint、DINOv2 和 OmniGlue 权重。LineTR 及其官方权重由 Git submodule 固定版本提供。默认自动使用 CUDA；没有可用 GPU 时使用 CPU，但模型推理会明显更慢。

## 快速开始：直接点线估计

```bash
image-stitch \
  examples/input/left.png \
  examples/input/right.png \
  --output examples/output/panorama.jpg \
  --artifacts-dir examples/output
```

第一个参数是源图，第二个参数是目标/参考图，输出矩阵满足 `p_destination ~ H @ p_source`。

## 使用已有初值并精修

`H_init` 可以是 3×3 `.npy` 文件，也可以是 JSON 数组；JSON 对象支持 `homography`、`H_init` 或 `H` 字段。

```bash
image-stitch local.jpg global.jpg \
  --initial-h H_init.json \
  --output panorama.jpg \
  --artifacts-dir output
```

该模式先把 local 图按 `H_init` 投影到 global 坐标，在有效重叠框中进行 OmniGlue + LineTR 匹配，估计增量矩阵，并输出：

```text
H_final = H_delta @ H_init
```

## 输出文件

指定 `--artifacts-dir` 后会生成：

- `metadata.json`：最终 H、画布变换、点线数量、优化迭代次数和误差；
- `point_line_matches.jpg`：通过几何过滤的点与线匹配；
- `initial_warp.jpg`：仅在使用 `--initial-h` 时生成。

示例结果：

| 源图 | 目标图 |
| --- | --- |
| ![source](examples/input/left.png) | ![destination](examples/input/right.png) |

![点线匹配](examples/output/point_line_matches.jpg)

## 常用参数

```text
--confidence 0.1                OmniGlue 置信度阈值
--max-match-width 1200          匹配阶段的最大图像宽度，0 表示不缩放
--point-weight 0.5              点残差权重
--line-weight 0.5               线残差权重
--ransac-threshold 4.0          点 RANSAC 阈值（匹配尺度像素）
--line-distance-threshold 12.0  线中点到目标直线阈值
--line-angle-threshold 15.0     线方向差阈值（度）
--device auto                   auto / cpu / cuda
--gpu 0                         同时提供给 TensorFlow 与 PyTorch 的物理 GPU
```

OmniGlue 同时使用 TensorFlow 和 PyTorch。CLI 默认只暴露第 0 张 GPU，并开启 TensorFlow 显存按需增长，以避免两个框架抢占显存；可通过 `--gpu` 改用其他卡。

## Python API

```python
import cv2 as cv
from image_stitch import LineTRLineMatcher, OmniGluePointMatcher, stitch_pair

source = cv.imread("local.jpg")
destination = cv.imread("global.jpg")

points = OmniGluePointMatcher("models", confidence_threshold=0.1)
lines = LineTRLineMatcher("third_party/LineTR", device="auto")
result = stitch_pair(source, destination, points, lines)

cv.imwrite("panorama.jpg", result.panorama)
print(result.estimation.homography)
```

## 测试

不下载模型也能运行几何单元测试：

```bash
python -m unittest discover -s tests -v
```

测试使用确定性的合成点线约束，覆盖离群点过滤、点线联合优化、画布计算和融合。仓库中的输入图由 `python scripts/generate_example.py` 可重复生成；已提交的输出样例则由完整 OmniGlue + LineTR 流程产生。

## 第三方项目

- OmniGlue：Apache-2.0，代码和模型安装方式以其官方仓库为准；
- LineTR：随 Git submodule 指向官方仓库；其许可证限定为学术或非营利机构的非商业研究用途，使用前请阅读并确认接受 `third_party/LineTR/LICENSE`。

如在论文或研究中使用相关匹配模型，请引用 OmniGlue 与 LineTR 的原论文。
