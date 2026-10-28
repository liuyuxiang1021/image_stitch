# 点线联合单应性算法

## 1. 输入和坐标约定

输入为源图（local/source）与目标图（global/destination）。最终矩阵满足：

```text
p_destination ~ H · p_source
```

实现提供两条入口：

- 直接模式：在两张原图上做点线匹配，直接估计 `H`；
- 初值修正模式：先用 `H_init` 把源图投影到目标坐标，在有效重叠区域估计 `H_delta`，最终得到 `H_final = H_delta @ H_init`。

## 2. 特征匹配

点特征使用 OmniGlue。它融合 SuperPoint 局部特征和 DINOv2 基础模型特征，对不同视角和不同图像域更稳健。低于 `--confidence` 的点匹配会被删除。

线特征使用 LineTR：LSD 检测线段，SuperPoint 为线上的采样位置提供局部描述，Line Transformer 生成上下文相关的线描述子并完成互相最近邻匹配。

## 3. 稳健初值和匹配过滤

当有效点不少于四个时，先用 RANSAC 从点匹配得到联合优化的初值，并删除点离群值。如果点不足而有至少四条线，则通过源线段端点到目标直线的代数约束求线性初值。

将源线段用初值投影后，再根据以下条件过滤线匹配：

- 投影线段中点到目标直线的距离；
- 投影线段与目标线段的无向夹角；
- 两侧线段的最小长度。

## 4. 点线联合优化

固定 `h22 = 1`，优化单应矩阵的其余八个自由度。点残差为二维重投影误差。每条源线段均匀采样若干点，线残差为这些投影点到匹配目标直线的有符号距离。

所有残差由匹配置信度、`point_weight` 或 `line_weight` 加权，再用 Huber 权重抑制剩余离群值。优化器使用带轻微阻尼的 Gauss–Newton 正规方程迭代更新。

## 5. 画布和融合

将源图四个角经最终单应矩阵投影，并与目标图角点共同计算画布边界。平移两张图以保留负坐标区域。图像与全白有效掩膜分别做透视变换，因此原图中的合法黑色像素不会被误判成空区域。

融合前，在 Local 与 Global 有效重叠区统计配对像素，按 B、G、R 通道估计 HHM 直方图映射，并将映射应用于投影后的 Local。这样只调整 Local 的颜色分布，不改变几何位置；映射系数可设为 0 关闭校正，CLI 也可用 `--no-color-correction` 完全关闭。

融合时先将 Local 和 Global 的像素归属掩膜按高斯核模糊，再把两侧权重归一化后加权求和。羽化只发生在两种来源的边界附近：

```text
local_weight = blur(local_owner_mask) ^ feather_power
global_weight = blur(global_owner_mask) ^ feather_power
weight_local = local_weight / (local_weight + global_weight)
weight_global = 1 - weight_local
result = weight_local * local + weight_global * global
```

默认羽化半径为 6 个输出像素，边界两侧各在约 6 px 范围内由一种来源平滑过渡到另一种。羽化带中的无效投影像素先用最近的有效源像素补齐，避免黑边。`--feather-width 0` 表示关闭羽化并直接覆盖。

## 6. 方法边界

该实现只估计一个全局单应矩阵，不使用深度或光流。它适合近似平面场景，或相机主要围绕同一光心旋转的场景。明显非平面视差、动态物体、大面积重复纹理以及极小重叠区都可能导致错配或重影。
