"""Build a browser gallery for the five saved point-line/SIFT examples."""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
from pathlib import Path

import cv2 as cv
import numpy as np

from benchmark_common import image_metrics, sift_homography
from point_line_stitch import compose_local_roi, load_homography
from point_line_stitch.color_correction import apply_hhm_luts, fit_hhm_for_regions


EXAMPLES = (
    "20201024084037322_cam11",
    "20201025083303988_cam22",
    "20210313151132256_cam22",
    "20220806141041707_cam32",
    "20220817171240976_cam35",
)
SCENES = (
    "运动场",
    "运动场",
    "展厅",
    "商街",
    "校园道路",
)
IMAGE_LABELS = (
    ("global_native_roi", "Global 原始 ROI"),
    ("local_input", "Local 原图"),
    ("global_x6", "Global 100% · x6"),
    ("point_line_final", "点线拼接 · 最终结果"),
    ("sift_final", "SIFT + RANSAC · 最终结果"),
    ("point_line_half", "点线 · Local/Global 50:50"),
    ("sift_half", "SIFT · Local/Global 50:50"),
)


def read_image(path: Path) -> np.ndarray:
    image = cv.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image


def save_jpeg(path: Path, image: np.ndarray, quality: int = 94) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(path), image, [cv.IMWRITE_JPEG_QUALITY, quality]):
        raise RuntimeError(f"Could not write {path}")


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def half_blend(
    local: np.ndarray,
    background: np.ndarray,
    homography: np.ndarray,
    roi: tuple[int, int, int, int],
    scale: int,
) -> np.ndarray:
    """Use the final renderer's warp and HHM, with 50:50 overlap for inspection."""
    x0, y0, _, _ = roi
    source_to_roi = (
        np.diag([float(scale), float(scale), 1.0])
        @ np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]])
        @ np.asarray(homography, dtype=np.float64).reshape(3, 3)
    )
    size = (background.shape[1], background.shape[0])
    warped = cv.warpPerspective(local, source_to_roi, size)
    valid = cv.warpPerspective(
        np.full(local.shape[:2], 255, dtype=np.uint8),
        source_to_roi, size, flags=cv.INTER_NEAREST,
    )
    luts, _ = fit_hhm_for_regions(background, [warped], [valid])
    corrected = apply_hhm_luts(warped, luts, valid)
    blended = cv.addWeighted(corrected, 0.5, background, 0.5, 0)
    blended[valid == 0] = background[valid == 0]
    return blended


def thumbnail(source: Path, destination: Path, max_width: int = 640) -> None:
    image = read_image(source)
    if image.shape[1] > max_width:
        height = int(round(image.shape[0] * max_width / image.shape[1]))
        image = cv.resize(image, (max_width, height), interpolation=cv.INTER_AREA)
    save_jpeg(destination, image, 86)


def build_example(root: Path, output: Path, name: str, scene: str) -> dict:
    source = root / name
    metadata = json.loads((source / "output/metadata.json").read_text(encoding="utf-8"))
    roi = tuple(int(value) for value in metadata["local_roi_output"]["global_roi_xyxy"])
    scale = int(metadata["local_roi_output"]["scale"])
    if scale != 6:
        raise ValueError(f"Expected a saved x6 ROI: {name}")

    images = {
        "local_input": source / "input/local.jpg",
        "global_x6": source / "output/global_roi_lanczos_x6.jpg",
        "point_line_final": source / "output/local_roi_x6.jpg",
    }
    folder = output / name
    folder.mkdir(parents=True, exist_ok=True)
    for key, path in images.items():
        images[key] = folder / f"{key}.jpg"
        link_or_copy(path, images[key])

    local = read_image(images["local_input"])
    global_frame = read_image(source / "input/global.jpg")
    x0, y0, x1, y1 = roi
    images["global_native_roi"] = folder / "global_native_roi.jpg"
    save_jpeg(images["global_native_roi"], global_frame[y0:y1, x0:x1])
    background = read_image(images["global_x6"])
    expected_shape = ((y1 - y0) * scale, (x1 - x0) * scale)
    if background.shape[:2] != expected_shape:
        raise ValueError(f"Global x6 dimensions mismatch: {name}")
    pure = read_image(images["point_line_final"])
    if pure.shape[:2] != expected_shape:
        raise ValueError(f"Point-line result dimensions mismatch: {name}")
    ix, iy, iw, ih = (int(value) for value in metadata["initial_projection_roi_xywh"])
    h_sift, match_count, inlier_count = sift_homography(
        local, global_frame, load_homography(source / "input/H_init.npz"),
        (ix, iy, ix + iw, iy + ih),
    )
    sift = compose_local_roi(
        local, global_frame, h_sift,
        scale=scale, roi_xyxy=roi, background_roi=background, feather_width=6.0,
    ).image
    images["sift_final"] = folder / "sift_final.jpg"
    images["sift_final"].unlink(missing_ok=True)
    save_jpeg(images["sift_final"], sift)
    methods = {
        "OmniGlue_LineTR_H": image_metrics(background, pure),
        "SIFT_RANSAC_H": {
            **image_metrics(background, read_image(images["sift_final"])),
            "ratio_test_matches": match_count,
            "ransac_inliers": inlier_count,
        },
    }

    homographies = {
        "point_line_half": metadata["homography_source_to_destination"],
        "sift_half": h_sift,
    }
    for key, homography in homographies.items():
        images[key] = folder / f"{key}.jpg"
        save_jpeg(images[key], half_blend(local, background, homography, roi, scale))

    entries = []
    for key, label in IMAGE_LABELS:
        path = images[key]
        thumb_path = folder / f"{key}_thumb.jpg"
        thumbnail(path, thumb_path)
        entries.append({
            "key": key, "label": label,
            "full": str(path.relative_to(output)),
            "thumbnail": str(thumb_path.relative_to(output)),
        })
    return {
        "sample": name, "scene": scene,
        "global_frame": json.loads((source / "sample.json").read_text(encoding="utf-8")).get("global_frame_index_zero_based"),
        "roi_native_xyxy": list(roi), "roi_x6_size": [expected_shape[1], expected_shape[0]],
        "methods": methods, "sift_homography_local_to_global": h_sift.tolist(),
        "images": entries,
    }


def render_html(samples: list[dict]) -> str:
    cards = []
    for sample in samples:
        rows = []
        for item in sample["images"]:
            rows.append(
                '<figure><a href="{}" target="_blank"><img loading="lazy" src="{}" alt="{}"></a>'
                '<figcaption>{}</figcaption></figure>'.format(
                    *(html.escape(item[key], quote=True) for key in ("full", "thumbnail", "label")),
                    html.escape(item["label"]),
                )
            )
        point = sample["methods"]["OmniGlue_LineTR_H"]
        sift = sample["methods"]["SIFT_RANSAC_H"]
        cards.append(
            '<section id="{id}"><h2>{scene} · {id}</h2><p>Global 帧 {frame} · '
            '同一 ROI {width} × {height}（x6） · 点线 PSNR/SSIM {pps:.3f}/{pss:.4f} · '
            'SIFT PSNR/SSIM {sps:.3f}/{sss:.4f}</p><div class="grid">{images}</div></section>'.format(
                id=html.escape(sample["sample"]), scene=html.escape(sample["scene"]),
                frame=sample["global_frame"], width=sample["roi_x6_size"][0],
                height=sample["roi_x6_size"][1],
                pps=point["psnr_db"], pss=point["ssim"],
                sps=sift["psnr_db"], sss=sift["ssim"], images="".join(rows),
            )
        )
    links = " ".join(
        f'<a href="#{html.escape(item["sample"])}">{html.escape(item["scene"])} · cam{item["sample"].split("cam")[-1]}</a>'
        for item in samples
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>纯拼接五组对比</title><style>
body{{font:16px/1.55 system-ui,sans-serif;background:#101820;color:#edf3f8;margin:0 auto;padding:24px;max-width:1550px}}
a{{color:#83c9ff}} nav{{display:flex;flex-wrap:wrap;gap:12px;margin:20px 0 32px}}
section{{border-top:1px solid #3d4b56;padding:18px 0 34px}} section p{{color:#b6c5d0;margin-top:0}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}
figure{{background:#1b2731;margin:0;padding:10px;border-radius:7px;min-width:0}}
figure img{{display:block;width:100%;height:auto;object-fit:contain;max-height:450px}}
figcaption{{padding:8px 2px 2px;font-weight:600}}
@media(max-width:720px){{.grid{{grid-template-columns:1fr}}body{{padding:12px}}}}
</style></head><body><h1>纯拼接：点线与 SIFT 对比</h1>
<p>每组使用同一 Local、Global 和 x6 Global ROI。两张最终结果均经过各自的 HHM 校色和 6 px 原画布边缘羽化；50:50 图显示同一投影内校色后的 Local 与 Global 各占一半，投影外保持 Global。点击图片查看原尺寸。</p>
<nav>{links}</nav>{''.join(cards)}</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, default=Path("examples/point_line_stitch"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    samples = [
        build_example(args.examples.resolve(), output, name, scene)
        for name, scene in zip(EXAMPLES, SCENES)
    ]
    (output / "manifest.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    (output / "index.html").write_text(render_html(samples), encoding="utf-8")
    print(output / "index.html")


if __name__ == "__main__":
    main()
