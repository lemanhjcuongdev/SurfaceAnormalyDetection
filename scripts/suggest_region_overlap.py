"""Suggest which captured OK images may show the same/overlapping surface region.

Runs fully locally. It never copies raw images and never writes the registry.
Two signals are computed for every pair of images inside one capture session:

* ``visual_similarity``: the same low-level appearance descriptor used by
  ``infer_capture_visual_groups.py`` (looks alike, not necessarily overlap).
* ``orb_inliers``: ORB keypoint matches that survive a RANSAC homography on the
  ROI. Many geometric inliers means the two photos share physical texture,
  i.e. they very likely overlap the same spot.

Output is a review aid. The researcher confirms and fills
``surface_region_id`` in the registry by hand.
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import base64
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_capture_visual_groups import DEFAULT_ROI, cosine_similarity, image_descriptor  # noqa: E402


def roi_gray(path: Path, roi, long_side: int = 1000) -> np.ndarray:
    x, y, w, h = roi
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("L")
        crop = rgb.crop((x, y, min(rgb.width, x + w), min(rgb.height, y + h)))
        scale = long_side / max(crop.size)
        crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.BILINEAR)
    return np.asarray(crop, dtype=np.uint8)


def thumb_b64(path: Path, roi, width: int = 260) -> str:
    x, y, w, h = roi
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        crop = rgb.crop((x, y, min(rgb.width, x + w), min(rgb.height, y + h)))
        crop.thumbnail((width, width))
        buffer = io.BytesIO()
        crop.save(buffer, format="JPEG", quality=80)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def orb_inliers(orb, matcher, left, right) -> int:
    kp1, d1 = left
    kp2, d2 = right
    if d1 is None or d2 is None or len(kp1) < 8 or len(kp2) < 8:
        return 0
    pairs = matcher.knnMatch(d1, d2, k=2)
    good = [m for m, *rest in pairs if rest and m.distance < 0.75 * rest[0].distance]
    if len(good) < 8:
        return len(good) // 2
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
    return int(mask.sum()) if mask is not None else 0


def shot_time(name: str) -> datetime | None:
    try:
        return datetime.strptime(Path(name).stem[:15], "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", action="append", required=True,
                        help="session_name=folder, ví dụ 20261016-S06=D:/.../RESOURCES/16102026")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--overlap-inliers", type=int, default=40,
                        help="Ngưỡng inlier để gợi ý 'chồng lấp' (khai báo trước)")
    parser.add_argument("--maybe-inliers", type=int, default=15)
    args = parser.parse_args()

    import openpyxl
    wb = openpyxl.load_workbook(args.registry, read_only=True, data_only=True)
    ws = wb["Anh_OK"]
    rows = list(ws.iter_rows(values_only=True))
    header = list(rows[0])
    ix = {k: i for i, k in enumerate(header)}
    sample_of = {(r[ix["capture_session"]], r[ix["file_name"]]): r[ix["sample_id"]] for r in rows[1:] if r[0]}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    orb = cv2.ORB_create(nfeatures=3000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pair_rows, image_rows, html_parts = [], [], []

    for spec in args.session:
        session, folder = spec.split("=", 1)
        files = sorted(Path(folder).glob("*.jpg"))
        feats, kps, thumbs = [], [], []
        for f in files:
            feats.append(image_descriptor(f, DEFAULT_ROI)["feature"])
            kp, des = orb.detectAndCompute(roi_gray(f, DEFAULT_ROI), None)
            kps.append((kp, des))
            thumbs.append(thumb_b64(f, DEFAULT_ROI))
        n = len(files)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(n):
            for j in range(i + 1, n):
                inl = orb_inliers(orb, matcher, kps[i], kps[j])
                sim = cosine_similarity(feats[i], feats[j])
                if inl >= args.overlap_inliers:
                    verdict = "nghi_chong_lap"
                    parent[find(j)] = find(i)
                elif inl >= args.maybe_inliers:
                    verdict = "can_xem"
                else:
                    continue
                t1, t2 = shot_time(files[i].name), shot_time(files[j].name)
                gap = int((t2 - t1).total_seconds()) if t1 and t2 else ""
                pair_rows.append({
                    "capture_session": session, "file_a": files[i].name, "file_b": files[j].name,
                    "sample_a": sample_of.get((session, files[i].name), ""),
                    "sample_b": sample_of.get((session, files[j].name), ""),
                    "consecutive": j == i + 1, "seconds_apart": gap,
                    "orb_inliers": inl, "visual_similarity": round(sim, 4), "suggestion": verdict,
                })

        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        ordered = sorted(groups.values(), key=lambda g: g[0])
        html_parts.append(f"<h2>{html.escape(session)} — {n} ảnh, {len(ordered)} nhóm vùng gợi ý</h2>")
        for k, members in enumerate(ordered, 1):
            gid = f"{session.split('-')[-1]}-G{k:03d}"
            samples = sorted({str(sample_of.get((session, files[m].name), "")) for m in members})
            for m in members:
                image_rows.append({
                    "capture_session": session, "file_name": files[m].name,
                    "sample_id": sample_of.get((session, files[m].name), ""),
                    "suggested_region_group": gid, "group_size": len(members),
                })
            warn = " ⚠ khác sample_id" if len(samples) > 1 else ""
            cards = "".join(
                f"<figure><img src='data:image/jpeg;base64,{thumbs[m]}'><figcaption>{html.escape(files[m].name)}<br>{html.escape(str(sample_of.get((session, files[m].name), '')))}</figcaption></figure>"
                for m in members)
            html_parts.append(f"<section><h3>{gid} · {len(members)} ảnh · {', '.join(samples)}{warn}</h3><div class=row>{cards}</div></section>")

    with (args.out_dir / "overlap_pairs.csv").open("w", newline="", encoding="utf-8-sig") as s:
        w = csv.DictWriter(s, fieldnames=list(pair_rows[0].keys()) if pair_rows else ["capture_session"])
        w.writeheader(); w.writerows(pair_rows)
    with (args.out_dir / "suggested_region_groups.csv").open("w", newline="", encoding="utf-8-sig") as s:
        w = csv.DictWriter(s, fieldnames=list(image_rows[0].keys()))
        w.writeheader(); w.writerows(image_rows)
    page = ("<!doctype html><meta charset=utf-8><title>Rà soát vùng S06/S07</title><style>"
            "body{font-family:sans-serif;margin:16px;background:#fafafa}section{background:#fff;border:1px solid #ddd;"
            "border-radius:8px;padding:8px 12px;margin:10px 0}.row{display:flex;flex-wrap:wrap;gap:8px}"
            "figure{margin:0;font-size:12px;text-align:center}img{height:170px;border-radius:4px}</style>"
            "<h1>Gợi ý nhóm vùng chồng lấp (chỉ để rà soát, không phải nhãn)</h1>"
            "<p>Ảnh trong cùng một khung bị máy nghi là chụp chồng lên cùng một chỗ. Ảnh đứng một mình = không thấy chồng lấp với ảnh nào.</p>"
            + "".join(html_parts))
    (args.out_dir / "review.html").write_text(page, encoding="utf-8")
    meta = {"created": datetime.now().isoformat(timespec="seconds"), "roi": DEFAULT_ROI,
            "overlap_inliers": args.overlap_inliers, "maybe_inliers": args.maybe_inliers,
            "sessions": args.session, "pairs_reported": len(pair_rows), "opencv": cv2.__version__}
    (args.out_dir / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
