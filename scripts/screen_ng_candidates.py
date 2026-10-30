"""Rank images of one capture session by leave-one-out PatchCore patch distance.

Protocol D-090: the ranking only orders images for human review. It assigns no
labels and no threshold. Each image is compared with the other images of the
same session, excluding itself and every image suggested as overlapping with it
(`overlap_pairs.csv` from suggest_region_overlap.py). All outputs stay local.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iad_w1.manifest_data import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from iad_w1.normal_calibration import smooth_map_3x3  # noqa: E402

ROI = (0, 0, 3000, 1957)
INPUT_WH = (768, 504)
EXCLUDE_VERDICTS = {"nghi_chong_lap", "can_xem"}


def load_roi(path: Path) -> Image.Image:
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        x, y, w, h = ROI
        return rgb.crop((x, y, min(rgb.width, x + w), min(rgb.height, y + h)))


def read_exclusions(pairs_csv: Path, names: list[str]) -> dict[str, set[str]]:
    excluded = {name: {name} for name in names}
    if pairs_csv.exists():
        with pairs_csv.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["suggestion"] in EXCLUDE_VERDICTS and row["file_a"] in excluded and row["file_b"] in excluded:
                    excluded[row["file_a"]].add(row["file_b"])
                    excluded[row["file_b"]].add(row["file_a"])
    return excluded


def jpeg_b64(image: Image.Image, width: int = 520) -> str:
    height = round(image.height * width / image.width)
    buffer = io.BytesIO()
    image.resize((width, height), Image.Resampling.BILINEAR).save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def overlay(roi: Image.Image, score_map: np.ndarray, scale_max: float) -> Image.Image:
    import cv2

    base = np.asarray(roi.resize((520, round(roi.height * 520 / roi.width)), Image.Resampling.BILINEAR))
    heat = cv2.resize(score_map, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
    colour = cv2.applyColorMap((np.clip(heat / scale_max, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    mixed = cv2.addWeighted(base, 0.55, cv2.cvtColor(colour, cv2.COLOR_BGR2RGB), 0.45, 0)
    return Image.fromarray(mixed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, help="Mã phiên, ví dụ 20261029-S08")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--overlap-pairs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {args.out_dir}")

    import torch
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = sorted(args.image_dir.glob("*.jpg"))
    names = [path.name for path in files]
    excluded = read_exclusions(args.overlap_pairs, names)

    model = PatchcoreModel(backbone="wide_resnet50_2", layers=["layer2", "layer3"], pre_trained=True).eval().to(device)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    embeddings, rois = [], []
    with torch.inference_mode():
        for path in files:
            roi = load_roi(path)
            rois.append(roi)
            array = np.asarray(roi.resize(INPUT_WH, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
            tensor = (torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device) - mean) / std
            features = model.feature_extractor(tensor)
            features = {layer: model.feature_pooler(value) for layer, value in features.items()}
            embedding = model.generate_embedding(features)[0]  # (C, H, W)
            grid = embedding.shape[1:]
            embeddings.append(embedding.reshape(embedding.shape[0], -1).T.contiguous().half())
        norms = [item.float().pow(2).sum(1) for item in embeddings]

        rows, maps = [], []
        for i, name in enumerate(names):
            query = embeddings[i].float()
            best = torch.full((query.shape[0],), float("inf"), device=device)
            references = [j for j, other in enumerate(names) if other not in excluded[name]]
            for j in references:
                bank = embeddings[j].float()
                distance = norms[i][:, None] + norms[j][None, :] - 2.0 * query @ bank.T
                best = torch.minimum(best, distance.min(dim=1).values)
            patch_map = best.clamp_min(0).sqrt().reshape(grid).cpu().numpy().astype(np.float32)
            smoothed = smooth_map_3x3(patch_map)
            flat = np.sort(smoothed.ravel())[::-1]
            top = flat[: max(1, int(round(0.01 * flat.size)))]
            maps.append(smoothed)
            rows.append({
                "file_name": name,
                "score_max_smoothed": float(flat[0]),
                "score_top1pct_mean": float(top.mean()),
                "reference_image_count": len(references),
                "excluded_overlap": ";".join(sorted(excluded[name] - {name})),
            })

    order = sorted(range(len(rows)), key=lambda k: rows[k]["score_max_smoothed"], reverse=True)
    for rank, k in enumerate(order, start=1):
        rows[k]["rank"] = rank
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["rank", "file_name", "score_max_smoothed", "score_top1pct_mean", "reference_image_count", "excluded_overlap"]
    with (args.out_dir / "ranking.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows[k] for k in order)

    scale_max = float(np.quantile([m.max() for m in maps], 0.9))
    cards = []
    for k in order:
        row = rows[k]
        cards.append(
            "<div class='card'><h3>#{rank} — {name}</h3><p>điểm max {s:.3f} · top 1% {t:.3f} · so với {n} ảnh{ex}</p>"
            "<img src='data:image/jpeg;base64,{a}'><img src='data:image/jpeg;base64,{b}'></div>".format(
                rank=row["rank"], name=html.escape(row["file_name"]), s=row["score_max_smoothed"],
                t=row["score_top1pct_mean"], n=row["reference_image_count"],
                ex=(" · loại ảnh chồng lấp: " + html.escape(row["excluded_overlap"])) if row["excluded_overlap"] else "",
                a=jpeg_b64(rois[k]), b=jpeg_b64(overlay(rois[k], maps[k], scale_max)),
            )
        )
    page = (
        "<!doctype html><meta charset='utf-8'><title>Gợi ý ảnh nghi NG — {s}</title>"
        "<style>body{{font-family:sans-serif;margin:16px}}.card{{border:1px solid #ccc;padding:8px;margin:8px 0}}"
        "img{{margin-right:8px;max-width:48%}}</style>"
        "<h1>Gợi ý ảnh nghi NG — {s}</h1><p>Xếp theo điểm giảm dần. Đây chỉ là thứ tự để xem, không phải nhãn. "
        "Trái: ROI gốc. Phải: heatmap (thang màu cố định cho cả phiên).</p>{c}"
    ).format(s=html.escape(args.session), c="".join(cards))
    (args.out_dir / "review.html").write_text(page, encoding="utf-8")
    metadata = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "protocol": "D-090",
        "session": args.session,
        "image_count": len(files),
        "roi": ROI,
        "input_wh": INPUT_WH,
        "patch_grid": list(grid),
        "device": str(device),
        "torch": torch.__version__,
        "colour_scale_max": scale_max,
        "excluded_pairs_source": str(args.overlap_pairs),
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
