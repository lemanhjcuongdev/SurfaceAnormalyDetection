"""Draw the two schematic figures of the paper: Figure 1 (inspection workflow) and Figure 2 (capture setup).

Both are diagrams, not photographs. Figure 2 uses the setup stated by the researcher on
08/11/2026 (D-133): camera on a fixed stand about 20 cm above the surface (estimate),
top-down with at most a few degrees of tilt, one ceiling lamp of about 5000 K, 2 s timer.
The field-of-view sizes are derived from the 69 mm equivalent focal length at 20 cm.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

FILL, EDGE, TEXT, GREY, TILT = "#edf3fa", "#2f557f", "#1f3550", "#8a8f96", "#b5651d"
plt.rcParams.update({"font.family": "Arial", "font.size": 9})

FULL_FRAME_DIAGONAL_MM = math.hypot(36, 24)


def field_of_view_cm(distance_cm: float, equivalent_focal_mm: float = 69.0) -> tuple[float, float]:
    """Short and long side of a 4:3 frame at the given distance, from the 35 mm equivalent focal length."""
    diagonal = distance_cm * FULL_FRAME_DIAGONAL_MM / equivalent_focal_mm
    return diagonal * 0.6, diagonal * 0.8


def box(ax, x, y, w, h, text, style="-", size=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=FILL, edgecolor=EDGE, linewidth=1.3, linestyle=style))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=TEXT, fontsize=size, linespacing=1.25)


def arrow(ax, start, end, style="-"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color=EDGE,
                                 linewidth=1.4, linestyle=style, shrinkA=0, shrinkB=0))


def figure1(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.2)
    ax.axis("off")
    w, gap = 2.72, 0.32
    xs = [0.05 + i * (w + gap) for i in range(5)]

    ax.text(xs[2], 7.85, "Chuẩn bị ngoại tuyến (trước khi vận hành)", fontsize=8.5, fontweight="bold", color=TEXT)
    box(ax, xs[2], 6.35, w, 1.15, "Ảnh OK xây dựng\nmô hình", style="--")
    box(ax, xs[3], 6.35, w, 1.15, "Ảnh OK hiệu chuẩn\n(vật mẫu riêng):\nngưỡng phân vị 95%", style="--", size=7.5)

    ax.text(xs[0], 5.35, "Vận hành (đề xuất)", fontsize=8.5, fontweight="bold", color=TEXT)
    y, h = 3.75, 1.35
    texts = ["Sản phẩm sau\ncông đoạn sơn", "Chụp vùng bề mặt\n(camera cố định),\ncắt ROI",
             "Mô hình: điểm\nbất thường và\nbản đồ nhiệt", "So với ngưỡng\nđã khóa: gợi ý\nOK hoặc NG",
             "Người kiểm tra\nxem bản đồ nhiệt\nvà xác nhận"]
    for x, text in zip(xs, texts):
        box(ax, x, y, w, h, text)
    for a, b in zip(xs, xs[1:]):
        arrow(ax, (a + w, y + h / 2), (b, y + h / 2))
    for x in (xs[2], xs[3]):
        arrow(ax, (x + w / 2, 6.35), (x + w / 2, y + h), style="--")

    box(ax, xs[4], 1.55, w, 1.25, "OK: công đoạn tiếp\nNG: xử lý theo quy\ntrình chất lượng", size=7.5)
    box(ax, xs[3], 1.55, w, 1.25, "Lưu vết: ảnh, điểm,\ngợi ý và quyết định")
    arrow(ax, (xs[4] + w / 2, y), (xs[4] + w / 2, 2.8))
    arrow(ax, (xs[4] + 0.25, y), (xs[3] + w - 0.25, 2.8))
    ax.text(xs[0], 0.55, "Ảnh NG và mặt nạ lỗi chỉ dùng để đánh giá, không dùng để xây dựng mô hình hay đặt ngưỡng.\n"
            "Sơ đồ là quy trình đề xuất; quy trình chất lượng tại xưởng chưa được khảo sát.", fontsize=7.5, color=GREY)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure2(path: Path, distance_cm: float = 20.0) -> None:
    short_cm, long_cm = field_of_view_cm(distance_cm)
    roi_fraction = 1957 / 4000
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.3, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})

    # (a) side view, 1 unit = 1 cm
    ax.set_xlim(-16, 16)
    ax.set_ylim(-3.6, 31)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.text(0.01, 0.97, "(a) Nhìn ngang", fontsize=9, fontweight="bold", color=TEXT, va="top")
    ax.plot([-16, 16], [30, 30], color=GREY, linewidth=1)
    ax.add_patch(Rectangle((-14, 28.7), 6, 1.3, facecolor="#fff6d6", edgecolor=EDGE, linewidth=1))
    ax.text(-15.6, 27.9, "Đèn trần, ánh trắng\nhơi vàng (~5000 K)", va="top", fontsize=7, color=TEXT)
    ax.add_patch(Rectangle((-16, -1.2), 32, 1.2, facecolor="#e4e6e9", edgecolor=GREY, linewidth=0.8))
    ax.add_patch(Rectangle((-7, 0), 14, 1.6, facecolor="#cfd8e3", edgecolor=EDGE, linewidth=1))
    ax.text(0, -3.2, "Vật mẫu (bề mặt sơn) trên mặt bàn", ha="center", fontsize=7.5, color=TEXT)

    surface, camera_y = 1.6, 1.6 + distance_cm
    ax.plot([13, 13], [0, camera_y + 1.2], color=EDGE, linewidth=2.2)
    ax.plot([13, 3.6], [camera_y + 1.2, camera_y + 1.2], color=EDGE, linewidth=2.2)
    ax.text(9.0, 9.0, "Giá đỡ\ncố định", fontsize=7.5, color=TEXT)
    ax.add_patch(Rectangle((-3.6, camera_y), 7.2, 0.9, facecolor="#33475b", edgecolor="black", linewidth=0.8))
    ax.text(-3.6, camera_y + 2.6, "Galaxy S23+, tele 3×,\nhẹn giờ 2 giây, 1 ảnh/lần", fontsize=7, color=TEXT, va="bottom")

    half = long_cm / 2
    ax.add_patch(Polygon([(0, camera_y), (-half, surface), (half, surface)], closed=True,
                         facecolor=EDGE, alpha=0.08, edgecolor="none"))
    for sign in (-1, 1):
        ax.plot([0, sign * half], [camera_y, surface], color=EDGE, linewidth=0.8, linestyle=":")
    ax.plot([0, 0], [camera_y, surface], color=EDGE, linewidth=1, linestyle="--")
    tilt = math.radians(4)
    ax.plot([0, distance_cm * math.tan(tilt)], [camera_y, surface], color=TILT, linewidth=1, linestyle="--")
    ax.text(2.0, 16.6, "trục camera, lệch\nvài độ so với\nphương thẳng đứng", fontsize=6.5, color=TILT)
    ax.add_patch(Rectangle((2.2, surface), 2.6, 0.35, facecolor="#555b63", alpha=0.6, edgecolor="none"))
    ax.annotate("bóng điện thoại", xy=(4.4, surface + 0.3), xytext=(5.0, 5.4), fontsize=6.5, color=GREY,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6))
    ax.annotate("", xy=(-8, surface), xytext=(-8, camera_y), arrowprops=dict(arrowstyle="<->", color=TEXT, lw=1))
    ax.text(-15.6, surface + distance_cm / 2, f"khoảng {distance_cm:.0f} cm\n(ước lượng)", fontsize=7.5,
            color=TEXT, va="center")

    # (b) image frame after EXIF rotation, portrait 3000 x 4000
    bx.set_xlim(-0.4, 4.4)
    bx.set_ylim(-1.5, 4.3)
    bx.set_aspect("equal")
    bx.axis("off")
    fig.text(0.62, 0.97, "(b) Khung ảnh và ROI-v0.1", fontsize=9, fontweight="bold", color=TEXT, va="top")
    frame_w, frame_h = 3.0, 4.0
    roi_h = frame_h * roi_fraction
    bx.add_patch(Rectangle((0, 0), frame_w, frame_h, facecolor="#e4e6e9", edgecolor=TEXT, linewidth=1))
    bx.add_patch(Rectangle((0, frame_h - roi_h), frame_w, roi_h, facecolor=FILL, edgecolor=EDGE, linewidth=1.6))
    bx.text(frame_w / 2, frame_h - roi_h / 2, "ROI-v0.1\nx = 0, y = 0\n3000 × 1957 px", ha="center", va="center",
            fontsize=8, color=TEXT)
    bx.text(frame_w / 2, (frame_h - roi_h) / 2, "Ngoài ROI\n(có bóng điện thoại)", ha="center", va="center",
            fontsize=7.5, color=GREY)
    bx.annotate("", xy=(0, -0.2), xytext=(frame_w, -0.2), arrowprops=dict(arrowstyle="<->", color=TEXT, lw=0.8))
    bx.text(frame_w / 2, -0.5, f"3000 px ≈ {short_cm:.1f} cm".replace(".", ","), ha="center", va="top",
            fontsize=7, color=TEXT)
    bx.annotate("", xy=(frame_w + 0.2, 0), xytext=(frame_w + 0.2, frame_h),
                arrowprops=dict(arrowstyle="<->", color=TEXT, lw=0.8))
    bx.text(frame_w + 0.3, frame_h / 2, f"4000 px\n≈ {long_cm:.0f} cm", va="center", fontsize=7, color=TEXT)
    bx.text(-0.4, -1.45, "Kích thước cm ước tính từ khoảng cách\n20 cm và tiêu cự tương đương 69 mm.",
            fontsize=6.5, color=GREY, va="bottom")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure1(args.out_dir / "hinh1_quy_trinh_kiem_tra.png")
    figure2(args.out_dir / "hinh2_bo_tri_chup.png")
    print("written to", args.out_dir)


if __name__ == "__main__":
    main()
