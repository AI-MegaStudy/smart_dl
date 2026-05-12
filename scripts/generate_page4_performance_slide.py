from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
VISUAL_DIR = ROOT / "Note" / "visuals"
CANVA_DIR = ROOT / "Note" / "canva_upload"
CANVA_DIR.mkdir(parents=True, exist_ok=True)

FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/NanumSquareR.ttf"),
    Path("C:/Windows/Fonts/NanumSquare.ttf"),
    Path("C:/Windows/Fonts/NanumGothic.ttf"),
    Path("C:/Windows/Fonts/malgun.ttf"),
]
BOLD_CANDIDATES = [
    Path("C:/Windows/Fonts/NanumSquareB.ttf"),
    Path("C:/Windows/Fonts/NanumSquareBold.ttf"),
    Path("C:/Windows/Fonts/NanumGothicBold.ttf"),
    Path("C:/Windows/Fonts/malgunbd.ttf"),
]

FONT_PATH = next(path for path in FONT_CANDIDATES if path.exists())
BOLD_PATH = next((path for path in BOLD_CANDIDATES if path.exists()), FONT_PATH)


def font(size):
    return ImageFont.truetype(str(FONT_PATH), size)


def bold(size):
    return ImageFont.truetype(str(BOLD_PATH), size)


W, H = 1600, 900
COLORS = {
    "bg": (251, 252, 254),
    "panel": (255, 255, 255),
    "ink": (23, 32, 42),
    "muted": (96, 110, 128),
    "line": (205, 214, 224),
    "green": (47, 125, 89),
    "green_soft": (235, 246, 239),
    "blue": (47, 103, 154),
    "blue_soft": (235, 243, 250),
    "red": (179, 71, 56),
    "yellow": (151, 111, 33),
}


SCORES = [
    ("view", "0.9177", "0.9162", "top/middle/side 분류"),
    ("top grade", "0.8605", "0.8612", "top 이미지 A/B/C"),
    ("middle grade", "0.9074", "0.9065", "middle 이미지 A/B/C"),
    ("side grade", "0.9136", "0.9136", "side 이미지 A/B/C"),
]


def crop_whitespace(img, threshold=248, margin=8):
    rgb = img.convert("RGB")
    px = rgb.load()
    w, h = rgb.size
    xs = []
    ys = []
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            if not (r > threshold and g > threshold and b > threshold):
                xs.append(x)
                ys.append(y)
    if not xs:
        return rgb
    box = (
        max(min(xs) - margin, 0),
        max(min(ys) - margin, 0),
        min(max(xs) + margin, w),
        min(max(ys) + margin, h),
    )
    return rgb.crop(box)


def fit_image(img, box):
    x1, y1, x2, y2 = box
    max_w = x2 - x1
    max_h = y2 - y1
    scale = min(max_w / img.width, max_h / img.height)
    new_size = (int(img.width * scale), int(img.height * scale))
    resized = img.resize(new_size, Image.Resampling.LANCZOS)
    x = x1 + (max_w - resized.width) // 2
    y = y1 + (max_h - resized.height) // 2
    return resized, (x, y)


def panel(draw, xy, title):
    draw.rounded_rectangle(xy, radius=18, fill=COLORS["panel"], outline=COLORS["line"], width=2)
    draw.text((xy[0] + 24, xy[1] + 18), title, font=bold(25), fill=COLORS["green"])


def draw_score_table(draw, xy):
    x1, y1, x2, y2 = xy
    panel(draw, xy, "최종 모델 balanced test 점수")
    table_x = x1 + 24
    table_y = y1 + 52
    col_w = [165, 120, 120, 310]
    row_h = 34

    headers = ["모델", "Accuracy", "Macro F1", "역할"]
    draw.rounded_rectangle((table_x, table_y, x2 - 24, table_y + row_h), radius=10, fill=COLORS["green_soft"], outline=COLORS["line"], width=1)
    x = table_x
    for header, width in zip(headers, col_w):
        draw.text((x + 10, table_y + 7), header, font=bold(17), fill=COLORS["ink"])
        x += width

    y = table_y + row_h + 8
    for idx, row in enumerate(SCORES):
        fill = (248, 250, 252) if idx % 2 == 0 else (255, 255, 255)
        draw.rounded_rectangle((table_x, y, x2 - 24, y + row_h), radius=8, fill=fill, outline=COLORS["line"], width=1)
        x = table_x
        for value, width in zip(row, col_w):
            color = COLORS["green"] if value in {"0.9177", "0.9162", "0.9074", "0.9065", "0.9136", "0.9136"} else COLORS["ink"]
            fnt = bold(17) if value.startswith("0.") else font(16)
            draw.text((x + 10, y + 7), value, font=fnt, fill=color)
            x += width
        y += row_h + 6


def main():
    img = Image.new("RGB", (W, H), COLORS["bg"])
    draw = ImageDraw.Draw(img)

    draw.text((60, 38), "성능 비교와 최종 모델 선정", font=bold(46), fill=COLORS["ink"])
    draw.rectangle((60, 135, W - 60, 138), fill=COLORS["line"])

    # Left-top: experiment comparison
    exp_box = (60, 165, 820, 520)
    panel(draw, exp_box, "실험별 성능 비교")
    exp_img = crop_whitespace(Image.open(VISUAL_DIR / "07_experiment_performance_comparison.png"))
    resized, pos = fit_image(exp_img, (85, 215, 800, 508))
    img.paste(resized, pos)

    # Left-bottom: score table
    draw_score_table(draw, (60, 545, 820, 840))

    # Right: confusion matrix
    cm_box = (850, 165, 1540, 840)
    panel(draw, cm_box, "최종 모델 confusion matrix")
    cm_img = crop_whitespace(Image.open(VISUAL_DIR / "09_final_confusion_matrices.png"))
    resized, pos = fit_image(cm_img, (875, 215, 1515, 820))
    img.paste(resized, pos)

    out_path = CANVA_DIR / "page_04_performance_selection.png"
    visual_path = VISUAL_DIR / "15_page4_performance_summary.png"
    img.save(out_path)
    img.save(visual_path)
    print(out_path)
    print(visual_path)


if __name__ == "__main__":
    main()
