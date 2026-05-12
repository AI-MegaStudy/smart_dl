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
    "red_soft": (250, 239, 236),
    "yellow": (151, 111, 33),
    "yellow_soft": (250, 246, 232),
    "code": (31, 42, 55),
}

JSON_LINES = [
    '{',
    '  "angle_label": "top",',
    '  "angle_confidence": 0.9697,',
    '  "model_grade": "C",',
    '  "freshness_score": 55.35,',
    '  "model_decision": "HOLD",',
    '  "action_required": "OWNER_REVIEW",',
    '}',
]


def crop_whitespace(img, threshold=248, margin=8):
    rgb = img.convert("RGB")
    px = rgb.load()
    w, h = rgb.size
    xs, ys = [], []
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            if not (r > threshold and g > threshold and b > threshold):
                xs.append(x)
                ys.append(y)
    if not xs:
        return rgb
    return rgb.crop((max(min(xs) - margin, 0), max(min(ys) - margin, 0), min(max(xs) + margin, w), min(max(ys) + margin, h)))


def fit_image(img, box):
    x1, y1, x2, y2 = box
    max_w = x2 - x1
    max_h = y2 - y1
    scale = min(max_w / img.width, max_h / img.height)
    new_size = (int(img.width * scale), int(img.height * scale))
    resized = img.resize(new_size, Image.Resampling.LANCZOS)
    return resized, (x1 + (max_w - resized.width) // 2, y1 + (max_h - resized.height) // 2)


def panel(draw, xy, title):
    draw.rounded_rectangle(xy, radius=18, fill=COLORS["panel"], outline=COLORS["line"], width=2)
    draw.text((xy[0] + 24, xy[1] + 18), title, font=bold(25), fill=COLORS["green"])


def draw_thresholds(draw, xy):
    panel(draw, xy, "앱 적용 기준")
    x1, y1, x2, _ = xy
    chips = [
        ("view confidence < 0.60", "RETAKE", COLORS["red_soft"], COLORS["red"]),
        ("grade confidence < 0.55", "OWNER_REVIEW", COLORS["yellow_soft"], COLORS["yellow"]),
        ("그 외 결과", "점주 최종 확인", COLORS["blue_soft"], COLORS["blue"]),
    ]
    y = y1 + 68
    for left, right, fill, outline in chips:
        draw.rounded_rectangle((x1 + 24, y, x2 - 24, y + 54), radius=12, fill=fill, outline=outline, width=2)
        draw.text((x1 + 42, y + 15), left, font=bold(17), fill=COLORS["ink"])
        rb = draw.textbbox((0, 0), right, font=bold(17))
        draw.text((x2 - 42 - (rb[2] - rb[0]), y + 15), right, font=bold(17), fill=outline)
        y += 68


def draw_json_box(draw, xy):
    panel(draw, xy, "결과 예시 JSON")
    x1, y1, x2, y2 = xy
    code_x1, code_y1 = x1 + 24, y1 + 62
    code_x2, code_y2 = x2 - 24, y2 - 14
    draw.rounded_rectangle((code_x1, code_y1, code_x2, code_y2), radius=12, fill=COLORS["code"], outline=COLORS["line"], width=1)
    y = code_y1 + 14
    code_font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 14) if Path("C:/Windows/Fonts/consola.ttf").exists() else font(14)
    for line in JSON_LINES:
        color = (229, 239, 250)
        if '"action_required"' in line or '"model_decision"' in line:
            color = (190, 235, 207)
        if '"freshness_score"' in line or '"model_grade"' in line:
            color = (255, 221, 164)
        draw.text((code_x1 + 18, y), line, font=code_font, fill=color)
        y += 17


def main():
    img = Image.new("RGB", (W, H), COLORS["bg"])
    draw = ImageDraw.Draw(img)

    draw.text((60, 38), "신선도 점수와 앱 적용 값", font=bold(46), fill=COLORS["ink"])
    draw.rectangle((60, 135, W - 60, 138), fill=COLORS["line"])

    draw_thresholds(draw, (60, 165, 575, 470))

    flow_box = (610, 165, 1540, 560)
    panel(draw, flow_box, "신선도 점수와 최종 판정 흐름")
    flow_img = crop_whitespace(Image.open(VISUAL_DIR / "12_freshness_decision_flow.png"))
    resized, pos = fit_image(flow_img, (635, 220, 1515, 540))
    img.paste(resized, pos)

    draw.rounded_rectangle((60, 500, 575, 805), radius=18, fill=COLORS["panel"], outline=COLORS["line"], width=2)
    draw.text((84, 523), "앱에서 사용하는 주요 값", font=bold(25), fill=COLORS["green"])
    key_items = [
        ("model_grade", "A/B/C 예측 등급"),
        ("freshness_score", "0~100 품질 보조 점수"),
        ("model_decision", "PASS/REVIEW/HOLD/RETAKE"),
        ("action_required", "앱 화면 분기 기준"),
        ("confidence", "낮으면 재촬영 또는 점주 확인"),
    ]
    y = 575
    for key, desc in key_items:
        draw.rounded_rectangle((84, y, 550, y + 38), radius=9, fill=(248, 250, 252), outline=COLORS["line"], width=1)
        draw.text((100, y + 8), key, font=bold(16), fill=COLORS["green"])
        draw.text((270, y + 8), desc, font=font(16), fill=COLORS["ink"])
        y += 48

    draw_json_box(draw, (610, 590, 1540, 805))

    out_path = CANVA_DIR / "page_05_api_app.png"
    visual_path = VISUAL_DIR / "16_page5_final_metrics_api_summary.png"
    img.save(out_path)
    img.save(visual_path)
    print(out_path)
    print(visual_path)


if __name__ == "__main__":
    main()
