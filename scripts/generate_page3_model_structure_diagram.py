from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Note" / "visuals"
CANVA_DIR = ROOT / "Note" / "canva_upload"
OUT_DIR.mkdir(parents=True, exist_ok=True)
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
    return ImageFont.truetype(str(BOLD_PATH if BOLD_PATH.exists() else FONT_PATH), size)


def semi_bold(size):
    return ImageFont.truetype(str(BOLD_PATH), size)


W, H = 1600, 900
COLORS = {
    "bg": (251, 252, 254),
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
    "panel": (255, 255, 255),
}


def text_size(draw, text, fnt):
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def wrap_text(draw, text, fnt, max_width):
    lines = []
    for raw_line in text.split("\n"):
        words = raw_line.split(" ")
        current = ""
        for word in words:
            candidate = word if not current else current + " " + word
            if text_size(draw, candidate, fnt)[0] <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines


def draw_centered_text(draw, box, text, fnt, fill, spacing=8):
    x1, y1, x2, y2 = box
    lines = text.split("\n")
    sizes = [text_size(draw, line, fnt) for line in lines]
    total_h = sum(h for _, h in sizes) + spacing * (len(lines) - 1)
    y = y1 + (y2 - y1 - total_h) / 2
    for line, (w, h) in zip(lines, sizes):
        x = x1 + (x2 - x1 - w) / 2
        draw.text((x, y), line, font=fnt, fill=fill)
        y += h + spacing


def rounded_box(draw, xy, title, body, fill, outline, title_color=None, title_size=24, body_size=19):
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=18, fill=fill, outline=outline, width=2)
    draw.text((x1 + 20, y1 + 16), title, font=bold(title_size), fill=title_color or outline)
    body_font = semi_bold(body_size)
    lines = wrap_text(draw, body, body_font, x2 - x1 - 40)
    y = y1 + 55
    for line in lines:
        draw.text((x1 + 20, y), line, font=body_font, fill=COLORS["ink"])
        y += body_size + 10


def arrow(draw, start, end, color=None, width=4):
    color = color or COLORS["muted"]
    draw.line((start, end), fill=color, width=width)
    x1, y1 = start
    x2, y2 = end
    if abs(x2 - x1) >= abs(y2 - y1):
        direction = 1 if x2 >= x1 else -1
        pts = [(x2, y2), (x2 - direction * 16, y2 - 10), (x2 - direction * 16, y2 + 10)]
    else:
        direction = 1 if y2 >= y1 else -1
        pts = [(x2, y2), (x2 - 10, y2 - direction * 16), (x2 + 10, y2 - direction * 16)]
    draw.polygon(pts, fill=color)


def small_chip(draw, xy, text, fill, outline):
    draw.rounded_rectangle(xy, radius=15, fill=fill, outline=outline, width=2)
    draw_centered_text(draw, xy, text, semi_bold(17), COLORS["ink"], spacing=4)


def main():
    img = Image.new("RGB", (W, H), COLORS["bg"])
    draw = ImageDraw.Draw(img)

    draw.text((60, 38), "모델 구조와 학습 방식", font=bold(46), fill=COLORS["ink"])
    draw.text((60, 96), "입력 이미지 1장에서 view를 먼저 판단한 뒤, view별 등급 모델로 A/B/C를 예측", font=semi_bold(24), fill=COLORS["muted"])
    draw.rectangle((60, 135, W - 60, 138), fill=COLORS["line"])

    # Row 1: data and preprocessing
    rounded_box(
        draw,
        (70, 180, 350, 330),
        "1. 입력",
        "사과 이미지 1장\nRGB 이미지",
        COLORS["green_soft"],
        COLORS["green"],
    )
    rounded_box(
        draw,
        (430, 180, 750, 330),
        "2. 전처리",
        "Resize/Crop 224x224\n증강: 회전, 밝기, 채도\nImageNet Normalize",
        COLORS["blue_soft"],
        COLORS["blue"],
    )
    rounded_box(
        draw,
        (830, 180, 1170, 330),
        "3. ResNet18 Backbone",
        "ImageNet pretrained\n마지막 fc layer만 교체",
        COLORS["yellow_soft"],
        COLORS["yellow"],
    )
    rounded_box(
        draw,
        (1250, 180, 1530, 330),
        "4. 출력",
        "view 또는 A/B/C\nsoftmax confidence",
        COLORS["red_soft"],
        COLORS["red"],
    )
    arrow(draw, (350, 255), (430, 255))
    arrow(draw, (750, 255), (830, 255))
    arrow(draw, (1170, 255), (1250, 255))

    # Backbone internals
    draw.rounded_rectangle((260, 385, 1340, 530), radius=18, fill=COLORS["panel"], outline=COLORS["line"], width=2)
    draw.text((290, 407), "ResNet18 내부 흐름", font=bold(26), fill=COLORS["green"])
    chips = [
        ("Conv2d\n7x7", COLORS["blue_soft"], COLORS["blue"]),
        ("BatchNorm", COLORS["blue_soft"], COLORS["blue"]),
        ("ReLU", COLORS["green_soft"], COLORS["green"]),
        ("MaxPool", COLORS["green_soft"], COLORS["green"]),
        ("Residual\nBlock x 8", COLORS["yellow_soft"], COLORS["yellow"]),
        ("AvgPool", COLORS["green_soft"], COLORS["green"]),
        ("FC\n512 -> class", COLORS["red_soft"], COLORS["red"]),
    ]
    x = 290
    y = 455
    for i, (label, fill, outline) in enumerate(chips):
        small_chip(draw, (x, y, x + 130, y + 58), label, fill, outline)
        if i < len(chips) - 1:
            arrow(draw, (x + 130, y + 29), (x + 165, y + 29), width=3)
        x += 165

    # Final routing
    draw.rounded_rectangle((70, 585, 1530, 795), radius=18, fill=COLORS["panel"], outline=COLORS["line"], width=2)
    draw.text((100, 610), "최종 추론 라우팅", font=bold(27), fill=COLORS["green"])

    small_chip(draw, (115, 675, 295, 735), "view 모델\n3-class", COLORS["green_soft"], COLORS["green"])
    small_chip(draw, (390, 625, 610, 685), "top grade 모델\nA/B/C", COLORS["blue_soft"], COLORS["blue"])
    small_chip(draw, (390, 700, 610, 760), "middle grade 모델\nA/B/C", COLORS["blue_soft"], COLORS["blue"])
    small_chip(draw, (390, 775, 610, 835), "side grade 모델\nA/B/C", COLORS["blue_soft"], COLORS["blue"])

    # Adjust because side chip overflows panel; move lower routing block items slightly upward.
    draw.rectangle((360, 610, 640, 850), fill=COLORS["panel"])
    small_chip(draw, (390, 625, 610, 680), "top grade 모델\nA/B/C", COLORS["blue_soft"], COLORS["blue"])
    small_chip(draw, (390, 695, 610, 750), "middle grade 모델\nA/B/C", COLORS["blue_soft"], COLORS["blue"])
    small_chip(draw, (390, 765, 610, 820), "side grade 모델\nA/B/C", COLORS["blue_soft"], COLORS["blue"])
    arrow(draw, (295, 705), (390, 652), width=3)
    arrow(draw, (295, 705), (390, 722), width=3)
    arrow(draw, (295, 705), (390, 792), width=3)

    rounded_box(
        draw,
        (735, 625, 1035, 820),
        "학습 설정",
        "batch_size=64\nepochs=10, patience=4\nlearning_rate=0.0003\nweight_decay=0.0001",
        COLORS["yellow_soft"],
        COLORS["yellow"],
        body_size=18,
    )
    rounded_box(
        draw,
        (1090, 625, 1495, 820),
        "학습 기준",
        "Loss: CrossEntropyLoss\nOptimizer: AdamW\nBest model: validation macro F1\nEarly stopping 적용",
        COLORS["red_soft"],
        COLORS["red"],
        body_size=18,
    )
    arrow(draw, (610, 722), (735, 722), width=3)
    arrow(draw, (1035, 722), (1090, 722), width=3)

    out_path = OUT_DIR / "14_resnet18_training_structure.png"
    canva_path = CANVA_DIR / "visual_page_03_resnet18_training_structure.png"
    img.save(out_path)
    img.save(canva_path)
    print(out_path)
    print(canva_path)


if __name__ == "__main__":
    main()
