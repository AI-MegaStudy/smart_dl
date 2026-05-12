from pathlib import Path
import shutil

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "Note" / "ppt_assets"
CANVA_DIR = ROOT / "Note" / "canva_upload"
ASSET_DIR.mkdir(parents=True, exist_ok=True)
CANVA_DIR.mkdir(parents=True, exist_ok=True)

W, H = 1600, 900
FONT_PATH = Path("C:/Windows/Fonts/malgun.ttf")
BOLD_PATH = Path("C:/Windows/Fonts/malgunbd.ttf")


def font(size):
    return ImageFont.truetype(str(FONT_PATH), size)


def bold(size):
    return ImageFont.truetype(str(BOLD_PATH if BOLD_PATH.exists() else FONT_PATH), size)


COLORS = {
    "bg": (251, 252, 254),
    "panel": (255, 255, 255),
    "visual": (247, 249, 251),
    "ink": (23, 32, 42),
    "muted": (99, 112, 131),
    "line": (216, 222, 230),
    "dash": (158, 178, 195),
    "accent": (47, 125, 89),
}


SLIDES = [
    {
        "title": "1. 데이터와 문제 정의",
        "subtitle": "사과 품질 이미지 데이터",
        "bullets": [
            "AI-Hub 계열 농산물 품질 이미지 데이터를 사용",
            "Kaggle에는 사과 데이터만 연결, 현재 모델도 사과 전용",
            "원본은 Training/Validation, 원천 이미지/라벨링 JSON으로 구성",
            "JSON에서 품종, 등급, 촬영 view, 같은 사과 개체 정보를 추출",
        ],
        "visual": "데이터 폴더 구조\n+ JSON 라벨 구조",
        "visual_note": "권장 자료: 01_data_folder_structure.png, 02_json_label_mapping.png",
        "footer": "출처와 데이터 구조를 먼저 보여주는 장",
        "file": "page_01_data_problem.png",
    },
    {
        "title": "2. 데이터 분리와 카테고리화",
        "subtitle": "group split + balanced subset",
        "bullets": [
            "등급은 A/B/C로 통일",
            "세부 각도 5개를 top/middle/side 3개 view로 재분류",
            "같은 사과가 train/test에 동시에 들어가지 않도록 group_id 기준 분리",
            "Training + Validation을 합친 뒤 7:3, 다시 8:2로 분리",
        ],
        "visual": "라벨 매핑\n+ group split 방식\n+ 불균형 보정 전/후",
        "visual_note": (
            "권장 자료: 03_label_mappings.png, "
            "04_group_split_vs_random_split.png, "
            "05_view_distribution_and_balancing.png"
        ),
        "footer": "데이터 누수 방지와 불균형 처리 방식을 보여주는 장",
        "file": "page_02_split_category.png",
    },
    {
        "title": "3. 모델 구조와 학습 방식",
        "subtitle": "ResNet18 전이학습",
        "bullets": [
            "PyTorch ResNet18에 ImageNet pretrained weight 사용",
            "마지막 fc layer만 class 수에 맞게 교체",
            "view 모델 1개가 top/middle/side를 먼저 판단",
            "view별 grade 모델 3개 중 하나가 A/B/C를 예측",
            "주요 설정: image 224, batch 64, epoch 10, lr 3e-4",
        ],
        "visual": "ResNet18 구조 다이어그램\n+ view 라우팅 흐름도",
        "visual_note": "권장 자료: ResNet18 구조 그림, 06_final_model_routing.png",
        "footer": "모델 구조는 그림 중심, parameter는 작게 표시",
        "file": "page_03_model_training.png",
    },
    {
        "title": "4. 성능 비교와 최종 모델 선정",
        "subtitle": "최종 models/apple_balanced",
        "bullets": [
            "random split은 점수가 높지만 같은 사과 누수 가능성 때문에 제외",
            "5개 각도 모델은 불균형과 앱 사용성 문제로 제외",
            "최종은 top/middle/side view 모델 + view별 grade 모델 3개",
            "balanced test: view F1 0.9162, top 0.8612, middle 0.9065, side 0.9136",
        ],
        "visual": "실험별 성능 비교 그래프\n+ 최종 confusion matrix",
        "visual_note": (
            "권장 자료: 07_experiment_performance_comparison.png, "
            "08_final_model_metrics.png, "
            "09_final_confusion_matrices.png"
        ),
        "footer": "최종 모델 선정 이유를 성능 그래프로 보여주는 장",
        "file": "page_04_performance_selection.png",
    },
    {
        "title": "5. 신선도 지표와 앱 적용",
        "subtitle": "API 응답과 화면 분기",
        "bullets": [
            "API는 이미지 1장을 받아 view, grade, confidence, 보조 점수를 반환",
            "freshness_score = 등급 60% + 색상 20% + 둥근 정도 10% + 멍 가능성 10%",
            "앱은 action_required로 RETAKE, OWNER_REVIEW, NONE을 분기",
            "view confidence 기준 0.60, grade confidence 기준 0.55",
        ],
        "visual": "신선도 계산/판정 흐름도\n+ 앱 API 응답 분기 화면",
        "visual_note": (
            "권장 자료: 12_freshness_decision_flow.png, "
            "13_api_app_branching.png, API 응답 예시 JSON 캡처"
        ),
        "footer": "앱 개발자가 실제로 사용할 값을 보여주는 장",
        "file": "page_05_api_app.png",
    },
]


def text_width(text, text_font):
    image = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), text, font=text_font)
    return bbox[2] - bbox[0]


def wrap_text(text, text_font, max_width):
    words = text.split(" ")
    lines = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        if text_width(candidate, text_font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def dashed_rect(draw, xy, radius, outline, width=3, dash=14, gap=10, fill=None):
    x1, y1, x2, y2 = xy
    if fill:
        draw.rounded_rectangle(xy, radius=radius, fill=fill)
    for x in range(x1 + radius, x2 - radius, dash + gap):
        draw.line((x, y1, min(x + dash, x2 - radius), y1), fill=outline, width=width)
        draw.line((x, y2, min(x + dash, x2 - radius), y2), fill=outline, width=width)
    for y in range(y1 + radius, y2 - radius, dash + gap):
        draw.line((x1, y, x1, min(y + dash, y2 - radius)), fill=outline, width=width)
        draw.line((x2, y, x2, min(y + dash, y2 - radius)), fill=outline, width=width)
    draw.arc((x1, y1, x1 + 2 * radius, y1 + 2 * radius), 180, 270, fill=outline, width=width)
    draw.arc((x2 - 2 * radius, y1, x2, y1 + 2 * radius), 270, 360, fill=outline, width=width)
    draw.arc((x1, y2 - 2 * radius, x1 + 2 * radius, y2), 90, 180, fill=outline, width=width)
    draw.arc((x2 - 2 * radius, y2 - 2 * radius, x2, y2), 0, 90, fill=outline, width=width)


def draw_multiline_center(draw, box, text, text_font, fill, spacing=12):
    lines = text.split("\n")
    bboxes = [draw.textbbox((0, 0), line, font=text_font) for line in lines]
    heights = [bbox[3] - bbox[1] for bbox in bboxes]
    total_h = sum(heights) + spacing * (len(lines) - 1)
    y = box[1] + (box[3] - box[1] - total_h) / 2
    for line, bbox, height in zip(lines, bboxes, heights):
        width = bbox[2] - bbox[0]
        x = box[0] + (box[2] - box[0] - width) / 2
        draw.text((x, y), line, font=text_font, fill=fill)
        y += height + spacing


def render_slide(index, slide):
    image = Image.new("RGB", (W, H), COLORS["bg"])
    draw = ImageDraw.Draw(image)

    draw.text((60, 38), slide["title"], font=bold(43), fill=COLORS["ink"])
    subtitle_bbox = draw.textbbox((0, 0), slide["subtitle"], font=bold(21))
    draw.text((W - 60 - (subtitle_bbox[2] - subtitle_bbox[0]), 52), slide["subtitle"], font=bold(21), fill=COLORS["accent"])
    draw.rectangle((60, 114, W - 60, 117), fill=COLORS["line"])

    draw.rounded_rectangle((60, 148, 675, 713), radius=14, fill=COLORS["panel"], outline=COLORS["line"], width=2)
    draw.text((95, 178), "핵심 설명", font=bold(28), fill=COLORS["accent"])
    y = 235
    bullet_font = font(24)
    for bullet in slide["bullets"]:
        lines = wrap_text(bullet, bullet_font, 505)
        draw.text((98, y), "-", font=bullet_font, fill=COLORS["ink"])
        for line_i, line in enumerate(lines):
            draw.text((125, y + line_i * 36), line, font=bullet_font, fill=COLORS["ink"])
        y += max(1, len(lines)) * 36 + 18

    dashed_rect(draw, (718, 148, 1540, 713), 14, COLORS["dash"], 3, fill=COLORS["visual"])
    draw.text((755, 178), "시각화 자료 자리", font=bold(28), fill=COLORS["accent"])
    draw_multiline_center(draw, (760, 275, 1495, 500), slide["visual"], bold(40), COLORS["muted"], 16)
    draw.rectangle((760, 598, 1498, 600), fill=COLORS["line"])

    note_lines = wrap_text(slide["visual_note"], font(19), 710)
    note_y = 622
    for line in note_lines:
        line_bbox = draw.textbbox((0, 0), line, font=font(19))
        draw.text((760 + (738 - (line_bbox[2] - line_bbox[0])) / 2, note_y), line, font=font(19), fill=COLORS["muted"])
        note_y += 28

    draw.text((60, 788), slide["footer"], font=font(17), fill=COLORS["muted"])
    page = f"{index} / 5"
    page_bbox = draw.textbbox((0, 0), page, font=font(17))
    draw.text((W - 60 - (page_bbox[2] - page_bbox[0]), 788), page, font=font(17), fill=COLORS["muted"])

    asset_path = ASSET_DIR / f"dl_apple_ppt_slide_{index}.png"
    canva_path = CANVA_DIR / slide["file"]
    image.save(asset_path)
    image.save(canva_path)
    return canva_path


def main():
    canva_paths = [render_slide(i, slide) for i, slide in enumerate(SLIDES, start=1)]
    images = [Image.open(path).convert("RGB") for path in canva_paths]
    pdf_path = CANVA_DIR / "DL_apple_freshness_canva_5pages.pdf"
    images[0].save(pdf_path, save_all=True, append_images=images[1:], resolution=150.0)

    readme_path = CANVA_DIR / "README_canva_upload.md"
    readme_path.write_text(
        """# Canva 업로드용 파일

Canva에서 바로 사용할 수 있는 5장 발표 자료입니다.

## 가장 간단한 방법

1. Canva 접속
2. `업로드` 또는 `파일 가져오기` 선택
3. `DL_apple_freshness_canva_5pages.pdf` 업로드
4. Canva가 PDF 5페이지를 슬라이드로 변환
5. 각 페이지의 시각화 자료 자리 위에 실제 그래프/그림을 배치

## 개별 페이지로 넣는 방법

PDF 변환이 마음에 들지 않으면 아래 PNG 파일을 16:9 프레젠테이션 페이지에 한 장씩 넣으면 됩니다.

- `page_01_data_problem.png`
- `page_02_split_category.png`
- `page_03_model_training.png`
- `page_04_performance_selection.png`
- `page_05_api_app.png`

## 시각화 자료 자리

각 페이지 오른쪽 점선 박스는 실제 시각화 자료를 넣을 공간입니다.
현재는 어떤 자료를 넣을지 표시만 해둔 상태입니다.
""",
        encoding="utf-8",
    )

    # Keep page names easy to find if someone browses ppt_assets directly.
    for slide in SLIDES:
        src = CANVA_DIR / slide["file"]
        dst = ASSET_DIR / slide["file"]
        shutil.copy2(src, dst)

    print(pdf_path)
    for path in canva_paths:
        print(path)


if __name__ == "__main__":
    main()
