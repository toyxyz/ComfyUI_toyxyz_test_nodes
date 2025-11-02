import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class VisualAreaMask:
    MAX_OUTPUTS = 16  # 최대 지원 출력 개수

    def __init__(self) -> None:
        pass

    @staticmethod
    def hsl_to_rgb(h, s, l):
        """HSL 색상을 RGB로 변환 (JavaScript의 HSL과 동일한 방식)"""
        h = h / 360.0
        s = s / 100.0
        l = l / 100.0

        if s == 0:
            r = g = b = l
        else:
            def hue_to_rgb(p, q, t):
                if t < 0:
                    t += 1
                if t > 1:
                    t -= 1
                if t < 1 / 6:
                    return p + (q - p) * 6 * t
                if t < 1 / 2:
                    return q
                if t < 2 / 3:
                    return p + (q - p) * (2 / 3 - t) * 6
                return p

            q = l * (1 + s) if l < 0.5 else l + s - l * s
            p = 2 * l - q
            r = hue_to_rgb(p, q, h + 1 / 3)
            g = hue_to_rgb(p, q, h)
            b = hue_to_rgb(p, q, h - 1 / 3)

        return int(r * 255), int(g * 255), int(b * 255)

    @staticmethod
    def generate_area_color(index, total_areas, alpha=80):
        """영역 인덱스에 따라 동적으로 HSL 색상 생성 (JavaScript와 동일)"""
        # JavaScript의 generateHslColor와 동일한 로직
        # hue = ((index + 1) % total_areas) / total_areas * 360
        hue = int((((index + 1) % total_areas) / total_areas) * 360)
        r, g, b = VisualAreaMask.hsl_to_rgb(hue, 100, 50)
        return (r, g, b, alpha)

    @staticmethod
    def wrap_text(text, font, max_width, draw):
        """텍스트를 여러 줄로 나누기 (문자 단위 줄바꿈 - 단어가 잘릴 수 있음)"""
        lines = []

        if not text:
            return [text]

        current_line = ""

        for char in text:
            # 현재 줄에 문자를 추가했을 때의 너비 계산
            test_line = current_line + char
            bbox = draw.textbbox((0, 0), test_line, font=font)
            test_width = bbox[2] - bbox[0]

            if test_width <= max_width:
                # 너비가 허용 범위 내면 추가
                current_line = test_line
            else:
                # 너비 초과하면 현재 줄 저장하고 새 줄 시작
                if current_line:  # 현재 줄이 비어있지 않으면 저장
                    lines.append(current_line)
                current_line = char

        # 마지막 줄 추가
        if current_line:
            lines.append(current_line)

        return lines if lines else [text]

    @staticmethod
    def wrap_text_with_fixed_font(text, font_size, max_width, max_height, draw):
        """고정된 폰트 크기로 텍스트를 여러 줄로 나누기"""
        # 폰트 로드
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()

        # 줄바꿈 적용
        lines = VisualAreaMask.wrap_text(text, font, max_width, draw)

        return font, lines

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 16,
                        "max": 16384,
                        "tooltip": "The width of the canvas. (only affects looks)."
                    }
                ),
                "image_height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 16,
                        "max": 16384,
                        "tooltip": "The height of the canvas. (only affects looks)."
                    }
                ),
                "area_number": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": cls.MAX_OUTPUTS,
                        "step": 1,
                        "tooltip": "The number of areas/masks to return."
                    }
                ),
                "mask_overlap_method": (
                    ["default", "subtract"],
                    {
                        "default": "default",
                        "tooltip": "How to handle overlapping masks. 'default': output masks as-is, 'subtract': subtract all other masks from each mask."
                    }
                ),
                "font_size": (
                    "INT",
                    {
                        "default": 40,
                        "min": 10,
                        "max": 200,
                        "step": 1,
                        "tooltip": "Font size for area labels."
                    }
                ),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            },
        }

    RETURN_TYPES = tuple(["IMAGE", "MASK"] + ["MASK" for _ in range(MAX_OUTPUTS)])
    RETURN_NAMES = tuple(["canvas_image", "combined_mask"] + [f"area_{i}" for i in range(MAX_OUTPUTS)])
    FUNCTION = "run_node"
    OUTPUT_NODE = False
    CATEGORY = "ToyxyzTestNodes"

    def create_canvas_image(self, image_width, image_height, conditioning_areas, area_number, font_size,
                            area_texts=None):
        """캔버스 이미지를 생성하는 함수"""
        # 흰색 배경 이미지 생성
        img = Image.new('RGB', (image_width, image_height), color='white')
        draw = ImageDraw.Draw(img, 'RGBA')

        # area_texts가 None이면 빈 딕셔너리로 초기화
        if area_texts is None:
            area_texts = {}

        # 각 영역 그리기
        for i in range(area_number):
            area_values = conditioning_areas[i]
            x_min, y_min, x_max, y_max, strength = area_values

            # 비율을 픽셀 단위로 변환
            x = int(x_min * image_width)
            y = int(y_min * image_height)
            w = int(x_max * image_width)
            h = int(y_max * image_height)

            x_end = min(x + w, image_width)
            y_end = min(y + h, image_height)

            # 색상 동적 생성 (JavaScript UI와 동일한 방식)
            color = self.generate_area_color(i, area_number, alpha=80)
            color_with_strength = (color[0], color[1], color[2], int(color[3] * strength))

            # 영역 채우기
            draw.rectangle([x, y, x_end, y_end], fill=color_with_strength, outline=None)

            # 테두리 그리기 (불투명)
            border_color = (color[0], color[1], color[2], 255)
            draw.rectangle([x, y, x_end, y_end], outline=border_color, width=3)

            # 텍스트 결정 (area_text가 있으면 사용, 없으면 숫자)
            display_text = area_texts.get(i, str(i))
            if not display_text or display_text.strip() == "":
                display_text = str(i)

            # 텍스트 크기에 따라 폰트 크기 조정
            box_width = x_end - x
            box_height = y_end - y

            # 박스가 너무 작으면 스킵
            if box_width < 10 or box_height < 10:
                continue

            # 고정된 폰트 크기로 줄바꿈된 텍스트 가져오기
            max_text_width = box_width * 0.9
            max_text_height = box_height * 0.9

            font, lines = self.wrap_text_with_fixed_font(
                display_text,
                font_size,
                max_text_width,
                max_text_height,
                draw
            )

            # 각 줄의 높이 계산
            line_heights = []
            line_widths = []
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                line_heights.append(bbox[3] - bbox[1])
                line_widths.append(bbox[2] - bbox[0])

            # 평균 줄 높이
            avg_line_height = sum(line_heights) // max(len(line_heights), 1)
            line_spacing = int(avg_line_height * 0.2)  # 줄 간격 20%

            # 전체 텍스트 블록 높이
            total_text_height = sum(line_heights) + line_spacing * (len(lines) - 1)

            # 텍스트 블록 시작 위치 (중앙 정렬)
            start_y = y + (box_height - total_text_height) // 2

            # 테두리 효과 두께
            outline_width = max(1, font_size // 20)

            # 각 줄 그리기
            current_y = start_y
            for idx, (line, line_width, line_height) in enumerate(zip(lines, line_widths, line_heights)):
                # 줄 중앙 정렬
                text_x = x + (box_width - line_width) // 2
                text_y = current_y

                # 텍스트 테두리 효과 (가독성 향상)
                for offset_x in range(-outline_width, outline_width + 1):
                    for offset_y in range(-outline_width, outline_width + 1):
                        if offset_x != 0 or offset_y != 0:
                            draw.text((text_x + offset_x, text_y + offset_y), line, fill=(255, 255, 255, 255),
                                      font=font)

                # 실제 텍스트
                draw.text((text_x, text_y), line, fill=(0, 0, 0, 255), font=font)

                # 다음 줄 위치
                current_y += line_height + line_spacing

        # PIL Image를 ComfyUI IMAGE 형식으로 변환 (torch tensor)
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]  # [1, H, W, C]

        return img_tensor

    def run_node(self, extra_pnginfo, unique_id, image_width, image_height, area_number, mask_overlap_method, font_size,
                 **kwargs):
        # extra_pnginfo에서 현재 노드의 conditioning 영역값 추출
        conditioning_areas: list[list[float]] = []
        for node in extra_pnginfo["workflow"]["nodes"]:
            if node["id"] == int(unique_id):
                conditioning_areas = node["properties"]["area_values"]
                break

        # conditioning_areas의 길이가 area_number와 일치하는지 확인
        if len(conditioning_areas) < area_number:
            raise ValueError(
                f"conditioning_areas의 영역 수({len(conditioning_areas)})가 요청된 area_number({area_number})보다 적습니다.")

        # kwargs에서 area_text 인풋 추출
        area_texts = {}
        for i in range(area_number):
            text_key = f"area_{i}_text"
            if text_key in kwargs and kwargs[text_key] is not None:
                area_texts[i] = kwargs[text_key]

        # 캔버스 이미지 생성 (font_size와 area_texts 전달)
        canvas_image = self.create_canvas_image(image_width, image_height, conditioning_areas, area_number, font_size,
                                                area_texts)

        masks = []

        for i in range(area_number):
            area_values = conditioning_areas[i]
            # 각 area_values가 5개의 요소를 가지는지 확인
            if len(area_values) != 5:
                raise ValueError(f"conditioning_areas[{i}]는 5개의 요소([x_min, y_min, x_max, y_max, strength])를 가져야 합니다.")

            x_min, y_min, x_max, y_max, strength = area_values

            # 비율을 픽셀 단위로 변환
            x = int(x_min * image_width)
            y = int(y_min * image_height)
            w = int(x_max * image_width)
            h = int(y_max * image_height)

            # 마스크 생성
            mask = torch.zeros((1, image_height, image_width), dtype=torch.float32, device="cpu")
            # 영역이 이미지 범위를 벗어나지 않도록 조정
            x_end = min(x + w, image_width)
            y_end = min(y + h, image_height)
            mask[:, y:y_end, x:x_end] = strength
            masks.append(mask)

        # subtract 모드 처리
        if mask_overlap_method == "subtract":
            subtracted_masks = []
            for i in range(len(masks)):
                # 현재 마스크에서 시작
                result_mask = masks[i].clone()

                # 다른 모든 마스크를 빼기
                for j in range(len(masks)):
                    if i != j:
                        # 마스크 값을 빼고, 음수가 되지 않도록 0으로 클램핑
                        result_mask = torch.clamp(result_mask - masks[j], min=0.0)

                subtracted_masks.append(result_mask)

            masks = subtracted_masks

        # 모든 마스크를 합친 combined_mask 생성
        combined_mask = torch.zeros((1, image_height, image_width), dtype=torch.float32, device="cpu")
        for mask in masks:
            # 각 마스크를 더하되, 최대값은 1.0을 넘지 않도록 클램핑
            combined_mask = torch.clamp(combined_mask + mask, min=0.0, max=1.0)

        # 최대 출력 개수만큼 튜플 생성, area_number 미만은 실제 마스크, 나머지는 None 할당
        outputs = tuple(
            [canvas_image, combined_mask] + [masks[i] if i < len(masks) else None for i in range(self.MAX_OUTPUTS)])

        return {
            "result": outputs
        }