import torch

class VisualAreaMask:
    MAX_OUTPUTS = 12  # 최대 지원 출력 개수

    def __init__(self) -> None:
        pass

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
                        "default": 2, 
                        "min": 2, 
                        "max": cls.MAX_OUTPUTS, 
                        "step": 1,
                        "tooltip": "The number of areas/masks to return."
                    }
                ),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO", 
                "unique_id": "UNIQUE_ID"
            },
        }

    RETURN_TYPES = tuple("MASK" for _ in range(MAX_OUTPUTS))
    RETURN_NAMES = tuple(f"area_{i}" for i in range(MAX_OUTPUTS))
    FUNCTION = "run_node"
    OUTPUT_NODE = False
    CATEGORY = "ToyxyzTestNodes"

    def run_node(self, extra_pnginfo, unique_id, image_width, image_height, area_number, **kwargs):
        # extra_pnginfo에서 현재 노드의 conditioning 영역값 추출
        conditioning_areas: list[list[float]] = []
        for node in extra_pnginfo["workflow"]["nodes"]:
            if node["id"] == int(unique_id):
                conditioning_areas = node["properties"]["area_values"]
                break

        
        # conditioning_areas의 길이가 area_number와 일치하는지 확인
        if len(conditioning_areas) < area_number:
            raise ValueError(f"conditioning_areas의 영역 수({len(conditioning_areas)})가 요청된 area_number({area_number})보다 적습니다.")
        
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

        # 최대 출력 개수만큼 튜플 생성, area_number 미만은 실제 마스크, 나머지는 None 할당
        outputs = tuple(masks[i] if i < len(masks) else None for i in range(self.MAX_OUTPUTS))

        return {
            "result": outputs
        }
        
NODE_CLASS_MAPPINGS = {
    "VisualAreaMask": VisualAreaMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VisualAreaMask": "Visual Area Mask"
}