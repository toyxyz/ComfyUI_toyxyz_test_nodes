import json
import torch
import numpy as np
import cv2
from .util import draw_pose_json, draw_pose

OpenposeJSON = dict

class OpenposeEditorNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "show_body": ("BOOLEAN", {"default": True}),
                "show_face": ("BOOLEAN", {"default": True}),
                "show_hands": ("BOOLEAN", {"default": True}),
                "resolution_x": ("INT", {"default": -1, "min": -1, "max": 12800}),
                "use_ground_plane": ("BOOLEAN", {"default": True}),
                "pose_marker_size": ("INT", { "default": 4, "min": 0, "max": 100 }),
                "face_marker_size": ("INT", { "default": 3, "min": 0, "max": 100 }),
                "hand_marker_size": ("INT", { "default": 2, "min": 0, "max": 100 }),
                "pelvis_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01 }),
                "torso_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01 }),
                "neck_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01 }),
                # --- 머리 및 눈 관련 스케일 ---
                "head_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01 }),
                "eye_distance_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01 }),
                "eye_height": ("FLOAT", { "default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1 }),
                "eyebrow_height": ("FLOAT", { "default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1 }), # 눈썹 높이 조절
                "left_eye_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01 }),
                "right_eye_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01 }),
                "left_eyebrow_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01 }),
                "right_eyebrow_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01 }),
                "mouth_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01 }),
                "nose_scale_face": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01 }),
                "face_shape_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01 }),
                # ---
                "shoulder_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01 }),
                "arm_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01 }),
                "leg_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01 }),
                "hands_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01 }),
                "overall_scale": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01 }),
                "rotate_angle": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "translate_x": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.1}),
                "translate_y": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.1}),
                "POSE_JSON": ("STRING", {"multiline": True}),
                "POSE_KEYPOINT": ("POSE_KEYPOINT",{"default": None}),
                "Target_pose_keypoint": ("POSE_KEYPOINT", {"default": None}),
            },
        }

    RETURN_NAMES = ("POSE_IMAGE", "POSE_KEYPOINT", "POSE_JSON")
    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT", "STRING")
    OUTPUT_NODE = True
    FUNCTION = "load_pose"
    CATEGORY = "ToyxyzTestNodes"

    def load_pose(self, show_body, show_face, show_hands, resolution_x, use_ground_plane,
                  pose_marker_size, face_marker_size, hand_marker_size,
                  pelvis_scale, torso_scale, neck_scale, head_scale, eye_distance_scale, eye_height, eyebrow_height,
                  left_eye_scale, right_eye_scale, left_eyebrow_scale, right_eyebrow_scale,
                  mouth_scale, nose_scale_face, face_shape_scale,
                  shoulder_scale, arm_scale, leg_scale, hands_scale, overall_scale,
                  rotate_angle, translate_x, translate_y,
                  POSE_JSON: str, POSE_KEYPOINT=None, Target_pose_keypoint=None) -> tuple[OpenposeJSON]:
        
        # 내부 함수인 process_pose에 Target_pose_keypoint를 전달하도록 수정
        def process_pose(pose_input_str_list, target_pose_obj=None, rot_angle=0, tr_x=0.0, tr_y=0.0):
            pose_imgs, final_keypoints_batch = draw_pose_json(
                pose_input_str_list, resolution_x, use_ground_plane, show_body, show_face, show_hands,
                pose_marker_size, face_marker_size, hand_marker_size,
                pelvis_scale, torso_scale, neck_scale, head_scale, eye_distance_scale, eye_height, eyebrow_height,
                left_eye_scale, right_eye_scale, left_eyebrow_scale, right_eyebrow_scale,
                mouth_scale, nose_scale_face, face_shape_scale,
                shoulder_scale, arm_scale, leg_scale, hands_scale, overall_scale,
                rot_angle, tr_x, tr_y,
                target_pose_keypoint_obj=target_pose_obj # util.py 함수로 Target_pose_keypoint 전달
            )
            
            if not pose_imgs: return None, None, None
            
            pose_imgs_np = np.array(pose_imgs).astype(np.float32) / 255
            final_json_str = json.dumps(final_keypoints_batch, indent=4)
            return torch.from_numpy(pose_imgs_np), final_keypoints_batch, final_json_str

        input_json_str = ""
        # 팔 길이 비교를 위해 POSE_KEYPOINT가 우선순위를 갖도록 순서 조정
        if POSE_KEYPOINT is not None:
            normalized_json_data = json.dumps(POSE_KEYPOINT, indent=4).replace("'",'"').replace('None','[]')
            if not isinstance(POSE_KEYPOINT, list):
                input_json_str = f'[{normalized_json_data}]'
            else:
                input_json_str = normalized_json_data
        elif POSE_JSON: 
            temp_json = POSE_JSON.replace("'",'"').replace('None','[]')
            try:
                parsed_json = json.loads(temp_json)
                input_json_str = f"[{temp_json}]" if not isinstance(parsed_json, list) else temp_json
            except json.JSONDecodeError: input_json_str = f"[{temp_json}]"
        
        if input_json_str:
            # process_pose 호출 시 Target_pose_keypoint 객체를 인자로 전달
            image_tensor, keypoint_obj_batch, json_str_batch = process_pose(input_json_str, Target_pose_keypoint, rotate_angle, translate_x, translate_y)
            if image_tensor is not None:
                return { "ui": {"POSE_JSON": [json_str_batch]}, "result": (image_tensor, keypoint_obj_batch, json_str_batch) }

        W, H = 512, 768
        blank_person = dict(pose_keypoints_2d=[], face_keypoints_2d=[], hand_left_keypoints_2d=[], hand_right_keypoints_2d=[])
        blank_output_keypoints = [{"people": [blank_person], "canvas_width": W, "canvas_height": H}]
        W_scaled = resolution_x if resolution_x >= 64 else W
        H_scaled = int(H*(W_scaled*1.0/W))
        blank_pose_for_draw = {"people": [blank_person]}
        pose_img = [draw_pose(blank_pose_for_draw, H_scaled, W_scaled, pose_marker_size, face_marker_size, hand_marker_size)]
        pose_img_np = np.array(pose_img).astype(np.float32) / 255
        return { "ui": {"POSE_JSON": [json.dumps(blank_output_keypoints)]}, "result": (torch.from_numpy(pose_img_np), blank_output_keypoints, json.dumps(blank_output_keypoints)) }


class PoseToMaskNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT",),
                "line_thickness": ("INT", {"default": 30, "min": 1, "max": 1000, "step": 1, "display": "number"}),
                "finger_line_thickness": ("INT", {"default": 10, "min": 1, "max": 1000, "step": 1, "display": "number"}),
                "torso_thickness": ("INT", {"default": 30, "min": 1, "max": 1000, "step": 1, "display": "number"}),
                "head_size": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 1000.0, "step": 0.01, "display": "number"}), # 얼굴 마스크 크기 조절
                "confidence_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number"}),
            }
            # "optional": { # 캔버스 크기 오버라이드 옵션 (필요시 유지)
            #     "override_canvas_width": ("INT", {"default": 0, "min": 0, "max": 8192}),
            #     "override_canvas_height": ("INT", {"default": 0, "min": 0, "max": 8192}),
            # }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASK",)
    FUNCTION = "create_body_parts_mask"
    CATEGORY = "ToyxyzTestNodes"

    # OpenPose 키포인트 인덱스 정의 (참고용)
    # Nose:0, Neck:1, RShoulder:2, RElbow:3, RWrist:4, LShoulder:5, LElbow:6, LWrist:7,
    # RHip:8, RKnee:9, RAnkle:10, LHip:11, LKnee:12, LAnkle:13
    
    # 팔, 다리, 목 연결선 정의
    LIMB_CONNECTIONS = [
        (0, 1),  # 코(0) - 목(1)
        (2, 3), (3, 4),  # 오른쪽 팔 (RShoulder-RElbow, RElbow-RWrist)
        (5, 6), (6, 7),  # 왼쪽 팔 (LShoulder-LElbow, LElbow-LWrist)
        (8, 9), (9, 10), # 오른쪽 다리 (RHip-RKnee, RKnee-RAnkle)
        (11, 12), (12, 13) # 왼쪽 다리 (LHip-LKnee, LAnkle)
    ]

    # 몸통을 구성하는 4개 꼭짓점 인덱스 (RShoulder, LShoulder, LHip, RHip 순서)
    TORSO_POLYGON_INDICES = [2, 5, 11, 8] 

    # 손가락 연결선 정의 (OpenPose hand keypoints 기준)
    # 손목 (0)에서 시작하여 각 손가락의 마디를 연결
    # 출처: OpenPose GitHub 또는 문서에서 hand keypoint index 참조
    HAND_CONNECTIONS = [
        # 엄지 (Thumb)
        (0, 1), (1, 2), (2, 3), (3, 4),
        # 검지 (Index Finger)
        (0, 5), (5, 6), (6, 7), (7, 8),
        # 중지 (Middle Finger)
        (0, 9), (9, 10), (10, 11), (11, 12),
        # 약지 (Ring Finger)
        (0, 13), (13, 14), (14, 15), (15, 16),
        # 새끼손가락 (Little Finger)
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]

    # 얼굴 키포인트 인덱스 정의
    FACE_KEYPOINT_INDICES = list(range(17)) # 0~16번까지의 얼굴 키포인트 사용

    def _get_point(self, keypoints_list, index, confidence_threshold):
        """Helper function to get a valid point's coordinates if confidence is high enough."""
        if not keypoints_list or (index * 3 + 2) >= len(keypoints_list):
            return None
        x, y, conf = keypoints_list[index * 3], keypoints_list[index * 3 + 1], keypoints_list[index * 3 + 2]
        # 유효한 좌표 (OpenPose에서 0,0은 종종 감지 안됨을 의미)이고 신뢰도가 문턱값 이상일 때
        if conf >= confidence_threshold and (x != 0 or y != 0):
            return (int(round(x)), int(round(y)))
        return None

    def create_body_parts_mask(self, pose_keypoint, line_thickness, finger_line_thickness, torso_thickness, head_size, confidence_threshold, override_canvas_width=0, override_canvas_height=0):
        if not pose_keypoint or not isinstance(pose_keypoint, list) or not pose_keypoint[0]:
            # 입력이 유효하지 않으면 기본 빈 마스크 반환
            h = override_canvas_height if override_canvas_height > 0 else 768
            w = override_canvas_width if override_canvas_width > 0 else 512
            mask = np.zeros((h, w), dtype=np.float32)
            return (torch.from_numpy(mask).unsqueeze(0),)

        frame_data = pose_keypoint[0] # 첫 번째 프레임의 데이터 사용 가정

        canvas_width = override_canvas_width if override_canvas_width > 0 else frame_data.get("canvas_width", 512)
        canvas_height = override_canvas_height if override_canvas_height > 0 else frame_data.get("canvas_height", 768)
        
        if canvas_width <= 0 or canvas_height <= 0: # 안전장치
            canvas_width = 512
            canvas_height = 768

        # 최종 마스크는 8비트 단일 채널
        mask_image = np.zeros((canvas_height, canvas_width), dtype=np.uint8) 
        # 몸통 마스크를 위한 임시 마스크
        torso_mask_temp = np.zeros((canvas_height, canvas_width), dtype=np.uint8)


        for person_data in frame_data.get("people", []):
            body_keypoints = person_data.get("pose_keypoints_2d")
            hand_left_keypoints = person_data.get("hand_left_keypoints_2d", [])
            hand_right_keypoints = person_data.get("hand_right_keypoints_2d", [])
            face_keypoints = person_data.get("face_keypoints_2d", [])

            if not body_keypoints:
                continue

            # 1. 몸통 마스크 (채워진 사각형/다각형) - 임시 마스크에 그립니다.
            torso_points_for_poly = []
            for idx in self.TORSO_POLYGON_INDICES:
                point = self._get_point(body_keypoints, idx, confidence_threshold)
                if point:
                    # 좌표가 캔버스 범위 내에 있는지 확인
                    if 0 <= point[0] < canvas_width and 0 <= point[1] < canvas_height:
                        torso_points_for_poly.append(point)
                else:
                    # 몸통 꼭짓점 중 하나라도 유효하지 않으면 몸통 마스크를 그리지 않음
                    torso_points_for_poly = [] # 리스트 비우기
                    break 
            
            if len(torso_points_for_poly) == 4: # 4개의 꼭짓점이 모두 유효할 때만 그림
                np_torso_points = np.array([torso_points_for_poly], dtype=np.int32) 
                cv2.fillConvexPoly(torso_mask_temp, np_torso_points, 255, lineType=cv2.LINE_AA) # 임시 마스크에 흰색(255)으로 채움

            # 몸통 마스크 확장 (dilate) - 임시 마스크에만 적용
            # torso_thickness 값에 따라 커널 크기 조정 (홀수로 유지)
            # torso_thickness가 1보다 작으면 1로 설정하여 최소한의 dilate 적용
            kernel_size_torso = max(1, torso_thickness // 2 * 2 + 1) 
            kernel_torso = np.ones((kernel_size_torso, kernel_size_torso), np.uint8)
            torso_mask_temp = cv2.dilate(torso_mask_temp, kernel_torso, iterations=1)

            # 확장된 몸통 마스크를 최종 마스크에 추가
            mask_image = cv2.bitwise_or(mask_image, torso_mask_temp)

            # 2. 팔, 다리, 목 마스크 (두꺼운 선) - 최종 마스크에 직접 그립니다.
            for p1_idx, p2_idx in self.LIMB_CONNECTIONS:
                p1 = self._get_point(body_keypoints, p1_idx, confidence_threshold)
                p2 = self._get_point(body_keypoints, p2_idx, confidence_threshold)

                if p1 and p2:
                    # 두 점이 모두 유효하고 캔버스 범위 내에 있을 경우 선 그리기
                    if (0 <= p1[0] < canvas_width and 0 <= p1[1] < canvas_height and
                        0 <= p2[0] < canvas_width and 0 <= p2[1] < canvas_height):
                        cv2.line(mask_image, p1, p2, 255, line_thickness, lineType=cv2.LINE_AA)
            
            # 3. 손가락 마스크 (새로 추가) - 최종 마스크에 직접 그립니다.
            for hand_keypoints in [hand_left_keypoints, hand_right_keypoints]:
                if not hand_keypoints:
                    continue
                
                # 손가락 연결선 그리기
                for p1_idx, p2_idx in self.HAND_CONNECTIONS:
                    p1 = self._get_point(hand_keypoints, p1_idx, confidence_threshold)
                    p2 = self._get_point(hand_keypoints, p2_idx, confidence_threshold)

                    if p1 and p2:
                        if (0 <= p1[0] < canvas_width and 0 <= p1[1] < canvas_height and
                            0 <= p2[0] < canvas_width and 0 <= p2[1] < canvas_height):
                            cv2.line(mask_image, p1, p2, 255, finger_line_thickness, lineType=cv2.LINE_AA)

            # 4. 얼굴 마스크 (타원)
            face_points = []
            for idx in self.FACE_KEYPOINT_INDICES:
                point = self._get_point(face_keypoints, idx, confidence_threshold)
                if point:
                    # 좌표가 캔버스 범위 내에 있는지 확인
                    if 0 <= point[0] < canvas_width and 0 <= point[1] < canvas_height:
                        face_points.append(point)

            if len(face_points) >= 5:  # 타원을 그리기 위한 최소 점 개수 (최소 5개의 점이 필요함)
                np_face_points = np.array(face_points, dtype=np.int32)
                
                # 볼록 껍질(Convex Hull) 계산
                hull = cv2.convexHull(np_face_points)
                
                # 볼록 껍질을 감싸는 최소 크기 타원 계산
                # cv2.fitEllipse는 최소 5개의 점이 필요합니다.
                (x, y), (major_axis, minor_axis), angle = cv2.fitEllipse(hull)

                # head_size에 따라 타원 크기 조정
                major_axis *= head_size
                minor_axis *= head_size

                # 타원 그리기
                cv2.ellipse(mask_image, (int(x), int(y)), (int(major_axis / 2), int(minor_axis / 2)), int(angle), 0, 360, 255, cv2.FILLED, lineType=cv2.LINE_AA)


        # NumPy 배열을 PyTorch 텐서로 변환하고 정규화 (H, W) -> (1, H, W)
        mask_tensor = torch.from_numpy(mask_image.astype(np.float32) / 255.0).unsqueeze(0)
        
        return (mask_tensor,)
