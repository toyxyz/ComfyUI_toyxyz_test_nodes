import math
import json
import numpy as np
import matplotlib
import cv2
from comfy.utils import ProgressBar

eps = 0.01

def scale(point, scale_factor, pivot):
    if not isinstance(point, np.ndarray): point = np.array(point)
    if not isinstance(pivot, np.ndarray): pivot = np.array(pivot)
    return pivot + (point - pivot) * scale_factor

def draw_pose_json(pose_json_str, resolution_x, use_ground_plane, show_body, show_face, show_hands,
                   pose_marker_size, face_marker_size, hand_marker_size,
                   pelvis_scale, torso_scale, neck_scale, head_scale, eye_distance_scale, eye_height, eyebrow_height,
                   left_eye_scale, right_eye_scale, left_eyebrow_scale, right_eyebrow_scale,
                   mouth_scale, nose_scale_face, face_shape_scale,
                   shoulder_scale, arm_scale, leg_scale, hands_scale, overall_scale,
                   rotate_angle, translate_x, translate_y,
                   target_pose_keypoint_obj=None):

    # 최종적으로 적용될 스케일 값을 초기화
    final_hands_scale = hands_scale
    final_torso_scale = torso_scale
    final_head_scale = head_scale
    final_neck_scale = neck_scale
    final_pelvis_scale = pelvis_scale
    final_shoulder_scale = shoulder_scale
    final_arm_scale = arm_scale
    final_leg_scale = leg_scale

    if target_pose_keypoint_obj and pose_json_str:
        try:
            source_pose_obj = json.loads(pose_json_str)

            # --- 공용 헬퍼 함수 정의 ---
            def get_point(kps_list, index):
                if index * 3 + 2 >= len(kps_list) or kps_list[index * 3 + 2] == 0:
                    return None
                return np.array([kps_list[index * 3], kps_list[index * 3 + 1]])

            def calculate_limb_length(kps_list, p1_idx, p2_idx):
                p1 = get_point(kps_list, p1_idx)
                p2 = get_point(kps_list, p2_idx)
                if p1 is not None and p2 is not None:
                    return np.linalg.norm(p1 - p2)
                return 0.0
            
            # --- 팔 길이 계산 ---
            def get_max_arm_length(pose_obj):
                try:
                    if not isinstance(pose_obj, list) or not pose_obj or 'people' not in pose_obj[0] or not pose_obj[0]['people']: return 0.0
                    keypoints = pose_obj[0]['people'][0].get('pose_keypoints_2d', [])
                    if not keypoints: return 0.0
                    right_arm_len = calculate_limb_length(keypoints, 2, 3) + calculate_limb_length(keypoints, 3, 4)
                    left_arm_len = calculate_limb_length(keypoints, 5, 6) + calculate_limb_length(keypoints, 6, 7)
                    return max(left_arm_len, right_arm_len)
                except (IndexError, TypeError): return 0.0

            target_arm_len = get_max_arm_length(target_pose_keypoint_obj)
            source_arm_len = get_max_arm_length(source_pose_obj)

            if source_arm_len > 0 and target_arm_len > 0:
                final_arm_scale = arm_scale * (target_arm_len / source_arm_len)

            # --- 다리 길이 계산 ---
            def get_max_leg_length(pose_obj):
                try:
                    if not isinstance(pose_obj, list) or not pose_obj or 'people' not in pose_obj[0] or not pose_obj[0]['people']: return 0.0
                    keypoints = pose_obj[0]['people'][0].get('pose_keypoints_2d', [])
                    if not keypoints: return 0.0
                    right_leg_len = calculate_limb_length(keypoints, 8, 9) + calculate_limb_length(keypoints, 9, 10)
                    left_leg_len = calculate_limb_length(keypoints, 11, 12) + calculate_limb_length(keypoints, 12, 13)
                    return max(left_leg_len, right_leg_len)
                except (IndexError, TypeError): return 0.0
            
            target_leg_len = get_max_leg_length(target_pose_keypoint_obj)
            source_leg_len = get_max_leg_length(source_pose_obj)

            if source_leg_len > 0 and target_leg_len > 0:
                final_leg_scale = leg_scale * (target_leg_len / source_leg_len)
            
            # --- 어깨 너비 계산 ---
            def get_shoulder_width(pose_obj):
                try:
                    if not isinstance(pose_obj, list) or not pose_obj or 'people' not in pose_obj[0] or not pose_obj[0]['people']: return 0.0
                    keypoints = pose_obj[0]['people'][0].get('pose_keypoints_2d', [])
                    if not keypoints: return 0.0
                    width = calculate_limb_length(keypoints, 2, 5)
                    return width
                except (IndexError, TypeError): return 0.0

            target_shoulder_width = get_shoulder_width(target_pose_keypoint_obj)
            source_shoulder_width = get_shoulder_width(source_pose_obj)
            
            if source_shoulder_width > 0 and target_shoulder_width > 0:
                final_shoulder_scale = shoulder_scale * (target_shoulder_width / source_shoulder_width)

            # --- 골반 너비 계산 ---
            def get_pelvis_width(pose_obj):
                try:
                    if not isinstance(pose_obj, list) or not pose_obj or 'people' not in pose_obj[0] or not pose_obj[0]['people']: return 0.0
                    keypoints = pose_obj[0]['people'][0].get('pose_keypoints_2d', [])
                    if not keypoints: return 0.0
                    width = calculate_limb_length(keypoints, 8, 11)
                    return width
                except (IndexError, TypeError): return 0.0
            
            target_pelvis_width = get_pelvis_width(target_pose_keypoint_obj)
            source_pelvis_width = get_pelvis_width(source_pose_obj)

            if source_pelvis_width > 0 and target_pelvis_width > 0:
                final_pelvis_scale = pelvis_scale * (target_pelvis_width / source_pelvis_width)
            
            # --- 목 길이 계산 ---
            def get_neck_length(pose_obj):
                try:
                    if not isinstance(pose_obj, list) or not pose_obj or 'people' not in pose_obj[0] or not pose_obj[0]['people']: return 0.0
                    keypoints = pose_obj[0]['people'][0].get('pose_keypoints_2d', [])
                    if not keypoints: return 0.0
                    length = calculate_limb_length(keypoints, 1, 0)
                    return length
                except (IndexError, TypeError): return 0.0

            target_neck_length = get_neck_length(target_pose_keypoint_obj)
            source_neck_length = get_neck_length(source_pose_obj)

            if source_neck_length > 0 and target_neck_length > 0:
                final_neck_scale = neck_scale * (target_neck_length / source_neck_length)
                
            # --- 머리 크기 계산 ---
            def get_head_size(pose_obj):
                try:
                    if not isinstance(pose_obj, list) or not pose_obj or 'people' not in pose_obj[0] or not pose_obj[0]['people']: return 0.0
                    keypoints = pose_obj[0]['people'][0].get('pose_keypoints_2d', [])
                    if not keypoints: return 0.0
                    
                    head_indices = [0, 14, 15, 16, 17]
                    valid_points = [p for i in head_indices if (p := get_point(keypoints, i)) is not None]

                    if len(valid_points) < 3: return 0.0

                    points_for_hull = np.array(valid_points, dtype=np.float32).reshape((-1, 1, 2))
                    hull = cv2.convexHull(points_for_hull)
                    area = cv2.contourArea(hull)
                    return area
                except (IndexError, TypeError): return 0.0

            target_head_size = get_head_size(target_pose_keypoint_obj)
            source_head_size = get_head_size(source_pose_obj)

            if source_head_size > 0 and target_head_size > 0:
                size_ratio = math.sqrt(target_head_size / source_head_size)
                final_head_scale = head_scale * size_ratio

            # --- 몸통 길이 계산 ---
            def get_torso_length(pose_obj):
                try:
                    if not isinstance(pose_obj, list) or not pose_obj or 'people' not in pose_obj[0] or not pose_obj[0]['people']: return 0.0
                    keypoints = pose_obj[0]['people'][0].get('pose_keypoints_2d', [])
                    if not keypoints: return 0.0
                    
                    right_hip = get_point(keypoints, 8)
                    left_hip = get_point(keypoints, 11)
                    neck = get_point(keypoints, 1)

                    if right_hip is None or left_hip is None or neck is None: return 0.0
                    
                    hip_midpoint = (right_hip + left_hip) / 2.0
                    length = np.linalg.norm(neck - hip_midpoint)
                    return length
                except (IndexError, TypeError): return 0.0
            
            target_torso_length = get_torso_length(target_pose_keypoint_obj)
            source_torso_length = get_torso_length(source_pose_obj)
            
            if source_torso_length > 0 and target_torso_length > 0:
                final_torso_scale = torso_scale * (target_torso_length / source_torso_length)
                
            # --- 손 크기 계산 (새로 추가된 부분) ---
            def get_hand_area(hand_kps_list):
                if not hand_kps_list: return 0.0
                valid_points = []
                for i in range(0, len(hand_kps_list), 3):
                    if hand_kps_list[i+2] > 0:
                        valid_points.append([hand_kps_list[i], hand_kps_list[i+1]])
                if len(valid_points) < 3: return 0.0
                points_for_hull = np.array(valid_points, dtype=np.float32).reshape((-1, 1, 2))
                hull = cv2.convexHull(points_for_hull)
                return cv2.contourArea(hull)

            def get_max_hand_size(pose_obj):
                try:
                    if not isinstance(pose_obj, list) or not pose_obj or 'people' not in pose_obj[0] or not pose_obj[0]['people']: return 0.0
                    person = pose_obj[0]['people'][0]
                    left_hand_kps = person.get('hand_left_keypoints_2d', [])
                    right_hand_kps = person.get('hand_right_keypoints_2d', [])
                    left_area = get_hand_area(left_hand_kps)
                    right_area = get_hand_area(right_hand_kps)
                    return max(left_area, right_area)
                except(IndexError, TypeError): return 0.0

            target_hand_size = get_max_hand_size(target_pose_keypoint_obj)
            source_hand_size = get_max_hand_size(source_pose_obj)

            if source_hand_size > 0 and target_hand_size > 0:
                size_ratio = math.sqrt(target_hand_size / source_hand_size)
                final_hands_scale = hands_scale * size_ratio


        except (json.JSONDecodeError, IndexError, TypeError):
            # 에러 발생 시 원래 값 유지
            final_hands_scale = hands_scale
            final_torso_scale = torso_scale
            final_head_scale = head_scale
            final_neck_scale = neck_scale
            final_pelvis_scale = pelvis_scale
            final_shoulder_scale = shoulder_scale
            final_arm_scale = arm_scale
            final_leg_scale = leg_scale

    pose_imgs = []
    all_frames_keypoints_output = []

    if pose_json_str:
        images_data_list = json.loads(pose_json_str)
        if not isinstance(images_data_list, list): images_data_list = [images_data_list]

        pbar = ProgressBar(len(images_data_list))
        
        KP = {
            "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
            "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
            "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
            "LEye": 15, "REar": 16, "LEar": 17
        }
        
        FACE_KP_GROUPS_INDICES = {
            "Left_Eye": [42, 43, 44, 45, 46, 47, 69],
            "Right_Eye": [36, 37, 38, 39, 40, 41, 68],
            "Left_Eyebrow": [22, 23, 24, 25, 26],
            "Right_Eyebrow": [17, 18, 19, 20, 21],
            "Mouth": list(range(48, 68)),
            "Nose_Face": list(range(27, 36)),
            "Face_Shape": list(range(0, 17))
        }
        
        INDIVIDUAL_FACE_SCALES = {
            "Left_Eye": left_eye_scale, "Right_Eye": right_eye_scale,
            "Left_Eyebrow": left_eyebrow_scale, "Right_Eyebrow": right_eyebrow_scale,
            "Mouth": mouth_scale, "Nose_Face": nose_scale_face,
            "Face_Shape": face_shape_scale
        }

        BODY_HEAD_PARTS = {KP["REye"], KP["LEye"], KP["REar"], KP["LEar"]}
        R_LEG_INDICES = {KP["RKnee"], KP["RAnkle"]}
        L_LEG_INDICES = {KP["LKnee"], KP["LAnkle"]}
        FEET_INDICES = {KP["RAnkle"], KP["LAnkle"]}

        for image_data in images_data_list:
            if 'people' not in image_data or not image_data['people']:
                pbar.update(1); continue
            
            figures = image_data['people']
            H = image_data['canvas_height']
            W = image_data['canvas_width']
            
            current_image_people_data_for_output = []
            all_scaled_candidates_for_drawing, all_scaled_faces_for_drawing, all_scaled_hands_for_drawing = [], [], []
            final_subset_for_drawing = [[]] 
            
            for fig_idx, figure in enumerate(figures):
                body_raw, face_raw, lhand_raw, rhand_raw = [figure.get(k, []) for k in ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']]

                if not body_raw or len(body_raw) < (KP["LEar"] + 1) * 3: continue
                
                initial_candidate = np.array([body_raw[i:i+2] for i in range(0, len(body_raw), 3)])
                confidence_scores_body = [body_raw[i*3+2] for i in range(len(initial_candidate))]
                scaled_candidate_np = initial_candidate.copy()

                r_hip_orig, l_hip_orig = initial_candidate[KP["RHip"]], initial_candidate[KP["LHip"]]
                hip_center_orig = (r_hip_orig + l_hip_orig) / 2 
                neck_orig, r_shoulder_orig, l_shoulder_orig, nose_orig = [initial_candidate[KP[k]] for k in ["Neck", "RShoulder", "LShoulder", "Nose"]]
                lwrist_orig, rwrist_orig = initial_candidate[KP["LWrist"]], initial_candidate[KP["RWrist"]]

                r_hip_final = scale(r_hip_orig, final_pelvis_scale, hip_center_orig)
                l_hip_final = scale(l_hip_orig, final_pelvis_scale, hip_center_orig)
                scaled_candidate_np[KP["RHip"]], scaled_candidate_np[KP["LHip"]] = r_hip_final, l_hip_final
                
                hip_center_final = (r_hip_final + l_hip_final) / 2
                neck_final = scale(neck_orig, final_torso_scale, hip_center_final)
                scaled_candidate_np[KP["Neck"]] = neck_final

                scaled_candidate_np[KP["RShoulder"]] = neck_final + (r_shoulder_orig - neck_orig) * final_shoulder_scale
                scaled_candidate_np[KP["LShoulder"]] = neck_final + (l_shoulder_orig - neck_orig) * final_shoulder_scale
                
                r_shoulder_final, l_shoulder_final = scaled_candidate_np[KP["RShoulder"]], scaled_candidate_np[KP["LShoulder"]]
                
                for i in [KP["RElbow"], KP["RWrist"]]: scaled_candidate_np[i] = r_shoulder_final + (initial_candidate[i] - r_shoulder_orig) * final_arm_scale
                for i in [KP["LElbow"], KP["LWrist"]]: scaled_candidate_np[i] = l_shoulder_final + (initial_candidate[i] - l_shoulder_orig) * final_arm_scale
                
                for i in R_LEG_INDICES: scaled_candidate_np[i] = r_hip_final + (initial_candidate[i] - r_hip_orig) * final_leg_scale
                for i in L_LEG_INDICES: scaled_candidate_np[i] = l_hip_final + (initial_candidate[i] - l_hip_orig) * final_leg_scale
                
                nose_body_final = neck_final + (nose_orig - neck_orig) * final_neck_scale
                scaled_candidate_np[KP["Nose"]] = nose_body_final
                
                effective_nose_translation = nose_body_final - nose_orig
                for i in BODY_HEAD_PARTS:
                    part_moved_with_nose = initial_candidate[i] + effective_nose_translation
                    scaled_candidate_np[i] = scale(part_moved_with_nose, final_head_scale, nose_body_final)

                face_points_scaled_current_fig = []
                if face_raw:
                    face_points_orig = [np.array(face_raw[i:i+2]) for i in range(0, len(face_raw), 3)]
                    num_face_points = len(face_points_orig)
                    
                    face_points_positioned = [p + effective_nose_translation for p in face_points_orig]
                    face_points_after_global_head_scale = [scale(p, final_head_scale, nose_body_final) for p in face_points_positioned]
                    face_points_scaled_current_fig = list(face_points_after_global_head_scale)

                    reye_pos_after_head_scale = scale(initial_candidate[KP["REye"]] + effective_nose_translation, final_head_scale, nose_body_final)
                    leye_pos_after_head_scale = scale(initial_candidate[KP["LEye"]] + effective_nose_translation, final_head_scale, nose_body_final)
                    eye_center = (reye_pos_after_head_scale + leye_pos_after_head_scale) / 2
                    reye_pos_after_dist_scale = scale(reye_pos_after_head_scale, eye_distance_scale, eye_center)
                    leye_pos_after_dist_scale = scale(leye_pos_after_head_scale, eye_distance_scale, eye_center)
                    right_dist_translation = reye_pos_after_dist_scale - reye_pos_after_head_scale
                    left_dist_translation = leye_pos_after_dist_scale - leye_pos_after_head_scale

                    eye_height_offset = np.array([0.0, 0.0])
                    eyebrow_height_offset = np.array([0.0, 0.0])
                    direction_vector = nose_body_final - neck_final
                    norm_direction = np.linalg.norm(direction_vector)
                    if norm_direction > eps:
                        unit_direction = direction_vector / norm_direction
                        if abs(eye_height) > eps: eye_height_offset = unit_direction * eye_height
                        if abs(eyebrow_height) > eps: eyebrow_height_offset = unit_direction * eyebrow_height
                    
                    group_translations = {
                        "Right_Eye": right_dist_translation + eye_height_offset,
                        "Left_Eye": left_dist_translation + eye_height_offset,
                        "Right_Eyebrow": right_dist_translation + eyebrow_height_offset,
                        "Left_Eyebrow": left_dist_translation + eyebrow_height_offset,
                    }

                    scaled_candidate_np[KP["REye"]] = reye_pos_after_dist_scale + eye_height_offset
                    scaled_candidate_np[KP["LEye"]] = leye_pos_after_dist_scale + eye_height_offset
                    
                    for group_name, indices in FACE_KP_GROUPS_INDICES.items():
                        group_scale_modifier = INDIVIDUAL_FACE_SCALES.get(group_name, 1.0)
                        valid_indices = [idx for idx in indices if idx < num_face_points]
                        if not valid_indices: continue

                        points_after_head_scale = [face_points_after_global_head_scale[idx] for idx in valid_indices]

                        if group_name in group_translations:
                            points_after_translation = [p + group_translations[group_name] for p in points_after_head_scale]
                        else:
                            points_after_translation = points_after_head_scale

                        if abs(group_scale_modifier - 1.0) > eps:
                            if group_name == "Face_Shape":
                                pivot = nose_body_final
                                direction_vector = neck_final - nose_body_final
                                norm_direction = np.linalg.norm(direction_vector)
                                if norm_direction > eps:
                                    unit_direction = direction_vector / norm_direction
                                    final_points = []
                                    for p in points_after_translation:
                                        point_vector = p - pivot
                                        proj_length = np.dot(point_vector, unit_direction)
                                        parallel_component = proj_length * unit_direction
                                        perpendicular_component = point_vector - parallel_component
                                        scaled_parallel_component = parallel_component * group_scale_modifier
                                        new_point = pivot + scaled_parallel_component + perpendicular_component
                                        final_points.append(new_point)
                                else:
                                    final_points = points_after_translation
                            else:
                                pivot = np.mean(points_after_translation, axis=0)
                                final_points = [scale(p, group_scale_modifier, pivot) for p in points_after_translation]
                        else:
                            final_points = points_after_translation

                        for i, idx in enumerate(valid_indices):
                            face_points_scaled_current_fig[idx] = final_points[i]
                
                lwrist_final_calc, rwrist_final_calc = scaled_candidate_np[KP["LWrist"]], scaled_candidate_np[KP["RWrist"]]
                
                # hands_scale 대신 계산된 final_hands_scale을 사용하도록 수정
                lhand_scaled_current_fig = [(scale(np.array(lhand_raw[i:i+2]), final_hands_scale, lwrist_orig) + (lwrist_final_calc - lwrist_orig)) if lhand_raw[i+2] > 0 else np.array([0.0, 0.0]) for i in range(0, len(lhand_raw), 3)] if lhand_raw else []
                rhand_scaled_current_fig = [(scale(np.array(rhand_raw[i:i+2]), final_hands_scale, rwrist_orig) + (rwrist_final_calc - rwrist_orig)) if rhand_raw[i+2] > 0 else np.array([0.0, 0.0]) for i in range(0, len(rhand_raw), 3)] if rhand_raw else []

                scales_to_check = [leg_scale, torso_scale, overall_scale, pelvis_scale, head_scale] 
                is_scaling_active = any(abs(s - 1.0) > 0.001 for s in scales_to_check)
                
                candidate_list_current_fig_np = scaled_candidate_np
                face_list_current_fig_np = np.array(face_points_scaled_current_fig) if face_points_scaled_current_fig else np.array([])
                lhand_list_current_fig_np = np.array(lhand_scaled_current_fig) if lhand_scaled_current_fig else np.array([])
                rhand_list_current_fig_np = np.array(rhand_scaled_current_fig) if rhand_scaled_current_fig else np.array([])
                
                if use_ground_plane and is_scaling_active:
                    ground_y_coord = H
                    orig_feet_coords = [initial_candidate[i] for i in FEET_INDICES if i < len(initial_candidate)]
                    orig_lowest_y = max(p[1] for p in orig_feet_coords) if orig_feet_coords else H
                    orig_dist_to_ground = ground_y_coord - orig_lowest_y

                    feet_coords_for_overall_pivot = [candidate_list_current_fig_np[i] for i in FEET_INDICES if i < len(candidate_list_current_fig_np)]
                    
                    if feet_coords_for_overall_pivot:
                        feet_pos_pivot = np.mean(feet_coords_for_overall_pivot, axis=0)
                        candidate_list_current_fig_np = np.array([scale(p, overall_scale, feet_pos_pivot) for p in candidate_list_current_fig_np])
                        if face_list_current_fig_np.size > 0: face_list_current_fig_np = np.array([scale(p, overall_scale, feet_pos_pivot) for p in face_list_current_fig_np])
                        if lhand_list_current_fig_np.size > 0: lhand_list_current_fig_np = np.array([scale(p, overall_scale, feet_pos_pivot) if np.sum(np.abs(p)) > eps else p for p in lhand_list_current_fig_np])
                        if rhand_list_current_fig_np.size > 0: rhand_list_current_fig_np = np.array([scale(p, overall_scale, feet_pos_pivot) if np.sum(np.abs(p)) > eps else p for p in rhand_list_current_fig_np])

                        final_feet_coords = [candidate_list_current_fig_np[i] for i in FEET_INDICES if i < len(candidate_list_current_fig_np)]
                        if final_feet_coords:
                            final_lowest_y = max(p[1] for p in final_feet_coords)
                            desired_final_y = ground_y_coord - orig_dist_to_ground
                            vertical_translation = desired_final_y - final_lowest_y
                            
                            candidate_list_current_fig_np = candidate_list_current_fig_np + np.array([0, vertical_translation])
                            if face_list_current_fig_np.size > 0: face_list_current_fig_np = face_list_current_fig_np + np.array([0, vertical_translation])
                            if lhand_list_current_fig_np.size > 0: lhand_list_current_fig_np = lhand_list_current_fig_np + np.array([0, vertical_translation])
                            if rhand_list_current_fig_np.size > 0: rhand_list_current_fig_np = rhand_list_current_fig_np + np.array([0, vertical_translation])
                else: 
                    center_pivot = [W * 0.5, H * 0.5]
                    candidate_list_current_fig_np = np.array([scale(p, overall_scale, center_pivot) for p in candidate_list_current_fig_np])
                    if face_list_current_fig_np.size > 0: face_list_current_fig_np = np.array([scale(p, overall_scale, center_pivot) for p in face_list_current_fig_np])
                    if lhand_list_current_fig_np.size > 0: lhand_list_current_fig_np = np.array([scale(p, overall_scale, center_pivot) if np.sum(np.abs(p)) > eps else p for p in lhand_list_current_fig_np])
                    if rhand_list_current_fig_np.size > 0: rhand_list_current_fig_np = np.array([scale(p, overall_scale, center_pivot) if np.sum(np.abs(p)) > eps else p for p in rhand_list_current_fig_np])
                
                # Rotation Logic (after all scaling, before translation)
                if abs(rotate_angle) > eps: # Only rotate if angle is significant
                    all_points_for_rotation_center = []
                    if candidate_list_current_fig_np.size > 0:
                        # 유효한 body 포인트만 중심 계산에 사용 (confidence 기반으로 필터링하는 것이 더 정확할 수 있으나, 여기서는 모든 점 사용)
                        all_points_for_rotation_center.extend(candidate_list_current_fig_np.tolist())
                    if face_list_current_fig_np.size > 0:
                        all_points_for_rotation_center.extend(face_list_current_fig_np.tolist())
                    
                    # 손 포인트 중 [0,0]이 아닌 유효한 포인트만 중심 계산에 사용
                    if lhand_list_current_fig_np.size > 0:
                        valid_lhand_points = [p.tolist() for p in lhand_list_current_fig_np if np.sum(np.abs(p)) > eps]
                        if valid_lhand_points:
                            all_points_for_rotation_center.extend(valid_lhand_points)
                    if rhand_list_current_fig_np.size > 0:
                        valid_rhand_points = [p.tolist() for p in rhand_list_current_fig_np if np.sum(np.abs(p)) > eps]
                        if valid_rhand_points:
                            all_points_for_rotation_center.extend(valid_rhand_points)

                    if all_points_for_rotation_center:
                        points_for_center_np = np.array(all_points_for_rotation_center)
                        center_x = np.mean(points_for_center_np[:, 0])
                        center_y = np.mean(points_for_center_np[:, 1])

                        angle_rad = math.radians(rotate_angle)
                        cos_a = math.cos(angle_rad)
                        sin_a = math.sin(angle_rad)

                        def apply_rotation_to_points(points_np, cx, cy, c_angle, s_angle):
                            if points_np.size == 0:
                                return points_np
                            
                            # 회전 적용할 포인트만 선택 (예: [0,0] 제외는 여기서 처리 안함, 모든 점 동일하게 회전)
                            # 원본 포인트를 복사하여 사용
                            rotated_points = points_np.copy()
                            
                            # 중심점으로 이동
                            translated_x = rotated_points[:, 0] - cx
                            translated_y = rotated_points[:, 1] - cy
                            
                            # 회전
                            rotated_x = translated_x * c_angle - translated_y * s_angle
                            rotated_y = translated_x * s_angle + translated_y * c_angle
                            
                            # 다시 원래 위치로 이동 (중심점 기준)
                            rotated_points[:, 0] = rotated_x + cx
                            rotated_points[:, 1] = rotated_y + cy
                            return rotated_points

                        if candidate_list_current_fig_np.size > 0:
                            candidate_list_current_fig_np = apply_rotation_to_points(candidate_list_current_fig_np, center_x, center_y, cos_a, sin_a)
                        if face_list_current_fig_np.size > 0:
                            face_list_current_fig_np = apply_rotation_to_points(face_list_current_fig_np, center_x, center_y, cos_a, sin_a)
                        if lhand_list_current_fig_np.size > 0:
                            # [0,0] 점들도 회전 중심에 대해 상대적으로 회전됨
                            lhand_list_current_fig_np = apply_rotation_to_points(lhand_list_current_fig_np, center_x, center_y, cos_a, sin_a)
                        if rhand_list_current_fig_np.size > 0:
                            # [0,0] 점들도 회전 중심에 대해 상대적으로 회전됨
                            rhand_list_current_fig_np = apply_rotation_to_points(rhand_list_current_fig_np, center_x, center_y, cos_a, sin_a)
                
                
                if abs(translate_x) > eps or abs(translate_y) > eps: # 실제로 이동이 필요한 경우에만 연산
                    translation_vector = np.array([translate_x, translate_y], dtype=np.float32)

                    if candidate_list_current_fig_np.size > 0:
                        candidate_list_current_fig_np = candidate_list_current_fig_np + translation_vector
                    
                    if face_list_current_fig_np.size > 0:
                        face_list_current_fig_np = face_list_current_fig_np + translation_vector
                    
                    if lhand_list_current_fig_np.size > 0:
                        lhand_list_current_fig_np = lhand_list_current_fig_np + translation_vector
                    
                    if rhand_list_current_fig_np.size > 0:
                        rhand_list_current_fig_np = rhand_list_current_fig_np + translation_vector
                
                
                body_kps_out_current_fig = [item for i, p in enumerate(candidate_list_current_fig_np) for item in [p[0], p[1], confidence_scores_body[i]]]
                face_kps_out_current_fig = [item for p in face_list_current_fig_np for item in [p[0], p[1], 1.0]] if face_list_current_fig_np.size > 0 else []
                
                original_lhand_confidences = [lhand_raw[i+2] for i in range(0, len(lhand_raw), 3)] if lhand_raw else []
                original_rhand_confidences = [rhand_raw[i+2] for i in range(0, len(rhand_raw), 3)] if rhand_raw else []
                
                lhand_kps_out_current_fig = [item for i, p in enumerate(lhand_list_current_fig_np) for item in [p[0], p[1], original_lhand_confidences[i]]] if lhand_list_current_fig_np.size > 0 else []
                rhand_kps_out_current_fig = [item for i, p in enumerate(rhand_list_current_fig_np) for item in [p[0], p[1], original_rhand_confidences[i]]] if rhand_list_current_fig_np.size > 0 else []
                
                current_image_people_data_for_output.append({
                    "pose_keypoints_2d": body_kps_out_current_fig, "face_keypoints_2d": face_kps_out_current_fig,
                    "hand_left_keypoints_2d": lhand_kps_out_current_fig, "hand_right_keypoints_2d": rhand_kps_out_current_fig,
                })

                all_scaled_candidates_for_drawing.extend(candidate_list_current_fig_np.tolist())
                if face_list_current_fig_np.size > 0: all_scaled_faces_for_drawing.extend(face_list_current_fig_np.tolist())
                if lhand_list_current_fig_np.size > 0: all_scaled_hands_for_drawing.append(lhand_list_current_fig_np.tolist())
                if rhand_list_current_fig_np.size > 0: all_scaled_hands_for_drawing.append(rhand_list_current_fig_np.tolist())

                if fig_idx == 0 and not final_subset_for_drawing[0]:
                    final_subset_for_drawing[0].extend([i if body_raw[i*3+2]>0 else -1 for i in range(len(candidate_list_current_fig_np))])
                else:
                    prev_candidate_count = len(all_scaled_candidates_for_drawing) - len(candidate_list_current_fig_np)
                    final_subset_for_drawing.append([prev_candidate_count+i if body_raw[i*3+2]>0 else -1 for i in range(len(candidate_list_current_fig_np))])
            
            current_frame_keypoint_object = { "people": current_image_people_data_for_output, "canvas_width": W, "canvas_height": H }
            all_frames_keypoints_output.append(current_frame_keypoint_object)
            
            candidate_norm, faces_norm = all_scaled_candidates_for_drawing, all_scaled_faces_for_drawing
            hands_norm_for_drawing = all_scaled_hands_for_drawing 
            
            if candidate_norm:
                candidate_np_norm = np.array(candidate_norm).astype(float); candidate_np_norm[...,0] /= float(W); candidate_np_norm[...,1] /= float(H)
                candidate_norm = candidate_np_norm.tolist()
            if faces_norm: 
                faces_np_norm = np.array(faces_norm).astype(float); 
                if faces_np_norm.size > 0: faces_np_norm[...,0] /= float(W); faces_np_norm[...,1] /= float(H)
                faces_norm = faces_np_norm.tolist()
            
            hands_final_norm_for_drawing = []
            if hands_norm_for_drawing: 
                for hand_kps_list in hands_norm_for_drawing:
                    current_normalized_hand = []
                    for point_list in hand_kps_list: 
                        if not isinstance(point_list, (list, np.ndarray)) or len(point_list) != 2: continue 
                        norm_point = np.array(point_list).astype(float)
                        if norm_point[0] > eps or norm_point[1] > eps:
                            norm_point[0] /= float(W)
                            norm_point[1] /= float(H)
                        current_normalized_hand.append(norm_point.tolist())
                    if current_normalized_hand : hands_final_norm_for_drawing.append(current_normalized_hand)
            
            bodies = dict(candidate=candidate_norm, subset=final_subset_for_drawing)
            original_face_exists = any(fig.get('face_keypoints_2d') for fig in figures)
            original_lhand_exists = any(fig.get('hand_left_keypoints_2d') for fig in figures)
            original_rhand_exists = any(fig.get('hand_right_keypoints_2d') for fig in figures)

            pose = dict(
                bodies=bodies if show_body else {'candidate':[], 'subset':[]}, 
                faces=faces_norm if show_face and original_face_exists else [], 
                hands=hands_final_norm_for_drawing if show_hands and (original_lhand_exists or original_rhand_exists) else []
            )
            W_scaled = resolution_x if resolution_x >= 64 else W
            H_scaled = int(H*(W_scaled*1.0/W))
            pose_imgs.append(draw_pose(pose, H_scaled, W_scaled, pose_marker_size, face_marker_size, hand_marker_size))
            pbar.update(1)

    return pose_imgs, all_frames_keypoints_output

def draw_pose(pose, H, W, pose_marker_size, face_marker_size, hand_marker_size):
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    body_render_info = pose.get('bodies', {})
    candidate = body_render_info.get('candidate', [])
    subset = body_render_info.get('subset', [])
    faces_data = pose.get('faces', []) 
    hands_data = pose.get('hands', [])

    if candidate and subset and np.array(candidate).size > 0 : canvas = draw_bodypose(canvas, np.array(candidate), np.array(subset), pose_marker_size)
    if hands_data and np.array(hands_data).size > 0 : canvas = draw_handpose(canvas, hands_data, hand_marker_size)
    if faces_data and np.array(faces_data).size > 0 : canvas = draw_facepose(canvas, faces_data, face_marker_size)
    return canvas

def draw_bodypose(canvas, candidate, subset, pose_marker_size):
    H, W, C = canvas.shape
    limbSeq = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]]
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    if candidate.ndim != 2 or candidate.shape[1] != 2: return canvas 
    for i in range(len(limbSeq)):
        for n in range(len(subset)):
            limb = limbSeq[i]
            if max(limb) >= subset.shape[1]: continue
            index = subset[n][np.array(limb)].astype(int)
            if -1 in index or max(index) >= len(candidate): continue
            Y, X = candidate[index, 0] * float(W), candidate[index, 1] * float(H)
            mX, mY = np.mean(X), np.mean(Y)
            length = np.linalg.norm(np.array([X[0], Y[0]]) - np.array([X[1], Y[1]]))
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            if length < 1: continue
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), pose_marker_size), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i % len(colors)])
    for n in range(len(subset)):
        for i in range(subset.shape[1]): 
            index = int(subset[n][i])
            if index == -1 or index >= len(candidate): continue
            x, y = candidate[index][0:2]
            x, y = int(x * W), int(y * H)
            cv2.circle(canvas, (x, y), pose_marker_size, colors[i % len(colors)], thickness=-1)
    return canvas

def draw_handpose(canvas, all_hand_peaks, hand_marker_size):
    H, W, C = canvas.shape
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    for peaks_list_for_one_hand in all_hand_peaks:
        peaks_np = np.array(peaks_list_for_one_hand)
        if peaks_np.ndim != 2 or peaks_np.shape[1] != 2: continue
        for ie, e in enumerate(edges):
            if e[0] >= len(peaks_np) or e[1] >= len(peaks_np): continue
            x1_coord, y1_coord = peaks_np[e[0]] 
            x2_coord, y2_coord = peaks_np[e[1]]
            if x1_coord < eps and y1_coord < eps or x2_coord < eps and y2_coord < eps: continue
            x1, y1 = int(x1_coord * W), int(y1_coord * H)
            x2, y2 = int(x2_coord * W), int(y2_coord * H)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(canvas, (x1, y1), (x2, y2), matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255, thickness=max(1, hand_marker_size))
        for i, keyponit in enumerate(peaks_np):
            x_coord, y_coord = keyponit
            x, y = int(x_coord * W), int(y_coord * H)
            if x > eps and y > eps: cv2.circle(canvas, (x, y), max(1, hand_marker_size) + 1, (0, 0, 255), thickness=-1)
    return canvas

def draw_facepose(canvas, all_lmks, face_marker_size):
    H, W, C = canvas.shape
    lmks_np = np.array(all_lmks) 
    if lmks_np.ndim != 2 or lmks_np.shape[1] != 2: return canvas
    for lmk in lmks_np:
        x_coord, y_coord = lmk
        x, y = int(x_coord * W), int(y_coord * H)
        if x > eps and y > eps: cv2.circle(canvas, (x, y), face_marker_size, (255, 255, 255), thickness=-1)
    return canvas
