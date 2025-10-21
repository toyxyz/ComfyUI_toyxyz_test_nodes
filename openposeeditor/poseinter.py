import torch
import cv2
import numpy as np
import copy
from matplotlib.colors import hsv_to_rgb
import json

DEFAULT_BODY_LIMB_THICKNESS = 6
DEFAULT_BODY_POINT_RADIUS = 5
DEFAULT_HAND_LIMB_THICKNESS = 2
DEFAULT_HAND_POINT_RADIUS = 3
DEFAULT_FACE_POINT_RADIUS = 2

# --- 스켈레톤, 색상, KP 딕셔너리 정의 ---
body_colors = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
    [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
    [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]
]
face_color = [255, 255, 255]
hand_keypoint_color = [0, 0, 255]
hand_limb_colors = [
    [255,0,0],[255,60,0],[255,120,0],[255,180,0], [180,255,0],[120,255,0],[60,255,0],[0,255,0],
    [0,255,60],[0,255,120],[0,255,180],[0,180,255], [0,120,255],[0,60,255],[0,0,255],[60,0,255],
    [120,0,255],[180,0,255],[255,0,180],[255,0,120]
]
body_skeleton = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
    [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]
]
face_skeleton = []
hand_skeleton = [
    [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]
]

KP = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "MidHip": 8, "RHip": 9,
    "RKnee": 10, "RAnkle": 11, "LHip": 12, "LKnee": 13, "LAnkle": 14,
    "REye": 15, "LEye": 16, "REar": 17, "LEar": 18, "LBigToe": 19,
    "LSmallToe": 20, "LHeel": 21, "RBigToe": 22, "RSmallToe": 23, "RHeel": 24
}

def calculate_bone_length(kps, p1_idx, p2_idx):
    if kps.shape[0] <= max(p1_idx, p2_idx): return 0.0
    if kps[p1_idx, 2] == 0 or kps[p2_idx, 2] == 0: return 0.0
    p1 = kps[p1_idx, :2]
    p2 = kps[p2_idx, :2]
    return np.linalg.norm(p1 - p2)

def get_valid_kps_coords(kps_np, confidence_threshold=0.1):
    if kps_np is None or kps_np.ndim != 2 or kps_np.shape[1] != 3 or kps_np.size == 0: return None
    valid_points = kps_np[kps_np[:, 2] > confidence_threshold][:, :2]
    return valid_points if valid_points.shape[0] > 0 else None

def get_bounding_box_area_and_center(kps_np, confidence_threshold=0.1):
    if kps_np is None or kps_np.ndim != 2 or kps_np.shape[1] != 3 or kps_np.size == 0:
        return 0.0, None
    valid_points_xy = []
    for i in range(kps_np.shape[0]):
        if kps_np[i, 2] > confidence_threshold:
            valid_points_xy.append(kps_np[i, :2])
    if not valid_points_xy or len(valid_points_xy) < 1:
        return 0.0, None
    valid_points_xy = np.array(valid_points_xy)
    min_x, min_y = np.min(valid_points_xy, axis=0)
    max_x, max_y = np.max(valid_points_xy, axis=0)
    width, height = max_x - min_x, max_y - min_y
    area = 0.0
    if valid_points_xy.shape[0] >=2 :
        area = width * height if width > 1e-6 and height > 1e-6 else 0.0
    center = (np.mean(valid_points_xy[:, 0]), np.mean(valid_points_xy[:, 1]))
    return area, center

def adjust_pose_to_reference_size(source_kps, ref_kps, confidence_threshold=0.1):
    if source_kps.size == 0 or ref_kps.size == 0:
        return source_kps

    adjusted_kps = source_kps.copy()

    RShoulder, LShoulder, RHip, LHip, Neck = KP["RShoulder"], KP["LShoulder"], KP["RHip"], KP["LHip"], KP["Neck"]
    required_indices = [RShoulder, LShoulder, RHip, LHip, Neck]
    if not all(idx < source_kps.shape[0] and source_kps[idx, 2] > confidence_threshold for idx in required_indices) or \
       not all(idx < ref_kps.shape[0] and ref_kps[idx, 2] > confidence_threshold for idx in required_indices):
        pass 
    else:
        src_shoulder_width = np.linalg.norm(source_kps[LShoulder, :2] - source_kps[RShoulder, :2])
        src_shoulder_center = 0.5 * (source_kps[LShoulder, :2] + source_kps[RShoulder, :2])
        src_hip_center = 0.5 * (source_kps[LHip, :2] + source_kps[RHip, :2])
        src_torso_height = np.linalg.norm(src_shoulder_center - src_hip_center)
        ref_shoulder_width = np.linalg.norm(ref_kps[LShoulder, :2] - ref_kps[RShoulder, :2])
        ref_shoulder_center = 0.5 * (ref_kps[LShoulder, :2] + ref_kps[RShoulder, :2])
        ref_hip_center = 0.5 * (ref_kps[LHip, :2] + ref_kps[RHip, :2])
        ref_torso_height = np.linalg.norm(ref_shoulder_center - ref_hip_center)
        x_ratio = ref_shoulder_width / src_shoulder_width if src_shoulder_width > 1e-6 else 1.0
        y_ratio = ref_torso_height / src_torso_height if src_torso_height > 1e-6 else 1.0
        neck_pos = adjusted_kps[Neck, :2].copy()
        for i in range(adjusted_kps.shape[0]):
            if adjusted_kps[i, 2] > 0: 
                vec_from_neck = adjusted_kps[i, :2] - neck_pos
                vec_from_neck[0] *= x_ratio
                vec_from_neck[1] *= y_ratio
                adjusted_kps[i, :2] = neck_pos + vec_from_neck

    bones_to_adjust = [
        (KP["Neck"], KP["Nose"], [KP["Nose"], KP["REye"], KP["LEye"], KP["REar"], KP["LEar"]]),
        (KP["RShoulder"], KP["RElbow"], [KP["RElbow"], KP["RWrist"]]),
        (KP["RElbow"], KP["RWrist"], [KP["RWrist"]]),
        (KP["LShoulder"], KP["LElbow"], [KP["LElbow"], KP["LWrist"]]),
        (KP["LElbow"], KP["LWrist"], [KP["LWrist"]]),
        (KP["RHip"], KP["RKnee"], [KP["RKnee"], KP["RAnkle"], KP["RBigToe"], KP["RSmallToe"], KP["RHeel"]]),
        (KP["RKnee"], KP["RAnkle"], [KP["RAnkle"], KP["RBigToe"], KP["RSmallToe"], KP["RHeel"]]),
        (KP["LHip"], KP["LKnee"], [KP["LKnee"], KP["LAnkle"], KP["LBigToe"], KP["LSmallToe"], KP["LHeel"]]),
        (KP["LKnee"], KP["LAnkle"], [KP["LAnkle"], KP["LBigToe"], KP["LSmallToe"], KP["LHeel"]]),
    ]
    for parent_idx, child_idx, children_indices in bones_to_adjust:
        if max(parent_idx, child_idx) >= adjusted_kps.shape[0] or max(parent_idx, child_idx) >= ref_kps.shape[0]: continue
        if adjusted_kps[parent_idx, 2] < confidence_threshold or adjusted_kps[child_idx, 2] < confidence_threshold or \
           ref_kps[parent_idx, 2] < confidence_threshold or ref_kps[child_idx, 2] < confidence_threshold:
            continue
        len_source = calculate_bone_length(adjusted_kps, parent_idx, child_idx)
        len_ref = calculate_bone_length(ref_kps, parent_idx, child_idx)
        if len_source == 0 or len_ref == 0: continue
        ratio = len_ref / len_source
        if abs(1.0 - ratio) < 0.01: continue 
        parent_pos = adjusted_kps[parent_idx, :2]
        child_pos_old = adjusted_kps[child_idx, :2]
        vector = child_pos_old - parent_pos
        vector_new = vector * ratio
        child_pos_new = parent_pos + vector_new
        offset = child_pos_new - child_pos_old
        for idx in children_indices:
            if idx < adjusted_kps.shape[0] and adjusted_kps[idx, 2] > 0: 
                adjusted_kps[idx, :2] += offset

    if Neck < adjusted_kps.shape[0] and Neck < ref_kps.shape[0] and \
       adjusted_kps[Neck, 2] > confidence_threshold and ref_kps[Neck, 2] > confidence_threshold:
        final_offset = ref_kps[Neck, :2] - adjusted_kps[Neck, :2]
        for i in range(adjusted_kps.shape[0]):
            if adjusted_kps[i, 2] > 0: 
                adjusted_kps[i, :2] += final_offset

    head_kp_indices = [KP["Nose"], KP["REye"], KP["LEye"], KP["REar"], KP["LEar"]]
    ref_head_points = np.array([ref_kps[i] for i in head_kp_indices if i < ref_kps.shape[0] and ref_kps[i, 2] > confidence_threshold])
    adj_head_points = np.array([adjusted_kps[i] for i in head_kp_indices if i < adjusted_kps.shape[0] and adjusted_kps[i, 2] > confidence_threshold])
    if ref_head_points.shape[0] >= 2 and adj_head_points.shape[0] >= 2: 
        ref_head_area, _ = get_bounding_box_area_and_center(ref_head_points, confidence_threshold)
        adj_head_area, adj_head_center = get_bounding_box_area_and_center(adj_head_points, confidence_threshold)
        if adj_head_area > 1e-6 and ref_head_area > 1e-6 and adj_head_center is not None:
            scale_factor_head = np.sqrt(ref_head_area / adj_head_area)
            if abs(1.0 - scale_factor_head) > 0.01: 
                center_x, center_y = adj_head_center
                for kp_idx in head_kp_indices:
                    if kp_idx < adjusted_kps.shape[0] and adjusted_kps[kp_idx, 2] > confidence_threshold:
                        x, y, _ = adjusted_kps[kp_idx]
                        adjusted_kps[kp_idx, 0] = center_x + (x - center_x) * scale_factor_head
                        adjusted_kps[kp_idx, 1] = center_y + (y - center_y) * scale_factor_head
    return adjusted_kps

def adjust_face_keypoints_size(full_face_kps_np, scale_factor, center_xy, confidence_threshold=0.1):
    adjusted_kps = full_face_kps_np.copy()
    if center_xy is None : return adjusted_kps
    center_x, center_y = center_xy
    for i in range(adjusted_kps.shape[0]):
        x, y, conf = adjusted_kps[i]
        if conf > confidence_threshold:
            adjusted_kps[i, :2] = (center_x + (x - center_x) * scale_factor, center_y + (y - center_y) * scale_factor)
    return adjusted_kps

def adjust_face_to_maintain_relative_offset(orig_target_body_kps, orig_target_face_kps, adjusted_body_kps, adjusted_face_kps, confidence_threshold=0.1):
    FACE_NOSE_INDEX = 8 
    if not (orig_target_body_kps.shape[0] > KP["Nose"] and orig_target_body_kps[KP["Nose"], 2] >= confidence_threshold and
            orig_target_face_kps.shape[0] > FACE_NOSE_INDEX and orig_target_face_kps[FACE_NOSE_INDEX, 2] >= confidence_threshold and
            adjusted_body_kps.shape[0] > KP["Nose"] and adjusted_body_kps[KP["Nose"], 2] >= confidence_threshold and
            adjusted_face_kps.shape[0] > FACE_NOSE_INDEX and adjusted_face_kps[FACE_NOSE_INDEX, 2] >= confidence_threshold):
        return adjusted_face_kps 

    orig_body_nose_pos = orig_target_body_kps[KP["Nose"], :2]
    orig_face_nose_pos = orig_target_face_kps[FACE_NOSE_INDEX, :2]
    original_offset = orig_face_nose_pos - orig_body_nose_pos

    adjusted_body_nose_pos = adjusted_body_kps[KP["Nose"], :2]
    current_face_nose_pos = adjusted_face_kps[FACE_NOSE_INDEX, :2] 
    
    desired_face_nose_pos = adjusted_body_nose_pos + original_offset
    translation_vector = desired_face_nose_pos - current_face_nose_pos
    
    final_face_kps = adjusted_face_kps.copy() 
    for i in range(final_face_kps.shape[0]):
        if final_face_kps[i, 2] > confidence_threshold:
            final_face_kps[i, :2] += translation_vector
    return final_face_kps

def calculate_hand_intrinsic_properties(hand_kps_np, confidence_threshold=0.1):
    if hand_kps_np is None or hand_kps_np.size == 0: return None
    hand_kp0_abs = None
    if hand_kps_np.shape[0] > 0 and hand_kps_np[0, 2] > confidence_threshold:
        hand_kp0_abs = hand_kps_np[0, :2].copy()
    
    valid_hand_coords_xy = get_valid_kps_coords(hand_kps_np, confidence_threshold)
    scale = 0.0
    if valid_hand_coords_xy is not None and valid_hand_coords_xy.shape[0] >= 2:
        min_x, min_y = np.min(valid_hand_coords_xy, axis=0)
        max_x, max_y = np.max(valid_hand_coords_xy, axis=0)
        width, height = max_x - min_x, max_y - min_y
        scale = np.sqrt(width**2 + height**2) if width > 1e-6 and height > 1e-6 else 0.0
    return {'scale': scale, 'kp0_abs': hand_kp0_abs}

def transform_hand_final(target_hand_kps_np_orig, target_body_wrist_pos_xy_adj, ref_hand_kps_np_orig, ref_body_wrist_pos_xy_orig, confidence_threshold=0.1):
    ref_props = calculate_hand_intrinsic_properties(ref_hand_kps_np_orig, confidence_threshold)
    target_orig_props = calculate_hand_intrinsic_properties(target_hand_kps_np_orig, confidence_threshold)

    if not all([ref_props, target_orig_props, 
                ref_props.get('kp0_abs') is not None, target_orig_props.get('kp0_abs') is not None, 
                ref_body_wrist_pos_xy_orig is not None, target_body_wrist_pos_xy_adj is not None]):
        return target_hand_kps_np_orig.flatten().tolist() if target_hand_kps_np_orig is not None and target_hand_kps_np_orig.size > 0 else []

    ref_hand_kp0_pos, ref_scale = ref_props['kp0_abs'], ref_props['scale']
    target_orig_hand_kp0_pos, target_orig_scale = target_orig_props['kp0_abs'], target_orig_props['scale']

    ref_offset_bodywrist_to_handkp0 = ref_hand_kp0_pos - ref_body_wrist_pos_xy_orig
    scale_factor = ref_scale / target_orig_scale if target_orig_scale > 1e-6 else 1.0

    scaled_target_hand_kps = target_hand_kps_np_orig.copy()
    pivot_for_scaling = target_orig_hand_kp0_pos.copy() 

    for i in range(scaled_target_hand_kps.shape[0]):
        if scaled_target_hand_kps[i, 2] > confidence_threshold:
            vec_from_pivot = scaled_target_hand_kps[i, :2] - pivot_for_scaling
            scaled_target_hand_kps[i, :2] = pivot_for_scaling + (vec_from_pivot * scale_factor)

    if scaled_target_hand_kps.shape[0] == 0 or scaled_target_hand_kps[0, 2] <= confidence_threshold:
        return scaled_target_hand_kps.flatten().tolist()

    current_abs_pos_of_scaled_target_hand_kp0 = scaled_target_hand_kps[0, :2]
    desired_abs_pos_for_target_hand_kp0 = target_body_wrist_pos_xy_adj + ref_offset_bodywrist_to_handkp0
    translation_vector = desired_abs_pos_for_target_hand_kp0 - current_abs_pos_of_scaled_target_hand_kp0

    for i in range(scaled_target_hand_kps.shape[0]):
        if scaled_target_hand_kps[i, 2] > confidence_threshold:
            scaled_target_hand_kps[i, :2] += translation_vector
            
    return scaled_target_hand_kps.flatten().tolist()

def draw_keypoints_and_skeleton(image, keypoints_data, skeleton_connections, 
                                colors_config, 
                                limb_thickness, 
                                point_radius,   
                                confidence_threshold=0.1,
                                is_body=False, is_face=False, is_hand=False,
                                hand_edges_count=0): 
    if not keypoints_data or len(keypoints_data) % 3 != 0: return 
    tri_tuples = [keypoints_data[i:i + 3] for i in range(0, len(keypoints_data), 3)]

    if skeleton_connections:
        for i, (joint_idx_a, joint_idx_b) in enumerate(skeleton_connections):
            if joint_idx_a >= len(tri_tuples) or joint_idx_b >= len(tri_tuples): continue
            
            a_x_f, a_y_f, a_confidence = tri_tuples[joint_idx_a]
            b_x_f, b_y_f, b_confidence = tri_tuples[joint_idx_b]

            if a_confidence >= confidence_threshold and b_confidence >= confidence_threshold:
                a_x, a_y = int(round(a_x_f)), int(round(a_y_f)) 
                b_x, b_y = int(round(b_x_f)), int(round(b_y_f))

                current_limb_color = None 
                if is_body:
                    current_limb_color = tuple(colors_config[i % len(colors_config)]) 
                elif is_hand:
                    if hand_edges_count > 0: 
                        rgb_color = hsv_to_rgb([i / float(hand_edges_count), 1.0, 1.0])
                        current_limb_color = tuple((np.array(rgb_color) * 255).astype(np.uint8).tolist()) 
                    else: 
                        current_limb_color = tuple(colors_config[i % len(colors_config)]) 
                elif is_face: 
                    if skeleton_connections: 
                        current_limb_color = tuple(colors_config) 

                if current_limb_color is not None:
                    if is_body: 
                        center_x, center_y = (a_x + b_x) // 2, (a_y + b_y) // 2
                        length = np.linalg.norm(np.array([a_x, a_y]) - np.array([b_x, b_y]))
                        if length >= 1: 
                            angle_rad = np.arctan2(b_y - a_y, b_x - a_x)
                            angle_deg = np.degrees(angle_rad)
                            ellipse_major_axis = max(1, int(length / 2))
                            ellipse_minor_axis = max(1, int(limb_thickness / 2))
                            axes = (ellipse_major_axis, ellipse_minor_axis)
                            polygon_points = cv2.ellipse2Poly((center_x, center_y), axes, int(angle_deg), 0, 360, 10)
                            cv2.fillConvexPoly(image, polygon_points, current_limb_color) 
                    else: 
                         cv2.line(image, (a_x, a_y), (b_x, b_y), current_limb_color, limb_thickness) 

    for i, (x_f, y_f, confidence) in enumerate(tri_tuples):
        if confidence >= confidence_threshold:
            current_point_color = None
            if is_body:
                current_point_color = body_colors[i % len(body_colors)] 
            elif is_hand:
                current_point_color = hand_keypoint_color 
            elif is_face:
                current_point_color = face_color 
            
            if current_point_color:
                 cv2.circle(image, (int(round(x_f)), int(round(y_f))), point_radius, tuple(current_point_color), -1) 

def gen_skeleton_with_face_hands(pose_keypoints_2d, face_keypoints_2d, hand_left_keypoints_2d, hand_right_keypoints_2d, 
                                 canvas_width, canvas_height, landmarkType, confidence_threshold=0.1):
    image = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    def scale_keypoints(keypoints, target_w, target_h, input_is_normalized):
        if not keypoints or len(keypoints) % 3 != 0 : return []
        scaled = []
        for i in range(0, len(keypoints), 3):
            x, y, conf = keypoints[i:i+3]
            scaled.extend([x * target_w, y * target_h, conf] if input_is_normalized else [x, y, conf])
        return scaled
        
    input_normalized = (landmarkType == "OpenPose") 

    scaled_pose = scale_keypoints(pose_keypoints_2d, canvas_width, canvas_height, input_normalized)
    scaled_face = scale_keypoints(face_keypoints_2d, canvas_width, canvas_height, input_normalized)
    scaled_hand_left = scale_keypoints(hand_left_keypoints_2d, canvas_width, canvas_height, input_normalized)
    scaled_hand_right = scale_keypoints(hand_right_keypoints_2d, canvas_width, canvas_height, input_normalized)

    draw_keypoints_and_skeleton(image, scaled_pose, body_skeleton, body_colors, 
                                DEFAULT_BODY_LIMB_THICKNESS, DEFAULT_BODY_POINT_RADIUS, 
                                confidence_threshold, is_body=True)
    
    if scaled_face: 
        draw_keypoints_and_skeleton(image, scaled_face, face_skeleton, face_color, 
                                    0, DEFAULT_FACE_POINT_RADIUS, 
                                    confidence_threshold, is_face=True)
    
    if scaled_hand_left: 
        draw_keypoints_and_skeleton(image, scaled_hand_left, hand_skeleton, hand_limb_colors, 
                                    DEFAULT_HAND_LIMB_THICKNESS, DEFAULT_HAND_POINT_RADIUS,
                                    confidence_threshold, is_hand=True, hand_edges_count=len(hand_skeleton))
    if scaled_hand_right: 
        draw_keypoints_and_skeleton(image, scaled_hand_right, hand_skeleton, hand_limb_colors, 
                                    DEFAULT_HAND_LIMB_THICKNESS, DEFAULT_HAND_POINT_RADIUS,
                                    confidence_threshold, is_hand=True, hand_edges_count=len(hand_skeleton))
    return image

def transform_all_keypoints(keypoints_1, keypoints_2, frames, interpolation="linear"):
    def interpolate_keypoint_set(kp1, kp2, num_frames, interp_method):
        kp1 = kp1 if kp1 is not None else []
        kp2 = kp2 if kp2 is not None else []

        if not kp1 and not kp2: return [[] for _ in range(num_frames)]
        
        len_kp1 = len(kp1)
        len_kp2 = len(kp2)

        if len_kp1 == 0 and len_kp2 > 0:
            kp1 = [0.0] * len_kp2
        elif len_kp2 == 0 and len_kp1 > 0:
            kp2 = [0.0] * len_kp1
        elif len_kp1 != len_kp2 :
            print(f"Warning: Keypoint list length mismatch. kp1 len: {len_kp1}, kp2 len: {len_kp2}. Interpolation might be unreliable.")
            max_len = max(len_kp1, len_kp2)
            if max_len % 3 != 0 : 
                 print(f"Error: Max keypoint list length {max_len} is not a multiple of 3. Returning empty interpolation.")
                 return [[] for _ in range(num_frames)]
            while len(kp1) < max_len: kp1.extend([0.0, 0.0, 0.0])
            while len(kp2) < max_len: kp2.extend([0.0, 0.0, 0.0])
        
        if not kp1 and not kp2: return [[] for _ in range(num_frames)] 

        num_kps1 = len(kp1) // 3
        num_kps2 = len(kp2) // 3

        if num_kps1 != num_kps2: 
             print(f"Critical Error: Mismatch in number of keypoints after padding. KPs1: {num_kps1}, KPs2: {num_kps2}")
             return [[] for _ in range(num_frames)]
        if num_kps1 == 0 : return [[] for _ in range(num_frames)]

        tri_tuples_1 = [kp1[i:i + 3] for i in range(0, len(kp1), 3)]
        tri_tuples_2 = [kp2[i:i + 3] for i in range(0, len(kp2), 3)]
        
        keypoints_sequence = []
        for j in range(num_frames):
            interpolated_kps_for_frame = []
            t = j / float(num_frames - 1) if num_frames > 1 else 0.0 
            
            if interp_method == "ease-in": interp_factor = t * t
            elif interp_method == "ease-out": interp_factor = 1 - (1 - t) * (1 - t)
            elif interp_method == "ease-in-out":
                interp_factor = 4 * t * t * t if t < 0.5 else 1.0 - pow(-2 * t + 2, 3) / 2 
            else: interp_factor = t 

            for i in range(num_kps1): 
                x1, y1, c1 = tri_tuples_1[i]; x2, y2, c2 = tri_tuples_2[i]
                new_x, new_y, new_c = 0.0, 0.0, 0.0
                
                if c1 > 0 and c2 > 0: 
                    new_x = x1 + (x2 - x1) * interp_factor
                    new_y = y1 + (y2 - y1) * interp_factor
                    new_c = c1 + (c2 - c1) * interp_factor 
                elif c1 > 0: 
                    new_x, new_y = x1, y1
                    new_c = c1 * (1.0 - interp_factor) 
                elif c2 > 0: 
                    new_x, new_y = x2, y2
                    new_c = c2 * interp_factor 
                interpolated_kps_for_frame.extend([new_x, new_y, new_c])
            keypoints_sequence.append(interpolated_kps_for_frame)
        return keypoints_sequence

    parts = ['pose', 'face', 'hand_left', 'hand_right']
    sequences = {}
    for part in parts:
        kp1_part = keypoints_1.get(f'{part}_keypoints_2d', [])
        kp2_part = keypoints_2.get(f'{part}_keypoints_2d', [])
        sequences[part] = interpolate_keypoint_set(kp1_part, kp2_part, frames, interpolation)
        
    combined_sequence = []
    for i in range(frames):
        combined_frame_data = {}
        valid_frame = False
        for part in parts:
            if i < len(sequences[part]) and sequences[part][i]: 
                combined_frame_data[f'{part}_keypoints_2d'] = sequences[part][i]
                valid_frame = True
            else: 
                combined_frame_data[f'{part}_keypoints_2d'] = []
        if valid_frame: 
            combined_sequence.append(combined_frame_data)
    return combined_sequence

def apply_confidence_threshold(keypoints_list, threshold):
    if not keypoints_list:
        return []
    filtered_kps = []
    for i in range(0, len(keypoints_list), 3):
        x, y, c = keypoints_list[i:i+3]
        if c < threshold:
            filtered_kps.extend([0.0, 0.0, 0.0]) 
        else:
            filtered_kps.extend([x, y, c])
    return filtered_kps

class Pose_Inter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        interpolation_methods = ["linear", "ease-in", "ease-out", "ease-in-out"]
        return {
            "required": {
                "pose_from": ("POSE_KEYPOINT", ), "pose_to": ("POSE_KEYPOINT", ),
                "interpolate_frames": ("INT", {"default": 12, "min": 1, "max": 99999, "step": 1}), 
                "interpolation": (interpolation_methods, {"default": "linear"}),
                "confidence_threshold": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}), 
                "adjust_body_shape": ("BOOLEAN", {"default": False}),
                "landmarkType": (["OpenPose", "DWPose"], {"default": "DWPose"}), 
                "include_face": ("BOOLEAN", {"default": True}),
                "include_hands": ("BOOLEAN", {"default": True}),
                "pick_frame": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}), # Allow negative for pick_frame list items
            },
        }

    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT",)
    RETURN_NAMES = ("image", "pose_keypoint",)
    FUNCTION = "run"
    CATEGORY = "ToyxyzTestNodes"

    def run(self, pose_from, pose_to, interpolate_frames, interpolation, confidence_threshold, landmarkType, include_face, include_hands, adjust_body_shape, pick_frame):
        if not pose_from or not pose_to:
            raise ValueError("Input 'pose_from' or 'pose_to' data is empty.")

        pose_from_list = pose_from if isinstance(pose_from, list) else [pose_from]
        pose_to_list = pose_to if isinstance(pose_to, list) else [pose_to]

        if not pose_from_list or not pose_to_list:
             raise ValueError("Input 'pose_from' or 'pose_to' list is empty after ensuring it's a list.")

        if len(pose_from_list) != len(pose_to_list):
            raise ValueError(f"Batch size mismatch: 'pose_from' has {len(pose_from_list)} items, 'pose_to' has {len(pose_to_list)} items. They must be equal.")

        batch_size = len(pose_from_list)
        
        pick_rules_for_each_item_in_batch = []
        if isinstance(pick_frame, list):
            # If pick_frame is a list, it applies per batch item if batch_size > 1
            # Or, if batch_size is 1, this list applies to that single item.
            if batch_size > 1 and len(pick_frame) != batch_size:
                 raise ValueError(f"'pick_frame' list has {len(pick_frame)} items, but batch size is {batch_size}. They must be equal if 'pick_frame' is a list for multi-item batch processing where each list item corresponds to a batch item.")
            for i in range(batch_size):
                if batch_size == 1:
                    # If batch is 1, the entire pick_frame list is the rule for this one item
                    pick_rules_for_each_item_in_batch.append(pick_frame)
                else:
                    # If batch > 1, then pick_frame[i] is the rule for pose_from_list[i]
                    # Ensure it's a list, even if it's a single int from the input pick_frame list
                    rule = pick_frame[i]
                    pick_rules_for_each_item_in_batch.append([rule] if isinstance(rule, int) else rule)
        elif isinstance(pick_frame, int):
            # If pick_frame is a single int, this int (wrapped in a list) becomes the rule for all batch items.
            pick_rules_for_each_item_in_batch = [[pick_frame]] * batch_size
        else:
            raise TypeError("'pick_frame' must be an INT or a LIST of INTs/LISTs.")


        final_output_images = []
        final_output_poses = []

        default_canvas_width = 512 
        default_canvas_height = 512
        if batch_size > 0 and pose_from_list[0] and isinstance(pose_from_list[0], dict):
            default_canvas_width = pose_from_list[0].get("canvas_width", default_canvas_width)
            default_canvas_height = pose_from_list[0].get("canvas_height", default_canvas_height)


        for i in range(batch_size):
            current_pose_from_dict = pose_from_list[i]
            current_pose_to_dict = pose_to_list[i]
            current_pick_rule_list_for_item = pick_rules_for_each_item_in_batch[i] 

            if not isinstance(current_pose_from_dict, dict) or "people" not in current_pose_from_dict or not current_pose_from_dict["people"]:
                print(f"Warning: Invalid or no people data in 'pose_from' for batch item {i}. Skipping.")
                continue
            if not isinstance(current_pose_to_dict, dict) or "people" not in current_pose_to_dict or not current_pose_to_dict["people"]:
                print(f"Warning: Invalid or no people data in 'pose_to' for batch item {i}. Skipping.")
                continue

            person_from = current_pose_from_dict["people"][0]
            person_to = current_pose_to_dict["people"][0]
            
            keypoints_from_current_pair = {
                'pose_keypoints_2d': person_from.get("pose_keypoints_2d", []),
                'face_keypoints_2d': person_from.get("face_keypoints_2d", []) if include_face else [],
                'hand_left_keypoints_2d': person_from.get("hand_left_keypoints_2d", []) if include_hands else [],
                'hand_right_keypoints_2d': person_from.get("hand_right_keypoints_2d", []) if include_hands else []
            }
            keypoints_to_current_pair = {
                'pose_keypoints_2d': person_to.get("pose_keypoints_2d", []),
                'face_keypoints_2d': person_to.get("face_keypoints_2d", []) if include_face else [],
                'hand_left_keypoints_2d': person_to.get("hand_left_keypoints_2d", []) if include_hands else [],
                'hand_right_keypoints_2d': person_to.get("hand_right_keypoints_2d", []) if include_hands else []
            }

            original_person_to_face_kps_current = person_to.get("face_keypoints_2d", []) if include_face else []
            original_person_to_hand_left_kps_current = person_to.get("hand_left_keypoints_2d", []) if include_hands else []
            original_person_to_hand_right_kps_current = person_to.get("hand_right_keypoints_2d", []) if include_hands else []

            kps_from_np_body = np.array(keypoints_from_current_pair['pose_keypoints_2d']).reshape(-1, 3) if keypoints_from_current_pair['pose_keypoints_2d'] else np.array([])
            kps_to_np_body_for_adjustment = np.array(keypoints_to_current_pair['pose_keypoints_2d']).reshape(-1, 3) if keypoints_to_current_pair['pose_keypoints_2d'] else np.array([])
            kps_to_np_body_final_for_interp = kps_to_np_body_for_adjustment.copy()

            if adjust_body_shape:
                temp_kps_to_body = kps_to_np_body_for_adjustment.copy() 
                if kps_from_np_body.size > 0 and temp_kps_to_body.size > 0:
                    adjusted_body_kps_intermediate = adjust_pose_to_reference_size(temp_kps_to_body, kps_from_np_body, confidence_threshold)
                    keypoints_to_current_pair['pose_keypoints_2d'] = adjusted_body_kps_intermediate.flatten().tolist()
                    kps_to_np_body_final_for_interp = adjusted_body_kps_intermediate 
                
                if include_face:
                    face_kps_from_np = np.array(keypoints_from_current_pair['face_keypoints_2d']).reshape(-1, 3) if keypoints_from_current_pair['face_keypoints_2d'] else np.array([])
                    face_kps_to_np_orig = np.array(original_person_to_face_kps_current).reshape(-1, 3) if original_person_to_face_kps_current else np.array([])
                    
                    if face_kps_from_np.size > 0 and face_kps_to_np_orig.size > 0:
                        scaled_face_kps_to = face_kps_to_np_orig.copy()
                        area_from, _ = get_bounding_box_area_and_center(face_kps_from_np, confidence_threshold)
                        area_to_orig, center_to_face_orig = get_bounding_box_area_and_center(face_kps_to_np_orig, confidence_threshold)
                        if area_to_orig > 1e-6 and area_from > 1e-6 and center_to_face_orig is not None:
                            scale_factor = np.sqrt(area_from / area_to_orig)
                            scaled_face_kps_to = adjust_face_keypoints_size(face_kps_to_np_orig, scale_factor, center_to_face_orig, confidence_threshold)
                        
                        final_adjusted_face_kps = adjust_face_to_maintain_relative_offset(
                            kps_to_np_body_for_adjustment, 
                            face_kps_to_np_orig,           
                            kps_to_np_body_final_for_interp, 
                            scaled_face_kps_to,            
                            confidence_threshold
                        )
                        keypoints_to_current_pair['face_keypoints_2d'] = final_adjusted_face_kps.flatten().tolist()

                if include_hands:
                    for hand_type in ["left", "right"]:
                        wrist_kp_name = "LWrist" if hand_type == "left" else "RWrist"
                        ref_body_wrist_pos_xy_orig, target_body_wrist_pos_xy_adj = None, None
                        
                        if kps_from_np_body.size > 0 and KP[wrist_kp_name] < kps_from_np_body.shape[0] and kps_from_np_body[KP[wrist_kp_name], 2] > confidence_threshold:
                            ref_body_wrist_pos_xy_orig = kps_from_np_body[KP[wrist_kp_name], :2]
                        
                        if kps_to_np_body_final_for_interp.size > 0 and KP[wrist_kp_name] < kps_to_np_body_final_for_interp.shape[0] and kps_to_np_body_final_for_interp[KP[wrist_kp_name], 2] > confidence_threshold:
                            target_body_wrist_pos_xy_adj = kps_to_np_body_final_for_interp[KP[wrist_kp_name], :2]

                        ref_hand_kps_list = keypoints_from_current_pair.get(f'hand_{hand_type}_keypoints_2d', [])
                        ref_hand_kps_np_orig = np.array(ref_hand_kps_list).reshape(-1, 3) if ref_hand_kps_list and len(ref_hand_kps_list) % 3 == 0 else np.array([])
                        
                        target_hand_kps_list_orig_current = original_person_to_hand_left_kps_current if hand_type == "left" else original_person_to_hand_right_kps_current
                        target_hand_kps_np_orig_current = np.array(target_hand_kps_list_orig_current).reshape(-1, 3) if target_hand_kps_list_orig_current and len(target_hand_kps_list_orig_current) % 3 == 0 else np.array([])

                        if ref_body_wrist_pos_xy_orig is not None and target_body_wrist_pos_xy_adj is not None and ref_hand_kps_np_orig.size > 0 and target_hand_kps_np_orig_current.size > 0:
                            adjusted_hand_kps_list = transform_hand_final(
                                target_hand_kps_np_orig_current, 
                                target_body_wrist_pos_xy_adj,    
                                ref_hand_kps_np_orig,            
                                ref_body_wrist_pos_xy_orig,      
                                confidence_threshold)
                            keypoints_to_current_pair[f'hand_{hand_type}_keypoints_2d'] = adjusted_hand_kps_list
            
            interpolated_sequence_for_current_pair = transform_all_keypoints(
                keypoints_from_current_pair, 
                keypoints_to_current_pair, 
                interpolate_frames, 
                interpolation
            )

            frames_to_render_this_item = []
            if not interpolated_sequence_for_current_pair:
                 print(f"Warning: Interpolation failed for batch item {i}. No frames to pick.")
            else:
                num_available_frames = len(interpolated_sequence_for_current_pair)
                if num_available_frames == 0:
                    print(f"Warning: Interpolation resulted in zero frames for batch item {i}.")
                else:
                    for pick_value in current_pick_rule_list_for_item: 
                        if not isinstance(pick_value, int):
                            print(f"Warning: Invalid non-integer pick_value '{pick_value}' for batch item {i}. Skipping this pick_value.")
                            continue

                        target_frame_num_1_based = pick_value
                        
                        if target_frame_num_1_based == 0: 
                            if interpolate_frames > 0 : 
                                frames_to_render_this_item.extend(interpolated_sequence_for_current_pair)
                        else:
                            if target_frame_num_1_based < 1:
                                target_frame_num_1_based = 1 # Clamp to 1st frame
                            if target_frame_num_1_based > num_available_frames:
                                target_frame_num_1_based = num_available_frames # Clamp to last frame
                            
                            frames_to_render_this_item.append(interpolated_sequence_for_current_pair[target_frame_num_1_based - 1])
            
            # 중복 프레임 제거 (선택적: 만약 [0, 1] 같은 rule로 인해 중복이 생기는 것을 방지하고 싶다면)
            # 이 경우, 추가된 순서가 중요하지 않다면 set으로 변환 후 list로 다시 만들 수 있지만,
            # 여기서는 사용자가 명시적으로 여러 번 같은 프레임을 요청할 수도 있으므로 중복 제거 안 함.

            canvas_width_current = current_pose_from_dict.get("canvas_width", default_canvas_width)
            canvas_height_current = current_pose_from_dict.get("canvas_height", default_canvas_height)

            for frame_data in frames_to_render_this_item: # 이미 선택/조정된 프레임 데이터 목록
                pose_output_for_this_frame = copy.deepcopy(current_pose_from_dict) 

                pose_kps_final = apply_confidence_threshold(frame_data.get('pose_keypoints_2d', []), confidence_threshold)
                face_kps_final = apply_confidence_threshold(frame_data.get('face_keypoints_2d', []), confidence_threshold)
                hand_left_kps_final = apply_confidence_threshold(frame_data.get('hand_left_keypoints_2d', []), confidence_threshold)
                hand_right_kps_final = apply_confidence_threshold(frame_data.get('hand_right_keypoints_2d', []), confidence_threshold)

                if "people" in pose_output_for_this_frame and pose_output_for_this_frame["people"]:
                    pose_output_for_this_frame["people"][0]["pose_keypoints_2d"] = pose_kps_final
                    pose_output_for_this_frame["people"][0]["face_keypoints_2d"] = face_kps_final
                    pose_output_for_this_frame["people"][0]["hand_left_keypoints_2d"] = hand_left_kps_final
                    pose_output_for_this_frame["people"][0]["hand_right_keypoints_2d"] = hand_right_kps_final
                
                pose_output_for_this_frame["canvas_width"] = canvas_width_current
                pose_output_for_this_frame["canvas_height"] = canvas_height_current
                
                final_output_poses.append(pose_output_for_this_frame)

                image_np = gen_skeleton_with_face_hands(
                    pose_kps_final,
                    face_kps_final,
                    hand_left_kps_final,
                    hand_right_kps_final,
                    canvas_width_current, 
                    canvas_height_current,
                    landmarkType,
                    confidence_threshold 
                )
                image_tensor = torch.from_numpy(image_np.astype(np.float32) / 255.0)
                final_output_images.append(image_tensor)

        if not final_output_images: 
            print("Warning: No images were generated across all batch items. Returning a single black image.")
            black_image_np = np.zeros((default_canvas_height, default_canvas_width, 3), dtype=np.float32)
            return (torch.from_numpy(black_image_np).unsqueeze(0), []) 

        return (torch.stack(final_output_images), final_output_poses)

class PoseKeypointToCoordStr: #
    def __init__(self): #
        pass #

    @classmethod
    def INPUT_TYPES(cls): #
        return { #
            "required": { #
                "pose_keypoint": ("POSE_KEYPOINT",), #
                "enable_body": ("BOOLEAN", {"default": True, "label_on": "Body Enabled", "label_off": "Body Disabled"}),
                "enable_face": ("BOOLEAN", {"default": True, "label_on": "Face Enabled", "label_off": "Face Disabled"}),
                "enable_hand": ("BOOLEAN", {"default": True, "label_on": "Hands Enabled", "label_off": "Hands Disabled"}),
                "enable_extra_points": ("BOOLEAN", {"default": False, "label_on": "Extra Body Points Enabled", "label_off": "Extra Body Points Disabled"}),
                "num_extra_points_per_bone": ("INT", {"default": 5, "min": 0, "max": 1000, "step": 1, "label": "Extra Points Per Bone"}),
            }
        }

    RETURN_TYPES = ("STRING",) # ComfyUI에서 문자열 리스트를 담는 단일 슬롯으로 처리될 수 있음
    RETURN_NAMES = ("coord_str",) #
    FUNCTION = "convert_to_coord_str" #
    CATEGORY = "ToyxyzTestNodes" #

    def convert_to_coord_str(self, pose_keypoint, enable_body, enable_face, enable_hand, enable_extra_points, num_extra_points_per_bone): # num_extra_points_per_bone 파라미터 추가
        if not pose_keypoint:
            return (["[]"],)

        pose_keypoint_list = pose_keypoint if isinstance(pose_keypoint, list) else [pose_keypoint]

        if not pose_keypoint_list:
            return (["[]"],)

        all_poses_kps_extracted = []
        max_kps_count_overall = 0
        
        # NUM_EXTRA_POINTS_PER_BONE 상수를 제거하고 입력 파라미터 사용

        for pose_data_idx, pose_data in enumerate(pose_keypoint_list):
            current_pose_all_coords_dicts_for_frame = []
            if pose_data and "people" in pose_data and pose_data["people"]:
                person_data = pose_data["people"][0]
                
                # Body points 처리
                if enable_body:
                    body_keypoints_flat = person_data.get("pose_keypoints_2d", [])
                    if body_keypoints_flat and len(body_keypoints_flat) > 0:
                        body_kps_triplets = []
                        for i in range(0, len(body_keypoints_flat), 3):
                            x = int(body_keypoints_flat[i])
                            y = int(body_keypoints_flat[i+1])
                            c = body_keypoints_flat[i+2]
                            body_kps_triplets.append({"x": x, "y": y, "c": c})
                            current_pose_all_coords_dicts_for_frame.append({"x": x, "y": y})

                        # enable_extra_points가 True이고, 사용자가 지정한 추가 포인트 수가 0보다 클 경우
                        if enable_extra_points and num_extra_points_per_bone > 0:
                            # body_skeleton은 전역 변수로 가정 (코드 상단에 정의된 것을 사용)
                            for p1_idx, p2_idx in body_skeleton:
                                if p1_idx < len(body_kps_triplets) and p2_idx < len(body_kps_triplets):
                                    p1 = body_kps_triplets[p1_idx]
                                    p2 = body_kps_triplets[p2_idx]

                                    if p1["c"] > 0 and p2["c"] > 0: # 두 원본 포인트의 신뢰도가 유효할 때만
                                        for j in range(1, num_extra_points_per_bone + 1):
                                            ratio = j / float(num_extra_points_per_bone + 1)
                                            extra_x = int(p1["x"] + (p2["x"] - p1["x"]) * ratio)
                                            extra_y = int(p1["y"] + (p2["y"] - p1["y"]) * ratio)
                                            current_pose_all_coords_dicts_for_frame.append({"x": extra_x, "y": extra_y})
                
                # Face points 처리
                if enable_face:
                    face_keypoints_flat = person_data.get("face_keypoints_2d", [])
                    if face_keypoints_flat and len(face_keypoints_flat) > 0:
                        for i in range(0, len(face_keypoints_flat), 3):
                            x = int(face_keypoints_flat[i])
                            y = int(face_keypoints_flat[i+1])
                            current_pose_all_coords_dicts_for_frame.append({"x": x, "y": y})
                
                # Hand points 처리 (left and right)
                if enable_hand:
                    for hand_type_key in ["hand_left_keypoints_2d", "hand_right_keypoints_2d"]:
                        hand_keypoints_flat = person_data.get(hand_type_key, [])
                        if hand_keypoints_flat and len(hand_keypoints_flat) > 0:
                            for i in range(0, len(hand_keypoints_flat), 3):
                                x = int(hand_keypoints_flat[i])
                                y = int(hand_keypoints_flat[i+1])
                                current_pose_all_coords_dicts_for_frame.append({"x": x, "y": y})
            
            all_poses_kps_extracted.append(current_pose_all_coords_dicts_for_frame)
            if len(current_pose_all_coords_dicts_for_frame) > max_kps_count_overall:
                max_kps_count_overall = len(current_pose_all_coords_dicts_for_frame)
        
        if max_kps_count_overall == 0:
             return (["[]"],)

        output_coord_groups_json_str = []
        for i in range(max_kps_count_overall):
            coords_for_this_track = []
            for single_pose_kps_list in all_poses_kps_extracted:
                if i < len(single_pose_kps_list):
                    coords_for_this_track.append(single_pose_kps_list[i])
                else:
                    if coords_for_this_track: 
                        coords_for_this_track.append({"x": coords_for_this_track[-1]["x"], "y": coords_for_this_track[-1]["y"]})
                    else:
                        coords_for_this_track.append({"x": 0, "y": 0})
            
            output_coord_groups_json_str.append(json.dumps(coords_for_this_track))

        return (output_coord_groups_json_str,)

class JoinPose:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_keypoint_1": ("POSE_KEYPOINT",),
                "pose_keypoint_2": ("POSE_KEYPOINT",),
            }
        }

    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_keypoint",)
    FUNCTION = "join_poses"
    CATEGORY = "ToyxyzTestNodes" 

    def join_poses(self, pose_keypoint_1, pose_keypoint_2):

        # 입력이 None일 경우 빈 리스트로 처리하여 오류 방지
        list_1 = pose_keypoint_1 if pose_keypoint_1 is not None else []
        list_2 = pose_keypoint_2 if pose_keypoint_2 is not None else []

        # 두 리스트를 순서대로 합침
        joined_list = list(list_1) + list(list_2)

        return (joined_list,)

