import numpy as np

# MediaPipe landmark numbers: (first point, joint, last point)
ELBOW = {"right": (12, 14, 16), "left": (11, 13, 15)}   # shoulder, elbow, wrist
# KNEE = {"right": (24, 26, 28), "left": (23, 25, 27)}    # hip, knee, ankle


# how one angle is calculated
def angle(a, b, c):
    """Angle at point b, in degrees, between the lines b->a and b->c.
    Works on single points or on arrays of points (last axis = x, y [, z]).
    Returns NaN if any point is NaN or a line has zero length."""
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    # arrow from elbow to shoulder
    u = a - b   
    # arrow from elbow to wrist
    v = c - b

    # Calculate vector lengths
    nu = np.linalg.norm(u, axis=-1)
    nv = np.linalg.norm(v, axis=-1)

    # Calculate angle between vectors using dot product formula
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (u * v).sum(axis=-1) / (nu * nv)      # 0/0 -> NaN
    # cos-->radians-->degrees
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


# applies the same angle calculation to every frame of a video.
def joint_angle(landmarks, joint, size=(640, 480)):
    """landmarks: (frames, 33, 4). joint: three landmark numbers.
    size: (width, height) of the video, so x and y become pixels.
    Returns one angle per frame, shape (frames,)."""
    i, j, k = joint
    scale = np.array(size, float)
    a = landmarks[:, i, :2] * scale
    b = landmarks[:, j, :2] * scale
    c = landmarks[:, k, :2] * scale
    return angle(a, b, c)


def elbow_angle(landmarks, side="right", size=(640, 480)):
    return joint_angle(landmarks, ELBOW[side], size)


# 1. පත්‍රිකාවේ 5 වන පිටුවට අනුකූලව ස්ථාවර සන්ධි අනුපිළිවෙල:
# Index 0: Shoulder (උරහිස), Index 1: Elbow (වැලමිට), Index 2: Wrist (මැණික් කටුව)
UPPER_BODY_ORDER = {
    'right': (12, 14, 16),  # (Right Shoulder, Right Elbow, Right Wrist)
    'left':  (11, 13, 15),  # (Left Shoulder,  Left Elbow,  Left Wrist)
}

def extract_ordered_arm_joints(world_landmarks, side='right', min_bone_len=0.10, max_bone_len=0.55):
    """
    Aguilar-Ortega et al. (Sensors 2023, Page 5 & Page 20) අනුව:
    1. සන්ධි තුන (0: උරහිස, 1: වැලමිට, 2: මැණික් කටුව) ස්ථාවර අනුපිළිවෙලට වෙන් කර ගනී.
    2. වම් සහ දකුණු අත් මාරු වී ඇති (Left-Right Swapped) Frames හඳුනාගෙන නිවැරදි කරයි.
    3. සන්ධි එක මත එක වැටුණු (Collapsed) හෝ අසාමාන්‍ය දුරක් ඇති Frames ඉවත් කරයි.
    """
    sh_idx, el_idx, wr_idx = UPPER_BODY_ORDER[side]
    opp_side = 'left' if side == 'right' else 'right'
    opp_sh_idx, opp_el_idx, opp_wr_idx = UPPER_BODY_ORDER[opp_side]

    # (N_frames, 3, 3) -> [Frame, Joint(0=Sh, 1=El, 2=Wr),XYZ]
    ordered_pts = np.stack([
        world_landmarks[:, sh_idx, :3],  # 0: Shoulder
        world_landmarks[:, el_idx, :3],  # 1: Elbow
        world_landmarks[:, wr_idx, :3],  # 2: Wrist
    ], axis=1).copy()

    # --- පියවර A: Left-Right Limb Swap පරීක්ෂාව (Anti-Swap Check) ---
    # පෙර Frame එකේ වැලමිට තිබූ තැනට වඩා විරුද්ධ අතේ වැලමිට ආසන්න නම් (Swap වී ඇත්නම්) එය ඉවත් කිරීම
    for t in range(1, len(ordered_pts)):
        prev_el = ordered_pts[t - 1, 1]
        curr_el = ordered_pts[t, 1]
        opp_el  = world_landmarks[t, opp_el_idx, :3]

        if np.all(np.isfinite(prev_el)) and np.all(np.isfinite(curr_el)) and np.all(np.isfinite(opp_el)):
            dist_same = np.linalg.norm(curr_el - prev_el)
            dist_opp  = np.linalg.norm(curr_el - opp_el)
            # වැලමිට එකවර 15cm කට වඩා පැන විරුද්ධ අත සමඟ පැටලී ඇත්නම්:
            if dist_same > 0.15 and dist_opp < 0.05:
                ordered_pts[t] = np.nan

    # --- පියවර B: භෞතික අස්ථි පරතරය පරීක්ෂාව (Anatomical Bone Length Gate) ---
    # 0 -> 1 (උඩුබාහුව) සහ 1 -> 2 (යටිබාහුව) අතර දුර 10cm ත් 55cm ත් අතර විය යුතුය
    vec_upper = ordered_pts[:, 0] - ordered_pts[:, 1]  # Shoulder(0) - Elbow(1)
    vec_fore  = ordered_pts[:, 2] - ordered_pts[:, 1]  # Wrist(2)    - Elbow(1)

    len_upper = np.linalg.norm(vec_upper, axis=1)
    len_fore  = np.linalg.norm(vec_fore, axis=1)

    invalid_anatomy = (
        (len_upper < min_bone_len) | (len_upper > max_bone_len) |
        (len_fore  < min_bone_len) | (len_fore  > max_bone_len)
    )
    ordered_pts[invalid_anatomy] = np.nan

    return ordered_pts


def elbow_angle_3d_ordered(world_landmarks, side='right'):
    """
    පත්‍රිකාවේ Page 11, Equation (2) අනුව ස්ථාවර සන්ධි 3 මඟින් 3D කෝණය ගණනය කිරීම.
    """
    pts = extract_ordered_arm_joints(world_landmarks, side=side)
    
    # A = Shoulder (0), B = Elbow (1), C = Wrist (2)
    A = pts[:, 0, :]
    B = pts[:, 1, :]
    C = pts[:, 2, :]

    # දෛශික දෙකම වැලමිටෙන් (B) ආරම්භ විය යුතුය
    BA = A - B
    BC = C - B

    norm_BA = np.linalg.norm(BA, axis=1)
    norm_BC = np.linalg.norm(BC, axis=1)

    # ශුන්‍යයෙන් බෙදීම වැළැක්වීම
    valid = (norm_BA > 1e-6) & (norm_BC > 1e-6) & np.all(np.isfinite(BA), axis=1) & np.all(np.isfinite(BC), axis=1)
    angles = np.full(len(pts), np.nan, dtype=float)

    dot_prod = np.sum(BA[valid] * BC[valid], axis=1)
    cos_angle = np.clip(dot_prod / (norm_BA[valid] * norm_BC[valid]), -1.0, 1.0)
    angles[valid] = np.degrees(np.arccos(cos_angle))

    return angles