import numpy as np


LANDMARK_COUNT = 21
COORDS_PER_LANDMARK = 3
XYZ_FEATURE_COUNT = LANDMARK_COUNT * COORDS_PER_LANDMARK
WRIST_INDEX = 0
FINGERTIP_INDICES = (4, 8, 12, 16, 20)
FINGERTIP_DISTANCE_PAIRS = ((4, 8), (8, 12), (12, 16), (16, 20))
DISTANCE_FEATURE_COUNT = len(FINGERTIP_DISTANCE_PAIRS) + len(FINGERTIP_INDICES)
FINGER_ANGLE_JOINTS = ((5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19))
ANGLE_FEATURE_COUNT = len(FINGER_ANGLE_JOINTS)
FEATURE_COUNT = XYZ_FEATURE_COUNT + DISTANCE_FEATURE_COUNT + ANGLE_FEATURE_COUNT


def _landmarks_to_xyz_array(landmarks):
    if isinstance(landmarks, np.ndarray):
        values = landmarks.astype(float, copy=False)
    else:
        values = np.array(
            [[landmark.x, landmark.y, landmark.z] for landmark in landmarks],
            dtype=float,
        )

    if values.ndim == 1:
        values = values[:XYZ_FEATURE_COUNT].reshape(LANDMARK_COUNT, COORDS_PER_LANDMARK)

    if values.shape != (LANDMARK_COUNT, COORDS_PER_LANDMARK):
        raise ValueError(
            f"Expected {LANDMARK_COUNT} landmarks with x, y, z coordinates; "
            f"got shape {values.shape}"
        )

    return values


def normalize_landmark_xyz(landmarks):
    xyz_values = _landmarks_to_xyz_array(landmarks)
    normalized = xyz_values - xyz_values[WRIST_INDEX]

    max_abs = np.max(np.abs(normalized))
    if max_abs == 0:
        max_abs = 1.0

    return normalized / max_abs


def extract_distance_features(normalized_xyz):
    distances = []

    for start_index, end_index in FINGERTIP_DISTANCE_PAIRS:
        distances.append(
            float(np.linalg.norm(normalized_xyz[start_index] - normalized_xyz[end_index]))
        )

    for fingertip_index in FINGERTIP_INDICES:
        distances.append(
            float(np.linalg.norm(normalized_xyz[WRIST_INDEX] - normalized_xyz[fingertip_index]))
        )

    return distances


def calculate_joint_angle(normalized_xyz, joint1_index, joint2_index, joint3_index):
    first_vector = normalized_xyz[joint2_index] - normalized_xyz[joint1_index]
    second_vector = normalized_xyz[joint3_index] - normalized_xyz[joint2_index]

    first_norm = np.linalg.norm(first_vector)
    second_norm = np.linalg.norm(second_vector)
    if first_norm == 0 or second_norm == 0:
        return 0.0

    cosine = np.dot(first_vector, second_vector) / (first_norm * second_norm)
    clipped_cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.arccos(clipped_cosine))


def extract_angle_features(normalized_xyz):
    return [
        calculate_joint_angle(normalized_xyz, joint1_index, joint2_index, joint3_index)
        for joint1_index, joint2_index, joint3_index in FINGER_ANGLE_JOINTS
    ]


def extract_features_from_landmarks(landmarks):
    normalized_xyz = normalize_landmark_xyz(landmarks)
    normalized_features = normalized_xyz.reshape(XYZ_FEATURE_COUNT).tolist()
    distance_features = extract_distance_features(normalized_xyz)
    angle_features = extract_angle_features(normalized_xyz)

    return normalized_features + distance_features + angle_features


def extract_feature_matrix(rows):
    return np.array([extract_features_from_landmarks(row) for row in rows], dtype=float)
