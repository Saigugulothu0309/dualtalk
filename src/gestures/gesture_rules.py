import math


NO_MIN_SAMPLE_COUNT = 5
NO_MIN_HORIZONTAL_RANGE = 0.028
NO_MAX_VERTICAL_RANGE = 0.12
NO_MIN_DIRECTION_CHANGES = 1
NO_MIN_DELTA = 0.0045
NO_MIN_SIGNIFICANT_MOVES = 3
NO_MIN_TOTAL_MOTION = 0.03
PALM_CENTER_LANDMARKS = (0, 5, 9, 13, 17)
FINGER_DEBUG_ORDER = ("Thumb", "Index", "Middle", "Ring", "Pinky")
FINGER_DEBUG_JOINTS = {
    "Thumb": (2, 3, 4),
    "Index": (5, 6, 8),
    "Middle": (9, 10, 12),
    "Ring": (13, 14, 16),
    "Pinky": (17, 18, 20),
}


def _distance_between_points(point_a, point_b):
    return math.sqrt(
        (float(point_a[0]) - float(point_b[0])) ** 2
        + (float(point_a[1]) - float(point_b[1])) ** 2
        + (float(point_a[2]) - float(point_b[2])) ** 2
    )


def _distance_between_landmarks(landmark_a, landmark_b):
    return _distance_between_points(
        (landmark_a.x, landmark_a.y, landmark_a.z),
        (landmark_b.x, landmark_b.y, landmark_b.z),
    )


def _distance_to_point(landmark, point):
    return _distance_between_points(
        (landmark.x, landmark.y, landmark.z),
        point,
    )


def _joint_angle_degrees(point_a, point_b, point_c):
    vector_ba = (
        float(point_a.x) - float(point_b.x),
        float(point_a.y) - float(point_b.y),
        float(point_a.z) - float(point_b.z),
    )
    vector_bc = (
        float(point_c.x) - float(point_b.x),
        float(point_c.y) - float(point_b.y),
        float(point_c.z) - float(point_b.z),
    )

    magnitude_ba = math.sqrt(sum(component * component for component in vector_ba))
    magnitude_bc = math.sqrt(sum(component * component for component in vector_bc))
    if magnitude_ba <= 1e-6 or magnitude_bc <= 1e-6:
        return 0.0

    dot_product = sum(
        component_a * component_c
        for component_a, component_c in zip(vector_ba, vector_bc)
    )
    cosine_value = max(
        -1.0,
        min(1.0, dot_product / (magnitude_ba * magnitude_bc)),
    )
    return math.degrees(math.acos(cosine_value))


def get_palm_center(landmarks):
    count = float(len(PALM_CENTER_LANDMARKS))
    return (
        sum(float(landmarks[index].x) for index in PALM_CENTER_LANDMARKS) / count,
        sum(float(landmarks[index].y) for index in PALM_CENTER_LANDMARKS) / count,
        sum(float(landmarks[index].z) for index in PALM_CENTER_LANDMARKS) / count,
    )


def is_finger_open(landmarks, finger_name):
    normalized_name = str(finger_name).strip().capitalize()
    if normalized_name not in FINGER_DEBUG_JOINTS:
        raise ValueError(f"Unsupported finger name: {finger_name}")

    base_index, hinge_index, tip_index = FINGER_DEBUG_JOINTS[normalized_name]
    palm_center = get_palm_center(landmarks)
    palm_span = max(
        _distance_between_landmarks(landmarks[5], landmarks[17]),
        _distance_between_landmarks(landmarks[0], landmarks[9]),
        1e-6,
    )

    base_joint = landmarks[base_index]
    hinge_joint = landmarks[hinge_index]
    fingertip = landmarks[tip_index]
    extension_margin = palm_span * (0.08 if normalized_name == "Thumb" else 0.12)
    angle_threshold = 145.0 if normalized_name == "Thumb" else 150.0

    return (
        _distance_to_point(fingertip, palm_center)
        > _distance_to_point(hinge_joint, palm_center) + extension_margin
        and _joint_angle_degrees(base_joint, hinge_joint, fingertip) >= angle_threshold
    )


def get_finger_states(landmarks):
    return {
        finger_name: "OPEN" if is_finger_open(landmarks, finger_name) else "CLOSED"
        for finger_name in FINGER_DEBUG_ORDER
    }


def is_open_palm(landmarks):
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    ring_mcp = landmarks[13]
    pinky_mcp = landmarks[17]

    return (
        index_tip.y < index_mcp.y
        and middle_tip.y < middle_mcp.y
        and ring_tip.y < ring_mcp.y
        and pinky_tip.y < pinky_mcp.y
    )


def is_fist(landmarks):
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]

    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    ring_mcp = landmarks[13]

    return (
        index_tip.y > index_mcp.y
        and middle_tip.y > middle_mcp.y
        and ring_tip.y > ring_mcp.y
    )


def is_no_handshape(landmarks):
    index_tip = landmarks[8]
    index_pip = landmarks[6]
    index_mcp = landmarks[5]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    middle_mcp = landmarks[9]
    ring_mcp = landmarks[13]
    pinky_mcp = landmarks[17]

    return (
        index_tip.y < index_pip.y
        and index_tip.y < index_mcp.y
        and middle_tip.y > middle_mcp.y
        and ring_tip.y > ring_mcp.y
        and pinky_tip.y > pinky_mcp.y
    )


def get_index_finger_tip_position(landmarks):
    index_tip = landmarks[8]
    return float(index_tip.x), float(index_tip.y)


def smooth_series(values, window_size=3):
    values = list(values)
    smoothed = []
    for index in range(len(values)):
        start_index = max(0, index - window_size + 1)
        window = values[start_index : index + 1]
        smoothed.append(sum(window) / len(window))
    return smoothed


def detect_no_motion(x_positions, y_positions):
    if len(x_positions) < NO_MIN_SAMPLE_COUNT or len(y_positions) < NO_MIN_SAMPLE_COUNT:
        return False

    smoothed_x_positions = smooth_series(x_positions)
    smoothed_y_positions = smooth_series(y_positions)
    deltas = [
        smoothed_x_positions[index] - smoothed_x_positions[index - 1]
        for index in range(1, len(smoothed_x_positions))
    ]
    significant_signs = [
        1 if delta > 0 else -1
        for delta in deltas
        if abs(delta) >= NO_MIN_DELTA
    ]

    if len(significant_signs) < NO_MIN_SIGNIFICANT_MOVES:
        return False

    direction_changes = sum(
        1
        for index in range(1, len(significant_signs))
        if significant_signs[index] != significant_signs[index - 1]
    )
    horizontal_range = max(smoothed_x_positions) - min(smoothed_x_positions)
    vertical_range = max(smoothed_y_positions) - min(smoothed_y_positions)
    total_motion = sum(abs(delta) for delta in deltas)

    return (
        horizontal_range >= NO_MIN_HORIZONTAL_RANGE
        and vertical_range <= NO_MAX_VERTICAL_RANGE
        and direction_changes >= NO_MIN_DIRECTION_CHANGES
        and total_motion >= NO_MIN_TOTAL_MOTION
    )


def detect_gesture(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]

    thumb_ip = landmarks[3]
    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]

    # YES: thumb up
    if (
        thumb_tip.y < thumb_ip.y
        and index_tip.y > index_mcp.y
        and middle_tip.y > middle_mcp.y
    ):
        return "YES"

    # HOLD: static open palm
    if is_open_palm(landmarks):
        return "HOLD"

    return "UNKNOWN"
