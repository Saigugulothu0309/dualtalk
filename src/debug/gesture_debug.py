from src.testing.gesture_inspector import main, run_gesture_inspector


def run_gesture_debug(camera_index, print_interval_seconds):
    return run_gesture_inspector(camera_index, print_interval_seconds)


if __name__ == "__main__":
    main()
