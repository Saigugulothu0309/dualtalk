#!/usr/bin/env python3
"""
DualTalk unified launcher.

Usage:
  python run.py server
  python run.py sender
  python run.py receiver
  python run.py normal
  python run.py offline
  python run.py web
  python run.py gesture-debug
  python run.py gesture-test
  python -m src.testing.gesture_inspector
"""

import runpy
import sys


COMMANDS = {
    "server": "src.communication.server",
    "sender": "src.communication.sender",
    "receiver": "src.communication.receiver",
    "normal": "src.communication.normal_user",
    "offline": "src.ui.offline_ui",
    "web": "src.web.app",
    "gesture-debug": "src.testing.gesture_inspector",
    "gesture-test": "src.testing.gesture_validation_mode",
}


def print_help():
    print(__doc__.strip())


def main():
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print_help()
        return

    command = sys.argv[1]
    module_name = COMMANDS.get(command)
    if module_name is None:
        print(f"Unknown command: {command}")
        print_help()
        raise SystemExit(2)

    sys.argv = [f"{sys.argv[0]} {command}", *sys.argv[2:]]
    runpy.run_module(module_name, run_name="__main__")


if __name__ == "__main__":
    main()
