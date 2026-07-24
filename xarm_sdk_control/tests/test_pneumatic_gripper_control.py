from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "10_move_above_rubiks_cube.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("move_above_rubiks_cube", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeArm:
    def __init__(self):
        self.calls = []

    def set_cgpio_digital_output_function(self, ionum, fun):
        self.calls.append(("set_cgpio_digital_output_function", ionum, fun))
        return 0

    def set_cgpio_digital(self, ionum, value):
        self.calls.append(("set_cgpio_digital", ionum, value))
        return 0


class PneumaticGripperControlTest(unittest.TestCase):
    def test_sets_output_function_then_digital_value(self):
        module = load_script_module()
        arm = FakeArm()

        module.set_pneumatic_gripper(
            arm=arm,
            ionum=9,
            value=1,
            label="close gripper",
            settle_sec=0,
        )

        self.assertEqual(
            arm.calls,
            [
                ("set_cgpio_digital_output_function", 9, 0),
                ("set_cgpio_digital", 9, 1),
            ],
        )


if __name__ == "__main__":
    unittest.main()
