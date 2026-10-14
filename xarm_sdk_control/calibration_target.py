"""One observation interface for ordinary chessboard and ChArUco targets."""

import numpy as np

from xarm_sdk_control.multiview_calibration import (
    chessboard_object_points,
    create_charuco_board,
    detect_charuco_corners,
    observe_chessboard,
)


class CalibrationTarget:
    def __init__(self, kind, spec):
        self.kind, self.spec = kind, dict(spec)
        if kind == "chessboard":
            self.pattern = (spec["inner_corners_cols"], spec["inner_corners_rows"])
            if self.pattern[0] % 2 == self.pattern[1] % 2 or min(self.pattern) < 3:
                raise ValueError(
                    "Paired chessboards need mixed-parity inner-corner counts >=3 (for example 11x8); use ChArUco otherwise"
                )
            self.objects = chessboard_object_points(
                *self.pattern, spec["square_size_mm"] / 1000
            )
            self.board = None
        elif kind == "charuco":
            self.board = create_charuco_board(
                spec["squares_x"],
                spec["squares_y"],
                spec["square_size_mm"] / 1000,
                spec["marker_size_mm"] / 1000,
                spec["dictionary"],
            )
            self.objects = np.asarray(self.board.getChessboardCorners(), float)
        else:
            raise ValueError("Unknown calibration target")

    @classmethod
    def from_pair(cls, pair):
        for kind in ("chessboard", "charuco"):
            if pair.get("type") == f"fixed_camera_pair_{kind}_extrinsics":
                return cls(kind, pair[kind])
        raise ValueError("Unsupported paired calibration")

    def detect(self, color):
        if self.kind == "charuco":
            return detect_charuco_corners(color, self.board)
        return observe_chessboard(
            color, self.pattern, self.spec["square_size_mm"] / 1000
        )

    def observation(self, sample, prefix):
        from xarm_sdk_control.multiview_calibration import BoardObservation

        ids = np.array(
            sample.get(prefix + "_corner_ids", np.arange(len(self.objects))), int
        )
        pixels = np.asarray(sample[prefix + "_corner_points_px"], float).reshape(
            -1, 1, 2
        )
        if (
            len(ids) != len(pixels)
            or len(set(ids)) != len(ids)
            or np.any(ids < 0)
            or np.any(ids >= len(self.objects))
        ):
            raise ValueError("Invalid or duplicated board corner IDs")
        return BoardObservation(pixels, ids, self.objects[ids], 0)
