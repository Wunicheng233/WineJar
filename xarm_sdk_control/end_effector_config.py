"""Validated configuration loader for the installed end effector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from xarm_sdk_control.suction_sensor import SuctionSensorConfig
from xarm_sdk_control.vacuum_control import VacuumIOConfig


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "end_effector.yaml"


@dataclass(frozen=True)
class EndEffectorConfig:
    name: str
    kind: str
    cup_count: int
    tcp_offset_mm_deg: tuple[float, float, float, float, float, float]
    home_pose_mm_deg: tuple[float, float, float, float, float, float]
    vacuum_io: VacuumIOConfig
    max_compression_mm: float
    default_compression_mm: float
    suction_sensor: SuctionSensorConfig | None = None


def load_end_effector_config(path: Path = DEFAULT_CONFIG_PATH) -> EndEffectorConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    tcp = tuple(float(value) for value in data["tcp_offset_mm_deg"])
    if len(tcp) != 6:
        raise ValueError("tcp_offset_mm_deg must contain x, y, z, roll, pitch, yaw")
    home = tuple(float(value) for value in data["home_pose_mm_deg"])
    if len(home) != 6:
        raise ValueError("home_pose_mm_deg must contain x, y, z, roll, pitch, yaw")
    cup_count = int(data["cup_count"])
    if cup_count < 1:
        raise ValueError("cup_count must be positive")
    compression = float(data.get("default_compression_mm", 0.0))
    max_compression = float(data["max_compression_mm"])
    if max_compression <= 0.0:
        raise ValueError("max_compression_mm must be positive")
    if not 0.0 <= compression <= max_compression:
        raise ValueError("default_compression_mm must remain within max_compression_mm")

    vacuum = data["vacuum_io"]
    suction: SuctionSensorConfig | None = None
    if "suction_sensor" in data:
        sensor = data["suction_sensor"]
        suction = SuctionSensorConfig(
            input_channel=int(sensor["input_channel"]),
            success_value=int(sensor["success_value"]),
            debounce_seconds=float(sensor["debounce_seconds"]),
            timeout_seconds=float(sensor["timeout_seconds"]),
        )
    return EndEffectorConfig(
        name=str(data["name"]),
        kind=str(data["type"]),
        cup_count=cup_count,
        tcp_offset_mm_deg=tcp,
        home_pose_mm_deg=home,
        vacuum_io=VacuumIOConfig(
            pump_channel=int(vacuum["pump_channel"]),
            valve_channel=int(vacuum["valve_channel"]),
            pump_on_value=int(vacuum["pump_on_value"]),
            valve_vacuum_value=int(vacuum["valve_vacuum_value"]),
        ),
        max_compression_mm=max_compression,
        default_compression_mm=compression,
        suction_sensor=suction,
    )
