"""Metres, column-vector transforms; OpenCV cameras (+X right,+Y down,+Z forward)."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Camera:
    name: str
    K: np.ndarray
    T_world_from_camera: np.ndarray
    width: int = 640
    height: int = 480

    def project(self, points):
        p = (np.asarray(points) - self.T_world_from_camera[:3, 3]) @ self.T_world_from_camera[:3, :3]
        q = p @ self.K.T
        return q[:, :2] / q[:, 2:3]

    def unproject(self, pixels, z):
        rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(self.K).T
        return (rays * np.asarray(z)[:, None]) @ self.T_world_from_camera[:3, :3].T + self.T_world_from_camera[:3, 3]


@dataclass
class Observation:
    camera: Camera
    pixels: np.ndarray
    points_world: np.ndarray | None = None


@dataclass
class Estimate:
    center: np.ndarray | None = None
    normal: np.ndarray | None = None
    radius: float | None = None
    accepted: bool = False
    reasons: list[str] = field(default_factory=list)
    residual: float | None = None
    covariance_center: np.ndarray | None = None
    condition: float | None = None
    elapsed_ms: float = 0.0
    diagnostics: dict = field(default_factory=dict)


def basis(normal):
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    a = np.eye(3)[np.argmin(np.abs(n))]
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def circle_points(center, normal, radius, count=180):
    u, v = basis(normal)
    t = np.arange(count) * (2 * np.pi / count)
    return np.asarray(center) + radius * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v)


def normal_positive_z(normal):
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    return n if n[2] >= 0 else -n


def look_at(name, position, target, width=640, height=480, fovy_deg=50):
    p = np.asarray(position, float)
    z = np.asarray(target, float) - p
    z /= np.linalg.norm(z)
    x = np.cross(z, [0., 0., 1.])
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1., 0., 0.])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    t = np.eye(4)
    t[:3, :3] = np.column_stack((x, y, z))
    t[:3, 3] = p
    f = height / (2 * np.tan(np.deg2rad(fovy_deg) / 2))
    k = np.array([[f, 0, (width - 1)/2], [0, f, (height - 1)/2], [0, 0, 1.]])
    return Camera(name, k, t, width, height)
