"""Fisheye lens for the virtual camera (Phase 7): the double-sphere model (Usenko, Demmel, Cremers, 3DV 2018) that
habitat-sim 0.3.3 renders with `FisheyeSensorDoubleSphereSpec`. numpy only, so both containers can project points.

Camera frame here = OpenCV: x right, y down, z forward (Habitat's camera: x right, y up, -z forward; convert with
`from_habitat`). A ray at angle theta from the optical axis lands at radius r(theta) from the principal point
(cx, cy), with fx, fy the focal lengths in pixels:

    d1 = |p|,  d2 = sqrt(x^2 + y^2 + (xi d1 + z)^2),  u = fx x / (alpha d2 + (1 - alpha)(xi d1 + z)) + cx  (v: fy, y, cy)

Rays outside the model's valid cone (z <= -w2 d1) cannot be projected. `circle_px` (optional) is the lens's image
circle: pixels farther than this from (cx, cy) are black, as on the real camera (dark corners).

Example (a 170 deg lens whose image circle has radius 95 px, fitted to the ideal equidistant curve r = f theta):

    lens = DoubleSphere.fit_equidistant(fov_deg=170.0, circle_px=95.0, width=160, height=120)
    uv, ok = lens.project(np.array([[0.0, 0.0, 1.0]]))     # -> [[80, 60]], [True]
"""

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


def from_habitat(points: np.ndarray) -> np.ndarray:
    """Points in Habitat's camera frame (x right, y up, -z forward) -> OpenCV frame (x right, y down, z forward)."""
    p = np.asarray(points, dtype=np.float64)
    return p * np.array([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class DoubleSphere:
    """Double-sphere fisheye: focal lengths and principal point in pixels, xi and alpha unitless, image size."""

    fx: float
    fy: float
    cx: float
    cy: float
    xi: float
    alpha: float
    width: int
    height: int
    circle_px: Optional[float] = None  # image-circle radius (pixels); None = no black ring

    # --- model --------------------------------------------------------------------------------------------------

    @property
    def w2(self) -> float:
        """Validity constant: a point projects only if z > -w2 * |p| (paper, eq. 43)."""
        a, xi = self.alpha, self.xi
        w1 = a / (1.0 - a) if a <= 0.5 else (1.0 - a) / a
        return (w1 + xi) / math.sqrt(2.0 * w1 * xi + xi * xi + 1.0)

    def project(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(N, 3) points in the OpenCV camera frame -> (N, 2) pixels (u right, v down) and a validity mask (inside
        the model's cone and, if circle_px is set, inside the image circle)."""
        p = np.atleast_2d(np.asarray(points, dtype=np.float64))
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        d1 = np.sqrt(x * x + y * y + z * z)
        k = self.xi * d1 + z
        d2 = np.sqrt(x * x + y * y + k * k)
        den = self.alpha * d2 + (1.0 - self.alpha) * k
        ok = (z > -self.w2 * d1) & (den > 1e-12)
        den = np.where(ok, den, 1.0)
        uv = np.stack([self.fx * x / den + self.cx, self.fy * y / den + self.cy], axis=1)
        if self.circle_px is not None:
            ok &= np.hypot(uv[:, 0] - self.cx, uv[:, 1] - self.cy) <= self.circle_px
        return uv, ok

    def unproject(self, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(N, 2) pixels -> (N, 3) unit rays (OpenCV frame) and a validity mask (paper, eqs. 47-50)."""
        q = np.atleast_2d(np.asarray(uv, dtype=np.float64))
        mx, my = (q[:, 0] - self.cx) / self.fx, (q[:, 1] - self.cy) / self.fy
        r2 = mx * mx + my * my
        a, xi = self.alpha, self.xi
        inner = 1.0 - (2.0 * a - 1.0) * r2
        ok = inner >= 0.0
        mz = (1.0 - a * a * r2) / (a * np.sqrt(np.where(ok, inner, 0.0)) + 1.0 - a)
        disc = mz * mz + (1.0 - xi * xi) * r2
        ok &= disc >= 0.0
        k = (mz * xi + np.sqrt(np.where(ok, disc, 0.0))) / (mz * mz + r2)
        rays = np.stack([k * mx, k * my, k * mz - xi], axis=1)
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        if self.circle_px is not None:
            ok &= np.hypot(q[:, 0] - self.cx, q[:, 1] - self.cy) <= self.circle_px
        return rays, ok

    def radius_px(self, theta: np.ndarray) -> np.ndarray:
        """Image radius (pixels, along x with fx) of rays at angle theta (radians) from the optical axis."""
        t = np.asarray(theta, dtype=np.float64)
        return radial(t, self.fx, self.xi, self.alpha)

    def circle_mask(self) -> Optional[np.ndarray]:
        """(H, W) bool, True inside the image circle; None without circle_px."""
        if self.circle_px is None:
            return None
        v, u = np.mgrid[0:self.height, 0:self.width]
        return np.hypot(u + 0.5 - self.cx, v + 0.5 - self.cy) <= self.circle_px

    def apply_circle(self, rgb: np.ndarray) -> np.ndarray:
        """The picture with everything outside the image circle set to 0 (black), as the real lens shows it."""
        mask = self.circle_mask()
        if mask is None:
            return rgb
        out = rgb.copy()
        out[~mask] = 0
        return out

    # --- construction -------------------------------------------------------------------------------------------

    @classmethod
    def fit_equidistant(cls, fov_deg: float, circle_px: float, width: int, height: int,
                        cx: Optional[float] = None, cy: Optional[float] = None, n: int = 200) -> "DoubleSphere":
        """The double sphere closest (least squares in pixels) to the ideal equidistant fisheye r = f theta whose
        image circle (radius circle_px) is the edge of a fov_deg field of view. Square pixels (fx = fy); the
        principal point defaults to the image centre."""
        theta_max = math.radians(fov_deg) / 2.0
        theta = np.linspace(1e-3, theta_max, n)
        target = circle_px * theta / theta_max
        best = None
        for xi in np.linspace(-0.6, 1.2, 181):
            for alpha in np.linspace(0.05, 0.95, 91):
                g = radial(theta, 1.0, xi, alpha)
                if not np.all(np.isfinite(g)) or np.any(np.diff(g) <= 0):  # must grow with theta (no fold)
                    continue
                f = float(g @ target / (g @ g))
                err = float(np.sqrt(np.mean((f * g - target) ** 2)))
                if best is None or err < best[0]:
                    best = (err, f, float(xi), float(alpha))
        if best is None:
            raise ValueError("no double-sphere parameters fit this lens")
        _, f, xi, alpha = best
        lens = cls(fx=f, fy=f, cx=width / 2.0 if cx is None else cx, cy=height / 2.0 if cy is None else cy,
                   xi=xi, alpha=alpha, width=width, height=height, circle_px=circle_px)
        edge = np.array([[math.sin(theta_max), 0.0, math.cos(theta_max)]])
        if not DoubleSphere(**{**asdict(lens), "circle_px": None}).project(edge)[1][0]:
            raise ValueError("fitted lens cannot project its own field-of-view edge")
        return lens

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "DoubleSphere":
        """From a config dict with the field names above (e.g. camera_fair.yaml `lens.double_sphere`)."""
        return cls(**{k: cfg[k] for k in ("fx", "fy", "cx", "cy", "xi", "alpha", "width", "height")},
                   circle_px=cfg.get("circle_px"))

    def to_config(self) -> Dict[str, Any]:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(self).items()}

    def fov_deg(self) -> float:
        """Full field of view across the image circle (or the model's valid cone without one), degrees."""
        theta = np.linspace(0.0, math.pi, 20001)
        ok = self.project(np.stack([np.sin(theta), np.zeros_like(theta), np.cos(theta)], axis=1))[1]
        if self.circle_px is None:
            return math.degrees(2.0 * theta[ok].max())
        r = self.radius_px(theta)
        return math.degrees(2.0 * theta[ok & (r <= self.circle_px)].max())

    def habitat_spec(self, uuid: str, position: Any, orientation: Any = None) -> Any:
        """habitat_sim.FisheyeSensorDoubleSphereSpec (colour) with these parameters."""
        import habitat_sim
        import magnum as mn

        s = habitat_sim.FisheyeSensorDoubleSphereSpec()
        s.uuid, s.sensor_type = uuid, habitat_sim.SensorType.COLOR
        s.sensor_model_type = habitat_sim.FisheyeSensorModelType.DOUBLE_SPHERE
        s.xi, s.alpha = self.xi, self.alpha
        s.focal_length = mn.Vector2(self.fx, self.fy)
        s.principal_point_offset = mn.Vector2(self.cx, self.cy)
        s.resolution = mn.Vector2i(self.height, self.width)
        s.position = position
        if orientation is not None:
            s.orientation = orientation
        return s


def radial(theta: np.ndarray, f: float, xi: float, alpha: float) -> np.ndarray:
    """Double-sphere image radius of rays at angle theta from the axis (unit-length ray, x = sin, z = cos)."""
    s, c = np.sin(theta), np.cos(theta)
    k = xi + c
    d2 = np.sqrt(s * s + k * k)
    with np.errstate(divide="ignore", invalid="ignore"):
        return f * s / (alpha * d2 + (1.0 - alpha) * k)


@dataclass(frozen=True)
class Equidistant:
    """Ideal equidistant fisheye r = f * theta (GoStanford stand-in: no calibration published)."""

    f: float
    cx: float
    cy: float
    width: int
    height: int
    circle_px: Optional[float] = None

    def project(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p = np.atleast_2d(np.asarray(points, dtype=np.float64))
        rho = np.hypot(p[:, 0], p[:, 1])
        theta = np.arctan2(rho, p[:, 2])
        r = self.f * theta
        s = np.where(rho > 1e-12, r / np.maximum(rho, 1e-12), 0.0)
        uv = np.stack([p[:, 0] * s + self.cx, p[:, 1] * s + self.cy], axis=1)
        ok = theta < math.pi / 2 + 0.3  # anything in front or slightly behind
        if self.circle_px is not None:
            ok &= r <= self.circle_px
        return uv, ok

    def unproject(self, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        q = np.atleast_2d(np.asarray(uv, dtype=np.float64))
        mx, my = q[:, 0] - self.cx, q[:, 1] - self.cy
        r = np.hypot(mx, my)
        theta = r / self.f
        s = np.where(r > 1e-12, np.sin(theta) / np.maximum(r, 1e-12), 1.0 / self.f)
        rays = np.stack([mx * s, my * s, np.cos(theta)], axis=1)
        ok = theta < math.pi
        if self.circle_px is not None:
            ok &= r <= self.circle_px
        return rays, ok


@dataclass(frozen=True)
class OpenCVPinhole:
    """Pinhole with OpenCV's radial-tangential distortion (k1, k2, p1, p2, k3); k = 0 is an ideal pinhole
    (Habitat's camera, RECON's published calibration)."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    def _distort(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r2 = x * x + y * y
        rad = 1.0 + self.k1 * r2 + self.k2 * r2 * r2 + self.k3 * r2 ** 3
        return (x * rad + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x),
                y * rad + self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y)

    def project(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p = np.atleast_2d(np.asarray(points, dtype=np.float64))
        ok = p[:, 2] > 1e-6
        z = np.where(ok, p[:, 2], 1.0)
        xd, yd = self._distort(p[:, 0] / z, p[:, 1] / z)
        return np.stack([self.fx * xd + self.cx, self.fy * yd + self.cy], axis=1), ok

    def unproject(self, uv: np.ndarray, iterations: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        q = np.atleast_2d(np.asarray(uv, dtype=np.float64))
        xd, yd = (q[:, 0] - self.cx) / self.fx, (q[:, 1] - self.cy) / self.fy
        x, y = xd.copy(), yd.copy()
        for _ in range(iterations):  # OpenCV undistortPoints iteration: remove tangential part, divide by radial
            r2 = x * x + y * y
            rad = 1.0 + self.k1 * r2 + self.k2 * r2 * r2 + self.k3 * r2 ** 3
            tx = 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
            ty = self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y
            x, y = (xd - tx) / rad, (yd - ty) / rad
        rays = np.stack([x, y, np.ones_like(x)], axis=1)
        back, _ = self.project(rays)
        ok = np.hypot(back[:, 0] - q[:, 0], back[:, 1] - q[:, 1]) < 0.5  # the inversion converged
        return rays / np.linalg.norm(rays, axis=1, keepdims=True), ok

    @classmethod
    def from_hfov(cls, hfov_deg: float, width: int, height: int) -> "OpenCVPinhole":
        """Habitat's centred pinhole (square pixels)."""
        f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        return cls(fx=f, fy=f, cx=width / 2.0, cy=height / 2.0, width=width, height=height)
