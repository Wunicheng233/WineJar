#!/usr/bin/env python3
"""Verify local derivatives and predeclared degeneracy examples; no pose search."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neck_localization.geometry import basis
from neck_localization.observability import radial_linearization, unprojection_jacobian, circle_information


def analyze():
    rng = np.random.default_rng(20261022)
    n = np.array([.08, -.05, 1.]); n /= np.linalg.norm(n)
    a = np.array([.01, -.02, .59])
    U = np.array(basis(n)).T
    beta = np.array([.052, 0., 40.])
    B = lambda s: np.c_[np.ones(len(s)), s, s*s]
    r = lambda s: B(s) @ beta
    rp = lambda s: beta[1]+2*beta[2]*s
    s = rng.uniform(-.012, .012, 200)
    phi = rng.uniform(-np.pi, np.pi, 200)
    radial = np.cos(phi)[:, None]*U[:, 0]+np.sin(phi)[:, None]*U[:, 1]
    points = a+s[:, None]*n+r(s)[:, None]*radial
    _, J = radial_linearization(points, a, n, r, rp, B)
    def residual(origin=a, normal=n, coeff=beta):
        d = points-origin
        normal = normal/np.linalg.norm(normal)
        axial = d @ normal
        return np.linalg.norm(d-axial[:, None]*normal, axis=1)-B(axial) @ coeff
    errors = {}
    for name, count, step in [("origin", 3, 1e-7), ("normal_tangent", 2, 1e-6), ("profile", 3, 1e-5)]:
        numerical = []
        for k in range(count):
            if name == "origin":
                d = np.eye(3)[k]*step
                plus, minus = residual(origin=a+d), residual(origin=a-d)
            elif name == "normal_tangent":
                d = U[:, k]*step
                plus, minus = residual(normal=n+d), residual(normal=n-d)
            else:
                d = np.eye(3)[k]*step
                plus, minus = residual(coeff=beta+d), residual(coeff=beta-d)
            numerical.append((plus-minus)/(2*step))
        errors[name] = float(np.max(np.abs(np.array(numerical).T-J[name])))
        if errors[name] > 1e-7:
            raise AssertionError((name, errors[name]))
    K = np.array([[515., 0, 319.5], [0, 515., 239.5], [0, 0, 1]])
    sample = np.array([401., 189., .8])
    unproject = lambda x: np.linalg.inv(K) @ np.r_[x[:2], 1.]*x[2]
    numerical = np.column_stack([(unproject(sample+np.eye(3)[k]*1e-5)-
                                  unproject(sample-np.eye(3)[k]*1e-5))/(2e-5) for k in range(3)])
    errors["unprojection"] = float(np.max(np.abs(numerical-unprojection_jacobian(*sample, K))))
    if errors["unprojection"] > 1e-8:
        raise AssertionError(errors)
    cases = {"full_circle": np.arange(120)*2*np.pi/120,
             "short_arc_30_deg": np.linspace(-np.pi/12, np.pi/12, 120),
             "one_repeated_direction": np.zeros(120),
             "two_opposed_directions": np.repeat([0., np.pi], 60),
             "three_balanced_directions": np.repeat(np.arange(3)*2*np.pi/3, 40)}
    examples = {}
    for name, angles in cases.items():
        I = circle_information(angles)
        eigenvalues = np.linalg.eigvalsh(I)
        rank = int(np.linalg.matrix_rank(I, tol=1e-8))
        examples[name] = {"n": len(angles), "rank": rank, "information_eigenvalues": eigenvalues.tolist(),
                          "condition": float(eigenvalues[-1]/eigenvalues[0]) if rank == 3 else None}
    return {"kind": "analytic_linearization_and_degeneracy_checks_not_localization_results",
            "seed": 20261022, "camera_pose_search": False, "rendered_noise_experiments": False,
            "maximum_absolute_derivative_difference": errors,
            "circle_examples": examples,
            "limits": "Unit radial variance and specified angular evidence only; not a camera placement or real accuracy prediction."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
