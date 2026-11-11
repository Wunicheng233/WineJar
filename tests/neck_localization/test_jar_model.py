"""Physical geometry invariants, without MuJoCo or device dependencies."""
from dataclasses import replace
import unittest

import numpy as np

from neck_localization.jar_model import JarSpec, build_jar_mesh, centerline, outer_radius, target_metadata


class JarModelTests(unittest.TestCase):
    def test_dimensions_and_circular_sections_are_independent(self):
        spec = JarSpec(height_m=.66, body_radius_m=.195, neck_radius_m=.055, lip_radius_m=.064,
                       neck_length_m=.045, neck_tilt_deg=(9., -12.), neck_offset_xy_m=(.015, -.008))
        mesh = build_jar_mesh(spec, angular_segments=48, vertical_samples=48)
        rings = mesh["vertices"][mesh["outer_ring_vertex_indices"]]
        centres = mesh["centerline_m"]
        np.testing.assert_allclose(np.linalg.norm(rings - centres[:, None], axis=2),
                                   np.broadcast_to(mesh["profile_radius_m"][:, None], rings.shape[:2]), atol=1e-14)
        np.testing.assert_allclose(centerline(spec, spec.height_m), [.015, -.008, .66], atol=1e-14)
        self.assertAlmostEqual(float(outer_radius(spec, target_metadata(spec)["s_m"])), .055)
        self.assertAlmostEqual(float(mesh["profile_radius_m"].max()), .195)
        self.assertAlmostEqual(mesh["bbox_min_m"][2], 0.)

    def test_local_tilt_preserves_vertical_body_and_changes_target_normal(self):
        spec = JarSpec(neck_tilt_deg=(12., 8.), neck_offset_xy_m=(.02, -.01))
        np.testing.assert_allclose(centerline(spec, np.array([.05, .2, .4])),
                                   [[0, 0, .05], [0, 0, .2], [0, 0, .4]], atol=1e-14)
        target = target_metadata(spec)
        normal = np.asarray(target["normal"])
        self.assertGreater(np.linalg.norm(normal[:2]), .2)
        self.assertAlmostEqual(np.linalg.norm(normal), 1.)
        self.assertGreater(normal[2], 0.)
        s = target["s_m"]
        numerical_axis = (centerline(spec, s + .001) - centerline(spec, s - .001)) / .002
        np.testing.assert_allclose(numerical_axis, normal, atol=1e-13)

    def test_flat_neck_returns_interval_midpoint_with_axial_length(self):
        spec = JarSpec(neck_profile="flat", neck_tilt_deg=(0, 15))
        target = target_metadata(spec)
        low, high = target["near_min_interval_s_m"]
        self.assertAlmostEqual(high - low, .010)
        # Raised lip has its own height, so shortening the neck must neither
        # consume it nor move the exposed neck into the mouth crown.
        self.assertAlmostEqual(spec.height_m - high, spec.lip_height_m)
        self.assertAlmostEqual(target["s_m"], (low + high) / 2)
        self.assertEqual(target["minimum_interval_s_m"], [low, high])
        np.testing.assert_allclose(outer_radius(spec, np.linspace(low, high, 21)), spec.neck_radius_m, atol=1e-15)
        self.assertAlmostEqual(np.linalg.norm(centerline(spec, high) - centerline(spec, low)),
                               target["near_min_interval_axial_length_m"])
        self.assertFalse(target["observability_verified"])

    def test_shallow_interval_is_threshold_set_not_mesh_minimum(self):
        spec = JarSpec(near_min_delta_m=.0007)
        target = target_metadata(spec)
        lo, hi = target["near_min_interval_s_m"]
        np.testing.assert_allclose(outer_radius(spec, [lo, hi]), spec.neck_radius_m + .0007, atol=1e-14)
        self.assertLess(outer_radius(spec, target["s_m"]), outer_radius(spec, lo))
        self.assertLessEqual(target["search_interval_s_m"][0], lo)
        self.assertLessEqual(hi, target["search_interval_s_m"][1])
        # Evaluation truth must not move with angular or axial mesh resolution.
        self.assertEqual(build_jar_mesh(spec, 24, 32)["target"], build_jar_mesh(spec, 64, 64)["target"])
        wider = target_metadata(replace(spec, near_min_delta_m=.0014))
        self.assertGreater(wider["near_min_interval_axial_length_m"], target["near_min_interval_axial_length_m"])

    def test_photo_reference_has_upper_belly_and_short_exposed_neck(self):
        spec = JarSpec()
        self.assertLess(outer_radius(spec, .10), outer_radius(spec, .30))
        self.assertLess(outer_radius(spec, .30), outer_radius(spec, .42))
        self.assertGreater(outer_radius(spec, .42), outer_radius(spec, .56))
        heights = np.linspace(0, .58, 1001)
        peak_height = heights[np.argmax(outer_radius(spec, heights))]
        self.assertTrue(.38 < peak_height < .46)
        start, stop = target_metadata(spec)["search_interval_s_m"]
        self.assertAlmostEqual(stop - start, .010)
        self.assertAlmostEqual(spec.height_m - stop, .006)
        # Dimension independence: changing the lip must not lengthen the neck.
        changed = target_metadata(replace(spec, lip_height_m=.008))
        self.assertAlmostEqual(np.diff(changed["search_interval_s_m"])[0], .010)

    def test_watertight_oriented_solid_has_open_cavity_and_no_degenerate_faces(self):
        mesh = build_jar_mesh(JarSpec(), angular_segments=32, vertical_samples=40)
        vertices, faces = mesh["vertices"], mesh["faces"]
        edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        np.testing.assert_array_equal(counts, 2)
        triangles = vertices[faces]
        area2 = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
        self.assertGreater(area2.min(), 1e-12)
        # Correct outward winding gives positive wall material volume.
        volume = np.sum(np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2]))) / 6
        self.assertGreater(volume, 0.)
        self.assertLess(volume, .02)
        # The crown is annular: there is no triangle crossing the opening centre.
        crown = triangles[mesh["face_regions"] == 1]
        self.assertTrue(np.all(np.linalg.norm(crown[:, :, :2], axis=2) > .04))
        self.assertEqual(mesh["face_texcoords"].shape, faces.shape)
        self.assertLess(mesh["face_texcoords"].max(), len(mesh["texcoords"]))

    def test_repeatable_geometry_and_invalid_parameters(self):
        a = build_jar_mesh(JarSpec(), 24, 32)
        b = build_jar_mesh(JarSpec(), 24, 32)
        np.testing.assert_array_equal(a["vertices"], b["vertices"])
        np.testing.assert_array_equal(a["faces"], b["faces"])
        for kwargs in ({"height_m": -1}, {"neck_radius_m": .2}, {"near_min_delta_m": -1},
                       {"neck_tilt_deg": (30, 0)}, {"neck_profile": "v_groove"}):
            with self.assertRaises(ValueError):
                JarSpec(**kwargs)

    def test_mild_tilt_changes_only_the_collar_not_the_body(self):
        upright = build_jar_mesh(JarSpec(), 48, 120)
        tilted = build_jar_mesh(JarSpec(neck_tilt_deg=(5., -3.)), 48, 120)
        below = upright["profile_s_m"] < .54
        np.testing.assert_array_equal(
            upright["vertices"][upright["outer_ring_vertex_indices"][below]],
            tilted["vertices"][tilted["outer_ring_vertex_indices"][below]])
        self.assertGreater(tilted["geometry_diagnostics"]["local_collar_blend_start_s_m"], .54)
        # The same lip ring has a real high side and low side, not a bent body.
        top = tilted["vertices"][tilted["outer_ring_vertex_indices"][-1]]
        self.assertTrue(.010 < np.ptp(top[:, 2]) < .016)

    def test_offset_against_neck_tilt_does_not_fold_shoulder(self):
        # This exact preparation case previously folded: a short centreline
        # bridge had kappa*r about 3 and left a black open-looking shoulder seam.
        spec = JarSpec(neck_tilt_deg=(12., -7.), neck_offset_xy_m=(.010, -.006))
        mesh = build_jar_mesh(spec, angular_segments=96, vertical_samples=160)
        diagnostic = mesh["geometry_diagnostics"]
        self.assertGreaterEqual(diagnostic["minimum_axial_jacobian"], .20)
        self.assertGreater(diagnostic["local_collar_blend_start_s_m"], .45)
        np.testing.assert_allclose(centerline(spec, [.1, .2]), [[0, 0, .1], [0, 0, .2]], atol=1e-14)
        np.testing.assert_allclose(centerline(spec, spec.height_m), [.010, -.006, .6], atol=1e-14)
        rings = mesh["vertices"][mesh["outer_ring_vertex_indices"]]
        centres = mesh["centerline_m"]
        # Outward triangle normals must retain a positive radial component on
        # BOTH sides of every exterior quad.  This directly catches a folded
        # shoulder; area and watertightness alone had passed the faulty mesh.
        for j in (0, 1):
            if j == 0:
                a, b, c = rings[:-1], np.roll(rings[:-1], -1, axis=1), rings[1:]
                triangle_centreline = (2 * centres[:-1, None] + centres[1:, None]) / 3
            else:
                a, b, c = np.roll(rings[:-1], -1, axis=1), np.roll(rings[1:], -1, axis=1), rings[1:]
                triangle_centreline = (centres[:-1, None] + 2 * centres[1:, None]) / 3
            face_normal = np.cross(b - a, c - a)
            radial = (a + b + c) / 3 - triangle_centreline
            self.assertGreater(float(np.min(np.sum(face_normal * radial, axis=2))), 0.)


if __name__ == "__main__":
    unittest.main()
