"""Analytic geometry checks; these ideal fixtures are not real accuracy evidence."""
import numpy as np
import pytest

from neck_localization.estimators import fit_depth, fit_depth_plane_rays, fit_multiview
from neck_localization.geometry import Camera, Observation, basis, circle_points, look_at


@pytest.fixture
def analytic_ring():
    center = np.array([.03, -.02, .31])
    normal = np.array([.17, -.24, 1.])
    normal /= np.linalg.norm(normal)
    radius = .057
    points = circle_points(center, normal, radius, count=120)
    cameras = [
        look_at("a", [.35, -.6, .6], center),
        look_at("b", [-.55, .2, .65], center),
        look_at("c", [0, .1, 1.0], center),
    ]
    return center, normal, radius, points, cameras


def _observations(points, cameras, depth_points=None):
    return [Observation(camera, camera.project(points), points.copy() if depth_points is None else depth_points.copy())
            for camera in cameras]


def test_depth_pca_plane_circle_recovers_ideal_ring(analytic_ring):
    center, normal, radius, points, cameras = analytic_ring
    result = fit_depth(_observations(points, cameras[:1]))
    assert result.accepted, result.reasons
    np.testing.assert_allclose(result.center, center, atol=1e-12)
    np.testing.assert_allclose(result.normal, normal, atol=1e-12)
    assert abs(result.radius - radius) < 1e-12
    assert result.residual < 1e-12
    assert result.normal[2] > 0


def test_depth_ransac_rejects_gross_outliers(analytic_ring):
    center, normal, radius, points, cameras = analytic_ring
    rng = np.random.default_rng(74)
    noisy = points + rng.normal(0, .0004, points.shape)
    outliers = center + rng.uniform(-.08, .08, (40, 3))
    contaminated = np.r_[noisy, outliers]
    observations = [Observation(cameras[0], np.zeros((len(contaminated), 2)), contaminated)]
    baseline = fit_depth(observations, robust=False)
    result = fit_depth(observations, robust=True)
    assert result.accepted, result.reasons
    assert np.linalg.norm(result.center - center) < .001
    assert abs(result.radius - radius) < .001
    assert result.normal @ normal > .999
    assert result.diagnostics["inlier_count"] < len(contaminated)
    assert not baseline.accepted
    # The robust implementation is deterministic, including its consensus set.
    again = fit_depth(observations, robust=True)
    np.testing.assert_array_equal(result.center, again.center)


@pytest.mark.parametrize("robust", [False, True])
def test_depth_rejects_collinear_and_insufficient_support(analytic_ring, robust):
    _, _, _, points, cameras = analytic_ring
    small = fit_depth(_observations(points[:5], cameras[:1]), robust=robust)
    assert not small.accepted
    assert "insufficient_finite_depth_points" in small.reasons
    line = np.column_stack((np.linspace(-.05, .05, 40), np.zeros(40), np.full(40, .3)))
    result = fit_depth(_observations(line, cameras[:1]), robust=robust)
    assert not result.accepted
    assert "degenerate_depth_geometry" in result.reasons


def test_depth_rejects_short_arc_even_when_fit_residual_is_zero(analytic_ring):
    center, normal, radius, _, cameras = analytic_ring
    u, v = basis(normal)
    angle = np.linspace(0, np.deg2rad(75), 60)
    points = center + radius * (np.cos(angle)[:, None] * u + np.sin(angle)[:, None] * v)
    result = fit_depth(_observations(points, cameras[:1]))
    assert not result.accepted
    assert result.residual < 1e-10
    assert "insufficient_angular_support" in result.reasons


@pytest.mark.parametrize("views", [2, 3])
def test_multiview_recovers_circle_with_biased_depth_initialization(analytic_ring, views):
    center, normal, radius, points, cameras = analytic_ring
    # Deliberately wrong depth pose and radius.  RGB alone must remove this
    # bias; using depth as final metric truth would fail the assertions.
    depth = center + 1.08 * (points - center) + [.012, -.009, .015]
    result = fit_multiview(_observations(points, cameras[:views], depth), robust=False)
    assert result.accepted, result.reasons
    np.testing.assert_allclose(result.center, center, atol=1e-7)
    np.testing.assert_allclose(result.normal, normal, atol=1e-6)
    assert abs(result.radius - radius) < 1e-7
    assert result.residual < 1e-5
    assert result.covariance_center.shape == (3, 3)
    assert np.linalg.eigvalsh(result.covariance_center).min() >= -1e-16
    assert "fixed camera calibration" in result.diagnostics["uncertainty_scope"]


def test_multiview_robust_fit_handles_bad_boundary_pixels(analytic_ring):
    center, normal, radius, points, cameras = analytic_ring
    observations = _observations(points, cameras)
    rng = np.random.default_rng(211)
    for observation in observations:
        observation.pixels += rng.normal(0, .15, observation.pixels.shape)
        observation.pixels[:15] += [16., -12.]
        observation.points_world += rng.normal(0, .0005, observation.points_world.shape)
    result = fit_multiview(observations, robust=True)
    baseline = fit_multiview(observations, robust=False)
    assert result.accepted, result.reasons
    assert np.linalg.norm(result.center - center) < .002
    assert abs(result.radius - radius) < .001
    assert np.degrees(np.arccos(np.clip(result.normal @ normal, -1, 1))) < 2.
    assert np.linalg.norm(result.center - center) < np.linalg.norm(baseline.center - center)


def test_balanced_multiview_solution_is_invariant_to_duplicated_view_samples(analytic_ring):
    _, _, _, points, cameras = analytic_ring
    observations = _observations(points, cameras[:2])
    # Fixed view-dependent bias makes unbalanced point counts matter.
    observations[1].pixels += [.6, -.3]
    original = fit_multiview(observations, equal_view_weight=True)
    dense = Observation(cameras[1], np.tile(observations[1].pixels, (5, 1)), points)
    duplicated = fit_multiview([observations[0], dense], equal_view_weight=True)
    assert original.accepted and duplicated.accepted
    np.testing.assert_allclose(original.center, duplicated.center, atol=1e-7)
    assert abs(original.radius - duplicated.radius) < 1e-7


def test_multiview_rejects_unobservable_input(analytic_ring):
    _, _, _, points, cameras = analytic_ring
    observation = _observations(points, cameras[:1])[0]
    mono = fit_multiview([observation])
    assert not mono.accepted
    assert "monocular_rgb_scale_ambiguity" in mono.reasons
    copy = Camera("same_origin", cameras[0].K.copy(), cameras[0].T_world_from_camera.copy())
    coincident = fit_multiview(_observations(points, [cameras[0], copy]))
    assert not coincident.accepted
    assert "degenerate_multiview_geometry" in coincident.reasons
    no_depth = [Observation(camera, camera.project(points)) for camera in cameras[:2]]
    missing = fit_multiview(no_depth)
    assert not missing.accepted
    assert "depth_initialization_unavailable" in missing.reasons


def test_multiview_rejects_too_few_or_nonfinite_pixels(analytic_ring):
    _, _, _, points, cameras = analytic_ring
    observations = _observations(points, cameras[:2])
    observations[1].pixels[:] = np.nan
    result = fit_multiview(observations)
    assert not result.accepted
    assert "insufficient_finite_image_points" in result.reasons


def test_multiview_rejects_tiny_observed_arc(analytic_ring):
    center, normal, radius, _, cameras = analytic_ring
    u, v = basis(normal)
    angle = np.linspace(0, np.deg2rad(70), 40)
    points = center + radius * (np.cos(angle)[:, None] * u + np.sin(angle)[:, None] * v)
    result = fit_multiview(_observations(points, cameras[:2]))
    assert not result.accepted
    assert "insufficient_angular_support" in result.reasons


def test_plane_rays_remove_in_plane_boundary_depth_mixing(analytic_ring):
    center, normal, radius, points, cameras = analytic_ring
    # Depth on an inward neighboring contour is still in the correct plane.
    # RGB boundary rays remain on the intended outer circle.
    mixed_depth = center + .94 * (points - center)
    observations = _observations(points, cameras, mixed_depth)
    baseline = fit_depth(observations)
    projected = fit_depth_plane_rays(observations)
    assert projected.accepted, projected.reasons
    assert abs(baseline.radius - radius) > .003
    np.testing.assert_allclose(projected.center, center, atol=1e-10)
    np.testing.assert_allclose(projected.normal, normal, atol=1e-10)
    assert abs(projected.radius - radius) < 1e-10
    assert "correlated inputs" in projected.diagnostics["observation_scope"]


def test_plane_rays_retain_common_depth_plane_bias_and_reject_nonplanarity(analytic_ring):
    center, normal, _, points, cameras = analytic_ring
    biased = fit_depth_plane_rays(_observations(points, cameras, points + .006 * normal))
    assert biased.center is not None
    assert abs((biased.center - center) @ normal) > .0059
    rng = np.random.default_rng(891)
    nonplanar = points + rng.normal(0, .015, (len(points), 1)) * normal
    rejected = fit_depth_plane_rays(_observations(points, cameras, nonplanar))
    assert not rejected.accepted
    assert "depth_support_not_planar" in rejected.reasons


def test_depth_fusion_can_improve_noisy_rgb_but_bias_accurate_rgb(analytic_ring):
    center, _, _, points, cameras = analytic_ring
    rng = np.random.default_rng(2934)
    accurate_depth = points + rng.normal(0, .00015, points.shape)
    noisy_rgb = _observations(points, cameras, accurate_depth)
    for observation in noisy_rgb:
        observation.pixels += rng.normal(0, .7, observation.pixels.shape)
    rgb = fit_multiview(noisy_rgb)
    fused = fit_multiview(noisy_rgb, depth_sigma_m=.0003)
    assert rgb.accepted and fused.accepted, (rgb.reasons, fused.reasons)
    assert np.linalg.norm(fused.center - center) < .8 * np.linalg.norm(rgb.center - center)
    assert "not an independent probabilistic likelihood" in fused.diagnostics["depth_fusion_scope"]

    # Overconfident weighting of a fixed depth offset is harmful even when RGB
    # and cameras are exact.  Averaging the same offset cannot remove this bias.
    biased_depth = _observations(points, cameras, points + [.008, -.006, .004])
    exact_rgb = fit_multiview(biased_depth)
    biased_fusion = fit_multiview(biased_depth, depth_sigma_m=.0002)
    assert exact_rgb.accepted
    assert np.linalg.norm(exact_rgb.center - center) < 1e-7
    assert np.linalg.norm(biased_fusion.center - center) > .003
    assert "not calibrated probability" in biased_fusion.diagnostics["uncertainty_scope"]


@pytest.mark.parametrize("sigma", [0., -.001, np.nan])
def test_depth_fusion_rejects_invalid_weight_scale(analytic_ring, sigma):
    _, _, _, points, cameras = analytic_ring
    result = fit_multiview(_observations(points, cameras), depth_sigma_m=sigma)
    assert not result.accepted
    assert "invalid_depth_fusion_scale" in result.reasons
