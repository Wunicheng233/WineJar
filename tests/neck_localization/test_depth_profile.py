"""Regressions for observed crown selection, failure handling and RGB extraction."""
import numpy as np
import pytest

from neck_localization.depth_profile import (
    ProfileRejection, ProfileSettings, _mouth_plane, _radius_profile, rgb_foreground)


def test_upper_crown_wins_over_a_much_larger_lower_plane():
    rng = np.random.default_rng(114)
    theta = rng.uniform(0, 2*np.pi, 800)
    radius = rng.uniform(.048, .062, 800)
    crown = np.c_[radius*np.cos(theta), radius*np.sin(theta), np.full(800, .60)]
    # Earlier global support maximization selected the lower shoulder patch.
    shoulder = np.c_[rng.uniform(-.10, .10, (20000, 2)), np.full(20000, .574)]
    origin, normal, support = _mouth_plane(np.r_[crown, shoulder], ProfileSettings(), rng)
    assert abs(origin[2]-.60) < 1e-10
    np.testing.assert_allclose(normal, [0, 0, 1], atol=1e-10)
    assert len(support) == len(crown)


def test_boundary_minimum_is_rejected_instead_of_inventing_an_interval():
    s = np.repeat(np.linspace(-.060, -.002, 500), 20)
    theta = np.tile(np.linspace(0, 2*np.pi, 20, endpoint=False), 500)
    radius = .06+.1*s  # Monotonic: there is no supported interior neck minimum.
    cloud = np.c_[radius*np.cos(theta), radius*np.sin(theta), s]
    with pytest.raises(ProfileRejection, match="search_boundary"):
        _radius_profile(cloud, np.zeros(3), np.array([0, 0, 1]), ProfileSettings())


def test_profile_midpoint_is_not_tied_to_a_single_minimum_sample():
    s = np.repeat(np.linspace(-.060, -.002, 1000), 20)
    theta = np.tile(np.linspace(0, 2*np.pi, 20, endpoint=False), 1000)
    radius = .051+15*(s+.0137)**2
    cloud = np.c_[radius*np.cos(theta), radius*np.sin(theta), s]
    midpoint, r, _ = _radius_profile(cloud, np.zeros(3), np.array([0, 0, 1]), ProfileSettings())
    assert abs(midpoint+.0137) < .0002
    assert abs(r-.051) < .00003


def test_rgb_extraction_requires_no_diagnostic_mask():
    rgb = np.full((60, 80, 3), [25, 40, 60], dtype=np.uint8)
    rgb[10:55, 25:55] = [180, 182, 183]
    mask = rgb_foreground(rgb)
    assert mask[30, 40]
    assert not mask[5, 5]
    assert mask.sum() == 45*30
