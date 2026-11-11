"""Seeded geometry sampling for a frozen-estimator audit, not factory statistics."""
from copy import deepcopy

import numpy as np


def random_audit_config(template, seed=2026102327, nominal=24, stress=12):
    """Two declared synthetic populations; no filtering using estimator outcomes.

    Jar base pose and camera layout stay fixed. Only individual local geometry
    and body proportions vary. Existing circular-section family is retained.
    """
    rng = np.random.default_rng(seed)
    cfg = deepcopy(template)
    cfg.update(seed=seed, cases=[], dataset_split='independent_random_frozen_algorithm_audit')
    for group, count in [('nominal', nominal), ('stress', stress)]:
        for index in range(count):
            hard = group == 'stress'
            h = rng.uniform(.55, .68) if hard else rng.uniform(.57, .65)
            radius = rng.uniform(.16, .205) if hard else rng.uniform(.165, .195)
            neck = rng.uniform(.046, .060) if hard else rng.uniform(.048, .057)
            tilt = rng.uniform(8, 12) if hard else rng.uniform(0, 7)
            azimuth = rng.uniform(-np.pi, np.pi)
            # Tilt components; normal's actual total tilt is derived by JarSpec.
            tilt_xy = tilt*np.array([np.cos(azimuth), np.sin(azimuth)])
            offset = rng.uniform(-.010 if hard else -.005, .010 if hard else .005, 2)
            profile = 'flat' if rng.random() < .5 else 'shallow'
            if hard:
                length = rng.uniform(.0045, .007) if index % 2 == 0 else rng.uniform(.016, .022)
            else:
                length = rng.uniform(.008, .014)
            low = rng.uniform(.54, .64)
            body_profile = [[0.,low],[.025,low+.04],[.12,rng.uniform(.70,.76)],
                [.30,rng.uniform(.82,.87)],[.52,rng.uniform(.93,.97)],
                [.72,1.],[.81,rng.uniform(.95,.99)],[.89,rng.uniform(.79,.87)],
                [.95,rng.uniform(.54,.64)]]
            jar = dict(height_m=float(h), body_radius_m=float(radius), neck_radius_m=float(neck),
                lip_radius_m=float(neck+rng.uniform(.008,.014)), neck_length_m=float(length),
                lip_height_m=float(rng.uniform(.0045,.008) if hard else rng.uniform(.005,.007)),
                neck_tilt_deg=tilt_xy.tolist(), neck_offset_xy_m=offset.tolist(), neck_profile=profile,
                shallow_relief_m=float(rng.uniform(.0004,.003) if hard else rng.uniform(.0012,.0024)),
                body_profile=body_profile)
            cfg['cases'].append(dict(name=f'{group}_{index:03d}', group=group,
                                     material='mottled_gray', jar=jar))
    cfg['audit_assumptions'] = dict(
        nominal='Assumed range around photo/user dimensions, NOT measured factory distribution',
        stress='Deliberately short/long neck, stronger local tilt and wider size range; NOT factory frequency',
        neck='Circular sections; single flat or shallow neck; no ellipticity/local dents/asymmetric rim',
        cameras='Frozen artificial three-camera rig and ideal true K/T; no layout search/no noise',
        placement='Body base position and orientation fixed; local geometry varies between jars',
        nominal_count=nominal, stress_count=stress)
    return cfg
