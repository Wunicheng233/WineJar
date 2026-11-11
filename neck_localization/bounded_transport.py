"""Set-membership update of a single conveyor displacement, in metres.

Bounds must come from independent characterization. This does not infer a
depth-camera bias bound from ICP residuals or claim real process acceptance.
"""
import numpy as np


def bounded_translation(measured_delta_m, prior_half_width_m, observation_error_bound_m, *, measurement_available=True):
    a,b=float(prior_half_width_m),float(observation_error_bound_m)
    if not np.isfinite([a,b]).all() or min(a,b)<0:
        raise ValueError('finite nonnegative independently justified bounds required')
    if not measurement_available or not np.isfinite(measured_delta_m):
        return dict(delta_m=0.,interval_m=[-a,a],status='nominal_fallback_no_observation',
                    process_release=False,conditional_on_bounds=True)
    low=max(-a,float(measured_delta_m)-b);high=min(a,float(measured_delta_m)+b)
    if low>high:
        return dict(delta_m=0.,interval_m=[-a,a],status='nominal_fallback_inconsistent_bounds',
                    process_release=False,conditional_on_bounds=True)
    return dict(delta_m=(low+high)/2,interval_m=[low,high],status='bounded_candidate',
                process_release=False,conditional_on_bounds=True)


def gaussian_shrinkage(measured_delta_m, prior_half_width_m, observation_error_bound_m):
    """Comparison only: zero-mean independent-uniform moment approximation.

    Fixed unknown biases violate its distributional interpretation; no claimed
    confidence or default deployment recommendation is attached to this mean.
    """
    a,b=float(prior_half_width_m),float(observation_error_bound_m)
    if not np.isfinite([a,b]).all() or min(a,b)<0:raise ValueError('invalid scales')
    gain=a*a/(a*a+b*b) if a+b else 0.
    return dict(delta_m=float(gain*measured_delta_m),gain=gain,status='zero_mean_shrinkage_comparison',process_release=False)
