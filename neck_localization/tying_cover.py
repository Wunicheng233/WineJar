"""Environment-only conforming opaque cover surrogate; no cloth/contact solver."""
import numpy as np
from .jar_model import centerline, neck_normal, outer_radius
from .station_tying import frame


def cover_mesh(spec, config):
    start = spec.height_m-spec.lip_height_m-spec.neck_length_m
    s = np.unique(np.r_[np.linspace(start, spec.height_m, 101),
                        spec.height_m-spec.lip_height_m, spec.height_m+config['cover_cap_height_m']])
    radius = outer_radius(spec, np.minimum(s, spec.height_m))+config['cover_radial_allowance_m']
    theta = np.linspace(0, 2*np.pi, 241)[:-1]
    vertices = np.stack([radius[:, None]*np.cos(theta)[None, :],
                         radius[:, None]*np.sin(theta)[None, :],
                         np.broadcast_to((s-start)[:, None], (len(s), len(theta)))], axis=-1).reshape(-1, 3)
    m, faces = len(theta), []
    for j in range(len(s)-1):
        for i in range(m):
            a, b, c, d = j*m+i, j*m+(i+1)%m, (j+1)*m+i, (j+1)*m+(i+1)%m
            faces.extend([(a, b, c), (b, d, c)])
    top = len(vertices)
    vertices = np.vstack([vertices, [0, 0, s[-1]-start]])
    for i in range(m):
        faces.append(((len(s)-1)*m+i, (len(s)-1)*m+(i+1)%m, top))
    return dict(vertices=vertices, faces=np.asarray(faces), origin=centerline(spec, start),
                rotation=frame(neck_normal(spec)), start_s_m=start)
