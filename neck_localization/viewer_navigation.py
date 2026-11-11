"""Framebuffer layout and hit regions for the offline inspection viewer."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

    def contains(self, x, y):
        return self.x <= x < self.x+self.width and self.y <= y < self.y+self.height


@dataclass(frozen=True)
class ViewerLayout:
    sidebar: Rect
    main: Rect
    inputs: Rect
    neck: Rect


def viewer_layout(width, height, large_neck=False):
    sidebar = 252 if width >= 1280 else 224
    right = min(390, max(270, int(height*.38)))
    scene_width = max(100, width-sidebar-right)
    main = Rect(sidebar, 62, scene_width, height-90)
    inset_width = min(700 if large_neck else 400, scene_width-28)
    inset_height = min(460 if large_neck else 280, int(main.height*(.58 if large_neck else .36)))
    neck = Rect(sidebar+14, height-44-inset_height, inset_width, inset_height)
    return ViewerLayout(Rect(0, 0, sidebar, height), main,
                        Rect(sidebar+scene_width, 0, right, height), neck)


def framebuffer_point(x, y, window_size, framebuffer_size):
    ww, hh = window_size
    fw, fh = framebuffer_size
    if min(ww, hh, fw, fh) <= 0:
        return None
    return x*fw/ww, y*fh/hh


@dataclass(frozen=True)
class NavButton:
    action: str
    label: str
    rect: Rect
    active: bool = False
    enabled: bool = True


def hit_button(buttons, x, y, content_top, content_bottom):
    if not content_top <= y < content_bottom:
        return None
    return next((b.action for b in buttons if b.enabled and b.rect.contains(x, y)), None)


CONDITION_LABELS = {
    'stereo_raw': '原始双目', 'stereo_spatial': '官方滤波',
    'stereo_gp': 'GP校正', 'stereo_gp_spatial': 'GP + 滤波',
    'ideal_rgb': '理想RGB深度', 'ideal_ir_registered': '理想IR对齐',
    'native_raw': '原生双目', 'native_spatial': '原生滤波',
    'native_ideal': '原生理想', 'native_ideal_z16': '原生理想量化',
}


def active_stereo_datasets(runs, group='true'):
    """Available saved comparisons only; no generation or estimation."""
    runs = Path(runs)
    if group == 'native':
        conditions = ['native_raw', 'native_spatial', 'native_ideal', 'native_ideal_z16']
    else:
        conditions = ['stereo_raw', 'stereo_spatial', 'stereo_gp', 'stereo_gp_spatial', 'ideal_rgb', 'ideal_ir_registered']
    result = []
    for condition in conditions:
        if group == 'native':
            folder = runs/'native_stereo_viewer_20261025_v2'/condition
            method = 'native_ir_rgb_mapped'
        else:
            if group == 'board' and condition.startswith('stereo_gp'):
                continue
            name = ('active_stereo_board_calibrated_test_20261025_v1' if group == 'board' else
                    'active_stereo_gp_adapted_test_20261025_v1' if condition.startswith('stereo_gp') else
                    'active_stereo_adapted_test_20261025_v1')
            folder = runs/name/'stereo_rgb_adapted'/condition
            method = 'stereo_rgb_adapted'
        if (folder/'records.json.gz').exists() and (folder/'scenes').is_dir():
            result.append(dict(condition=condition, label=CONDITION_LABELS[condition],
                               scenes=str(folder/'scenes'), records=str(folder/'records.json.gz'), method=method))
    return result
