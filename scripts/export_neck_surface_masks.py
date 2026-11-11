#!/usr/bin/env python3
"""Additional GT diagnostic: visible OUTSIDE surface, excluding the jar cavity.

Does not alter RGB/depth or provide a target-region mask. Normal RGB mode must
not read these labels. The source scene and segmentation IDs are generator GT.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
import json
from pathlib import Path
from hashlib import sha256

import cv2
import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, required=True)
    args = parser.parse_args()
    count = 0
    for source in sorted(args.scenes.glob("*/scene.xml")):
        folder = source.parent / "observations"
        metadata = json.loads((folder / "camera_true_parameters.json").read_text())
        outputs = [folder / (c["name"]+"_exterior_mask_diagnostic.png") for c in metadata["cameras"]]
        if any(path.exists() for path in outputs):
            raise FileExistsError("Refusing to replace existing diagnostic masks")
        model = mujoco.MjModel.from_xml_path(str(source))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        exterior = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "jar_surface")
        camera = metadata["cameras"][0]
        with mujoco.Renderer(model, camera["height"], camera["width"]) as renderer:
            renderer.enable_segmentation_rendering()
            for item, path in zip(metadata["cameras"], outputs):
                renderer.update_scene(data, camera=item["name"])
                labels = renderer.render()
                mask = ((labels[:, :, 0] == exterior) &
                        (labels[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)))
                cv2.imwrite(str(path), mask.astype(np.uint8)*255)
                count += 1
    provenance = {"kind": "additional_exterior_surface_gt_diagnostic_only",
                  "masks": count, "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                  "target_ring_mask": False, "rgb_depth_modified": False}
    output = args.scenes / "exterior_mask_diagnostic_provenance.json"
    output.write_text(json.dumps(provenance, indent=2)+"\n")
    print(json.dumps(provenance))


if __name__ == "__main__":
    main()
