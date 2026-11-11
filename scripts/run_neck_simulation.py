#!/usr/bin/env python3
"""Offline benchmark. No device imports, motion, or hardware calibration writes."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import mujoco
import numpy as np
import scipy
from neck_localization.calibration import virtual_calibrate
from neck_localization.estimators import fit_depth, fit_depth_plane_rays, fit_multiview
from neck_localization.evaluation import score, summarize
from neck_localization.perturbation import perturb
from neck_localization.rendering import render, extract_observations, exact_edge_diagnostic


def write_json(path,obj):
    def default(x):
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=default,allow_nan=False)+"\n")


def scenarios(config,split):
    seed=config["seed_"+split]
    rng=np.random.default_rng(seed)
    count=config[split+"_cases_per_geometry"]
    for style in ("coupon","groove"):
        for index in range(count):
            yield {"id":f"{split}_{style}_{index:02d}","split":split,"geometry":style,
                   "center_m":rng.uniform([-.045,-.045,.35],[.045,.045,.45]).tolist(),
                   "rotation_deg":rng.uniform([-20,-20,-180],[20,20,180]).tolist(),
                   "radius_m":float(rng.uniform(.047,.064)),
                   "layout":["wide","balanced","narrow"][index%3],
                   "distance_m":float(rng.uniform(.65,.85)),"light":float(rng.uniform(.65,1)),
                   "occlusion":bool(index>=6 and index%3==0),"resolution":config["resolution"],
                   "segments":config["segments"],"offsamples":0,"index":index}


def estimate_all(observations,ablations=False):
    result={"depth_ls":fit_depth(observations,robust=False),
            "depth_plane_rays":fit_depth_plane_rays(observations),
            "mv_robust":fit_multiview(observations,robust=True,equal_view_weight=True),
            "mv_depth_fused":fit_multiview(observations,robust=True,equal_view_weight=True,depth_sigma_m=.002)}
    if ablations:
        result.update(depth_robust=fit_depth(observations,robust=True),
                      mv_l2=fit_multiview(observations,robust=False,equal_view_weight=True),
                      mv_unbalanced=fit_multiview(observations,robust=True,equal_view_weight=False))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"experiments/neck_localization/configs/benchmark.json")
    parser.add_argument("--output",type=Path,required=True,help="New directory; existing output is refused")
    parser.add_argument("--quick",action="store_true",help="Smoke suite; not the final benchmark")
    parser.add_argument("--development-only",action="store_true")
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    if args.quick:
        config["test_cases_per_geometry"]=3
        config["development_cases_per_geometry"]=1
        config["factors"]=[f for f in config["factors"] if f["name"] in ("clean","outliers_15pct","common_frame_translation_5mm","two_side_views")]
    args.output.mkdir(parents=True,exist_ok=False)
    frames=args.output/"frames"
    frames.mkdir()
    start=time.perf_counter()
    write_json(args.output/"effective_config.json",config)
    results,truths,specs,calibrations=[],{},[],{}
    splits=["development"] if args.development_only else ["development","test"]

    def evaluate(observations,spec,evidence,factor,ablations=False):
        for method,estimate in estimate_all(observations,ablations).items():
            row={"case":spec["id"],"split":spec["split"],"geometry":spec["geometry"],
                 "layout":spec["layout"],"evidence":evidence,"factor":factor,"method":method,
                 "view_counts":{o.camera.name:len(o.pixels) for o in observations}}
            row["nominal_circle_reference_only"]=factor=="shape_mismatch_ellipticity_4pct"
            row.update(score(estimate,truths[spec["id"]]))
            results.append(row)

    for split in splits:
        for spec in scenarios(config,split):
            print(f"render {spec['id']} ({spec['layout']})",flush=True)
            scene=render(spec)
            specs.append(spec)
            truths[spec["id"]]=scene.truth
            # Image arrays and exact truth are saved separately; estimators get Observation only.
            if spec["index"]==0:
                folder=frames/spec["id"]
                folder.mkdir()
                (folder/"scene.xml").write_text(scene.xml)
                for camera,rgb,depth,mask in zip(scene.cameras,scene.rgb,scene.depth,scene.masks):
                    cv2.imwrite(str(folder/(camera.name+".png")),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                    np.savez_compressed(folder/(camera.name+"_depth.npz"),depth_z_m=depth)
                    cv2.imwrite(str(folder/(camera.name+"_mask_diagnostic.png")),mask)
                write_json(folder/"cameras_true.json",[{"name":c.name,"K":c.K,"T_world_from_camera":c.T_world_from_camera} for c in scene.cameras])
            for evidence in ("exact_edge_diagnostic","mask_diagnostic","rgb"):
                observations=exact_edge_diagnostic(scene) if evidence=="exact_edge_diagnostic" else extract_observations(scene,spec["geometry"],evidence)
                evaluate(observations,spec,evidence,"clean")
                if split=="test" and spec["index"]<3 and evidence=="rgb":
                    for j,factor in enumerate(config["factors"]):
                        if factor["name"]=="clean": continue
                        altered=perturb(observations,factor,config["seed_test"]+1000*spec["index"]+j)
                        evaluate(altered,spec,evidence,factor["name"],ablations=True)
                if split=="test" and spec["index"]==0 and evidence=="rgb" and not args.quick:
                    # Same observed image/Z, independently estimated camera models.
                    for name,params in [("virtual_ideal",{"corner_sigma_px":0}),
                                        ("virtual_corners_0.1px",{"corner_sigma_px":.1}),
                                        ("virtual_board_scale_1pct",{"corner_sigma_px":0,"board_scale_error":.01}),
                                        ("virtual_board_warp_0.5mm",{"corner_sigma_px":0,"board_warp_m":.0005})]:
                        cams,diagnostic=virtual_calibrate(scene.cameras,seed=config["seed_test"],**params)
                        calibrations[spec["id"]+"_"+name]=diagnostic
                        altered=perturb(observations,{},config["seed_test"],calibrated_cameras=cams)
                        evaluate(altered,spec,evidence,name)
            if split=="test" and spec["index"]==1 and not args.quick:
                mismatch=dict(spec,ellipticity=.04)
                mismatched=render(mismatch)
                # The nominal radius is a declared reference; no exact circle exists.
                ob=extract_observations(mismatched,spec["geometry"],"rgb")
                evaluate(ob,spec,"rgb","shape_mismatch_ellipticity_4pct",ablations=True)
            if split=="test" and spec["index"]==2 and spec["geometry"]=="groove" and not args.quick:
                natural=render(dict(spec,contrast_band=False))
                evaluate(extract_observations(natural,"groove","rgb"),spec,"rgb","no_artificial_contrast")
    summary=summarize(results)
    write_json(args.output/"scenarios.json",specs)
    write_json(args.output/"truth_evaluation_only.json",truths)
    write_json(args.output/"records.json",results)
    write_json(args.output/"calibration_diagnostics.json",calibrations)
    write_json(args.output/"summary.json",summary)
    with (args.output/"summary.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    source_files=sorted((ROOT/"neck_localization").glob("*.py"))+[Path(__file__).resolve()]
    write_json(args.output/"provenance.json",{
        "python":platform.python_version(),"mujoco":mujoco.__version__,"numpy":np.__version__,"scipy":scipy.__version__,"opencv":cv2.__version__,
        "elapsed_seconds":time.perf_counter()-start,"quick":args.quick,"development_only":args.development_only,
        "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "config_sha256":hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "limitations":["No hardware validation","Groove band has artificial contrast","Exact edges include invisible ring: diagnostic only",
                        "Virtual calibration uses synthetic identified corners, not rendered corner detection","No physical RealSense RGB-depth baseline occlusion model",
                        "Conditional covariance excludes calibration, model and selection bias","Noise magnitudes without measurement evidence are assumptions"]})
    print(json.dumps({"output":str(args.output),"records":len(results),"seconds":time.perf_counter()-start}),flush=True)


if __name__=="__main__":
    main()
