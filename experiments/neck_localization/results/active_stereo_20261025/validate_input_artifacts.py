#!/usr/bin/env python3
"""Read-only audit of saved RGB-D, image masks, training clouds and frozen artifacts."""
from collections import Counter
from hashlib import sha256
import gzip,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from neck_localization.stereo_robust import extract_stereo_masks
from scripts.run_depth_profile_baseline import load_frames
OUT=ROOT/'experiments/neck_localization/results/active_stereo_20261025'
RUNS=ROOT/'experiments/neck_localization/runs'
NAMES=['active_stereo_adapted_test_20261025_v1','active_stereo_gp_adapted_test_20261025_v1','active_stereo_board_calibrated_test_20261025_v1']

def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def rows(path):return json.load(gzip.open(path/'records.json.gz','rt'))
def strip_times(value):
 if isinstance(value,dict):return {k:strip_times(v) for k,v in value.items() if k not in {'runtime_ms','elapsed_ms','capture_processing_ms','correction_and_registration_ms'}}
 if isinstance(value,list):return [strip_times(v) for v in value]
 return value

def compare_rows(a,b):
 ra,rb=rows(a),rows(b)
 ka=lambda r:(r['method'],r['condition_name'],r['case'])
 aa={ka(r):strip_times(r) for r in ra};bb={ka(r):strip_times(r) for r in rb}
 changed=[list(k) for k in aa.keys()&bb.keys() if aa[k]!=bb[k]]
 return dict(original_rows=len(ra),reproduced_rows=len(rb),keys_equal=aa.keys()==bb.keys(),
             exact_equal_except_runtime=aa==bb,changed_rows=changed)

def compare_observations(a,b):
 count=rgb_equal=depth_equal=camera_equal=0
 differences=[]
 for pa in sorted(a.glob('*/scenes/*/observations/camera_true_parameters.json')):
  pb=b/pa.relative_to(a)
  if not pb.exists():differences.append(dict(path=str(pa.relative_to(a)),reason='missing camera metadata'));continue
  ma,mb=json.loads(pa.read_text()),json.loads(pb.read_text())
  camera_equal+=int(ma==mb)
  for cam in ma['cameras']:
   name=cam['name'];count+=1
   for suffix,kind in [('_rgb.png','rgb'),('_depth_z_m.npz','depth')]:
    x,y=pa.parent/(name+suffix),pb.parent/(name+suffix)
    if kind=='rgb':equal=digest(x)==digest(y);rgb_equal+=int(equal)
    else:
     with np.load(x) as dx,np.load(y) as dy:equal=np.array_equal(dx['depth_z_m'],dy['depth_z_m'],equal_nan=True)
     depth_equal+=int(equal)
    if not equal:differences.append(dict(path=str(x.relative_to(a)),reason=kind+' values differ'))
 return dict(camera_frames=count,rgb_byte_equal=rgb_equal,depth_array_equal=depth_equal,
             camera_metadata_groups_equal=camera_equal,differences=differences)

freeze=json.loads((OUT/'freeze.json').read_text())
result=dict(scope='Saved inference inputs only; no hardware; no evaluation labels used in this audit',
 script_sha256=digest(__file__), runs=[], source_checks=[], failures=[])
for collection in ['sources','evaluation_sources','gp_extra_sources']:
 for path,expected in freeze[collection].items():
  actual=digest(ROOT/path);result['source_checks'].append(dict(collection=collection,path=path,expected=expected,actual=actual,passed=actual==expected))
  if actual!=expected:result['failures'].append('frozen source changed: '+path)
model=RUNS/'gp_depth_calibration_20261024_v1/models/sim_d435_preset.npz'
board=RUNS/'rendered_calibration_20261025_v3/estimated_camera_parameters.json'
expected_board='9f9d25bccbcc333a0d8dc14cfc2c94e9f82d8c1f2d4eb4a29134a10323c63bf7'
result['calibration_artifacts']=dict(gp_model_sha256=digest(model),gp_expected_sha256=freeze['gp_model_sha256'],
 board_estimates_sha256=digest(board),board_expected_locked_sha256=expected_board)
if digest(model)!=freeze['gp_model_sha256']:result['failures'].append('GP model changed')
if digest(board)!=expected_board:result['failures'].append('Board estimated parameters changed')
board_parameters={c['name']:c for c in json.loads(board.read_text())['cameras']}
for name in NAMES:
 run=RUNS/name;local=rows(run);details=[];missing=[];points_total=0;max_error=0.;max_pixel_error=0.;masks_total=0;fit_camera_groups=0
 for r in local:
  key=f"{r['method']}/{r['condition_name']}/{r['case']}";folder=run/r['method']/r['condition_name']/'scenes'/r['case']
  frames=load_frames(folder/'observations','rgb')
  assert all(f.diagnostic_mask is None for f in frames),'diagnostic truth mask loaded'
  masks,meta=extract_stereo_masks(frames)
  saved=r['diagnostics']['foreground']
  assert meta==saved,(key,'RGB-only mask metadata differs')
  for f,mask,item in zip(frames,masks,saved['cameras']):
   image=cv2.imread(str(folder/'foreground_masks'/(f.camera.name+'_foreground.png')),cv2.IMREAD_GRAYSCALE)>0
   assert np.array_equal(image,mask),(key,'saved foreground mask differs')
   assert item['mask_sha256']==sha256(mask.astype(np.uint8).tobytes()).hexdigest()
   assert item['rgb_sha256']==sha256(np.ascontiguousarray(f.rgb).tobytes()).hexdigest()
   if r['camera_parameter_mode']=='rendered_board_estimated':
    c=board_parameters[f.camera.name]
    assert np.array_equal(f.camera.K,np.array(c['K'])) and np.array_equal(f.camera.T_world_from_camera,np.array(c['T_world_from_camera']))
   masks_total+=1
  d=r['diagnostics']
  if not d.get('fit_point_positions_m'):
   missing.append(dict(key=key,reasons=r['reasons'],attempts=d.get('attempts',[])));continue
  p=np.array(d['fit_point_positions_m']);ids=np.array(d['fit_camera_ids'])
  assert np.issubdtype(ids.dtype,np.integer)
  assert d['fit_camera_names']==[f.camera.name for f in frames]
  assert len(p)==len(ids)==d['fit_points']==len(d['fit_point_weights'])
  assert np.all(np.isfinite(p)) and np.all((ids>=0)&(ids<len(frames)))
  row_error=0.;row_pixel_error=0.
  for index,(f,mask) in enumerate(zip(frames,masks)):
   chosen=p[ids==index]
   if not len(chosen):continue
   uv=f.camera.project(chosen);pixels=np.rint(uv).astype(int)
   assert np.all((pixels[:,0]>=0)&(pixels[:,0]<f.camera.width)&(pixels[:,1]>=0)&(pixels[:,1]<f.camera.height))
   z=f.depth_z_m[pixels[:,1],pixels[:,0]]
   assert np.all(np.isfinite(z)&(z>0))
   assert np.all(mask[pixels[:,1],pixels[:,0]]),(key,'fit points outside actual RGB mask')
   rebuilt=f.camera.unproject(pixels,z)
   error=float(np.max(np.linalg.norm(rebuilt-chosen,axis=1)))
   pixel_error=float(np.max(abs(uv-pixels)))
   assert error<1e-8,(key,f.camera.name,error)
   assert pixel_error<1e-6,(key,f.camera.name,pixel_error)
   row_error=max(row_error,error);row_pixel_error=max(row_pixel_error,pixel_error);fit_camera_groups+=1
  points_total+=len(p);max_error=max(max_error,row_error);max_pixel_error=max(max_pixel_error,row_pixel_error)
  details.append(dict(key=key,fit_points=len(p),max_roundtrip_m=row_error,max_pixel_rounding_difference_px=row_pixel_error,
                      accepted=r['accepted']))
 result['runs'].append(dict(run=name,record_groups=len(local),groups_with_fit_points=len(details),missing_fit_points=len(missing),
  checked_fit_points=points_total,checked_fit_camera_groups=fit_camera_groups,checked_rgb_mask_groups=masks_total,
  max_roundtrip_m=max_error,max_pixel_rounding_difference_px=max_pixel_error,missing_fit_point_records=missing,groups=details))
repro=RUNS/'active_stereo_full_reproduction_20261025_v1'
result['full_reproduction']=dict(completed=(repro/'reproduction.json').exists(),snapshot_only=True,rows={},observations={})
for original,child in [('active_stereo_test_20261025_v1','observations'),(NAMES[0],'adapted'),(NAMES[1],'gp_adapted'),(NAMES[2],'board_adapted')]:
 if (repro/child/'records.json.gz').exists():result['full_reproduction']['rows'][child]=compare_rows(RUNS/original,repro/child)
 else:result['full_reproduction']['rows'][child]=dict(status='not complete at audit snapshot')
for original,child in [('active_stereo_test_20261025_v1','observations'),('active_stereo_gp_test_20261025_v1','gp_observations')]:
 if (repro/child/'provenance.json').exists():result['full_reproduction']['observations'][child]=compare_observations(RUNS/original,repro/child)
 else:result['full_reproduction']['observations'][child]=dict(status='not complete at audit snapshot')
result['totals']=dict(record_groups=sum(x['record_groups'] for x in result['runs']),groups_with_fit_points=sum(x['groups_with_fit_points'] for x in result['runs']),
 checked_fit_points=sum(x['checked_fit_points'] for x in result['runs']), checked_rgb_mask_groups=sum(x['checked_rgb_mask_groups'] for x in result['runs']),
 max_roundtrip_m=max(x['max_roundtrip_m'] for x in result['runs']),missing_fit_points=sum(x['missing_fit_points'] for x in result['runs']))
result['truth_separation']=dict(mask='All saved masks exactly regenerated from original RGB only; hashes and fit-mask inclusion checked',
 fit_cloud='All saved training points exactly unproject from same saved depth/K/T at integer pixels; no direct mesh sample substitute',
 ideal_depth_conditions=['ideal_rgb','ideal_ir_registered'],
 ideal_depth_caveat='Those explicitly labelled ideal diagnostics use rendered ideal depth by design; do not call them realistic sensor outputs',
 inference_source='frozen evaluator opens target.json only after estimator and band proposal; observation loader RGB branch never loads diagnostic masks')
result['passed']=not result['failures']
(OUT/'input_validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:result[k] for k in ['passed','totals','calibration_artifacts','full_reproduction']},indent=2))
