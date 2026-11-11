#!/usr/bin/env python3
"""Frozen study harness plus explicit failure rows for legacy exceptions.

The frozen estimator is unchanged. A failure to return is industrial failure,
never skipped, converted to a geometric success or supplied with true values.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from neck_localization.geometry import Estimate
from scripts import run_structural_study as original


def run(args):
    frozen=original.estimate_stereo_robust;exceptions=[]
    def safe(frames):
        start=perf_counter()
        try:return frozen(frames)
        except (ValueError,IndexError) as error:
            record=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
            exceptions.append(record)
            return Estimate(reasons=['estimator_exception:'+type(error).__name__+':'+str(error)],
                elapsed_ms=(perf_counter()-start)*1000,diagnostics=dict(method='stereo_rgb_adapted',
                    estimator_exception=record,semantics='unchanged legacy method failed to return; counted as failure'))
    original.estimate_stereo_robust=safe
    original.run(args.run,args.output,args.modes,args.methods,args.freeze)
    (args.output/'exception_capture.json').write_text(json.dumps(dict(exceptions=exceptions,
        wrapper_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),frozen_estimator_unchanged=True),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--modes',nargs='+',default=['stereo_raw']);p.add_argument('--methods',nargs='+',default=['baseline','portfolio'])
    p.add_argument('--freeze',type=Path,required=True);run(p.parse_args())
