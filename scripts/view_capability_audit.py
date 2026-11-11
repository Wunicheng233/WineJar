#!/usr/bin/env python3
"""Interactive read-only point cloud and result replay for all audit methods."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout',choices=['close','far'],default='close')
    parser.add_argument('--case',default='structural_test_nominal_000')
    parser.add_argument('--method',choices=['simsense','foundation','p2p','p2p-input','lndf','ideal'],default='foundation')
    parser.add_argument('--condition',choices=['noise','clean','raw','ideal'],default='noise')
    parser.add_argument('--snapshot',type=Path)
    parser.add_argument('--snapshot-inputs',type=Path)
    args=parser.parse_args()
    import matplotlib
    matplotlib.use('Agg' if args.snapshot else 'TkAgg')
    from neck_localization.capability_viewer import CapabilityViewer
    if args.method in ['simsense','foundation']:
        if args.condition not in ['noise','clean']:
            parser.error('Stereo methods require --condition noise or clean')
        option=('SimSense' if args.method=='simsense' else 'FoundationStereo')+' / '+args.condition
    elif args.method=='ideal':
        option='Ideal IR / control'
    else:
        condition='ideal' if args.condition=='ideal' else 'raw'
        option={'p2p':'P2P restored','p2p-input':'P2P input','lndf':'L-NDF'}[args.method]+' / '+condition
    viewer=CapabilityViewer(args.layout,args.case,option)
    if args.snapshot_inputs:
        args.snapshot_inputs.parent.mkdir(parents=True,exist_ok=True)
        viewer.show_inputs(args.snapshot_inputs)
    if args.snapshot:
        viewer.save(args.snapshot)
    else:
        viewer.plt.show()


if __name__=='__main__':
    main()
