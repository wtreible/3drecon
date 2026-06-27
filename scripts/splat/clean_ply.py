#!/usr/bin/env python3
"""Cull floater / giant gaussians from a splatfacto-exported 3DGS .ply so it renders
cleanly in third-party viewers (superspl.at, MeshLab, PlayCanvas).

nerfstudio's rasterizer blends huge / faint gaussians toward invisible, so its own
renders look fine — but other viewers don't apply the same treatment, so a few
enormous "sheet" gaussians and far-field sky floaters make the export look terrible.
This drops gaussians by max-axis scale, position outliers, and (optionally) opacity,
preserving all per-gaussian properties.

    recon dl python scripts/splat/clean_ply.py \
        --in projects/demo_lighthouse/exports/splat.ply \
        --out projects/demo_lighthouse/exports/splat_clean.ply
"""
import argparse, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _progress import stage

T2NP = {'float': '<f4', 'float32': '<f4', 'double': '<f8', 'uchar': 'u1',
        'uint8': 'u1', 'int': '<i4', 'uint': '<u4', 'short': '<i2',
        'ushort': '<u2', 'char': 'i1'}
NP2T = {'float32': 'float', 'float64': 'double', 'uint8': 'uchar', 'int32': 'int',
        'uint32': 'uint', 'int16': 'short', 'uint16': 'ushort', 'int8': 'char'}


def read_ply(path):
    f = open(path, 'rb')
    assert f.readline().strip() == b'ply', "not a ply"
    props, n, fmt = [], 0, None
    while True:
        ln = f.readline().decode('latin1').strip()
        if ln.startswith('format'): fmt = ln.split()[1]
        elif ln.startswith('element vertex'): n = int(ln.split()[-1])
        elif ln.startswith('property'):
            _, t, name = ln.split(); props.append((name, T2NP[t]))
        elif ln == 'end_header': break
    assert fmt == 'binary_little_endian', "only binary_little_endian supported"
    data = np.fromfile(f, dtype=np.dtype(props), count=n)
    return data


def write_ply(path, data):
    with open(path, 'wb') as f:
        f.write(b'ply\nformat binary_little_endian 1.0\n')
        f.write(b'comment cleaned by clean_ply.py\n')
        f.write(f'element vertex {len(data)}\n'.encode())
        for name in data.dtype.names:
            f.write(f'property {NP2T[data.dtype[name].name]} {name}\n'.encode())
        f.write(b'end_header\n')
        data.tofile(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-opacity", type=float, default=0.12)
    ap.add_argument("--max-needle", type=float, default=8.0,
                    help="drop spikes: longest axis > this × second-longest")
    ap.add_argument("--max-scale-frac", type=float, default=0.04,
                    help="drop gaussians whose max-axis size exceeds this fraction of the scene diagonal")
    ap.add_argument("--keep-percentile", type=float, default=98.0,
                    help="position bbox kept between (100-p) and p per axis")
    ap.add_argument("--margin", type=float, default=0.15, help="expand position bbox by this fraction")
    args = ap.parse_args()

    stage(1, 3, f"Reading {args.inp}")
    d = read_ply(args.inp)
    n0 = len(d)
    xyz = np.stack([d['x'], d['y'], d['z']], 1)

    stage(2, 3, "Computing cull masks")
    keep = np.ones(n0, bool); reasons = {}
    # position bbox from robust percentiles, expanded by margin
    lo = np.percentile(xyz, 100 - args.keep_percentile, 0)
    hi = np.percentile(xyz, args.keep_percentile, 0)
    ctr, ext = (lo + hi) / 2, (hi - lo)
    lo2, hi2 = ctr - ext * (1 + args.margin) / 2, ctr + ext * (1 + args.margin) / 2
    diag = float(np.linalg.norm(hi - lo))
    m_pos = np.all((xyz >= lo2) & (xyz <= hi2), 1)
    reasons['position outlier'] = int((~m_pos).sum()); keep &= m_pos
    # scale culls (need all 3 axes)
    sc_names = [c for c in ('scale_0', 'scale_1', 'scale_2') if c in d.dtype.names]
    if sc_names:
        S = np.exp(np.stack([d[c] for c in sc_names], 1))
        ss = np.sort(S, 1)[:, ::-1]                       # s0>=s1>=s2 per gaussian
        # oversized: largest axis too big vs the scene
        m_sc = ss[:, 0] <= args.max_scale_frac * diag
        reasons[f'oversized (> {args.max_scale_frac*diag:.3f})'] = int((~m_sc).sum()); keep &= m_sc
        # needle/spike: longest axis >> second-longest (kills spikes, spares flat disks)
        needle = ss[:, 0] / np.maximum(ss[:, 1], 1e-9)
        m_need = needle <= args.max_needle
        reasons[f'needle/spike (s0/s1 > {args.max_needle:g})'] = int((~m_need).sum()); keep &= m_need
    # opacity cull
    if 'opacity' in d.dtype.names:
        op = 1 / (1 + np.exp(-d['opacity']))
        m_op = op >= args.min_opacity
        reasons[f'opacity < {args.min_opacity}'] = int((~m_op).sum()); keep &= m_op

    stage(3, 3, f"Writing {args.out}")
    write_ply(args.out, d[keep])
    print(f"scene diagonal ~{diag:.2f} units")
    for k, v in reasons.items():
        print(f"  culled {v:>8,} ({v/n0*100:4.1f}%)  — {k}")
    print(f"kept {keep.sum():,}/{n0:,} gaussians ({keep.sum()/n0*100:.1f}%) -> {args.out}")


if __name__ == "__main__":
    main()
