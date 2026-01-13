#!/usr/bin/env python3
"""
Shard an AnnData (.h5ad) into multiple files with at most `--chunk-size` observations each.

The shards are written into the same directory as the input file with names:
  adata_part0.h5ad, adata_part1.h5ad, ...

Usage:
  python shard_adata.py /path/to/adata.h5ad --chunk-size 100000

Notes:
- The script uses `anndata.read_h5ad(..., backed='r')` to avoid loading the whole dataset into memory.
- Each shard is created by slicing the original AnnData and writing that subset to a new .h5ad file.
- Slicing in backed mode returns a small in-memory AnnData for the slice (so memory usage is proportional to chunk size).
"""

import os
import sys
import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Shard an AnnData (.h5ad) into smaller .h5ad files")
    p.add_argument('input', type=str, help='Path to input .h5ad file')
    p.add_argument('--chunk-size', type=int, default=50000, help='Maximum number of observations per shard (default: 100000)')
    p.add_argument('--prefix', type=str, default='adata_part', help='Output filename prefix (default: adata_part)')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing shard files')
    p.add_argument('--start-index', type=int, default=0, help='Start index for numbering output parts')
    p.add_argument('--no-progress', action='store_true', help='Disable progress prints')
    return p.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(2)

    try:
        import anndata
    except Exception as e:
        print("Failed to import anndata. Install with `pip install anndata`.")
        raise

    print(f"Opening {input_path} in backed='r' mode (will not load all data into memory)...")
    adata = anndata.read_h5ad(str(input_path), backed='r')

    try:
        n_obs = adata.n_obs
    except Exception:
        # fallback
        n_obs = adata.shape[0]

    if not args.no_progress:
        print(f"Dataset has {n_obs} observations. Chunk size = {args.chunk_size}")

    out_dir = input_path.parent

    part_idx = args.start_index
    written = 0
    for start in range(0, n_obs, args.chunk_size):
        end = min(start + args.chunk_size, n_obs)
        shard_name = f"{args.prefix}{part_idx}.h5ad"
        out_path = out_dir / shard_name

        if out_path.exists() and not args.overwrite:
            print(f"Skipping existing file {out_path} (use --overwrite to force)")
            part_idx += 1
            continue

        if not args.no_progress:
            print(f"Creating shard {part_idx}: rows [{start}:{end}) -> {out_path}")

        # Slice the backed AnnData
        X = adata.X[start:end]
        obs = adata.obs.iloc[start:end].copy()
        var = adata.var.copy()
        import anndata as _an
        adata_slice = _an.AnnData(X=X, obs=obs, var=var)
        # Try to also copy layers, obsm, varm if present
        try:
            if hasattr(adata, 'layers') and adata.layers:
                adata_slice.layers = {k: v[start:end].copy() for k, v in adata.layers.items()}
            if hasattr(adata, 'obsm') and adata.obsm:
                adata_slice.obsm = {k: v[start:end].copy() for k, v in adata.obsm.items()}
            if hasattr(adata, 'varm') and adata.varm:
                adata_slice.varm = {k: v.copy() for k, v in adata.varm.items()}
            if hasattr(adata, 'uns') and adata.uns:
                adata_slice.uns = adata.uns.copy()
        except Exception:
            # best-effort copy; ignore failures
            pass

        # Write shard
        try:
            adata_slice.write_h5ad(str(out_path))
        except Exception as e:
            # Try write using anndata.AnnData.write_h5ad fallback
            try:
                import anndata as _an
                _an.AnnData(adata_slice.X, obs=adata_slice.obs, var=adata_slice.var, uns=getattr(adata_slice, 'uns', None)).write_h5ad(str(out_path))
            except Exception as e2:
                print(f"Failed to write shard {out_path}: {e} / {e2}")
                raise

        written += 1
        part_idx += 1

    if not args.no_progress:
        print(f"Done. Written {written} shard(s) to {out_dir}")


if __name__ == '__main__':
    main()
