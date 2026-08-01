#!/usr/bin/env python3
"""
Cached MdfData loading for batch validation.

Parsed signal arrays are gzip-pickled per MF4 file into `.cache/mdf_signals/`.
Cache key = (file mtime_ns, file size) + sha256 of mdf_reader.py source, so any
change to the signal reader invalidates the cache automatically. On a cache hit
the MF4 file is never opened — repeated batch runs skip all channel parsing.

The cached object additionally carries array-wise auxiliary grids so batch
consumers never need the raw MDF:
  md._pz_grid         : Psi_PrkgInfo.PrkgZone_u8 interpolated on t_grid (or None)
  md._sel_id_grid     : Driver_Selected_Parking_Space_ID on t_grid (or None)
  md._slot_type_grid  : resolved slot type per sample (array-wise equivalent of
                        get_selected_slot_type) or NaN where unavailable
"""

import gzip
import hashlib
import os
import pickle
import time

import numpy as np

import mdf_reader as mr

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_PROJECT_DIR, '.cache', 'mdf_signals')


def _reader_hash():
    """Hash of mdf_reader.py source — reader changes invalidate cached data."""
    with open(os.path.join(_PROJECT_DIR, 'mdf_reader.py'), 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _cache_path(mf4_path):
    base = os.path.splitext(os.path.basename(mf4_path))[0]
    return os.path.join(CACHE_DIR, base + '.pkl.gz')


def _slot_type_grid(mdf, t_grid, sel_id_grid):
    """Array-wise equivalent of mdf_reader.get_selected_slot_type per sample."""
    from mdf_reader import find_channel, read_simple, nearest_interp_1d

    n = len(t_grid)
    out = np.full(n, np.nan)
    if sel_id_grid is not None:
        # Method 1: PFSM selected slot ID → TP_100ms PS_SlotType._<id>_ lookup
        prefix = 'TP_100ms_ZF_Fusion_Parking_Slot_Set_i_obv.PS_SlotType._'
        for ch_n in mdf.channels_db:
            if not ch_n.startswith(prefix):
                continue
            try:
                slot_id = int(ch_n[len(prefix):-1])
            except ValueError:
                continue
            ch_g, ch_c = mdf.channels_db[ch_n][0][0], mdf.channels_db[ch_n][0][1]
            ts, vals = read_simple(mdf, ch_n, ch_g, ch_c)
            if ts is None:
                continue
            vals_g = nearest_interp_1d(ts, vals, t_grid)
            mask = (np.round(sel_id_grid) == slot_id) & (vals_g > 0)
            out[mask] = vals_g[mask]
    # Method 2: fallback SA_Ego_PS_SlotType_NU where unresolved
    ch, grp, cidx = find_channel(mdf, 'SA_Ego_PS_SlotType_NU')
    if ch is not None:
        ts, vals = read_simple(mdf, ch, grp, cidx)
        if ts is not None:
            vals_g = nearest_interp_1d(ts, vals, t_grid)
            nanmask = np.isnan(out)
            out[nanmask] = vals_g[nanmask]
    return out


def _build(mf4_path, veh, t_step):
    """Full parse path (cache miss): open MDF, build MdfData + auxiliary grids.

    Returns (md, None) on success or (None, error_string)."""
    from asammdf import MDF
    from mdf_reader import (build_time_grid, MdfData, find_channel,
                            read_simple, nearest_interp_1d)
    from extract_scenarios import has_required_signals

    mdf = MDF(mf4_path)
    try:
        ok, reason = has_required_signals(mdf)
        if not ok:
            return None, f"required signals: {reason}"
        t_grid = build_time_grid(mdf, step=t_step)
        if len(t_grid) == 0:
            return None, "empty time grid"
        md = MdfData(mdf, t_grid, veh)
        if md.trigger is None:
            return None, "no trigger signal data"
        # Auxiliary grids (so cache hits never need to re-open the MDF)
        md._pz_grid = None
        ch, grp, cidx = find_channel(mdf, 'Psi_PrkgInfo.PrkgZone_u8')
        if ch is not None:
            ts, vals = read_simple(mdf, ch, grp, cidx)
            if ts is not None:
                md._pz_grid = nearest_interp_1d(ts, vals, t_grid)
        md._sel_id_grid = None
        ch, grp, cidx = find_channel(mdf, 'Driver_Selected_Parking_Space_ID')
        if ch is not None:
            ts, vals = read_simple(mdf, ch, grp, cidx)
            if ts is not None:
                md._sel_id_grid = nearest_interp_1d(ts, vals, t_grid)
        md._slot_type_grid = _slot_type_grid(mdf, t_grid, md._sel_id_grid)
        return md, None
    finally:
        mdf.close()


def load_md(mf4_path, veh, t_step=0.05, use_cache=True):
    """Return (MdfData, None) or (None, error_string). Opens the MF4 only on a
    cache miss; on a hit the parsed arrays are restored from the pickled file."""
    key = (os.stat(mf4_path).st_mtime_ns, os.stat(mf4_path).st_size)
    rkey = _reader_hash()
    cpath = _cache_path(mf4_path)

    if use_cache and os.path.exists(cpath):
        try:
            with gzip.open(cpath, 'rb') as f:
                obj = pickle.load(f)
            if obj.get('key') == key and obj.get('rkey') == rkey:
                md = object.__new__(mr.MdfData)
                md.__dict__.update(obj['data'])
                md.veh = veh
                return md, None
        except Exception:
            pass  # corrupted/stale cache → rebuild

    t0 = time.time()
    md, err = _build(mf4_path, veh, t_step)
    if md is not None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        data = {k: v for k, v in vars(md).items() if k not in ('mdf', 'veh')}
        tmp = cpath + '.tmp'
        try:
            with gzip.open(tmp, 'wb') as f:
                pickle.dump({'key': key, 'rkey': rkey, 'data': data}, f,
                            protocol=4)
            os.replace(tmp, cpath)
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
    return md, err
