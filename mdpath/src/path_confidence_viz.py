"""Confidence Path Visualization --- :mod:`mdpath.src.path_confidence_viz`
===============================================================================

This module contains the class :class:`ConfidencePathVisualizer` which turns the
per-edge multi-replica confidence produced by
:class:`mdpath.src.confidence.EdgeConfidenceCalculator` into a single,
self-contained ChimeraX viewer for the top-ranked pathways.

The pathways are drawn as spline tubes where

* **colour** encodes confidence (blue = high, red = low) via a robust diverging
  normalization computed once over all rendered paths,
* **thickness** encodes how many of the rendered paths share each connection
  (the same usage-based scaling as MDPath's original spline export), with the
  absolute radius range **auto-scaled to the size of the structure** so it looks
  right on any system, and
* every point is a pickable ChimeraX *marker* carrying its confidence, so
  hovering shows the value (``CONF nn`` = nn %, exact value in the B-factor).

The output is a standalone ``.py`` ChimeraX script with the structure and the
geometry embedded as base64 -- the user just drags it onto a ChimeraX window.

Classes
--------

:class:`ConfidencePathVisualizer`
"""

import os
import json
import base64
import textwrap

import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline, PchipInterpolator
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize, to_hex


class ConfidencePathVisualizer:
    """Build a self-contained ChimeraX viewer for confidence-coloured paths.

    All methods are static; the entry point is :meth:`write_chimerax_script`.

    Attributes:
        NAN_RGB (tuple): RGB (0-1) used for connections with undefined confidence.

        AUTOSCALE_FACTOR (float): Fraction of the structure bounding-box diagonal
            used as the maximum tube radius when radii are auto-scaled.

        AUTOSCALE_RATIO (float): Minimum/maximum tube-radius ratio.

        NULL_METRIC_FIELDS (tuple): Field names, in order, of the per-edge
            matched block-reference metric tuple produced by
            :meth:`null_lookup_from_df` and carried through to the ChimeraX
            markers: ``(c_obs, c_null_med, c_null_iqr_lo, c_null_iqr_hi, rho,
            c_corr, c_min_R)``.
    """

    NAN_RGB = (0.6, 0.6, 0.6)
    AUTOSCALE_FACTOR = 0.0145
    AUTOSCALE_RATIO = 0.25
    RADIUS_ABS_BOUNDS = (0.12, 4.0)  # absolute clamp (A) so tiny/huge systems stay sane

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def conf_lookup_from_df(edge_confidence_df: pd.DataFrame) -> dict:
        """Map canonical ``(min, max)`` residue pairs to their confidence value."""
        lut = {}
        for _, row in edge_confidence_df.iterrows():
            a, b = int(row["Residue1"]), int(row["Residue2"])
            lut[(min(a, b), max(a, b))] = float(row["confidence"])
        return lut

    # Order of the per-edge null tuple carried through to the ChimeraX markers.
    NULL_METRIC_FIELDS = (
        "c_obs",
        "c_null_med",
        "c_null_iqr_lo",
        "c_null_iqr_hi",
        "rho",
        "c_corr",
        "c_min_R",
    )

    @staticmethod
    def null_lookup_from_df(edge_null_df: pd.DataFrame | None) -> dict:
        """Map edges to matched block-reference metrics used in ChimeraX hover text.

        The value tuple follows :attr:`NULL_METRIC_FIELDS`:
        ``(c_obs, c_null_med, c_null_iqr_lo, c_null_iqr_hi, rho, c_corr, c_min_R)``.
        The corrected slot reads the ``c_corr_experimental`` column (legacy
        ``c_corr`` is accepted); it is ``NaN`` for below-reference/undefined edges.
        The ``c_null`` inter-quartile band (``c_null_iqr_lo``/``c_null_iqr_hi``)
        and the finite-replica lower bound (``c_min_R``) are optional -- when the
        supplied table lacks them they are filled with ``NaN`` and omitted from the
        hover text.
        """
        if edge_null_df is None:
            return {}
        columns = set(edge_null_df.columns)
        required = {"res_u", "res_v", "c_obs", "c_null_med", "rho"}
        if not required <= columns:
            raise ValueError(
                f"edge_null_df is missing columns: {sorted(required - columns)}"
            )
        corr_col = (
            "c_corr_experimental"
            if "c_corr_experimental" in columns
            else ("c_corr" if "c_corr" in columns else None)
        )
        if corr_col is None:
            raise ValueError(
                "edge_null_df needs a 'c_corr_experimental' (or legacy 'c_corr') column"
            )

        def _optional(row, name):
            return float(getattr(row, name)) if name in columns else float("nan")

        return {
            (
                min(int(row.res_u), int(row.res_v)),
                max(int(row.res_u), int(row.res_v)),
            ): (
                float(row.c_obs),
                float(row.c_null_med),
                _optional(row, "c_null_iqr_lo"),
                _optional(row, "c_null_iqr_hi"),
                float(row.rho),
                _optional(row, corr_col),
                _optional(row, "c_min_R"),
            )
            for row in edge_null_df.itertuples()
        }

    @staticmethod
    def edge_and_node_confidence(path, conf_lookup):
        """Return ``(residues, edge_conf, node_conf)`` for a single path.

        Edge confidence is looked up per consecutive residue pair; node (control
        point) confidence lifts edges to points by averaging the two adjacent
        edges (endpoints take their single neighbour). NaN-tolerant.
        """
        res = [int(x) for x in path]
        edge = np.array(
            [
                conf_lookup.get((min(a, b), max(a, b)), np.nan)
                for a, b in zip(res[:-1], res[1:])
            ],
            float,
        )
        node = np.empty(len(res))
        node[0], node[-1] = edge[0], edge[-1]
        for k in range(1, len(res) - 1):
            seg = edge[k - 1 : k + 1]
            node[k] = np.nanmean(seg) if np.isfinite(seg).any() else np.nan
        return res, edge, node

    @classmethod
    def auto_radius_range(cls, residue_coordinates: dict, scale: float = 1.0):
        """Derive a pleasant ``(radius_min, radius_max)`` from the structure size.

        The maximum radius is a small fraction of the bounding-box diagonal of all
        CA coordinates (so tubes are proportional to the scene on any system),
        clamped to sane absolute bounds. ``scale`` multiplies the result.
        """
        coords = []
        for c in residue_coordinates.values():
            try:
                coords.append(np.asarray(c[0], float))
            except (TypeError, IndexError):
                continue
        lo_abs, hi_abs = cls.RADIUS_ABS_BOUNDS
        if len(coords) < 2:
            rmax = 1.0 * scale
        else:
            arr = np.array(coords)
            diag = float(np.linalg.norm(arr.max(0) - arr.min(0)))
            rmax = diag * cls.AUTOSCALE_FACTOR * scale
        rmax = float(np.clip(rmax, max(lo_abs, 0.4), hi_abs))
        rmin = float(np.clip(rmax * cls.AUTOSCALE_RATIO, lo_abs, rmax))
        return rmin, rmax

    @staticmethod
    def build_norm(values, vmin=None, vcenter=None, vmax=None):
        """Robust diverging normalization (2nd/98th percentile, median centre)."""
        v = np.asarray([x for x in values if np.isfinite(x)], float)
        if v.size == 0:
            return Normalize(0.0, 1.0)
        lo = np.quantile(v, 0.02) if vmin is None else vmin
        hi = np.quantile(v, 0.98) if vmax is None else vmax
        ctr = np.median(v) if vcenter is None else vcenter
        if not lo < hi:
            lo, hi = float(v.min()), float(v.max())
        if not lo < hi:
            return Normalize(lo - 1e-6, hi + 1e-6)
        ctr = min(max(ctr, lo + 1e-9), hi - 1e-9)
        return TwoSlopeNorm(vmin=lo, vcenter=ctr, vmax=hi)

    @classmethod
    def _rgb(cls, c, norm, cmap):
        if not np.isfinite(c):
            return cls.NAN_RGB
        lo, hi = norm.vmin, norm.vmax
        return tuple(cmap(norm(np.clip(c, lo, hi)))[:3])

    @staticmethod
    def _interp_conf(node_conf, n_samples):
        """Shape-preserving interpolation of one per-node confidence series.

        Split out of :meth:`_spline` so a second series (e.g. the production
        confidence carried alongside a differently-coloured one) can be sampled on
        the same parameterization without recomputing the geometry splines.

        Args:
            node_conf (array-like): Per-control-point confidence, may contain NaN.
            n_samples (int): Number of samples along the path.

        Returns:
            np.ndarray: The interpolated series clipped to ``[0, 1]``, or all-NaN
            when fewer than two control points are finite.
        """
        values = np.asarray(node_conf, float)
        finite = np.isfinite(values)
        if finite.sum() < 2:
            return np.full(n_samples, np.nan)
        t = np.linspace(0, 1, len(values))
        tf = np.linspace(0, 1, n_samples)
        return np.clip(PchipInterpolator(t[finite], values[finite])(tf), 0.0, 1.0)

    @classmethod
    def _spline(cls, coords, node_conf, node_radius, n_samples, rmin, rmax):
        """Co-parameterized spline: cubic geometry, shape-preserving conf/radius."""
        coords = np.asarray(coords, float)
        K = len(coords)
        t = np.linspace(0, 1, K)
        tf = np.linspace(0, 1, n_samples)
        kk = min(3, K - 1)
        pts = np.column_stack(
            [make_interp_spline(t, coords[:, d], k=kk)(tf) for d in range(3)]
        )
        cf = cls._interp_conf(node_conf, n_samples)
        rf = np.clip(PchipInterpolator(t, node_radius)(tf), rmin, rmax)
        return pts, cf, rf

    # ----------------------------------------------------------------- main API
    @classmethod
    def write_chimerax_script(
        cls,
        top_pathways,
        residue_coordinates: dict,
        edge_confidence_df: pd.DataFrame,
        pdb_file: str = "first_frame.pdb",
        out_path: str = "confidence_paths_chimerax.py",
        top_n: int = 25,
        n_samples: int = 70,
        cmap_name: str = "RdBu",
        radius_min=None,
        radius_max=None,
        scale: float = 1.0,
        embed_structure: bool = True,
        edge_null_df: pd.DataFrame | None = None,
        color_by: str = "confidence",
    ) -> dict:
        """Write a single self-contained ChimeraX ``.py`` viewer.

        Args:
            top_pathways (list[list[int]]): Rank-sorted paths (residue IDs).
            residue_coordinates (dict): ``{res_id: [CA xyz, ...]}`` (MDPath output).
            edge_confidence_df (pd.DataFrame): Per-edge confidence
                (:meth:`EdgeConfidenceCalculator.build_edge_confidence_df`).
            pdb_file (str): Structure the paths were computed on (embedded).
            out_path (str): Output ``.py`` path.
            top_n (int): Number of top paths to render.
            n_samples (int): Spline samples per path.
            cmap_name (str): Matplotlib diverging colormap (default blue=high/red=low).
            radius_min, radius_max (float | None): Tube radius range in A. If either
                is ``None`` the range is auto-scaled to the structure size.
            scale (float): Multiplier applied to the auto-scaled radii.
            embed_structure (bool): Embed ``pdb_file`` so the output is portable.
            edge_null_df (pd.DataFrame | None): Optional matched block-reference
                edge table. When supplied, hover text includes the matched observed
                and within-source reference stability, ``rho``, and the experimental
                corrected score for the exact edge, and also surfaces the reference
                inter-quartile band and the finite-replica lower bound whenever
                those columns are present (see :meth:`null_lookup_from_df`).
            color_by (str): Which per-edge quantity drives the tube **colour**.
                ``"confidence"`` (default) colours by the production confidence
                ``P``; ``"c_corr"`` colours by the EXPERIMENTAL corrected score
                ``c_corr_experimental`` (``C``) and requires ``edge_null_df``
                (undefined/below-reference edges are gray). Only the colouring
                changes -- the hover text (``P``/``O``/``N``/``R``/``C``/``F``) and
                the pickable atom attributes are identical either way, so the two
                viewers differ solely in tube colour.

        Returns:
            dict: Summary ``{paths, sample_points, radius_min, radius_max, vmin,
            vcenter, vmax, out_path, color_by}``.
        """
        if color_by not in ("confidence", "c_corr"):
            raise ValueError("color_by must be 'confidence' or 'c_corr'")
        cmap = (
            plt.colormaps[cmap_name]
            if hasattr(plt, "colormaps")
            else plt.get_cmap(cmap_name)
        )
        lut = cls.conf_lookup_from_df(edge_confidence_df)
        null_lut = cls.null_lookup_from_df(edge_null_df)

        # Colour source: production confidence (P) or experimental corrected score (C).
        if color_by == "c_corr":
            if not null_lut:
                raise ValueError(
                    "color_by='c_corr' requires edge_null_df with a c_corr column"
                )
            corr_index = cls.NULL_METRIC_FIELDS.index("c_corr")
            color_lut = {
                edge: values[corr_index]
                for edge, values in null_lut.items()
                if np.isfinite(values[corr_index])
            }
            if not color_lut:
                raise ValueError(
                    "color_by='c_corr' but no edge is resolved (c_corr is NaN everywhere); "
                    "nothing to colour. Use color_by='confidence' instead."
                )
            color_desc = (
                "EXPERIMENTAL matched-window c_corr heuristic; "
                "undefined/below-reference edges gray"
            )
        else:
            color_lut = lut
            color_desc = "production confidence"

        # Matched design point (constant across edges) for the header, if present.
        design_desc = ""
        if (
            edge_null_df is not None
            and {"block_len_ns", "R_obs", "n_sources"} <= set(edge_null_df.columns)
            and len(edge_null_df)
        ):
            row = edge_null_df.iloc[0]
            design_desc = (
                "  DESIGN = block reference matched at L_obs=%g ns, R_obs=%d, %d source(s)\n"
                % (float(row["block_len_ns"]), int(row["R_obs"]), int(row["n_sources"]))
            )

        # --- per-path control points + confidence ---
        paths = []
        all_color_conf = []
        for idx, raw in enumerate(top_pathways[:top_n]):
            res, edge, node = cls.edge_and_node_confidence(raw, lut)
            _, _, color_node = cls.edge_and_node_confidence(raw, color_lut)
            try:
                coords = np.array(
                    [np.asarray(residue_coordinates[r][0], float) for r in res]
                )
            except KeyError:
                continue
            if len(coords) < 2:
                continue
            metrics = [
                null_lut.get(
                    (min(a, b), max(a, b)), (np.nan,) * len(cls.NULL_METRIC_FIELDS)
                )
                for a, b in zip(res[:-1], res[1:])
            ]
            # Per-edge colour validity: an edge is only coloured if its colour
            # source has a finite value. This is enforced per sample below so the
            # NaN-tolerant spline interpolation cannot bleed a neighbour's colour
            # across an undefined/below-reference edge.
            edge_color_valid = [
                bool(np.isfinite(color_lut.get((min(a, b), max(a, b)), np.nan)))
                for a, b in zip(res[:-1], res[1:])
            ]
            paths.append(
                dict(
                    res=res,
                    coords=coords,
                    node=node,
                    color_node=color_node,
                    metrics=metrics,
                    edge_color_valid=edge_color_valid,
                )
            )
            all_color_conf.extend(color_node[np.isfinite(color_node)].tolist())
        if not paths:
            raise ValueError("No renderable paths (no residue coordinates matched).")

        # --- auto-scaled radius range (unless overridden) ---
        if radius_min is None or radius_max is None:
            rmin, rmax = cls.auto_radius_range(residue_coordinates, scale=scale)
            if radius_min is not None:
                rmin = float(radius_min)
            if radius_max is not None:
                rmax = float(radius_max)
        else:
            rmin, rmax = float(radius_min), float(radius_max)

        # --- thickness = how many rendered paths share each edge ---
        from collections import Counter

        edge_count = Counter()
        for p in paths:
            for a, b in zip(p["res"][:-1], p["res"][1:]):
                edge_count[(min(a, b), max(a, b))] += 1
        max_count = max(edge_count.values()) if edge_count else 1

        def radius_for_count(c):
            if max_count <= 1:
                return 0.5 * (rmin + rmax)
            return rmin + (rmax - rmin) * (c - 1) / (max_count - 1)

        for p in paths:
            er = np.array(
                [
                    radius_for_count(edge_count[(min(a, b), max(a, b))])
                    for a, b in zip(p["res"][:-1], p["res"][1:])
                ],
                float,
            )
            nr = np.empty(len(p["res"]))
            nr[0], nr[-1] = er[0], er[-1]
            for k in range(1, len(p["res"]) - 1):
                nr[k] = 0.5 * (er[k - 1] + er[k])
            p["node_radius"] = nr

        # --- normalization + geometry payload ---
        norm = cls.build_norm(all_color_conf)
        vmin, vmax = float(norm.vmin), float(norm.vmax)
        vcenter = float(getattr(norm, "vcenter", 0.5 * (vmin + vmax)))

        data = []
        n_pts = 0
        for p in paths:
            pts, cf_color, rf = cls._spline(
                p["coords"], p["color_node"], p["node_radius"], n_samples, rmin, rmax
            )
            # Same parameterization, so only the extra series needs interpolating.
            cf_prod = cls._interp_conf(p["node"], n_samples)
            samples = []
            for i in range(len(pts)):
                conf = float(cf_prod[i]) if np.isfinite(cf_prod[i]) else -1.0
                edge_index = min(
                    int(i * (len(p["res"]) - 1) / max(1, len(pts) - 1)),
                    len(p["res"]) - 2,
                )
                # Force samples on a colour-invalid edge to gray so interpolation
                # cannot carry a valid neighbour's colour across it.
                if p["edge_color_valid"][edge_index]:
                    rr, gg, bb = cls._rgb(cf_color[i], norm, cmap)
                else:
                    rr, gg, bb = cls.NAN_RGB
                res_u, res_v = sorted((p["res"][edge_index], p["res"][edge_index + 1]))
                metrics = [
                    value if np.isfinite(value) else -1.0
                    for value in p["metrics"][edge_index]
                ]
                samples.append(
                    [
                        round(float(pts[i, 0]), 2),
                        round(float(pts[i, 1]), 2),
                        round(float(pts[i, 2]), 2),
                        int(round(rr * 255)),
                        int(round(gg * 255)),
                        int(round(bb * 255)),
                        round(conf, 3),
                        round(float(rf[i]), 3),
                        res_u,
                        res_v,
                        *[round(float(value), 3) for value in metrics],
                    ]
                )
            data.append(samples)
            n_pts += len(samples)
        data_json = json.dumps(data, separators=(",", ":"))

        # --- structure + colour key ---
        if embed_structure and os.path.exists(pdb_file):
            with open(pdb_file, "rb") as fh:
                pdb_b64 = _wrap_b64(fh.read())
            pdb_block = '_PDB_B64 = """\n%s\n"""' % pdb_b64
            open_structure = (
                "pdb_path = os.path.join(tempfile.gettempdir(), 'mdpath_conf_structure.pdb')\n"
                "with open(pdb_path, 'wb') as fh:\n"
                "    fh.write(_decode(_PDB_B64))\n"
                'run(session, \'open "%s"\' % pdb_path.replace(os.sep, "/"))'
            )
        else:
            pdb_block = "_PDB_B64 = None"
            open_structure = "# (structure not embedded)"

        data_b64 = _wrap_b64(data_json.encode("utf-8"))
        c_lo, c_mid, c_hi = (
            to_hex(cmap(norm(vmin))),
            to_hex(cmap(norm(vcenter))),
            to_hex(cmap(norm(vmax))),
        )
        key_cmd = (
            "key %s:%.2f %s:%.2f %s:%.2f pos 0.06,0.08 size 0.27,0.03 fontSize 14"
            % (c_lo, vmin, c_mid, vcenter, c_hi, vmax)
        )
        label_cmd = (
            '2dlabels create mdpcap text "MDPath: coloured by %s  (blue = high, red = low)" '
            "xpos 0.06 ypos 0.13 size 14" % color_desc
        )

        script = _CHIMERAX_TEMPLATE.format(
            top_n=len(paths),
            vmin=vmin,
            vcenter=vcenter,
            vmax=vmax,
            rmin=rmin,
            rmax=rmax,
            pdb_block=pdb_block,
            open_structure=open_structure,
            data_b64=data_b64,
            key_cmd=repr(key_cmd),
            label_cmd=repr(label_cmd),
            color_desc=color_desc,
            design_desc=design_desc,
        )
        with open(out_path, "w") as f:
            f.write(script)

        return dict(
            paths=len(paths),
            sample_points=n_pts,
            radius_min=rmin,
            radius_max=rmax,
            vmin=vmin,
            vcenter=vcenter,
            vmax=vmax,
            out_path=out_path,
            color_by=color_by,
        )


def _wrap_b64(raw: bytes) -> str:
    return "\n".join(textwrap.wrap(base64.b64encode(raw).decode("ascii"), 120))


_CHIMERAX_TEMPLATE = '''\
"""
MDPath confidence-colored pathways  ---  ChimeraX viewer (self-contained)
========================================================================
Drag this file onto a ChimeraX window (or:  File > Open, or  open <thisfile>).

Shows the protein together with the top {top_n} MDPath pathways as spline tubes:

  COLOR  = {color_desc}      blue = HIGH, red = LOW, gray = undefined
{design_desc}  WIDTH  = how many paths share a connection (auto-scaled to the structure size;
           radius {rmin:.2f}-{rmax:.2f} A, thin thread .. thick "trunk")
  HOVER  = mouse over any path segment to read its edge and confidence metrics
           (identical regardless of which quantity colours the tube).
           P = production confidence; O = matched observed confidence;
           N = block-reference median with its [IQR lo-hi] band; R = rho
           (cv_obs/cv_null); C = experimental corrected score (Cna = unresolved,
           observed below reference); F = finite-replica lower bound.

Normalization (robust, diverging) baked into this file:
      vmin (red)    = {vmin:.3f}
      vcenter(white)= {vcenter:.3f}
      vmax (blue)   = {vmax:.3f}

The pathways are a ChimeraX *marker model*, so every point is pickable.
Generated by MDPath ConfidencePathVisualizer.
"""
import os, base64, json, tempfile
from chimerax.core.commands import run

{pdb_block}
_DATA_B64 = """
{data_b64}
"""

def _decode(b64):
    return base64.b64decode("".join(b64.split()))

def _safe(cmd):
    try:
        run(session, cmd)
    except Exception as exc:
        print("MDPath: skipped '%s' (%s)" % (cmd, exc))

# ---- structure: gray cartoon (done BEFORE markers exist, so they're untouched) ----
{open_structure}
_safe("hide atoms")
_safe("show cartoon")
_safe("color #b8b8b8 target c")

# ---- confidence pathways as a pickable marker model ----
from chimerax.markers import MarkerSet
try:
    from chimerax.markers import create_link
except Exception:
    from chimerax.markers.markers import create_link
try:
    from chimerax.atomic import Atom
    Atom.register_attr(session, "mdpath_confidence", "MDPath", attr_type=float)
    Atom.register_attr(session, "mdpath_c_obs", "MDPath", attr_type=float)
    Atom.register_attr(session, "mdpath_c_null", "MDPath", attr_type=float)
    Atom.register_attr(session, "mdpath_c_null_iqr_lo", "MDPath", attr_type=float)
    Atom.register_attr(session, "mdpath_c_null_iqr_hi", "MDPath", attr_type=float)
    Atom.register_attr(session, "mdpath_rho", "MDPath", attr_type=float)
    Atom.register_attr(session, "mdpath_c_corr", "MDPath", attr_type=float)
    Atom.register_attr(session, "mdpath_c_min_r", "MDPath", attr_type=float)
    Atom.register_attr(session, "mdpath_edge", "MDPath", attr_type=str)
except Exception:
    pass

data = json.loads(_decode(_DATA_B64).decode("utf-8"))
mset = MarkerSet(session, "MDPath confidence paths")
n_markers = 0
for path in data:
    prev, prev_rgba, prev_rad = None, None, None
    for (x, y, z, R, G, B, conf, rad, res_u, res_v,
         c_obs, c_null, c_null_lo, c_null_hi, rho, c_corr, c_min_r) in path:
        rgba = (R, G, B, 255)
        m = mset.create_marker((x, y, z), rgba, rad)
        n_markers += 1
        if conf >= 0.0:
            try: m.bfactor = conf
            except Exception: pass
            try: m.mdpath_confidence = conf
            except Exception: pass
            try:
                if c_null >= 0.0:
                    name = "E%d-%d_P%.3f_O%.3f_N%.3f" % (res_u, res_v, conf, c_obs, c_null)
                    if c_null_lo >= 0.0 and c_null_hi >= 0.0:
                        name += "[%.3f-%.3f]" % (c_null_lo, c_null_hi)
                    name += ("_R%.2f" % rho) if rho >= 0.0 else "_Rna"
                    # c_corr is left unresolved (below-reference) as a negative sentinel;
                    # show "Cna" rather than a misleading corrected value.
                    name += ("_C%.3f" % c_corr) if c_corr >= 0.0 else "_Cna"
                    if c_min_r >= 0.0:
                        name += "_F%.3f" % c_min_r
                    m.name = name
                    m.residue.name = "MDP"
                    m.residue.number = n_markers
                    m.mdpath_c_obs = c_obs
                    m.mdpath_c_null = c_null
                    m.mdpath_edge = "%d-%d" % (res_u, res_v)
                    if rho >= 0.0:
                        m.mdpath_rho = rho
                    if c_corr >= 0.0:
                        m.mdpath_c_corr = c_corr
                    if c_null_lo >= 0.0:
                        m.mdpath_c_null_iqr_lo = c_null_lo
                    if c_null_hi >= 0.0:
                        m.mdpath_c_null_iqr_hi = c_null_hi
                    if c_min_r >= 0.0:
                        m.mdpath_c_min_r = c_min_r
                else:
                    m.residue.name = "CONF"
                    m.residue.number = int(round(conf * 100))
            except Exception:
                pass
        if prev is not None:
            try:
                create_link(prev, m, prev_rgba, min(rad, prev_rad))
            except Exception:
                pass
        prev, prev_rgba, prev_rad = m, rgba, rad
try:
    if mset.id is None:
        session.models.add([mset])
except Exception as exc:
    print("MDPath: models.add note (%s)" % exc)

# ---- scene + legend ----
_safe("set bgColor white")
_safe("lighting soft")
_safe("graphics silhouettes true")
run(session, "view")
_safe({key_cmd})
_safe({label_cmd})

print("MDPath: {top_n} confidence-colored pathways as %d markers. "
      "Hover a segment for edge, production, observed, null (with IQR band), rho, "
      "experimental corrected score, and the finite-replica lower bound." % n_markers)
'''
