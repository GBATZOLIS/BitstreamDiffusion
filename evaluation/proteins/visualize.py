"""Protein structure visualization renderers for the evaluation pipeline.

This module gathers the figure-producing helpers used when inspecting generated
backbones, self-consistency refolds, denoising trajectories, and motif designs.
Every renderer lazily imports its rendering backend inside the function that
needs it and raises a clear, actionable error when the backend is missing, so
the module itself imports with only numpy (and torch) present.

Two families of renderer are provided. The native cartoon renderers drive
external structure viewers (UCSF ChimeraX, PyMOL, or py3Dmol) that compute
secondary structure and ribbons themselves. The lightweight renderers draw
backbone traces with matplotlib and colour them from per-residue quantities
that this module computes with numpy, such as pLDDT read from the B-factor
column or per-residue self-consistency deviation.

Supported colour modes across the renderers:
    'plddt'   colour by the per-residue pLDDT stored in the B-factor column,
              using the AlphaFold confidence palette.
    'scrmsd'  colour by per-residue self-consistency deviation (a green to red
              gradient from small to large deviation).
    'ss'      colour by secondary structure (helix, strand, coil).
    'rainbow' colour from the N terminus to the C terminus as a rainbow.
"""

from __future__ import annotations

import importlib
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Sequence, Union

import numpy as np

COLOR_MODES = ("plddt", "scrmsd", "ss", "rainbow")

PathLike = Union[str, Path]

_INSTALL_REFERENCE = (
    "See the protein-eval extra (uv sync --extra protein-eval) and the dplm-inference extra (uv sync --extra dplm-inference), or run "
    "scripts/proteins/setup/setup_evaluation.sh, for the protein evaluation dependencies."
)

# AlphaFold confidence palette, expressed as RGB triples in the 0-1 range and
# keyed by the lower bound of each pLDDT confidence band.
_PLDDT_VERY_HIGH = np.array([0x00, 0x53, 0xD6], dtype=np.float64) / 255.0
_PLDDT_CONFIDENT = np.array([0x65, 0xCB, 0xF3], dtype=np.float64) / 255.0
_PLDDT_LOW = np.array([0xFF, 0xDB, 0x13], dtype=np.float64) / 255.0
_PLDDT_VERY_LOW = np.array([0xFF, 0x7D, 0x45], dtype=np.float64) / 255.0

# Anchor colours for the self-consistency deviation gradient (green, yellow, red).
_SCRMSD_ANCHORS = np.array(
    [[0.13, 0.55, 0.13], [1.00, 0.84, 0.00], [0.80, 0.10, 0.10]],
    dtype=np.float64,
)

__all__ = [
    "COLOR_MODES",
    "chimerax_cartoon",
    "pymol_cartoon",
    "py3dmol_view",
    "self_consistency_overlay",
    "denoising_trajectory",
    "motif_render",
    "contact_sheet",
]


# ---------------------------------------------------------------------------
# Backend acquisition helpers
# ---------------------------------------------------------------------------
def _require_python(import_name: str, install_cmd: str):
    """Import an optional Python backend or raise an actionable RuntimeError."""
    try:
        return importlib.import_module(import_name)
    except ImportError as exc:
        raise RuntimeError(
            f"The '{import_name}' package is required for this renderer but is not "
            f"installed. Install it with: {install_cmd}. {_INSTALL_REFERENCE}"
        ) from exc


def _require_tool(candidates: Sequence[str], install_hint: str) -> str:
    """Return the path to the first available command-line tool or raise."""
    for name in candidates:
        resolved = shutil.which(name)
        if resolved:
            return resolved
    raise RuntimeError(
        f"None of the command-line tools {tuple(candidates)} were found on PATH. "
        f"{install_hint}"
    )


def _matplotlib():
    """Import matplotlib with a headless backend or raise an actionable error."""
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError(
            "The 'matplotlib' package is required for backbone rendering but is not "
            f"installed. Install it with: pip install matplotlib. {_INSTALL_REFERENCE}"
        ) from exc
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    # matplotlib >= 3.8 (the pinned floor) auto-registers the '3d' projection
    # used below, so no explicit mpl_toolkits.mplot3d import is needed.
    return matplotlib, plt


def _run_render_command(cmd: Sequence[str], out_png: Path) -> Path:
    """Run an external rendering command and confirm the PNG was produced."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out_png.exists():
        raise RuntimeError(
            "Structure rendering command failed.\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout: {result.stdout.strip()}\n"
            f"stderr: {result.stderr.strip()}"
        )
    return out_png


def _validate_mode(color_by: str, allowed: Sequence[str]) -> str:
    """Normalise and validate a colour-mode string against the allowed set."""
    mode = str(color_by).lower()
    if mode not in allowed:
        raise ValueError(
            f"color_by={color_by!r} is not supported here; choose one of {tuple(allowed)}."
        )
    return mode


# ---------------------------------------------------------------------------
# PDB parsing and geometry (numpy only)
# ---------------------------------------------------------------------------
def _coerce_pdb_text(source: PathLike) -> str:
    """Return PDB text whether the source is a path or a literal PDB string."""
    if isinstance(source, Path):
        return source.read_text()
    text = str(source)
    stripped = text.lstrip()
    if "\n" in text or stripped.startswith(
        ("ATOM", "HETATM", "HEADER", "MODEL", "REMARK", "CRYST", "TITLE")
    ):
        return text
    return Path(text).read_text()


def _parse_backbone(source: PathLike) -> dict:
    """Parse the first model of a PDB into per-residue backbone arrays.

    Returns a dictionary with keys ``ca``, ``n``, ``c``, ``o`` (each an
    ``[L, 3]`` float array, with NaN for absent atoms), ``bfactor`` (``[L]``,
    the CA temperature factor used as a pLDDT proxy), and ``resid`` (``[L]``
    integer residue numbers). Only residues that carry a CA atom are kept.
    """
    text = _coerce_pdb_text(source)
    residues: dict = {}
    order: list = []
    for line in text.splitlines():
        record = line[:6].strip()
        if record == "ENDMDL":
            break
        if record not in ("ATOM", "HETATM"):
            continue
        atom = line[12:16].strip()
        if atom not in ("N", "CA", "C", "O"):
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        try:
            bfactor = float(line[60:66])
        except ValueError:
            bfactor = 0.0
        try:
            resid = int(line[22:26])
        except ValueError:
            resid = len(order)
        key = (line[21], line[22:27])
        if key not in residues:
            residues[key] = {"resid": resid, "atoms": {}, "b": {}}
            order.append(key)
        residues[key]["atoms"][atom] = (x, y, z)
        residues[key]["b"][atom] = bfactor

    order = [key for key in order if "CA" in residues[key]["atoms"]]
    if not order:
        raise ValueError("No CA atoms were found while parsing the backbone.")

    length = len(order)
    result = {
        name: np.full((length, 3), np.nan, dtype=np.float64)
        for name in ("n", "ca", "c", "o")
    }
    bfactor = np.zeros(length, dtype=np.float64)
    resid = np.zeros(length, dtype=np.int64)
    for index, key in enumerate(order):
        entry = residues[key]
        for atom, name in (("N", "n"), ("CA", "ca"), ("C", "c"), ("O", "o")):
            if atom in entry["atoms"]:
                result[name][index] = entry["atoms"][atom]
        resid[index] = entry["resid"]
        bfactor[index] = entry["b"].get(
            "CA", np.mean(list(entry["b"].values()))
        )
    result["bfactor"] = bfactor
    result["resid"] = resid
    return result


def _coerce_ca(frame) -> np.ndarray:
    """Coerce a trajectory frame into an ``[L, 3]`` array of CA coordinates."""
    if isinstance(frame, (str, Path)):
        return _parse_backbone(frame)["ca"]
    array = np.asarray(frame, dtype=np.float64)
    if array.ndim == 3 and array.shape[-1] == 3:
        # Backbone atom tensor [L, atoms, 3]; use the CA slot (index 1) when the
        # canonical N, CA, C, O ordering is present, otherwise the first atom.
        return array[:, 1, :] if array.shape[1] >= 2 else array[:, 0, :]
    if array.ndim == 2 and array.shape[-1] == 3:
        return array
    raise ValueError(
        f"Cannot interpret a frame of shape {array.shape} as CA coordinates."
    )


def _kabsch_align(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Superpose ``mobile`` onto ``reference`` with the Kabsch algorithm."""
    mobile_center = mobile.mean(axis=0)
    reference_center = reference.mean(axis=0)
    m = mobile - mobile_center
    r = reference - reference_center
    covariance = m.T @ r
    u, _, vt = np.linalg.svd(covariance)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return (m @ rotation.T) + reference_center


def _per_residue_deviation(
    mobile: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Return per-residue CA distances after optimal superposition."""
    length = min(mobile.shape[0], reference.shape[0])
    aligned = _kabsch_align(mobile[:length], reference[:length])
    return np.linalg.norm(aligned - reference[:length], axis=1)


# ---------------------------------------------------------------------------
# Colour helpers (numpy only)
# ---------------------------------------------------------------------------
def _rainbow_rgb(count: int) -> np.ndarray:
    """Return ``[count, 3]`` RGB colours running from red (N) to blue (C)."""
    if count <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    if count == 1:
        return np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    hue = np.linspace(0.0, 2.0 / 3.0, count)
    sextant = np.floor(hue * 6.0).astype(int)
    fraction = hue * 6.0 - sextant
    sextant = sextant % 6
    red = np.choose(
        sextant,
        [1.0, 1.0 - fraction, 0.0, 0.0, fraction, np.ones_like(fraction)],
    )
    green = np.choose(
        sextant,
        [fraction, 1.0, 1.0, 1.0 - fraction, 0.0, np.zeros_like(fraction)],
    )
    blue = np.choose(
        sextant,
        [np.zeros_like(fraction), 0.0, fraction, 1.0, 1.0, 1.0 - fraction],
    )
    return np.clip(np.stack([red, green, blue], axis=1), 0.0, 1.0)


def _plddt_rgb(values: np.ndarray) -> np.ndarray:
    """Map pLDDT values to the four-band AlphaFold confidence palette."""
    values = np.asarray(values, dtype=np.float64)
    colors = np.tile(_PLDDT_VERY_LOW, (values.shape[0], 1))
    colors[values >= 50.0] = _PLDDT_LOW
    colors[values >= 70.0] = _PLDDT_CONFIDENT
    colors[values >= 90.0] = _PLDDT_VERY_HIGH
    return colors


def _gradient_rgb(
    values: np.ndarray,
    vmin: float,
    vmax: float,
    anchors: np.ndarray = _SCRMSD_ANCHORS,
) -> np.ndarray:
    """Interpolate values into a multi-anchor colour gradient."""
    values = np.asarray(values, dtype=np.float64)
    span = max(float(vmax) - float(vmin), 1e-8)
    fraction = np.clip((values - float(vmin)) / span, 0.0, 1.0)
    channels = np.stack(
        [
            np.interp(
                fraction, np.linspace(0.0, 1.0, len(anchors)), anchors[:, k]
            )
            for k in range(3)
        ],
        axis=1,
    )
    return channels


def _secondary_structure(backbone: dict) -> np.ndarray:
    """Return a per-residue secondary-structure label array using biotite.

    Labels are single characters: ``a`` for helix, ``b`` for strand, and ``c``
    for coil. Biotite is imported lazily because it is only needed for the
    ``'ss'`` colour mode of the matplotlib renderers.
    """
    struc = _require_python(
        "biotite.structure",
        "pip install biotite (also listed in the dplm-inference extra (uv sync --extra dplm-inference))",
    )
    ca = backbone["ca"]
    length = ca.shape[0]
    array = struc.AtomArray(length)
    array.coord = ca.astype(np.float32)
    array.chain_id = np.full(length, "A")
    array.res_id = backbone["resid"].astype(int)
    array.res_name = np.full(length, "GLY")
    array.atom_name = np.full(length, "CA")
    array.element = np.full(length, "C")
    return struc.annotate_sse(array)


def _residue_rgb(
    mode: str, backbone: dict, deviations: np.ndarray = None
) -> np.ndarray:
    """Build a per-residue RGB array for the matplotlib renderers."""
    length = backbone["ca"].shape[0]
    if mode == "plddt":
        return _plddt_rgb(backbone["bfactor"])
    if mode == "rainbow":
        return _rainbow_rgb(length)
    if mode == "scrmsd":
        if deviations is None:
            raise ValueError(
                "color_by='scrmsd' requires per-residue deviations."
            )
        upper = float(np.nanmax(deviations)) if deviations.size else 1.0
        return _gradient_rgb(deviations, 0.0, max(upper, 2.0))
    if mode == "ss":
        labels = _secondary_structure(backbone)
        palette = {
            "a": [0.85, 0.15, 0.15],
            "b": [0.95, 0.80, 0.10],
            "c": [0.55, 0.55, 0.55],
        }
        return np.array(
            [palette.get(str(label), palette["c"]) for label in labels],
            dtype=np.float64,
        )
    raise ValueError(f"Unsupported colour mode {mode!r}.")


# ---------------------------------------------------------------------------
# Matplotlib backbone drawing helper
# ---------------------------------------------------------------------------
def _draw_backbone(
    ax, ca: np.ndarray, colors: np.ndarray, linewidth: float = 2.0
) -> None:
    """Draw a CA backbone trace on a 3d axis with per-residue colours."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    if ca.shape[0] >= 2:
        segments = np.stack([ca[:-1], ca[1:]], axis=1)
        collection = Line3DCollection(
            segments, colors=colors[:-1], linewidths=linewidth
        )
        ax.add_collection3d(collection)
    ax.scatter(ca[:, 0], ca[:, 1], ca[:, 2], c=colors, s=18, depthshade=False)


def _set_equal_aspect(ax, ca: np.ndarray, bounds: np.ndarray = None) -> None:
    """Apply equal data aspect and axis limits to a 3d backbone axis."""
    points = ca if bounds is None else bounds
    lower = np.nanmin(points, axis=0)
    upper = np.nanmax(points, axis=0)
    center = (lower + upper) / 2.0
    radius = max(float(np.max(upper - lower)) / 2.0, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_axis_off()


# ---------------------------------------------------------------------------
# Native cartoon renderers
# ---------------------------------------------------------------------------
def _chimerax_color_command(mode: str) -> str:
    """Return the ChimeraX colouring command for a supported colour mode."""
    if mode == "plddt":
        return (
            "color byattribute bfactor palette "
            "0,#ff7d45:50,#ffdb13:70,#65cbf3:90,#0053d6"
        )
    if mode == "rainbow":
        return "rainbow residues palette rainbow"
    if mode == "ss":
        return "color coil gray; color helix #d02020; color strand #e0c020"
    raise ValueError(
        f"color_by={mode!r} is not available for a single-structure ChimeraX cartoon; "
        "use self_consistency_overlay for per-residue deviation colouring."
    )


def chimerax_cartoon(
    pdb_path: PathLike, out_png: PathLike, color_by: str = "plddt"
) -> Path:
    """Render a publication-quality cartoon with UCSF ChimeraX offscreen.

    This drives the ``chimerax`` command-line executable in headless offscreen
    mode. Supported colour modes are ``'plddt'`` (B-factor palette), ``'ss'``
    (secondary structure) and ``'rainbow'`` (N to C). ChimeraX must be on the
    PATH; a clear RuntimeError is raised otherwise. Returns the output path.
    """
    mode = _validate_mode(color_by, ("plddt", "ss", "rainbow"))
    executable = _require_tool(
        ("chimerax", "ChimeraX"),
        "Install UCSF ChimeraX from https://www.cgl.ucsf.edu/chimerax/ and ensure "
        "'chimerax' is on PATH.",
    )
    pdb_path = Path(pdb_path)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    script = "\n".join(
        [
            f"open {pdb_path}",
            "hide atoms",
            "show cartoon",
            _chimerax_color_command(mode),
            "lighting soft",
            "set bgColor white",
            "view",
            f"save {out_png} width 1200 height 900 supersample 3",
            "exit",
        ]
    )
    with tempfile.NamedTemporaryFile(
        "w", suffix=".cxc", delete=False
    ) as handle:
        handle.write(script)
        script_path = handle.name
    return _run_render_command(
        [executable, "--nogui", "--offscreen", "--silent", script_path],
        out_png,
    )


def _pymol_color_commands(mode: str) -> list:
    """Return the PyMOL colouring commands for a supported colour mode."""
    if mode == "plddt":
        # Low pLDDT maps to red, high pLDDT to blue, matching AlphaFold intent.
        return [
            "spectrum b, red_yellow_blue, structure, minimum=50, maximum=90"
        ]
    if mode == "rainbow":
        return ["spectrum count, rainbow, structure and name CA"]
    if mode == "ss":
        return [
            "dss structure",
            "color red, structure and ss H",
            "color yellow, structure and ss S",
            "color cyan, structure and (ss L+'')",
        ]
    raise ValueError(
        f"color_by={mode!r} is not available for a single-structure PyMOL cartoon; "
        "use self_consistency_overlay for per-residue deviation colouring."
    )


def pymol_cartoon(
    pdb_path: PathLike, out_png: PathLike, color_by: str = "plddt"
) -> Path:
    """Render a cartoon with PyMOL as a fallback when ChimeraX is unavailable.

    This drives the ``pymol`` command-line executable in quiet, no-GUI mode with
    ray tracing enabled. Supported colour modes match chimerax_cartoon:
    ``'plddt'``, ``'ss'`` and ``'rainbow'``. PyMOL must be on the PATH; a clear
    RuntimeError is raised otherwise. Returns the output path.
    """
    mode = _validate_mode(color_by, ("plddt", "ss", "rainbow"))
    executable = _require_tool(
        ("pymol",),
        "Install PyMOL (for example: conda install -c conda-forge pymol-open-source) "
        "and ensure 'pymol' is on PATH.",
    )
    pdb_path = Path(pdb_path)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    commands = [
        f"load {pdb_path}, structure",
        "hide everything",
        "show cartoon",
        "bg_color white",
        "set ray_opaque_background, 0",
        *_pymol_color_commands(mode),
        "orient",
        f"png {out_png}, width=1200, height=900, dpi=150, ray=1",
        "quit",
    ]
    with tempfile.NamedTemporaryFile(
        "w", suffix=".pml", delete=False
    ) as handle:
        handle.write("\n".join(commands))
        script_path = handle.name
    return _run_render_command([executable, "-cq", script_path], out_png)


def py3dmol_view(
    pdb_str: str,
    color_by: str = "plddt",
    width: int = 640,
    height: int = 480,
    return_html: bool = False,
):
    """Build an interactive py3Dmol cartoon view for notebooks or HTML export.

    Accepts a PDB string (or path) and returns the py3Dmol view object, or its
    standalone HTML when ``return_html`` is True. Supported colour modes are
    ``'plddt'`` (B-factor gradient), ``'ss'`` (secondary structure computed by
    3Dmol) and ``'rainbow'`` (N to C spectrum). Per-residue self-consistency
    colouring is not available here; use self_consistency_overlay instead.
    py3Dmol must be installed; a clear RuntimeError is raised otherwise.
    """
    mode = _validate_mode(color_by, ("plddt", "ss", "rainbow"))
    py3dmol = _require_python("py3Dmol", "pip install py3Dmol")
    text = _coerce_pdb_text(pdb_str)
    view = py3dmol.view(width=width, height=height)
    view.addModel(text, "pdb")
    if mode == "plddt":
        style = {
            "cartoon": {
                "colorscheme": {
                    "prop": "b",
                    "gradient": "roygb",
                    "min": 50,
                    "max": 90,
                }
            }
        }
    elif mode == "ss":
        style = {"cartoon": {"colorscheme": "ssJmol"}}
    else:
        style = {"cartoon": {"color": "spectrum"}}
    view.setStyle({}, style)
    view.setBackgroundColor("white")
    view.zoomTo()
    if return_html:
        return view._make_html()
    return view


# ---------------------------------------------------------------------------
# Analysis-driven matplotlib renderers
# ---------------------------------------------------------------------------
def self_consistency_overlay(
    gen_pdb: PathLike, refold_pdb: PathLike, out_png: PathLike
) -> dict:
    """Overlay a generated backbone and its refold, coloured by deviation.

    The refolded backbone is superposed onto the generated one, and the
    generated CA trace is coloured by the per-residue CA deviation (a green to
    red gradient). The refold is drawn as a faint reference trace. The figure is
    saved to ``out_png``. Returns a dictionary with the output path, the
    per-residue deviations, and the mean and maximum deviation.
    """
    _matplotlib()
    import matplotlib.pyplot as plt

    gen = _parse_backbone(gen_pdb)
    refold = _parse_backbone(refold_pdb)
    length = min(gen["ca"].shape[0], refold["ca"].shape[0])
    gen_ca = gen["ca"][:length]
    refold_ca = refold["ca"][:length]
    aligned_refold = _kabsch_align(refold_ca, gen_ca)
    deviations = np.linalg.norm(aligned_refold - gen_ca, axis=1)
    colors = _gradient_rgb(deviations, 0.0, max(float(deviations.max()), 2.0))

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(7, 6))
    ax = figure.add_subplot(111, projection="3d")
    if length >= 2:
        ax.plot(
            aligned_refold[:, 0],
            aligned_refold[:, 1],
            aligned_refold[:, 2],
            color="0.7",
            linewidth=1.0,
        )
    _draw_backbone(ax, gen_ca, colors, linewidth=2.5)
    _set_equal_aspect(ax, np.concatenate([gen_ca, aligned_refold], axis=0))
    ax.set_title(
        f"Self-consistency overlay (mean CA deviation {deviations.mean():.2f} A)"
    )
    figure.tight_layout()
    figure.savefig(out_png, dpi=200)
    plt.close(figure)
    return {
        "out_png": out_png,
        "deviations": deviations,
        "mean_deviation": float(deviations.mean()),
        "max_deviation": float(deviations.max()),
    }


def denoising_trajectory(frames: Iterable, out_dir: PathLike) -> list:
    """Render the backbone at each reverse-diffusion step to a directory.

    ``frames`` is an ordered iterable of backbone frames, where each frame is a
    PDB path or string, an ``[L, 3]`` CA array, or an ``[L, atoms, 3]`` backbone
    tensor. Every frame is rendered as an N-to-C rainbow CA trace with shared
    axis limits so the frames are directly comparable, and written to
    ``out_dir/frame_XXXX.png``. Returns the list of output paths.
    """
    _matplotlib()
    import matplotlib.pyplot as plt

    coordinate_frames = [_coerce_ca(frame) for frame in frames]
    if not coordinate_frames:
        raise ValueError("denoising_trajectory requires at least one frame.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stacked = np.concatenate(coordinate_frames, axis=0)
    outputs = []
    for index, ca in enumerate(coordinate_frames):
        figure = plt.figure(figsize=(6, 6))
        ax = figure.add_subplot(111, projection="3d")
        _draw_backbone(ax, ca, _rainbow_rgb(ca.shape[0]), linewidth=2.0)
        _set_equal_aspect(ax, ca, bounds=stacked)
        ax.set_title(f"reverse step {index}")
        figure.tight_layout()
        out_png = out_dir / f"frame_{index:04d}.png"
        figure.savefig(out_png, dpi=150)
        plt.close(figure)
        outputs.append(out_png)
    return outputs


def motif_render(
    design_pdb: PathLike, motif_residues: Sequence[int], out_png: PathLike
) -> Path:
    """Render a design backbone with its scaffolded motif residues highlighted.

    The full backbone is drawn as a faint grey trace and the motif residues are
    overlaid as a bold red trace with markers. ``motif_residues`` entries are
    matched against both the 0-based position in the parsed chain and the PDB
    residue numbers, so either convention works. The figure is saved to
    ``out_png``, which is returned.
    """
    _matplotlib()
    import matplotlib.pyplot as plt

    backbone = _parse_backbone(design_pdb)
    ca = backbone["ca"]
    length = ca.shape[0]
    wanted = set(int(value) for value in motif_residues)
    resid = backbone["resid"]
    is_motif = np.array(
        [
            (position in wanted) or (int(resid[position]) in wanted)
            for position in range(length)
        ],
        dtype=bool,
    )

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(6, 6))
    ax = figure.add_subplot(111, projection="3d")
    if length >= 2:
        ax.plot(ca[:, 0], ca[:, 1], ca[:, 2], color="0.75", linewidth=1.5)
    if is_motif.any():
        motif_ca = ca[is_motif]
        ax.plot(
            motif_ca[:, 0],
            motif_ca[:, 1],
            motif_ca[:, 2],
            color="#d02020",
            linewidth=3.0,
        )
        ax.scatter(
            motif_ca[:, 0],
            motif_ca[:, 1],
            motif_ca[:, 2],
            color="#d02020",
            s=40,
            depthshade=False,
        )
    _set_equal_aspect(ax, ca)
    ax.set_title(f"Motif design ({int(is_motif.sum())} motif residues)")
    figure.tight_layout()
    figure.savefig(out_png, dpi=200)
    plt.close(figure)
    return out_png


def contact_sheet(
    image_paths: Sequence[PathLike], out_png: PathLike, cols: int = 4
) -> Path:
    """Assemble rendered images into a single grid contact sheet with PIL.

    Loads the given images, arranges them into a grid ``cols`` wide (rows are
    added as needed), pastes each into a uniformly sized cell on a white canvas,
    and writes the result to ``out_png``. Pillow must be installed; a clear
    RuntimeError is raised otherwise. Returns the output path.
    """
    pil_image = _require_python("PIL.Image", "pip install pillow")
    paths = [Path(item) for item in image_paths]
    if not paths:
        raise ValueError("contact_sheet requires at least one image path.")
    cols = max(1, int(cols))
    images = [pil_image.open(path).convert("RGB") for path in paths]
    cell_width = max(image.width for image in images)
    cell_height = max(image.height for image in images)
    rows = math.ceil(len(images) / cols)
    pad = 8
    canvas_width = cols * cell_width + (cols + 1) * pad
    canvas_height = rows * cell_height + (rows + 1) * pad
    sheet = pil_image.new("RGB", (canvas_width, canvas_height), color="white")
    for index, image in enumerate(images):
        row, col = divmod(index, cols)
        x = pad + col * (cell_width + pad) + (cell_width - image.width) // 2
        y = pad + row * (cell_height + pad) + (cell_height - image.height) // 2
        sheet.paste(image, (x, y))
        image.close()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    return out_png
