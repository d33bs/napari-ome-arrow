"""
Minimal napari reader for OME-Arrow sources (stack patterns, OME-Zarr, OME-Parquet,
OME-TIFF) plus a fallback .npy example.

Behavior:
    * If NAPARI_OME_ARROW_LAYER_TYPE is set to "image" or "labels",
      that choice is used.
    * Otherwise, in a GUI/Qt context, the user is prompted with a modal
      dialog asking whether to load as image or labels.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from ome_arrow.core import OMEArrow

PathLike = Union[str, Path]
LayerData = Tuple[np.ndarray, Dict[str, Any], str]


def _maybe_set_viewer_3d(arr: np.ndarray) -> None:
    """
    If the array has a Z axis with size > 1, switch the current napari viewer
    to 3D (ndisplay = 3).

    Assumes OME-Arrow's TCZYX convention or a subset, i.e., Z is always
    the third-from-last axis. No-op if there's no active viewer.
    """
    # Need at least (Z, Y, X)
    if arr.ndim < 3:
        return

    z_size = arr.shape[-3]
    if z_size <= 1:
        return

    try:
        import napari

        viewer = napari.current_viewer()
    except Exception:
        # no viewer / not in GUI context → silently skip
        return

    if viewer is not None:
        viewer.dims.ndisplay = 3


# --------------------------------------------------------------------- #
#  Mode selection (env var + GUI prompt)
# --------------------------------------------------------------------- #


def _get_layer_mode(sample_path: str) -> str:
    """
    Decide whether to load as 'image' or 'labels'.

    Priority:
      1. NAPARI_OME_ARROW_LAYER_TYPE env var (image/labels)
      2. If in a Qt GUI context, show a modal dialog asking the user
      3. Otherwise, default to 'image'
    """
    mode = os.environ.get("NAPARI_OME_ARROW_LAYER_TYPE")
    if mode is not None:
        mode = mode.lower()
        if mode in {"image", "labels"}:
            return mode
        raise RuntimeError(
            f"Invalid NAPARI_OME_ARROW_LAYER_TYPE={mode!r}; expected 'image' or 'labels'."
        )

    # No env var → try to prompt in GUI context
    try:
        from qtpy import QtWidgets
    except Exception:
        # no Qt, probably headless: default to image
        warnings.warn(
            "NAPARI_OME_ARROW_LAYER_TYPE not set and Qt not available; "
            "defaulting to 'image'."
        )
        return "image"

    app = QtWidgets.QApplication.instance()
    if app is None:
        # Again, likely headless or non-Qt usage
        warnings.warn(
            "NAPARI_OME_ARROW_LAYER_TYPE not set and no QApplication instance; "
            "defaulting to 'image'."
        )
        return "image"

    # Build a simple modal choice dialog
    msg = QtWidgets.QMessageBox()
    msg.setWindowTitle("napari-ome-arrow: choose layer type")
    msg.setText(
        f"<p align='left'>How should '{Path(sample_path).name}' be loaded?<br><br>"
        "You can also set NAPARI_OME_ARROW_LAYER_TYPE=image or labels to skip this prompt.</p>"
    )

    # Use ActionRole for ALL buttons so Qt does NOT reorder them
    image_button = msg.addButton("Image", QtWidgets.QMessageBox.ActionRole)
    labels_button = msg.addButton("Labels", QtWidgets.QMessageBox.ActionRole)
    cancel_button = msg.addButton("Cancel", QtWidgets.QMessageBox.ActionRole)

    # If you want Esc to behave as Cancel:
    msg.setEscapeButton(cancel_button)

    msg.exec_()
    clicked = msg.clickedButton()

    if clicked is labels_button:
        return "labels"
    if clicked is image_button:
        return "image"

    raise RuntimeError("User cancelled napari-ome-arrow load dialog.")


# --------------------------------------------------------------------- #
#  Helper utilities
# --------------------------------------------------------------------- #


def _as_labels(arr: np.ndarray) -> np.ndarray:
    """Convert any array into an integer label array."""
    if arr.dtype.kind == "f":
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.round(arr).astype(np.int32, copy=False)
    elif arr.dtype.kind not in ("i", "u"):
        arr = arr.astype(np.int32, copy=False)
    return arr


def _looks_like_ome_source(path_str: str) -> bool:
    """Basic extension / pattern sniffing for OME-Arrow supported formats."""
    s = path_str.strip().lower()
    p = Path(path_str)

    looks_stack = any(c in path_str for c in "<>*")
    looks_zarr = (
        s.endswith(".ome.zarr")
        or s.endswith(".zarr")
        or ".zarr/" in s
        or (p.exists() and p.is_dir() and p.suffix.lower() == ".zarr")
    )
    looks_parquet = s.endswith(
        (".ome.parquet", ".parquet", ".pq")
    ) or p.suffix.lower() in {
        ".parquet",
        ".pq",
    }
    looks_tiff = s.endswith(
        (".ome.tif", ".ome.tiff", ".tif", ".tiff")
    ) or p.suffix.lower() in {
        ".tif",
        ".tiff",
    }
    looks_npy = s.endswith(".npy")
    return (
        looks_stack or looks_zarr or looks_parquet or looks_tiff or looks_npy
    )


# --------------------------------------------------------------------- #
#  napari entry point: napari_get_reader
# --------------------------------------------------------------------- #


def napari_get_reader(path: Union[PathLike, Sequence[PathLike]]):
    """
    Napari plugin hook: return a reader callable if this plugin can read `path`.

    This MUST return a function object (e.g. `reader_function`) or None.
    """
    # napari may pass a list/tuple or a single path
    first = str(path[0] if isinstance(path, (list, tuple)) else path).strip()

    if _looks_like_ome_source(first):
        return reader_function
    return None


# --------------------------------------------------------------------- #
#  Reader implementation: reader_function
# --------------------------------------------------------------------- #


def _read_one(src: str, mode: str) -> LayerData:
    """
    Read a single source into (data, add_kwargs, layer_type),
    obeying `mode` = 'image' or 'labels'.
    """
    s = src.lower()
    p = Path(src)

    looks_stack = any(c in src for c in "<>*")
    looks_zarr = (
        s.endswith(".ome.zarr")
        or s.endswith(".zarr")
        or ".zarr/" in s
        or (p.exists() and p.is_dir() and p.suffix.lower() == ".zarr")
    )
    looks_parquet = s.endswith(
        (".ome.parquet", ".parquet", ".pq")
    ) or p.suffix.lower() in {
        ".parquet",
        ".pq",
    }
    looks_tiff = s.endswith(
        (".ome.tif", ".ome.tiff", ".tif", ".tiff")
    ) or p.suffix.lower() in {
        ".tif",
        ".tiff",
    }
    looks_npy = s.endswith(".npy")

    add_kwargs: Dict[str, Any] = {"name": p.name}

    # ---- OME-Arrow-backed sources -----------------------------------
    if looks_stack or looks_zarr or looks_parquet or looks_tiff:
        obj = OMEArrow(src)
        arr = obj.export(how="numpy", dtype=np.uint16)  # TCZYX
        info = obj.info()  # may contain 'shape': (T, C, Z, Y, X)

        # Recover from accidental 1D flatten
        if getattr(arr, "ndim", 0) == 1:
            T, C, Z, Y, X = info.get("shape", (1, 1, 1, 0, 0))
            if Y and X and arr.size == Y * X:
                arr = arr.reshape((1, 1, 1, Y, X))
            else:
                raise ValueError(
                    f"Flat array with unknown shape for {src}: size={arr.size}"
                )

        if mode == "image":
            # Image: preserve channels
            if arr.ndim >= 5:
                add_kwargs["channel_axis"] = 1  # TCZYX
            elif arr.ndim == 4:
                add_kwargs["channel_axis"] = 0  # CZYX
            layer_type = "image"
        else:
            # Labels: squash channels, ensure integer dtype
            if arr.ndim == 5:  # (T, C, Z, Y, X)
                arr = arr[:, 0, ...]
            elif arr.ndim == 4:  # (C, Z, Y, X)
                arr = arr[0, ...]
            arr = _as_labels(arr)
            add_kwargs.setdefault("opacity", 0.7)
            layer_type = "labels"

        # 🔹 Ask viewer to switch to 3D if there is a real Z-stack
        _maybe_set_viewer_3d(arr)

        return arr, add_kwargs, layer_type

    # ---- bare .npy fallback -----------------------------------------
    if looks_npy:
        arr = np.load(src)
        if arr.ndim == 1:
            n = int(np.sqrt(arr.size))
            if n * n == arr.size:
                arr = arr.reshape(n, n)
            else:
                raise ValueError(
                    f".npy is 1D and not a square image: {arr.shape}"
                )

        if mode == "image":
            if arr.ndim == 3 and arr.shape[0] <= 6:
                add_kwargs["channel_axis"] = 0
            layer_type = "image"
        else:
            # labels
            if arr.ndim == 3:  # treat as (C, Y, X) → first channel
                arr = arr[0, ...]
            arr = _as_labels(arr)
            add_kwargs.setdefault("opacity", 0.7)
            layer_type = "labels"

        # 🔹 Same 3D toggle for npy-based data
        _maybe_set_viewer_3d(arr)

        return arr, add_kwargs, layer_type

    raise ValueError(f"Unrecognized path for napari-ome-arrow reader: {src}")


def reader_function(
    path: Union[PathLike, Sequence[PathLike]],
) -> List[LayerData]:
    """
    The actual reader callable napari will use.

    It reads one or more paths, prompting the user (or using the env var)
    to decide 'image' vs 'labels', and returns a list of LayerData tuples.
    """
    paths: List[str] = [
        str(p) for p in (path if isinstance(path, (list, tuple)) else [path])
    ]
    layers: List[LayerData] = []

    # Use the first path as context for the dialog label
    try:
        mode = _get_layer_mode(sample_path=paths[0])  # 'image' or 'labels'
    except RuntimeError as e:
        # If user canceled the dialog, propagate a clean error for napari
        raise ValueError(str(e)) from e

    for src in paths:
        try:
            layers.append(_read_one(src, mode=mode))
        except Exception as e:
            warnings.warn(f"Failed to read '{src}' with napari-ome-arrow: {e}")

    if not layers:
        raise ValueError("No readable inputs found for given path(s).")
    return layers
