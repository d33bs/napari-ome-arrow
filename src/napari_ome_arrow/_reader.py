"""
Minimal napari reader for OME-Arrow sources (stack patterns, OME-Zarr, OME-Parquet,
OME-TIFF) plus a fallback .npy example. Returns Image layers ready for napari.
"""

from __future__ import annotations
from typing import Any, Dict, List, Sequence, Tuple, Union
from pathlib import Path
import warnings
import numpy as np

# Adjust import if your package layout differs
from ome_arrow.core import OMEArrow

PathLike = Union[str, Path]
LayerData = Tuple[np.ndarray, Dict[str, Any], str]


def napari_get_reader(path: Union[PathLike, Sequence[PathLike]]):
    """Return a callable if the path looks readable by this plugin."""
    first = str(path[0] if isinstance(path, (list, tuple)) else path).strip()
    p = Path(first)
    s = first.lower()

    looks_stack = any(c in first for c in "<>*")
    looks_zarr = s.endswith(".ome.zarr") or s.endswith(".zarr") or ".zarr/" in s or (p.exists() and p.is_dir() and p.suffix.lower() == ".zarr")
    looks_parquet = s.endswith((".parquet", ".pq")) or p.suffix.lower() in {".parquet", ".pq"}
    looks_tiff = s.endswith((".tif", ".tiff")) or p.suffix.lower() in {".tif", ".tiff"}
    looks_npy = s.endswith(".npy")

    if looks_stack or looks_zarr or looks_parquet or looks_tiff or looks_npy:
        return reader_function
    return None


def reader_function(path: Union[Path, Sequence[Path]]) -> List[LayerData]:
    paths: List[str] = [str(p) for p in (path if isinstance(path, (list, tuple)) else [path])]
    layers: List[LayerData] = []

    for src in paths:
        s = src.lower()
        p = Path(src)
        looks_stack = any(c in src for c in "<>*")
        looks_zarr = s.endswith(".ome.zarr") or s.endswith(".zarr") or ".zarr/" in s or (p.exists() and p.is_dir() and p.suffix.lower() == ".zarr")
        looks_parquet = s.endswith((".ome.parquet", ".parquet", ".pq")) or p.suffix.lower() in {".parquet", ".pq"}
        looks_tiff = s.endswith((".ome.tif", ".ome.tiff", ".tif", ".tiff")) or p.suffix.lower() in {".tif", ".tiff"}
        looks_npy = s.endswith(".npy")

        try:
            add_kwargs: Dict[str, Any] = {"name": p.name}

            if looks_stack or looks_zarr or looks_parquet or looks_tiff:
                obj = OMEArrow(src)
                arr = obj.export(how="numpy", dtype=np.uint16)  # expected TCZYX
                info = obj.info()  # has 'shape': (T, C, Z, Y, X)

                # If something upstream flattened the data to 1D, try to recover YX
                if getattr(arr, "ndim", 0) == 1:
                    T, C, Z, Y, X = info.get("shape", (1, 1, 1, 0, 0))
                    if Y and X and Y * X == arr.size:
                        arr = arr.reshape((1, 1, 1, Y, X))  # TCZYX minimal
                    else:
                        raise ValueError(f"Flat array with unknown shape for {src}: size={arr.size}")

                # Map channel axis if present (TCZYX)
                if arr.ndim >= 5:
                    add_kwargs["channel_axis"] = 1  # C
                elif arr.ndim == 4:
                    # Often (C, Z, Y, X) — treat first dim as channels
                    add_kwargs["channel_axis"] = 0
                elif arr.ndim == 3:
                    # (Z, Y, X) or (C, Y, X)
                    if arr.shape[0] <= 6:  # small first dim -> channels
                        add_kwargs["channel_axis"] = 0
                elif arr.ndim == 2:
                    pass  # (Y, X)
                else:
                    raise ValueError(f"Unsupported array dimensionality {arr.ndim} for {src}")

                layers.append((arr, add_kwargs, "image"))
                continue

            if looks_npy:
                arr = np.load(src)
                if arr.ndim == 1:
                    # Heuristic recovery: perfect square -> reshape to YX
                    n = int(np.sqrt(arr.size))
                    if n * n == arr.size:
                        arr = arr.reshape(n, n)
                    else:
                        raise ValueError(f".npy is 1D and not a square image: {arr.shape}")

                if arr.ndim == 3 and arr.shape[0] <= 6:
                    add_kwargs["channel_axis"] = 0
                layers.append((arr, add_kwargs, "image"))
                continue

            warnings.warn(f"Skipping unrecognized path: {src}")

        except Exception as e:
            warnings.warn(f"Failed to read '{src}': {e}")

    if not layers:
        raise ValueError("No readable inputs found for given path(s).")
    return layers