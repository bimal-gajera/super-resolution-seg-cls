"""Turn raw AI4SmallFarms tiles into the fixed geometry the models require.

Three things happen here, per tile:

1. **Regrid to true 10 m square pixels.** The delivered rasters are on a UTM
   grid with ~9.35 x 9.94 m pixels. PixelDiT is tied to a 480 m footprint at
   48 px, so a 48 px window on the native grid would be 449 x 477 m -- about
   6% off in x, and not square. We reproject once, to exactly 10.0 m.

2. **Rasterize the labels at 0.5 m.** The shipped ``masks/*.tif`` are field
   *boundary lines* rasterized at 10 m (verified: positive fraction 0.207
   against 0.207 for ``*_lines.gpkg``, IoU 0.72; field *areas* are 0.494).
   We go back to the vector source and rasterize both products at 0.5 m:

       band 1  areas  -- field interior, the training target
       band 2  lines  -- boundaries, buffered to ``boundary_width_m``

   Band 1 is what SEED-SR's ``mIoU_S`` measures and what watershed needs to
   produce instances. Band 2 is carried so a boundary target or a
   boundary-aware loss term costs no re-run.

3. **Enumerate windows.** 48 x 48 at stride 48 over the 10 m grid. The label
   grid is exactly 20x finer (10.0 / 0.5), so window ``(r, c)`` in LR pixels
   maps to ``(20r, 20c, 960, 960)`` in label pixels with no rounding -- which
   is the practical reason 0.5 m is a convenient choice.

Output layout::

    data/prepared/<split>/<tile>_s2_10m.tif       4-band uint16, ~701 x 759
    data/prepared/<split>/<tile>_label_0p5m.tif   2-band uint8, exactly 20x
    data/prepared/manifest.parquet                one row per window
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject

from ..constants import LABEL_GSD, LR_GSD, LR_PATCH

LABEL_SCALE = int(round(LR_GSD / LABEL_GSD))  # 20


@dataclass(frozen=True)
class Window:
    tile: str
    split: str
    row: int          # LR pixel offset into the 10 m grid
    col: int
    label_row: int    # == row * LABEL_SCALE
    label_col: int
    area_frac: float  # fraction of the window that is field, for QC/filtering


def window_starts(extent: int, patch: int = LR_PATCH, stride: int = LR_PATCH) -> list[int]:
    """Minimal set of start offsets covering ``extent``, all fully in bounds.

    Regular stride, plus a final window flush with the far edge when the last
    regular one does not already land there. That final window overlaps its
    neighbour rather than running off the tile -- the only overlap in the
    default stride-48 configuration.
    """
    if extent < patch:
        return []
    starts = list(range(0, extent - patch + 1, stride))
    if starts[-1] != extent - patch:
        starts.append(extent - patch)
    return starts


def regrid_to_10m(src_path: Path) -> tuple[np.ndarray, Affine, rasterio.crs.CRS, list[str]]:
    """Reproject a tile onto exactly 10.0 m square pixels in its own CRS.

    Keeps the upper-left corner fixed and grows the grid to cover the original
    footprint, so no data is cropped.
    """
    with rasterio.open(src_path) as src:
        west, south, east, north = src.bounds
        width = int(np.ceil((east - west) / LR_GSD))
        height = int(np.ceil((north - south) / LR_GSD))
        dst_transform = Affine(LR_GSD, 0.0, west, 0.0, -LR_GSD, north)

        dst = np.zeros((src.count, height, width), dtype=src.dtypes[0])
        for band in range(src.count):
            reproject(
                source=rasterio.band(src, band + 1),
                destination=dst[band],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=src.crs,
                resampling=Resampling.bilinear,
            )
        descriptions = [d or f"B{i + 1}" for i, d in enumerate(src.descriptions)]
        return dst, dst_transform, src.crs, descriptions


def rasterize_labels(
    areas_path: Path,
    lines_path: Path,
    crs: rasterio.crs.CRS,
    transform_10m: Affine,
    shape_10m: tuple[int, int],
    boundary_width_m: float = 2.0,
) -> tuple[np.ndarray, Affine]:
    """Burn field areas and buffered boundary lines onto the 0.5 m grid.

    The 0.5 m grid is *derived* from the 10 m one by exact integer refinement,
    never computed independently, so window offsets stay aligned.
    """
    h10, w10 = shape_10m
    shape = (h10 * LABEL_SCALE, w10 * LABEL_SCALE)
    transform = Affine(
        LABEL_GSD, 0.0, transform_10m.c,
        0.0, -LABEL_GSD, transform_10m.f,
    )

    out = np.zeros((2, *shape), dtype=np.uint8)

    areas = gpd.read_file(areas_path).to_crs(crs)
    if len(areas):
        out[0] = rasterize(
            ((geom, 1) for geom in areas.geometry if not geom.is_empty),
            out_shape=shape, transform=transform, fill=0, dtype="uint8",
        )

    lines = gpd.read_file(lines_path).to_crs(crs)
    if len(lines):
        # Rasterizing a bare LineString at 0.5 m gives a 0.5 m wide boundary,
        # which is thinner than any real field bund and makes the positive
        # class vanishingly rare. Buffer to a physical width instead.
        buffered = lines.geometry.buffer(boundary_width_m / 2.0)
        out[1] = rasterize(
            ((geom, 1) for geom in buffered if not geom.is_empty),
            out_shape=shape, transform=transform, fill=0, dtype="uint8",
        )

    return out, transform


def _write(path: Path, array: np.ndarray, transform: Affine, crs, descriptions=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count, height, width = array.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=count,
        dtype=array.dtype, crs=crs, transform=transform,
        tiled=True, blockxsize=512, blockysize=512, compress="deflate", predictor=2,
    ) as dst:
        dst.write(array)
        if descriptions:
            for i, name in enumerate(descriptions, 1):
                dst.set_band_description(i, name)


def prepare_tile(
    tile: str,
    split: str,
    image_path: Path,
    areas_path: Path,
    lines_path: Path,
    out_dir: Path,
    boundary_width_m: float = 2.0,
    overwrite: bool = False,
) -> list[Window]:
    """Regrid, rasterize, enumerate. Returns one ``Window`` per patch."""
    s2_out = out_dir / split / f"{tile}_s2_10m.tif"
    label_out = out_dir / split / f"{tile}_label_0p5m.tif"

    if overwrite or not (s2_out.exists() and label_out.exists()):
        s2, transform_10m, crs, descriptions = regrid_to_10m(image_path)
        _write(s2_out, s2, transform_10m, crs, descriptions)
        labels, label_transform = rasterize_labels(
            areas_path, lines_path, crs, transform_10m, s2.shape[1:], boundary_width_m
        )
        _write(label_out, labels, label_transform, crs, ["areas", "lines"])
    else:
        with rasterio.open(s2_out) as src:
            s2 = src.read()

    _, h10, w10 = s2.shape
    with rasterio.open(label_out) as src:
        areas_band = src.read(1)

    windows = []
    for row in window_starts(h10):
        for col in window_starts(w10):
            lr, lc = row * LABEL_SCALE, col * LABEL_SCALE
            size = LR_PATCH * LABEL_SCALE
            patch = areas_band[lr : lr + size, lc : lc + size]
            windows.append(
                Window(
                    tile=tile, split=split, row=row, col=col,
                    label_row=lr, label_col=lc,
                    area_frac=float(patch.mean()),
                )
            )
    return windows


def write_manifest(windows: list[Window], path: Path) -> pd.DataFrame:
    df = pd.DataFrame([asdict(w) for w in windows])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df
