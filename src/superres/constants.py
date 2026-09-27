"""Fixed geometry and band conventions for the whole project.

Every magic number that more than one module needs lives here. The chain is
rigid because PixelDiT is rigid: it was trained on one physical footprint and
one pixel count, and nothing downstream is free to disagree.

    LR input     48 x 48     @ 10.000 m/px  = 480 m across
    PixelDiT SR  1584 x 1584 @  0.303 m/px  = 480 m across
    DINOv3       99 x 99 tokens (patch 16)  = 4.85 m/token
    decoder out  960 x 960   @  0.500 m/px  = 480 m across
    labels       960 x 960   @  0.500 m/px  (rasterized from polygons)

Flow 2 differs only in that the 1584x1584 image is a bicubic stretch of the
48x48 input rather than a PixelDiT generation -- same pixel count, no new
information.
"""

from __future__ import annotations

# --- the 480 m frame PixelDiT was trained on -------------------------------
TILE_METRES = 480.0
LR_PATCH = 48                      # LR pixels across one tile
LR_GSD = TILE_METRES / LR_PATCH    # 10.0 m/px

# --- PixelDiT output -------------------------------------------------------
HR_SIZE = 1584                     # must stay 33 * PIXELDIT_PATCH
PIXELDIT_PATCH = 48                # the model's internal patch size
COARSE = HR_SIZE // PIXELDIT_PATCH  # 33 x 33 patch tokens
HR_GSD = TILE_METRES / HR_SIZE     # 0.30303 m/px

# --- DINOv3 ----------------------------------------------------------------
DINOV3_INPUT = HR_SIZE             # feed the SR at native resolution
DINOV3_PATCH = 16
TOKEN_GRID = DINOV3_INPUT // DINOV3_PATCH   # 99
TOKEN_GSD = TILE_METRES / TOKEN_GRID        # 4.85 m/token
DINOV3_EMBED_DIM = 1024            # ViT-L/16
DINOV3_TAPS = (5, 11, 17, 23)      # 0-indexed blocks of 24, for the DPT head

# DINOv3 SAT-493M normalization. NOT ImageNet -- this checkpoint was distilled
# on satellite imagery and has its own statistics.
SAT493M_MEAN = (0.430, 0.411, 0.296)
SAT493M_STD = (0.213, 0.156, 0.143)

# --- label / prediction grid ----------------------------------------------
# 0.5 m matches the GSD SEED-SR reports at, and is conservative about the
# positional accuracy of the AI4SmallFarms polygons (digitized on high-res
# basemaps, so good to a few metres, not sub-metre).
LABEL_GSD = 0.5
LABEL_SIZE = int(round(TILE_METRES / LABEL_GSD))   # 960

# --- Sentinel-2 bands ------------------------------------------------------
# AI4SmallFarms GeoTIFFs carry band descriptions ('B2','B3','B4','B8').
# Natural colour is B4/B3/B2 (red/green/blue).
RGB_BAND_NAMES = ("B4", "B3", "B2")

# Indices into the 12-band s2_band_stats.json, whose order is
# B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12 (L2A convention, no B10).
# So B4 -> 3, B3 -> 2, B2 -> 1.
RGB_STATS_INDICES = (3, 2, 1)

# --- dataset ---------------------------------------------------------------
AI4SMALLFARMS_DOI = "doi:10.17026/dans-xy6-ngg6"
DANS_BASE = "https://phys-techsciences.datastations.nl"
SPLITS = ("train", "validate", "test")


# --- classification track (torchgeo-bench) ---------------------------------
# PixelDiT's 12 conditioning channels, in the order the checkpoint expects.
# Mirrors the s2_band_stats.json order documented above; B10 (cirrus) is
# absent because the model was trained on L2A, which does not ship it.
S2_12BAND_CODES = (
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B11", "B12",
)

# The same twelve as torchgeo-bench canonical band names. Every classification
# dataset we use declares all twelve, and ``BenchDataset.select_band_specs``
# preserves the requested order -- so asking for this tuple by name yields the
# right channels in the right order without any index arithmetic, and without
# trusting each wrapper's own band ordering.
PIXELDIT_BAND_NAMES = (
    "coastal_aerosol",   # B01
    "blue",              # B02
    "green",             # B03
    "red",               # B04
    "red_edge_1",        # B05
    "red_edge_2",        # B06
    "red_edge_3",        # B07
    "nir",               # B08
    "red_edge_4",        # B8A
    "water_vapour",      # B09
    "swir_1",            # B11
    "swir_2",            # B12
)
assert len(PIXELDIT_BAND_NAMES) == len(S2_12BAND_CODES) == 12

# Which PixelDiT patch block the REPA head was aligned against (1-indexed, as
# the training config names it -- so patch_blocks[ALIGN_LAYER - 1] in code).
# Reading a different block would produce activations the head never calibrated
# on, which is silent: the shapes still match.
ALIGN_LAYER = 8

# Index of R, G, B within PIXELDIT_BAND_NAMES, for the DINOv3 arms.
RGB_IN_PIXELDIT_BANDS = (3, 2, 1)

# DINOv3 input sizes used by the classification arms. 528 gives a 33x33 token
# grid, matching PixelDiT's own patch grid and the REPA head's geometry; 1584
# gives 99x99 and is the SR image at native resolution. Both are run, because
# once an image has been generated a second encoder pass is nearly free, and
# the two token geometries are not interchangeable for a pooled descriptor.
DINOV3_CLS_SIZES = (528, 1584)
