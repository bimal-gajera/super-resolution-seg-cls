"""Pull AI4SmallFarms from the DANS data station.

The repository is a Dataverse instance. Its whole-dataset zip endpoint
(``/api/access/dataset/:persistentId``) returns HTTP 500 for this record --
too large -- so we enumerate the file listing and fetch each file by id.
Nothing here needs an account; the record is public and CC-BY-4.0.

Files land under ``dest/<directoryLabel>/<filename>``, mirroring the layout
the record declares:

    sentinel-2-asia/{train,validate,test}/images/<tile>.tif   4-band S2 composite
    sentinel-2-asia/{train,validate,test}/masks/<tile>.tif    binary mask @ ~10 m
    sentinel-2-asia/reference/<tile>_areas.gpkg               field polygons
    sentinel-2-asia/reference/<tile>_lines.gpkg               boundary lines
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ..constants import AI4SMALLFARMS_DOI, DANS_BASE

_TIMEOUT = 120


# Dataverse names its digest algorithms in a `checksum.type` field. This
# record uses SHA-1; other instances use MD5. Verify with whatever it declares
# rather than assuming.
_HASH_ALGOS = {"MD5": "md5", "SHA-1": "sha1", "SHA-256": "sha256", "SHA-512": "sha512"}


@dataclass(frozen=True)
class RemoteFile:
    file_id: int
    filename: str
    directory: str
    size: int
    checksum: str | None
    checksum_algo: str | None

    @property
    def relpath(self) -> Path:
        return Path(self.directory) / self.filename

    @property
    def tile(self) -> str:
        """'0_vietnam.tif' -> '0_vietnam'; '0_vietnam_areas.gpkg' -> '0_vietnam'."""
        stem = Path(self.filename).stem
        for suffix in ("_areas", "_lines"):
            if stem.endswith(suffix):
                return stem[: -len(suffix)]
        return stem

    @property
    def country(self) -> str:
        return "vietnam" if "vietnam" in self.filename else "cambodia"

    @property
    def split(self) -> str | None:
        """'sentinel-2-asia/train/images' -> 'train'. None for reference/."""
        parts = Path(self.directory).parts
        return parts[1] if len(parts) >= 3 else None

    @property
    def kind(self) -> str:
        """images | masks | reference | other"""
        parts = Path(self.directory).parts
        if parts and parts[-1] in ("images", "masks", "reference"):
            return parts[-1]
        return "other"


def list_files(doi: str = AI4SMALLFARMS_DOI) -> list[RemoteFile]:
    """Fetch the record's file listing."""
    url = f"{DANS_BASE}/api/datasets/:persistentId/?persistentId={doi}"
    with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:
        payload = json.load(resp)
    if payload.get("status") != "OK":
        raise RuntimeError(f"unexpected Dataverse response: {payload.get('status')!r}")

    out = []
    for entry in payload["data"]["latestVersion"]["files"]:
        df = entry["dataFile"]
        declared = df.get("checksum", {})
        algo = _HASH_ALGOS.get(declared.get("type", ""))
        value = declared.get("value")
        if df.get("md5"):
            algo, value = "md5", df["md5"]
        out.append(
            RemoteFile(
                file_id=df["id"],
                filename=df["filename"],
                directory=entry.get("directoryLabel", ""),
                size=df.get("filesize", 0),
                checksum=value,
                checksum_algo=algo,
            )
        )
    return out


def select(
    files: list[RemoteFile],
    *,
    region: str = "sentinel-2-asia",
    country: str | None = "vietnam",
    kinds: tuple[str, ...] = ("images", "masks", "reference"),
) -> list[RemoteFile]:
    """Narrow the listing. Defaults to everything we need for Vietnam.

    The record also carries a Netherlands set (``sentinel-2-nl``) which this
    project does not use.
    """
    picked = [
        f
        for f in files
        if f.directory.startswith(region)
        and f.kind in kinds
        and (country is None or f.country == country)
    ]
    return sorted(picked, key=lambda f: (f.kind, f.filename))


def _is_complete(path: Path, remote: RemoteFile) -> bool:
    """Already downloaded and intact? Size first, digest only if the size matches.

    An unrecognised digest algorithm degrades to the size check rather than
    failing the download outright.
    """
    if not path.exists():
        return False
    if remote.size and path.stat().st_size != remote.size:
        return False
    if remote.checksum and remote.checksum_algo:
        digest = hashlib.new(remote.checksum_algo)
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest().lower() == remote.checksum.lower()
    return True


def download(files: list[RemoteFile], dest: Path, *, verbose: bool = True) -> list[Path]:
    """Fetch each file, skipping any already present and intact."""
    dest = Path(dest)
    written = []
    for i, remote in enumerate(files, 1):
        target = dest / remote.relpath
        if _is_complete(target, remote):
            if verbose:
                print(f"  [{i}/{len(files)}] have {remote.relpath}")
            written.append(target)
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{DANS_BASE}/api/access/datafile/{remote.file_id}"
        tmp = target.with_suffix(target.suffix + ".part")
        if verbose:
            print(f"  [{i}/{len(files)}] get  {remote.relpath} ({remote.size / 1e6:.1f} MB)")
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp, tmp.open("wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
        tmp.replace(target)

        if not _is_complete(target, remote):
            raise RuntimeError(
                f"verification failed after download: {remote.relpath} "
                f"({remote.checksum_algo or 'size only'})"
            )
        written.append(target)
    return written
