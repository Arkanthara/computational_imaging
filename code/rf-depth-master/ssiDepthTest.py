"""Batch inference helper for the Python SSI depth pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from scipy.io import savemat
from skimage import io

from ssiDepthDetect import ssiDepthDetect


def ssiDepthTest(im_path: str | Path, model: Mapping[str, Any], out_dir: str | Path | None = None) -> None:
    """Run depth inference over a directory of images.

    Parameters
    ----------
    im_path : str or pathlib.Path
        Directory containing input images.
    model : Mapping[str, Any]
        Model dictionary from ``ssiDepthTrain``.
    out_dir : str or pathlib.Path, optional
        Output directory for ``.mat`` depth files. Defaults to
        ``outfolder/<dataset>`` under the script directory.
    """
    script_dir = Path(__file__).resolve().parent
    im_path = Path(im_path)

    if out_dir is None:
        dataset = str(model["opts"].get("dataSet", "make3d"))
        out_dir = script_dir / "outfolder" / dataset
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted([p for p in im_path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])

    for img_path in image_files:
        image = io.imread(img_path)
        depth = ssiDepthDetect(image, model)
        savemat(out_dir / f"{img_path.stem}.mat", {"depth": depth})
