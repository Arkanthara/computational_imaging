"""Shared options for the Python SSI depth pipeline.

This module keeps configuration separate from training logic so the code is
easier to read, maintain, and reuse in demo/inference scripts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _resolve_path(path_like: str | Path, base_dir: Path) -> Path:
    """Resolve absolute or relative paths.

    Parameters
    ----------
    path_like : str or pathlib.Path
        Path to resolve.
    base_dir : pathlib.Path
        Base directory used for relative paths.

    Returns
    -------
    pathlib.Path
        Resolved absolute-like path.
    """
    p = Path(path_like)
    return p if p.is_absolute() else (base_dir / p)


@dataclass
class SSIDepthTrainOptions:
    """Structured depth options converted from MATLAB defaults.

    Notes
    -----
    The field names intentionally preserve the original naming from the
    MATLAB codebase to keep serialized models compatible.
    """

    imWidth: int = 32
    gtWidth: int = 16
    nSamp: int = 500000
    nImgs: float = np.inf
    nTrees: int = 8
    fracFtrs: float = 0.25
    minCount: int = 1
    minChild: int = 4
    maxDepth: int = 64
    discretize: str = "pca"
    nClasses: int = 2
    split: str = "gini"
    chnSmooth: float = 2.0
    simSmooth: float = 8.0
    shrink: int = 2
    shrinkCol: int = 8
    nCells: int = 5
    nCellsCol: int = 2
    stride: int = 2
    nTreesEval: int = 4
    nThreads: int = 4
    seed: int = 1
    useParfor: int = 0
    modelDir: str = "models"
    modelFnm: str = "model"
    imResize: tuple[int, int] = (256, 336)
    trainImDir: str = "Train400Im"
    gtMatDir: str = "Train400Depth"
    dataSet: str = "make3d"
    refine: int = 0
    samplesPerImage: int = 1500

    nChns: int = 1
    nChnFtrs: int = 0
    nColChnFtrs: int = 0
    nSimFtrs: int = 0
    nColSimFtrs: int = 0
    nTotFtrs: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert options to a JSON-serializable dictionary.

        Returns
        -------
        dict of str to Any
            Dictionary representation suitable for model serialization.
        """
        out = asdict(self)
        out["imResize"] = [int(self.imResize[0]), int(self.imResize[1])]
        if np.isinf(self.nImgs):
            out["nImgs"] = "inf"
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SSIDepthTrainOptions":
        """Build an option object from a dictionary.

        Parameters
        ----------
        data : dict of str to Any
            Serialized options dictionary.

        Returns
        -------
        SSIDepthTrainOptions
            Parsed options object.
        """
        raw = dict(data)
        if raw.get("nImgs") == "inf":
            raw["nImgs"] = np.inf
        if "imResize" in raw:
            im_resize = raw["imResize"]
            raw["imResize"] = (int(im_resize[0]), int(im_resize[1]))
        return cls(**raw)

    def resolve_paths(self, base_dir: Path) -> "SSIDepthTrainOptions":
        """Resolve relative paths against a base directory.

        Parameters
        ----------
        base_dir : pathlib.Path
            Base directory used to resolve ``modelDir``, ``trainImDir``, and
            ``gtMatDir`` when they are relative.

        Returns
        -------
        SSIDepthTrainOptions
            The same instance updated in-place.
        """
        self.modelDir = str(_resolve_path(self.modelDir, base_dir))
        self.trainImDir = str(_resolve_path(self.trainImDir, base_dir))
        self.gtMatDir = str(_resolve_path(self.gtMatDir, base_dir))
        return self

    def update_feature_counts(self, sample_image: np.ndarray) -> None:
        """Populate derived feature-count fields from a sample image.

        Parameters
        ----------
        sample_image : numpy.ndarray
            RGB or grayscale image used only to probe channel dimensions.

        Returns
        -------
        None
            Updates derived feature-count fields on this instance.
        """
        from ssiDepthChns import ssiDepthChns

        (
            _,
            _,
            _,
            _,
            n_chn_ftrs,
            n_col_chn_ftrs,
            n_sim_ftrs,
            n_col_sim_ftrs,
            n_chns,
        ) = ssiDepthChns(sample_image, self)

        self.nChns = int(n_chns)
        self.nChnFtrs = int(n_chn_ftrs)
        self.nColChnFtrs = int(n_col_chn_ftrs)
        self.nSimFtrs = int(n_sim_ftrs)
        self.nColSimFtrs = int(n_col_sim_ftrs)
        self.nTotFtrs = int(n_chn_ftrs + n_col_chn_ftrs + n_sim_ftrs + n_col_sim_ftrs)
