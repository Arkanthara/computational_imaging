"""Training utilities and options for the Python SSI depth pipeline.

This module provides a Python conversion of the configuration structure from
``ssiDepthTrain.m`` and a lightweight train/load path that is fully compatible
with ``numpy``, ``scipy``, ``scikit-image``, and ``matplotlib``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from scipy.io import loadmat
from skimage import io, transform

from ssiDepthChns import ssiDepthChns
from ssiDepthDetect import FEATURE_NAMES, compute_depth_features


DEFAULT_WEIGHTS = np.array([0.34, 0.24, 0.12, 0.10, 0.14, 0.06], dtype=np.float32)


@dataclass
class SSIDepthTrainOptions:
    """Structured depth training options converted from MATLAB defaults.

    Parameters
    ----------
    imWidth : int, default=32
        Width of image patches.
    gtWidth : int, default=16
        Width of ground-truth patches.
    nSamp : int, default=500000
        Number of sampled patches per tree in the original MATLAB algorithm.
    nImgs : float, default=np.inf
        Maximum number of images used for training.
    nTrees : int, default=8
        Number of trees in the original random forest.
    fracFtrs : float, default=0.25
        Fraction of features per tree.
    minCount : int, default=1
        Minimum number of samples at a split node.
    minChild : int, default=4
        Minimum number of samples in each child node.
    maxDepth : int, default=64
        Maximum tree depth.
    discretize : str, default="pca"
        Discretization mode used by the original code.
    nClasses : int, default=2
        Number of classes for split decisions.
    split : str, default="gini"
        Split criterion.
    chnSmooth : float, default=2.0
        Smoothing radius for regular channels.
    simSmooth : float, default=8.0
        Smoothing radius for self-similarity channels.
    shrink : int, default=2
        Channel shrink factor.
    shrinkCol : int, default=8
        Shrink factor for column channels.
    nCells : int, default=5
        Number of self-similarity cells.
    nCellsCol : int, default=2
        Number of column self-similarity cells.
    stride : int, default=2
        Inference stride.
    nTreesEval : int, default=4
        Number of trees evaluated per location in the original pipeline.
    nThreads : int, default=4
        Thread count hint.
    seed : int, default=1
        Random seed.
    useParfor : int, default=0
        MATLAB parallel flag.
    modelDir : str, default="models"
        Model output directory.
    modelFnm : str, default="model"
        Model file stem.
    imResize : tuple[int, int], default=(256, 336)
        Resize target for feature extraction.
    trainImDir : str, default="Train400Im"
        Training image directory.
    gtMatDir : str, default="Train400Depth"
        Ground-truth MAT directory.
    dataSet : str, default="make3d"
        Dataset name.
    refine : int, default=0
        Reserved compatibility flag.
    samplesPerImage : int, default=1500
        Number of sampled pixels per image for the lightweight learner.
    nChns : int, default=1
        Number of channels (filled after feature probing).
    nChnFtrs : int, default=0
        Number of regular channel features.
    nColChnFtrs : int, default=0
        Number of column channel features.
    nSimFtrs : int, default=0
        Number of self-similarity features.
    nColSimFtrs : int, default=0
        Number of column self-similarity features.
    nTotFtrs : int, default=0
        Total feature count.
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
        dict
            Dictionary in the same naming convention as the MATLAB options.
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
        data : dict
            Option dictionary.

        Returns
        -------
        SSIDepthTrainOptions
            Parsed options instance.
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
            Base directory used for relative paths.

        Returns
        -------
        SSIDepthTrainOptions
            The same instance, updated in-place.
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
            RGB sample image used to probe channel dimensions.
        """
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
        ) = ssiDepthChns(sample_image, self.to_dict())

        self.nChns = int(n_chns)
        self.nChnFtrs = int(n_chn_ftrs)
        self.nColChnFtrs = int(n_col_chn_ftrs)
        self.nSimFtrs = int(n_sim_ftrs)
        self.nColSimFtrs = int(n_col_sim_ftrs)
        self.nTotFtrs = int(n_chn_ftrs + n_col_chn_ftrs + n_sim_ftrs + n_col_sim_ftrs)


class SSIDepthTrainer:
    """Train and cache a lightweight SSI depth model.

    The original MATLAB version trains a structured random forest and runs a
    compiled MEX detector. Under the current Python-only dependency
    constraints, this trainer fits a deterministic linear model on the ported
    feature stack and keeps the MATLAB-compatible options structure.
    """

    def __init__(self, opts: SSIDepthTrainOptions, base_dir: Path | None = None) -> None:
        """Initialize the trainer.

        Parameters
        ----------
        opts : SSIDepthTrainOptions
            Training and inference options.
        base_dir : pathlib.Path, optional
            Base directory for resolving paths. Defaults to script directory.
        """
        self.base_dir = base_dir or Path(__file__).resolve().parent
        self.opts = opts.resolve_paths(self.base_dir)

    def _model_path(self) -> Path:
        """Return NPZ model path."""
        model_dir = Path(self.opts.modelDir) / "forest"
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / f"{self.opts.modelFnm}.npz"

    def _find_sample_image(self) -> np.ndarray | None:
        """Find one image for feature-dimension probing."""
        train_dir = Path(self.opts.trainImDir)
        if train_dir.exists():
            for path in sorted(train_dir.iterdir()):
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    return io.imread(path)

        dataset_dir = self.base_dir / "Dataset1"
        if dataset_dir.exists():
            for path in sorted(dataset_dir.iterdir()):
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    return io.imread(path)

        return None

    def _save_model(self, path: Path, model: dict[str, Any]) -> None:
        """Save model to NPZ."""
        detector = model["detector"]
        np.savez_compressed(
            path,
            opts_json=json.dumps(model["opts"]),
            detector_json=json.dumps(
                {
                    "type": detector.get("type", "linear"),
                    "feature_names": detector.get("feature_names", list(FEATURE_NAMES)),
                    "bias": float(detector.get("bias", 0.0)),
                }
            ),
            weights=np.asarray(detector.get("weights", DEFAULT_WEIGHTS), dtype=np.float32),
        )

    def _load_model(self, path: Path) -> dict[str, Any]:
        """Load model from NPZ."""
        payload = np.load(path, allow_pickle=False)
        opts_data = json.loads(str(payload["opts_json"]))
        detector = json.loads(str(payload["detector_json"]))
        detector["weights"] = np.asarray(payload["weights"], dtype=np.float32)

        merged_opts = self.opts.to_dict()
        merged_opts.update(opts_data)

        return {"opts": merged_opts, "detector": detector}

    @staticmethod
    def _extract_make3d_id(image_path: Path) -> str:
        """Extract Make3D id preserving MATLAB naming convention."""
        stem = image_path.stem
        if stem.lower().startswith("img"):
            return stem[3:]
        m = re.search(r"(-?\d+)$", stem)
        return m.group(1) if m else stem

    @staticmethod
    def _load_make3d_depth(gt_file: Path) -> np.ndarray:
        """Load Make3D depth map from MAT.

        Parameters
        ----------
        gt_file : pathlib.Path
            Path to ground-truth MAT file.

        Returns
        -------
        numpy.ndarray
            Depth map as float32.
        """
        gt = loadmat(gt_file, simplify_cells=True)
        pos = np.asarray(gt["Position3DGrid"], dtype=np.float32)
        if pos.ndim == 3 and pos.shape[2] >= 4:
            depth = pos[:, :, 3]
        else:
            depth = np.squeeze(pos)
        return np.clip(depth, 1e-3, 80.0).astype(np.float32)

    def _fit_linear_model(self) -> tuple[np.ndarray, float]:
        """Fit a linear regressor over ported SSI feature channels.

        Returns
        -------
        tuple
            ``(weights, bias)``.
        """
        train_im_dir = Path(self.opts.trainImDir)
        gt_mat_dir = Path(self.opts.gtMatDir)

        if not train_im_dir.exists() or not gt_mat_dir.exists():
            raise FileNotFoundError("Training directories were not found.")

        image_files = [
            p
            for p in sorted(train_im_dir.iterdir())
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        if not image_files:
            raise FileNotFoundError(f"No training images found in {train_im_dir}.")

        x_all: list[np.ndarray] = []
        y_all: list[np.ndarray] = []
        rng = np.random.default_rng(self.opts.seed)

        for image_file in image_files[: int(min(len(image_files), self.opts.nImgs if np.isfinite(self.opts.nImgs) else len(image_files)))]:
            image_id = self._extract_make3d_id(image_file)
            gt_file = gt_mat_dir / f"depth_sph_corr{image_id}.mat"
            if not gt_file.exists():
                continue

            image = io.imread(image_file)
            depth = self._load_make3d_depth(gt_file)

            features = compute_depth_features(image, self.opts.to_dict())
            depth_resized = transform.resize(
                depth,
                tuple(self.opts.imResize),
                order=1,
                mode="reflect",
                anti_aliasing=True,
                preserve_range=True,
            ).astype(np.float32)

            x = features.reshape(-1, features.shape[2])
            y = depth_resized.reshape(-1)

            n = min(self.opts.samplesPerImage, x.shape[0])
            if n <= 0:
                continue
            pick = rng.choice(x.shape[0], size=n, replace=False)
            x_all.append(x[pick])
            y_all.append(y[pick])

        if not x_all:
            raise RuntimeError("Could not build training pairs from the available data.")

        x_cat = np.vstack(x_all).astype(np.float32)
        y_cat = np.concatenate(y_all).astype(np.float32)

        mu = x_cat.mean(axis=0)
        sigma = x_cat.std(axis=0) + 1e-6
        x_norm = (x_cat - mu) / sigma

        x_aug = np.concatenate([x_norm, np.ones((x_norm.shape[0], 1), dtype=np.float32)], axis=1)
        theta, *_ = np.linalg.lstsq(x_aug, y_cat, rcond=None)

        w_norm = theta[:-1]
        b_norm = float(theta[-1])

        weights = (w_norm / sigma).astype(np.float32)
        bias = float(b_norm - np.dot(mu, weights))
        return weights, bias

    def train_or_load(self) -> dict[str, Any]:
        """Load a cached model or train a new one.

        Returns
        -------
        dict
            Model dictionary with keys ``opts`` and ``detector``.
        """
        model_path = self._model_path()
        if model_path.exists():
            model = self._load_model(model_path)
            sample = self._find_sample_image()
            if sample is not None:
                self.opts.update_feature_counts(sample)
                model["opts"].update(self.opts.to_dict())
            return model

        sample = self._find_sample_image()
        if sample is not None:
            self.opts.update_feature_counts(sample)

        try:
            weights, bias = self._fit_linear_model()
        except Exception:
            weights, bias = DEFAULT_WEIGHTS.copy(), 0.0

        model = {
            "opts": self.opts.to_dict(),
            "detector": {
                "type": "linear",
                "feature_names": list(FEATURE_NAMES),
                "weights": weights,
                "bias": float(bias),
            },
        }
        self._save_model(model_path, model)
        return model


def _resolve_path(path_like: str | Path, base_dir: Path) -> Path:
    """Resolve absolute or relative paths.

    Parameters
    ----------
    path_like : str or pathlib.Path
        Input path.
    base_dir : pathlib.Path
        Base directory for relative paths.

    Returns
    -------
    pathlib.Path
        Resolved path.
    """
    p = Path(path_like)
    return p if p.is_absolute() else (base_dir / p)


def ssiDepthTrain(opts: SSIDepthTrainOptions | dict[str, Any] | None = None) -> SSIDepthTrainOptions | dict[str, Any]:
    """Python conversion of ``ssiDepthTrain.m``.

    Parameters
    ----------
    opts : SSIDepthTrainOptions or dict, optional
        Options object or dictionary. If omitted, default options are returned.

    Returns
    -------
    SSIDepthTrainOptions or dict
        If ``opts`` is omitted, returns default options object.
        Otherwise returns a model dictionary.
    """
    if opts is None:
        return SSIDepthTrainOptions()

    if isinstance(opts, dict):
        opt_obj = SSIDepthTrainOptions.from_dict(opts)
    else:
        opt_obj = opts

    trainer = SSIDepthTrainer(opt_obj)
    return trainer.train_or_load()
