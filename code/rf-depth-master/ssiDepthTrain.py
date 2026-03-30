"""Training utilities for the Python SSI depth pipeline.

Options are intentionally defined in ``ssiDepthOptions.py`` so training,
loading, and inference can stay clean and easy to understand.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any
import warnings

import numpy as np
from scipy.io import loadmat
from skimage import io, transform

from ssiDepthDetect import FEATURE_NAMES, compute_depth_features
from ssiDepthOptions import SSIDepthTrainOptions


DEFAULT_WEIGHTS = np.array([0.34, 0.24, 0.12, 0.10, 0.14, 0.06], dtype=np.float32)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


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

    def model_path(self) -> Path:
        """Return the expected on-disk path for the serialized model.

        Returns
        -------
        pathlib.Path
            Absolute or resolved path to ``models/forest/<model>.npz``.
        """
        model_dir = Path(self.opts.modelDir) / "forest"
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / f"{self.opts.modelFnm}.npz"

    def has_cached_model(self) -> bool:
        """Check whether a serialized model already exists.

        Returns
        -------
        bool
            ``True`` if the model file exists, else ``False``.
        """
        return self.model_path().exists()

    def _find_sample_image(self) -> np.ndarray | None:
        """Find one image for feature-dimension probing.

        Returns
        -------
        numpy.ndarray or None
            First image found in training directory (preferred), then
            ``Dataset1`` fallback, else ``None``.
        """
        train_dir = Path(self.opts.trainImDir)
        if train_dir.exists():
            for path in sorted(train_dir.iterdir()):
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    return io.imread(path)

        dataset_dir = self.base_dir / "Dataset1"
        if dataset_dir.exists():
            for path in sorted(dataset_dir.iterdir()):
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    return io.imread(path)

        return None

    def _save_model(self, path: Path, model: dict[str, Any]) -> None:
        """Serialize model metadata and detector parameters to NPZ.

        Parameters
        ----------
        path : pathlib.Path
            Output file path.
        model : dict of str to Any
            Model payload with ``opts`` and ``detector`` keys.
        """
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
        """Load model from NPZ.

        Parameters
        ----------
        path : pathlib.Path
            Input model file.

        Returns
        -------
        dict of str to Any
            Model dictionary with ``opts`` and ``detector`` keys.
        """
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

        image_files = [p for p in sorted(train_im_dir.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
        if not image_files:
            raise FileNotFoundError(f"No training images found in {train_im_dir}.")

        x_all: list[np.ndarray] = []
        y_all: list[np.ndarray] = []
        rng = np.random.default_rng(self.opts.seed)

        max_images = len(image_files) if not np.isfinite(self.opts.nImgs) else int(min(len(image_files), self.opts.nImgs))
        for image_file in image_files[:max_images]:
            image_id = self._extract_make3d_id(image_file)
            gt_file = gt_mat_dir / f"depth_sph_corr{image_id}.mat"
            if not gt_file.exists():
                continue

            image = io.imread(image_file)
            depth = self._load_make3d_depth(gt_file)

            features = compute_depth_features(image, self.opts)
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

    def load_cached_model(self) -> dict[str, Any]:
        """Load an existing model from disk.

        Returns
        -------
        dict of str to Any
            Loaded model dictionary.

        Raises
        ------
        FileNotFoundError
            If no cached model exists at :meth:`model_path`.
        """
        model_path = self.model_path()
        if not model_path.exists():
            raise FileNotFoundError(f"Cached model not found: {model_path}")

        model = self._load_model(model_path)
        sample = self._find_sample_image()
        if sample is not None:
            self.opts.update_feature_counts(sample)
            model["opts"].update(self.opts.to_dict())
        return model

    def train_and_save_model(self) -> dict[str, Any]:
        """Train a model and save it in the cache location.

        Returns
        -------
        dict of str to Any
            Trained model dictionary.

        Notes
        -----
        If training data is unavailable, a deterministic fallback detector is
        used so inference can still run.
        """
        sample = self._find_sample_image()
        if sample is not None:
            self.opts.update_feature_counts(sample)

        try:
            weights, bias = self._fit_linear_model()
        except Exception as exc:
            warnings.warn(
                f"Training failed ({exc}); using default detector weights.",
                RuntimeWarning,
                stacklevel=2,
            )
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
        model_path = self.model_path()
        self._save_model(model_path, model)
        return model

def ssiDepthTrain(opts: SSIDepthTrainOptions | dict[str, Any] | None = None) -> SSIDepthTrainOptions | dict[str, Any]:
    """Train and save a depth model.

    Parameters
    ----------
    opts : SSIDepthTrainOptions or dict, optional
        Options object or dictionary. If omitted, default options are returned.

    Returns
    -------
    SSIDepthTrainOptions or dict
        If ``opts`` is omitted, returns default options object.
        Otherwise returns a newly trained model dictionary.
    """
    if opts is None:
        return SSIDepthTrainOptions()

    if isinstance(opts, dict):
        opt_obj = SSIDepthTrainOptions.from_dict(opts)
    else:
        opt_obj = opts

    trainer = SSIDepthTrainer(opt_obj)
    return trainer.train_and_save_model()
