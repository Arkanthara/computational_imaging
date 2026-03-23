import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from skimage import io

from ssiDepthDetect import ssiDepthDetect
from ssiDepthTrain import SSIDepthTrainOptions, ssiDepthTrain


def _build_parser() -> argparse.ArgumentParser:
    """Create argument parser for the SSI depth demo.

    Returns
    -------
    argparse.ArgumentParser
        Parser instance.
    """
    parser = argparse.ArgumentParser(description="SSI depth demo (Python conversion)")
    parser.add_argument(
        "--image",
        type=str,
        default="Dataset1/img-2.jpg",
        help="Input image path. Relative paths are resolved from the script folder.",
    )
    parser.add_argument(
        "--model-fnm",
        type=str,
        default="modelMake3d",
        help="Model stem used in models/forest/<name>.npz.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Model directory.",
    )
    parser.add_argument(
        "--train-im-dir",
        type=str,
        default="Train400Im",
        help="Training image directory (Make3D naming expected).",
    )
    parser.add_argument(
        "--gt-mat-dir",
        type=str,
        default="Train400Depth",
        help="Ground-truth MAT directory (Make3D naming expected).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="make3d",
        help="Dataset label saved into model options.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Optional output figure path.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not show interactive window.",
    )
    return parser


def _resolve_local(path_str: str, base_dir: Path) -> Path:
    """Resolve path relative to script directory when needed."""
    p = Path(path_str)
    return p if p.is_absolute() else (base_dir / p)


def main() -> None:
    """Run the Python SSI depth demo."""
    args = _build_parser().parse_args()
    base_dir = Path(__file__).resolve().parent

    opts = ssiDepthTrain()
    if not isinstance(opts, SSIDepthTrainOptions):
        opts = SSIDepthTrainOptions.from_dict(opts["opts"])

    opts.modelFnm = args.model_fnm
    opts.modelDir = str(_resolve_local(args.model_dir, base_dir))
    opts.trainImDir = str(_resolve_local(args.train_im_dir, base_dir))
    opts.gtMatDir = str(_resolve_local(args.gt_mat_dir, base_dir))
    opts.dataSet = args.dataset

    model = ssiDepthTrain(opts)

    image_path = _resolve_local(args.image, base_dir)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    image = io.imread(image_path)
    depth = ssiDepthDetect(image, model)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    im = axes[1].imshow(depth, cmap="viridis")
    axes[1].set_title("Estimated Depth")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()

    if args.save:
        save_path = _resolve_local(args.save, base_dir)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
