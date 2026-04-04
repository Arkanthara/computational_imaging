"""
=========================================================
PIXEL-WISE DEPTH ESTIMATION USING RANDOM FOREST
=========================================================

This script:
- Trains a pixel-wise Random Forest on Make3D dataset
- Saves/loads trained model
- Predicts full depth maps (not single value)
- Tests on images from Dataset1
- Displays results (image + depth map + colorbar)

NOTE:
- Dataset MUST already be downloaded and extracted
- Expected structure:
    dataset_path/
        Train400/
        Depth/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import joblib

from skimage import io, color, transform, util
import skimage
from skimage.restoration import denoise_bilateral
from skimage.filters import gaussian
from sklearn.ensemble import RandomForestRegressor
from scipy.ndimage import minimum_filter


# =========================
# FFT Convolution
# =========================

def fft_convolve2d(img, kernel):
    H, W = img.shape
    kh, kw = kernel.shape
    
    pad = np.zeros((H, W))
    pad[:kh, :kw] = kernel
    
    # center kernel
    pad = np.roll(pad, -kh//2, axis=0)
    pad = np.roll(pad, -kw//2, axis=1)
    
    img_fft = np.fft.fft2(img)
    ker_fft = np.fft.fft2(pad)
    
    result = np.fft.ifft2(img_fft * ker_fft).real
    return result


# =========================
# Filters
# =========================

def get_laws_filters():
    L3 = np.array([1, 2, 1]) / 4
    E3 = np.array([-1, 0, 1])
    S3 = np.array([-1, 2, -1])
    
    filters = []
    for v1 in [L3, E3, S3]:
        for v2 in [L3, E3, S3]:
            filters.append(np.outer(v1, v2))
    return filters


def get_navatia_babu_filters():
    NB1 = np.array([
        [-100,-100,0,100,100],
        [-100,-100,0,100,100],
        [-100,-100,0,100,100],
        [-100,-100,0,100,100],
        [-100,-100,0,100,100]
    ]) / 2000

    NB2 = np.array([
        [-100,32,100,100,100],
        [-100,-78,92,100,100],
        [-100,-100,0,100,100],
        [-100,-100,-92,78,100],
        [-100,-100,-100,-32,100]
    ]) / 2000

    NB3 = -NB2.T
    NB4 = -NB1.T
    NB5 = -NB3[::-1, :]
    NB6 = NB5.T

    return [NB1, NB2, NB3, NB4, NB5, NB6]


# =========================
# Filter Banks FFT
# =========================

def calculate_filter_banks_fft(img):
    ycbcr = color.rgb2ycbcr(img)
    Y = ycbcr[..., 0]
    Cb = ycbcr[..., 1]
    Cr = ycbcr[..., 2]

    outputs = []

    # Laws on Y (9)
    for f in get_laws_filters():
        outputs.append(fft_convolve2d(Y, f))

    # Laws on Cb, Cr
    L3 = np.array([1, 2, 1]) / 4
    L3L3 = np.outer(L3, L3)

    outputs.append(fft_convolve2d(Cb, L3L3))
    outputs.append(fft_convolve2d(Cr, L3L3))

    # Navatia-Babu (6)
    for f in get_navatia_babu_filters():
        outputs.append(fft_convolve2d(Y, f))

    return np.abs(np.stack(outputs, axis=-1))  # [H,W,17]


# =========================
# Utils
# =========================

def resize_img(img, scale=None, size=None):
    if size is not None:
        return transform.resize(img, size, order=0, preserve_range=True, anti_aliasing=False)
    elif scale is not None:
        new_size = (int(img.shape[0]*scale), int(img.shape[1]*scale))
        return transform.resize(img, new_size, order=1, preserve_range=True, anti_aliasing=True)


def smooth_triangular(img, radius):
    if radius <= 1:
        return img
    
    size = int(radius)
    k = np.arange(1, size+1)
    k = np.concatenate([k, k[::-1][1:]])
    k = k / k.sum()
    kernel = np.outer(k, k)

    out = np.zeros_like(img)
    for c in range(img.shape[2]):
        out[..., c] = fft_convolve2d(img[..., c], kernel)
    return out


def rgb2hsi(img):
    r, g, b = img[...,0], img[...,1], img[...,2]
    intensity = (r + g + b) / 3
    
    min_rgb = np.minimum(np.minimum(r,g),b)
    saturation = 1 - min_rgb/(intensity + 1e-6)
    
    num = 0.5*((r-g)+(r-b))
    den = np.sqrt((r-g)**2 + (r-b)*(g-b)) + 1e-6
    theta = np.arccos(num/den)
    
    hue = np.where(b <= g, theta, 2*np.pi-theta)
    hue /= (2*np.pi)
    
    return np.stack([hue, saturation, intensity], axis=-1)


def dark_channel(img, size=15):
    return minimum_filter(img.min(axis=2), size=size)


# =========================
# MAIN FUNCTION
# =========================


def ssi_depth_chns(I, opts, with_names=False):
    """
    Compute all feature channels used by the depth pipeline.

    Returns:
        chns: H x W x C
        names: list[str] (optional)
    """
    I = util.img_as_float(I)
    I = resize_img(I, size=opts["imResize"])

    shrink = opts["shrink"]

    # ---- representations ----
    Irgb = I
    Iluv = color.rgb2luv(I)
    Ihsi = rgb2hsi(I)

    Irgb_s = resize_img(Irgb, scale=1/shrink)
    Iluv_s = resize_img(Iluv, scale=1/shrink)
    Ihsi_s = resize_img(Ihsi, scale=1/shrink)

    H, W = Irgb_s.shape[:2]

    # ---- prior ----
    prior = np.linspace(0, 1, H).reshape(H, 1).repeat(W, axis=1)

    channels = []
    names = [] if with_names else None

    def add_channel(ch, name):
        channels.append(ch)
        if with_names:
            names.append(name)

    # ---- base channels ----
    add_channel(prior[..., None], "prior")

    # RGB
    add_channel(Irgb_s[..., 0:1], "Irgb_R")
    add_channel(Irgb_s[..., 1:2], "Irgb_G")
    add_channel(Irgb_s[..., 2:3], "Irgb_B")

    # HSI
    add_channel(Ihsi_s[..., 0:1], "Ihsi_hue")
    add_channel(Ihsi_s[..., 1:2], "Ihsi_sat")
    add_channel(Ihsi_s[..., 2:3], "Ihsi_intensity")

    # LUV
    add_channel(Iluv_s[..., 0:1], "Iluv_L")
    add_channel(Iluv_s[..., 1:2], "Iluv_u")
    add_channel(Iluv_s[..., 2:3], "Iluv_v")

    # ---- multi-scale filters ----
    for s in [1, 2]:
        if s == shrink:
            I2 = Irgb_s
        else:
            I2 = resize_img(Irgb, scale=1/s)

        filters = calculate_filter_banks_fft(I2)  # [H,W,17]
        dark = dark_channel(I2)

        filters = resize_img(filters, scale=s/shrink)
        dark = resize_img(dark[..., None], scale=s/shrink)

        for k in range(filters.shape[-1]):
            ch = filters[..., k:k+1]

            if with_names:
                if k < 9:
                    name = f"s={s}_laws_{k+1}"
                elif k == 9:
                    name = f"s={s}_Cb_L3L3"
                elif k == 10:
                    name = f"s={s}_Cr_L3L3"
                else:
                    name = f"s={s}_nav_{k-10}"
            else:
                name = None

            add_channel(ch, name)

        add_channel(dark, f"s={s}_dark")

    chns = np.concatenate(channels, axis=2)

    if with_names:
        return chns, names
    
    # ---- smoothing (edge-preserving via FFT triangular kernel) ----
    chns = smooth_triangular(chns, opts["chnSmooth"] / opts["shrink"])

    return chns


def plot_depth_filters_structured(I, opts, include=("colors", "laws", "babu")):
    if isinstance(include, str):
        include_set = {include}
    else:
        include_set = set(include)
    chns, names = ssi_depth_chns(I, opts, with_names=True)

    name_to_idx = {n: i for i, n in enumerate(names)}

    def get(name):
        return chns[..., name_to_idx[name]]

    # ---- base channels ----
    rgb = np.stack([
        get("Irgb_R"),
        get("Irgb_G"),
        get("Irgb_B")
    ], axis=-1)

    R = get("Irgb_R")
    G = get("Irgb_G")
    B = get("Irgb_B")

    H = get("Ihsi_hue")
    S = get("Ihsi_sat")
    I_int = get("Ihsi_intensity")

    L = get("Iluv_L")
    u = get("Iluv_u")
    v = get("Iluv_v")

    figures = {}

    # =========================================================
    # 1. COLOR FIGURE
    # =========================================================
    if "colors" in include_set:
        fig1, axes1 = plt.subplots(3, 3, figsize=(10, 8))

        axes1[0, 0].imshow(R, cmap="Reds")
        axes1[0, 0].set_title("R")

        axes1[0, 1].imshow(G, cmap="Greens")
        axes1[0, 1].set_title("G")

        axes1[0, 2].imshow(B, cmap="Blues")
        axes1[0, 2].set_title("B")

        axes1[1, 0].imshow(H, cmap="hsv")
        axes1[1, 0].set_title("Hue")

        axes1[1, 1].imshow(S, cmap="viridis")
        axes1[1, 1].set_title("Saturation")

        axes1[1, 2].imshow(I_int, cmap="gray")
        axes1[1, 2].set_title("Intensity")

        axes1[2, 0].imshow(L, cmap="gray")
        axes1[2, 0].set_title("L")

        axes1[2, 1].imshow(u, cmap="coolwarm")
        axes1[2, 1].set_title("U")

        axes1[2, 2].imshow(v, cmap="coolwarm")
        axes1[2, 2].set_title("V")

        for ax in axes1.flat:
            ax.axis("off")

        fig1.tight_layout()
        figures["colors"] = fig1

    # =========================================================
    # 2. LAWS FIGURE
    # =========================================================
    if "laws" in include_set:
        laws = [get(f"s=1_laws_{i}") for i in range(1, 10)]
        laws.append(get("s=1_Cb_L3L3"))
        laws.append(get("s=1_Cr_L3L3"))

        fig2, axes2 = plt.subplots(3, 4, figsize=(10, 8))

        axes2[0, 0].imshow(I)
        axes2[0, 0].set_title("Original")

        for i, law in enumerate(laws):
            r = (i + 1) // 4
            c = (i + 1) % 4
            axes2[r, c].imshow(law, cmap="gray")
            if i < 9:
                axes2[r, c].set_title(f"Laws {i+1}")
            elif i == 9:
                axes2[r, c].set_title("Cb L3L3")
            elif i == 10:
                axes2[r, c].set_title("Cr L3L3")

        for ax in axes2.flat:
            ax.axis("off")

        fig2.tight_layout()
        figures["laws"] = fig2

    # =========================================================
    # 3. NAVATIA-BABU FIGURE
    # =========================================================
    if "babu" in include_set:
        babu = [get(f"s=1_nav_{i}") for i in range(1, 7)]

        fig3, axes3 = plt.subplots(2, 3, figsize=(8, 5))

        for i, nb in enumerate(babu):
            r = i // 3
            c = i % 3
            axes3[r, c].imshow(nb, cmap="gray")
            axes3[r, c].set_title(f"NB {i+1}")
            axes3[r, c].axis("off")

        fig3.tight_layout()
        figures["babu"] = fig3

    # =========================================================
    # RETURN
    # =========================================================
    return figures


# =========================================================
# PIXEL DATASET
# =========================================================

def build_pixel_dataset(chns, depth):

    X = chns.reshape(-1, chns.shape[-1])
    y = depth.reshape(-1)

    return X, y


# =========================================================
# RANDOM FOREST MODEL
# =========================================================

class SSI_RF_Model:

    def __init__(self, opts):
        self.opts = opts
        self.model = RandomForestRegressor(
            n_estimators=40,
            max_depth=20,
            n_jobs=-1
        )

    def train(self, images, depths, max_pixels=50000):

        X_all, y_all = [], []

        for img, depth in zip(images, depths):

            chns = ssi_depth_chns(img, self.opts)
            depth = resize_img(depth, size=chns.shape[:2])

            X, y = build_pixel_dataset(chns, depth)

            idx = np.random.choice(len(X),
                                   min(max_pixels, len(X)),
                                   replace=False)

            X_all.append(X[idx])
            y_all.append(y[idx])

        X_all = np.vstack(X_all)
        y_all = np.hstack(y_all)

        print("Training samples:", X_all.shape)

        self.model.fit(X_all, y_all)

    def predict(self, img):
        
        # Save original size
        orig_h, orig_w = img.shape[:2]

        # Compute features (resized internally)
        chns = ssi_depth_chns(img, self.opts, with_names=False)
        H, W, C = chns.shape

        X = chns.reshape(-1, C)

        pred = self.model.predict(X)
        pred = pred.reshape(H, W)

        # 🔥 FIX: resize back to original image size
        pred = resize_img(pred, size=(orig_h, orig_w))

        # # 🔥 Edge-preserving smoothing
        # pred = denoise_bilateral(
        #     pred,
        #     sigma_color=0.05,
        #     sigma_spatial=5,
        #     channel_axis=None
        # )

        return pred


# =========================================================
# DATA LOADING (Make3D already present)
# =========================================================

def load_dataset(dataset_path, max_samples=30):
    """
    Load dataset where:
    - Images are .jpg
    - Depth maps are .dat
    - Both are in the same folder
    """

    images = []
    depths = []

    files = sorted(os.listdir(dataset_path))

    img_files = [f for f in files if f.lower().endswith(".jpg")][:max_samples]

    for fname in img_files:

        img_path = os.path.join(dataset_path, fname)
        depth_path = os.path.join(
            dataset_path,
            fname.replace(".jpg", ".dat")
        )

        if not os.path.exists(depth_path):
            print(f"Missing depth for {fname}")
            continue

        try:
            # Load image
            img = io.imread(img_path)

            # Load depth (.dat)
            depth = np.loadtxt(depth_path)

            # Resize depth to match feature pipeline
            depth = resize_img(depth, size=(240, 320))

            images.append(img)
            depths.append(depth)

        except Exception as e:
            print("Skipped:", fname, e)

    print(f"Loaded {len(images)} samples")

    return images, depths

def export_sample_to_utils(images, depths, out_dir="utils"):
    """
    Save:
    - image.jpg
    - depth.jpg (same size as image)
    - law_filter.jpg
    - babu_filter.jpg
    """

    os.makedirs(out_dir, exist_ok=True)

    if len(images) == 0:
        raise ValueError("Dataset is empty")

    # ---- pick first sample (deterministic) ----
    img = images[0]
    depth = depths[0]

    H, W = img.shape[:2]

    # =========================================================
    # IMAGE
    # =========================================================
    if img.dtype != np.uint8:
        if img.max() <= 1:
            img_uint8 = (img * 255).astype(np.uint8)
        else:
            img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img_uint8 = img

    io.imsave(os.path.join(out_dir, "image.jpg"), img_uint8)

    # =========================================================
    # DEPTH (resize to image size)
    # =========================================================
    depth_resized = resize_img(depth, size=(H, W))

    d_min, d_max = depth_resized.min(), depth_resized.max()
    depth_norm = (depth_resized - d_min) / (d_max - d_min + 1e-8)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)

    io.imsave(os.path.join(out_dir, "depth.jpg"), depth_uint8)

    # =========================================================
    # Y CHANNEL (for filters)
    # =========================================================
    ycbcr = color.rgb2ycbcr(img_uint8 / 255.0)
    Y = ycbcr[..., 0]

    # =========================================================
    # LAW FILTER (take first one for example)
    # =========================================================
    law_filter = get_laws_filters()[0]
    law_response = fft_convolve2d(Y, law_filter)

    # normalize for visualization
    v = np.max(np.abs(law_response)) + 1e-6
    law_norm = (law_response / v + 1) / 2  # map [-v,v] → [0,1]
    law_uint8 = (law_norm * 255).astype(np.uint8)

    io.imsave(os.path.join(out_dir, "law_filter.jpg"), law_uint8)

    # =========================================================
    # NAVATIA-BABU FILTER (take first one)
    # =========================================================
    babu_filter = get_navatia_babu_filters()[0]
    babu_response = fft_convolve2d(Y, babu_filter)

    v = np.max(np.abs(babu_response)) + 1e-6
    babu_norm = (babu_response / v + 1) / 2
    babu_uint8 = (babu_norm * 255).astype(np.uint8)

    io.imsave(os.path.join(out_dir, "babu_filter.jpg"), babu_uint8)

    print(f"Saved sample to '{out_dir}/'")

# =========================================================
# MODEL LOAD / TRAIN
# =========================================================

def get_or_train_model(opts, model_path, dataset_path):

    if os.path.exists(model_path):
        print("Loading existing model...")
        return joblib.load(model_path)

    print("Training model...")

    images, depths = load_dataset(dataset_path)

    model = SSI_RF_Model(opts)
    model.train(images, depths)

    joblib.dump(model, model_path)

    print("Model saved.")

    return model


# =========================================================
# TESTING + VISUALIZATION
# =========================================================

def test_on_dataset(model, dataset_folder):

    files = [f for f in os.listdir(dataset_folder)
             if f.lower().endswith((".jpg",".png"))]

    for fname in files:

        path = os.path.join(dataset_folder, fname)
        img = io.imread(path)

        pred = model.predict(img)

        plt.figure(figsize=(12,5))

        # Original image
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.title("Image")
        plt.axis("off")

        # Depth map
        plt.subplot(1,2,2)
        im = plt.imshow(pred, cmap='inferno')
        plt.title("Predicted Depth")
        plt.axis("off")

        plt.colorbar(im, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    opts = {
        "imResize": (340, 256),
        "shrink": 1,
        "shrinkCol": 4,
        "chnSmooth": 2,
        "simSmooth": 4,
        "nCells": 4
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "./make3d")
    model_path = os.path.join(script_dir, "ssi_rf_model.pkl")
    test_folder = os.path.join(script_dir, "./Dataset1")
    test = False  # Set to False to visualize filters instead
    visualize_filters = False
    create_utils_sample = False  # Set to True to export sample image + depth + filters to utils/

    model = get_or_train_model(opts, model_path, dataset_path)

    if test:
        test_on_dataset(model, test_folder)
    elif visualize_filters:
        # Visualize filters on a sample image
        sample_img_path = os.path.join(test_folder, "img-2.jpg")
        sample_img = io.imread(sample_img_path)
        fig = plot_depth_filters_structured(sample_img, opts)
        fig["colors"].show()
        fig["laws"].show()
        fig["babu"].show()
        input("Press Enter to exit...")
        plt.close("all")
    elif create_utils_sample:
        out_dir = os.path.join(script_dir, "utils")
        images, depths = load_dataset(dataset_path, max_samples=1)
        export_sample_to_utils(images, depths, out_dir)
        # Run one prediction to generate sample output depth map in utils/
        pred = model.predict(images[0])

        # Normalize to [0, 1]
        pred_norm = (pred - pred.min()) / (pred.max() - pred.min())
        io.imsave(os.path.join(out_dir, "predicted_depth.jpg"), util.img_as_ubyte(pred_norm))