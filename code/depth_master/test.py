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
from sklearn.ensemble import RandomForestRegressor
from scipy.ndimage import minimum_filter, gaussian_filter


# =========================================================
# FFT CONVOLUTION
# =========================================================

def fft_convolve2d(img, kernel):
    H, W = img.shape
    kh, kw = kernel.shape

    pad = np.zeros((H, W))
    pad[:kh, :kw] = kernel

    pad = np.roll(pad, -kh // 2, axis=0)
    pad = np.roll(pad, -kw // 2, axis=1)

    return np.fft.ifft2(np.fft.fft2(img) * np.fft.fft2(pad)).real


# =========================================================
# FILTERS
# =========================================================

def get_laws_filters():
    L3 = np.array([1, 2, 1]) / 4
    E3 = np.array([-1, 0, 1])
    S3 = np.array([-1, 2, -1])

    return [np.outer(a, b) for a in [L3, E3, S3] for b in [L3, E3, S3]]


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


# =========================================================
# FILTER BANK
# =========================================================

def filter_bank_fft(img):
    ycbcr = color.rgb2ycbcr(img)
    Y, Cb, Cr = ycbcr[...,0], ycbcr[...,1], ycbcr[...,2]

    out = []

    for f in get_laws_filters():
        out.append(fft_convolve2d(Y, f))

    L3 = np.array([1,2,1]) / 4
    L3L3 = np.outer(L3, L3)

    out.append(fft_convolve2d(Cb, L3L3))
    out.append(fft_convolve2d(Cr, L3L3))

    for f in get_navatia_babu_filters():
        out.append(fft_convolve2d(Y, f))

    return np.abs(np.stack(out, axis=-1))


# =========================================================
# UTILITIES
# =========================================================

def resize_img(img, scale=None, size=None):
    if size is not None:
        return transform.resize(img, size, order=1, preserve_range=True)
    else:
        h, w = img.shape[:2]
        return transform.resize(img, (int(h*scale), int(w*scale)),
                                order=3, preserve_range=True)


def rgb2hsi(img):
    r,g,b = img[...,0], img[...,1], img[...,2]
    I = (r+g+b)/3
    minv = np.minimum(np.minimum(r,g),b)
    S = 1 - minv/(I+1e-6)

    num = 0.5*((r-g)+(r-b))
    den = np.sqrt((r-g)**2 + (r-b)*(g-b)) + 1e-6
    theta = np.arccos(num/den)

    H = np.where(b<=g, theta, 2*np.pi-theta)
    H = H/(2*np.pi)

    return np.stack([H,S,I],axis=-1)


def dark_channel(img, size=15):
    return minimum_filter(img.min(axis=2), size=size)


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def ssi_depth_chns(I, opts):

    I = util.img_as_float(I)
    I = resize_img(I, size=opts["imResize"])

    shrink = opts["shrink"]

    Irgb = I
    Iluv = color.rgb2luv(I)
    Ihsi = rgb2hsi(I)

    Irgb_s = resize_img(Irgb, scale=1/shrink)
    Iluv_s = resize_img(Iluv, scale=1/shrink)
    Ihsi_s = resize_img(Ihsi, scale=1/shrink)

    H, W = Irgb_s.shape[:2]

    prior = np.linspace(0,1,H).reshape(H,1).repeat(W,axis=1)

    channels = [
        prior[...,None],
        Irgb_s,
        Ihsi_s,
        Iluv_s
    ]

    for s in [1,2]:
        I2 = resize_img(Irgb, scale=1/s)
        fb = filter_bank_fft(I2)
        dc = dark_channel(I2)

        fb = resize_img(fb, scale=s/shrink)
        dc = resize_img(dc[...,None], scale=s/shrink)

        channels.append(fb)
        channels.append(dc)

    return np.concatenate(channels, axis=2)


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
        chns = ssi_depth_chns(img, self.opts)
        H, W, C = chns.shape

        X = chns.reshape(-1, C)

        pred = self.model.predict(X)
        pred = pred.reshape(H, W)

        # 🔥 FIX: resize back to original image size
        pred = resize_img(pred, size=(orig_h, orig_w))

        pred = gaussian_filter(pred, sigma=2)

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
            print("depth shape:", depth.shape)

            # Resize depth to match feature pipeline
            depth = resize_img(depth, size=(240, 320))

            images.append(img)
            depths.append(depth)

        except Exception as e:
            print("Skipped:", fname, e)

    print(f"Loaded {len(images)} samples")

    return images, depths


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
        "imResize": (240,320),
        "shrink": 2,
        "shrinkCol": 4,
        "chnSmooth": 2,
        "simSmooth": 4,
        "nCells": 4
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "./make3d")
    model_path = os.path.join(script_dir, "ssi_rf_model.pkl")
    test_folder = os.path.join(script_dir, "./Dataset1")

    model = get_or_train_model(opts, model_path, dataset_path)

    test_on_dataset(model, test_folder)