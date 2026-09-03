// Main report file
#import "template.typ": make-report, report-footnote
#import "metadata.typ": my-report
#import "@preview/theofig:0.1.0": definition
#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node
#import "@preview/cetz:0.4.2": canvas, draw
// #import "graph_utils.typ": *
#import "neural-viz/lib.typ": *
#import emoji: camera

#import "@preview/algorithmic:1.0.7"
#import algorithmic: style-algorithm, algorithm-figure

#show: style-algorithm

// Main content
#show: make-report.with(my-report)

#show raw.where(block: true): set block(fill: luma(240), inset: 1em, radius: 0.5em, width: 100%)

#set math.equation(numbering: "(1)")

```python
%| refresh: true
%| echo: false
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *
```

= Introduction on Diffusion Models

Generative models have revolutionized the field of artificial intelligence, enabling machines to create content that is indistinguishable from human-generated data or to improve the quality of data through various transformations.
Among these models, diffusion models have emerged as a powerful class of generative models that have shown remarkable performance in tasks such as data generation, image super-resolution, and data denoising.

Diffusion models are based on the concept of a diffusion process, which is a stochastic process that describes how data points evolve over time. The core idea behind diffusion models is to model the data generation process as a gradual transformation from noise to structured data. This is achieved by defining a forward diffusion process that adds noise to the data and a reverse diffusion process that learns to remove the noise and recover the original data.

Among the various types of diffusion models, Denoising Diffusion Probabilistic Models (DDPMs) have gained significant attention due to their ability to generate high-quality samples.
DDPMs work by iteratively adding noise to the data and then learning to reverse this process, effectively denoising the data step by step.

But are these models very suitable for all types of data, especially for quantized data?
Quantized data, which is often used in scenarios like in computational imaging, presents unique challenges for generative modeling.
The discrete nature of quantized data can make it difficult for diffusion models to learn the underlying data distribution effectively.

In answer to this problem, researchers have proposed various approaches to adapt diffusion models for quantized data, such as Binary Diffusion Models (BDMs) and Quantized Diffusion Models (QDMs).
These models are designed to handle the discrete nature of quantized data while still leveraging the powerful generative capabilities of diffusion models.

#pagebreak()

= Methodology

There are several types of generative models, including:
- Generative Adversarial Networks (GANs)
- Variational Autoencoders (VAEs)
- Normalizing Flows
- Diffusion Models
Each of these models has its own strengths and weaknesses, and the choice of model often depends on the specific application and the type of data being modeled.

Diffusion models, in particular, have shown great promise in generating high-quality samples and have been successfully applied to a wide range of tasks.
However, their performance can be limited when dealing with quantized data, which is often encountered in scenarios like computational imaging.
To address this issue, researchers have developed specialized diffusion models that are designed to handle quantized data effectively.

Before going into the details of these specialized diffusion models, let's first understand the foundational principles of diffusion models and how they work.

Diffusion models are Markov chains that consist of two main processes: the forward diffusion process and the reverse diffusion process.
The forward diffusion process gradually adds noise to the data, while the reverse diffusion process learns to remove the noise and recover the original data.
The training of diffusion models involves optimizing a variational bound on the data likelihood, which allows the model to learn the underlying data distribution effectively.

== Forward Diffusion Process

Suppose we have an original image $X_0$ that follows a data distribution $q$.

At each time step $t$, noise is added to the data according to a predefined schedule, which is typically controlled by a parameter $beta_t$ that determines the amount of noise added at each step, as shown in @eq1 where $epsilon_t ~ cal(N)(0, I)$ is a standard Gaussian noise term and $X_t$ is the noisy version of the data at time step $t$.

$ X_t = sqrt(1-beta_t) X_(t-1) + sqrt(beta_t) epsilon_t $ <eq1>

Indeed, if $beta_t = 1$ in @eq1, then $X_t$ would be pure noise whereas if $beta_t = 0$, then $X_t$ would be the original data $X_(t-1)$.

This process can be visualized as a gradual transformation of the original data into noise, where the data becomes increasingly corrupted as more noise is added at each time step, as illustrated in the @fig1.

#let src_dir = "../../.."

#let make-image-node = set-image-node-defaults(options: (image-pad: 0, image-width: 3.5, image-height: 4, border: false, title-size: 1em))

#figure(
  {
  let nodes = (
    image-node("x0", title: $X_0$, src: src_dir + "/img/noisy_images/tangled.png", cover: true, image-size: (2.5, 3)),
    image-node("x1", title: $X_1$, src: src_dir + "/img/noisy_images/noisy_image_0.3.png", cover: true, image-size: (2.5, 3)),
    image-node("x2", title: $X_2$, src: src_dir + "/img/noisy_images/noisy_image_0.5.png", cover: true, image-size: (2.5, 3)),
    text-node("dots", " ... ", title-size: 1em, node-size: (2, 3)),
    image-node("xT", title: $X_T$, src: src_dir + "/img/noisy_images/noisy_image_10.0.png", cover: true, image-size: (2.5, 3)),
  )

  let edges = (
    ml-edge("x0", "x1", label: "Adding noise", label-sep: 0.2em),
    ml-edge("x1", "x2"),
    ml-edge("x2", "dots"),
    ml-edge("dots", "xT"),
  )

  ml-diagram(nodes, edges: edges, spacing: 4.5em, label-size: 0.7em)
  },
  caption: "Forward diffusion process",
) <fig1>

We can also express $X_t$ in term of the data distribution $q$ as described in @eq2, where $cal(N)(X; mu, sigma^2)$ denotes a Gaussian distribution with mean $mu$ and variance $sigma^2$ applied to $X$.

$ q(X_t | X_(t-1)) = cal(N)(X_t; sqrt(1 - beta_t) X_(t-1), beta_t I) $ <eq2>

By iterating this process over $T$ time steps, we can obtain a sequence of noisy data points $(X_1, X_2, ..., X_T)$, where $X_T$ is the noisiest version of the data wich can be considered as pure noise.

This can be further simplified to a closed-form expression that directly relates $X_T$ to the original data $X_0$, as shown in @eq3.

$ q(X_T | X_0) = q(X_1 | X_0) times q(X_2 | X_1) times ... times q(X_T | X_(T-1)) = product_(t=1)^(T) q(X_t | X_(t-1)) $ <eq3>

We note $alpha_t = 1 - beta_t$, so the forward process can be written as:

$
  X_t &= sqrt(alpha_t) X_(t-1) + sqrt(1 - alpha_t) epsilon_t \
  &= sqrt(alpha_t) (sqrt(alpha_(t-1)) X_(t-2) + sqrt(1 - alpha_(t-1)) epsilon_(t-1)) + sqrt(1 - alpha_t) epsilon_t \
  &= sqrt(alpha_t) sqrt(alpha_(t-1)) X_(t-2) + sqrt(alpha_t) sqrt(1 - alpha_(t-1)) epsilon_(t-1) + sqrt(1 - alpha_t) epsilon_t \
  &= sqrt(alpha_t alpha_(t-1)) X_(t-2) + sqrt(alpha_t (1 - alpha_(t-1))) epsilon_(t-1) + sqrt(1 - alpha_t) epsilon_t \
  &= sqrt(alpha_t alpha_(t-1)) (sqrt(alpha_(t-2)) X_(t-3) + sqrt(1 - alpha_(t-2)) epsilon_(t-2)) + sqrt(alpha_t (1 - alpha_(t-1))) epsilon_(t-1) + sqrt(1 - alpha_t) epsilon_t \
  &= sqrt(alpha_t alpha_(t-1) alpha_(t-2)) X_(t-3) + sqrt(alpha_t alpha_(t-1) (1 - alpha_(t-2))) epsilon_(t-2) + sqrt(alpha_t (1 - alpha_(t-1))) epsilon_(t-1) + sqrt(1 - alpha_t) epsilon_t \
  &= ... \
  &= sqrt(product_(i=1)^(t) alpha_i) X_0 + sum_(k=1)^(t) sqrt((1 - alpha_k) product_(j=k+1)^(t) alpha_j) epsilon_k \
  &= sqrt(overline(alpha)_t) X_0 + sum_(k=1)^(t) sqrt((1 - alpha_k) product_(j=k+1)^(t) alpha_j) epsilon_k text("      with ") overline(alpha)_t = product_(i=1)^(t) alpha_i \
$

We note $N_t = sum_(k=1)^(t) sqrt((1 - alpha_k) product_(j=k+1)^(t) alpha_j) epsilon_k$.
We want to express $N_t$ as a single noise term since a sum of independent Gaussian noise terms is also a Gaussian noise term, which will allow us to obtain a closed-form expression for $X_t$ in terms of $X_0$ and a single noise term, as shown in @eq6.

The variance of $N_t$ can be calculated as in @eq5, where we use the fact that $epsilon_1, epsilon_2, ..., epsilon_T ~ cal(N)(0, I)$ are independent standard Gaussian noise terms.

#let Var = math.op("Var")

$
  Var(N_t) &= Var(sum_(k=1)^(t) sqrt((1 - alpha_k) product_(j=k+1)^(t) alpha_j) epsilon_k) \
  &= sum_(k=1)^(t) (1 - alpha_k) product_(j=k+1)^(t) alpha_j Var(cal(N)(0, I)) \
  &= sum_(k=1)^(t) (1 - alpha_k) product_(j=k+1)^(t) alpha_j \
  &= (1 - alpha_t) + alpha_t (1 - alpha_(t-1)) + alpha_t alpha_(t-1) (1 - alpha_(t-2)) + ... + product_(i=2)^(t) alpha_i (1 - alpha_1) \
  &= 1 - alpha_t + alpha_t - alpha_t alpha_(t-1) + alpha_t alpha_(t-1) - alpha_t alpha_(t-1) alpha_(t-2) + ... + product_(i=2)^(t) alpha_i - product_(i=1)^(t) alpha_i \
  &= 1 - product_(i=1)^(t) alpha_i \
  &= 1 - overline(alpha)_t \
$ <eq5>

We obtain the final expression for $X_t$ in @eq6 with $epsilon ~ cal(N)(0, I)$ being a single Gaussian noise term with variance $1 - overline(alpha)_t$.

$ X_t = sqrt(overline(alpha)_t) X_0 + sqrt(1 - overline(alpha)_t) epsilon $ <eq6>

This closed-form expression is particularly useful because it allows us to sample $X_t$ directly from $X_0$ without having to iterate through all the intermediate steps, which can be computationally expensive when $T$ is large.

== Reverse Diffusion Process

The reverse diffusion process consists of going from the noisy data back to the original data by learning to reverse the noise addition process, as illustrated in @fig2.

#let src_dir = "../../.."

#let make-image-node = set-image-node-defaults(options: (image-pad: 0, image-width: 3.5, image-height: 4, border: false, title-size: 1em))

#let src_dir = "../../.."

#let make-image-node = set-image-node-defaults(options: (image-pad: 0, image-width: 3.5, image-height: 4, border: false, title-size: 1em))

#figure(
  {
  let nodes = (
    image-node("x0", title: $X_0$, src: src_dir + "/img/noisy_images/tangled.png", cover: true, image-size: (2.5, 3)),
    image-node("x1", title: $X_1$, src: src_dir + "/img/noisy_images/noisy_image_0.3.png", cover: true, image-size: (2.5, 3), pos: right-of("x0")),
    image-node("x2", title: $X_2$, src: src_dir + "/img/noisy_images/noisy_image_0.5.png", cover: true, image-size: (2.5, 3), pos: right-of("x1")),
    text-node("dots", " ... ", title-size: 1em, pos: right-of("x2"), node-size: (2, 3)),
    image-node("xT", title: $X_T$, src: src_dir + "/img/noisy_images/noisy_image_10.0.png", pos: right-of("dots"), cover: true, image-size: (2.5, 3)),
  )

  let edges = (
    ml-edge("xT", "dots", label: "Removing noise", label-sep: 0.2em),
    ml-edge("dots", "x2"),
    ml-edge("x2", "x1"),
    ml-edge("x1", "x0"),

  )

  ml-diagram(nodes, edges: edges, label-size: 0.6em, spacing: 4.5em)
  },
  caption: "Illustration of the reverse diffusion process",
) <fig2>

As we cannot estimate $q(X_(t-1) | X_t)$ directly, we train a model $p_theta (X_(t-1) | X_t)$ to approximate it, where $theta$ denotes the parameters of the model.
This conditional distribution can be represented as a Gaussian distribution with mean $mu_theta (X_t, t)$ and covariance $Sigma_theta (X_t, t)$, as shown in @eq7.
$ p_theta (X_(t-1) | X_t) = cal(N)(X_(t-1); mu_theta (X_t, t), Sigma_theta (X_t, t))  approx q(X_(t-1) | X_t) $ <eq7>

The $mu_theta (X_t, t)$ aims to predict the expected value of $X_(t-1)$ given $X_t$, while $Sigma_theta (X_t, t)$ captures the uncertainty in this prediction.

As detailled in the very nice blog post by Lilian Weng on diffusion models @diffusion, the mean $mu_theta (X_t, t)$ can be expressed in terms of the noise prediction $epsilon_theta (X_t, t)$ as shown in @eq8, where $alpha_t$ and $overline(alpha)_t$ are defined as in the forward diffusion process.
In fact, the model aims to predict the noise $epsilon_theta (X_t, t)$ that was added to the data at time step $t$ in the forward diffusion process, and then uses this prediction to compute the mean $mu_theta (X_t, t)$ for the reverse diffusion process.

$ mu_theta (X_t, t) = 1/sqrt(alpha_t)(X_t - (1 - alpha_t) / sqrt(1 - overline(alpha)_t) epsilon_theta (X_t, t)) $ <eq8>

A common choice for the covariance $Sigma_theta (X_t, t)$ is to set it to a fixed value, such as $Sigma_theta (X_t, t) = beta_t I$, which corresponds to the variance of the noise added in the forward diffusion process at time step $t$.
Indeed, the covariance will indicate how much noise should be removed at each step of the reverse diffusion process, that's why it is often set to a value that matches the noise level in the forward diffusion process to ensure that the model learns to effectively reverse the noise addition process.

=== Intuitive interpretation

We can imagine that the data are represented by a mountain with $X_0$ at the peak.
The forward diffusion process can be seen as a hiker starting at the peak and gradually descending the mountain, adding noise to the data at each step.
The reverse diffusion process can be thought of as a hiker starting at the base of the mountain (the noisiest data) and trying to climb back up to the peak (the original data) by following the path of least resistance, which is determined by the learned model $p_theta$.

The $mu_theta (X_t, t)$ indicates the direction in which the hiker should move to climb back up the mountain, while $Sigma_theta (X_t, t)$ represents the uncertainty in this direction, which indicates how many different paths the hiker can take to reach the peak.

// So if $mu_theta$ is zero, the hiker don't know which direction to go and can take any path, while if $mu_theta$ is large, the hiker has a clear direction to follow.
// And if $Sigma_theta$ is small, the hiker has only one path to follow, while if $Sigma_theta$ is large, the hiker has many different paths to choose from.

For instance, in the case of image generation, $mu_theta$ indicates that the image should be a bear, while $Sigma_theta$ indicates if it should be a black bear, a brown bear, or a polar bear.

So the model aims to minimize the difference between the predicted distribution $p_theta (X_(t-1) | X_t)$ and the true distribution $q(X_(t-1))$ to learn how to effectively reverse the noise addition process and recover the original data from the noisy data.

The complete process can be represented as in @fig3, where the model learns to predict the original data $hat(X)_0_t$ and the predicted noise $hat(z)_t$ from the noisy data $X_t$ at each time step $t$.

#let src_dir = "../../.."

#let make-image-node = set-image-node-defaults(options: (image-pad: 0, image-width: 3.5, image-height: 4, border: false, title-size: 1em))

#let src_dir = "../../.."

#let make-image-node = set-image-node-defaults(options: (image-pad: 0, image-width: 3.5, image-height: 4, border: false, title-size: 1em))

#figure(
  {
  let nodes = (
    image-node("x0", title: $X_0$, src: src_dir + "/img/noisy_images/tangled.png", cover: true, image-size: (2.5, 3)),
    image-node("x1", title: $X_1$, src: src_dir + "/img/noisy_images/noisy_image_0.3.png", cover: true, image-size: (2.5, 3), pos: right-of("x0")),
    image-node("x2", title: $X_2$, src: src_dir + "/img/noisy_images/noisy_image_0.5.png", cover: true, image-size: (2.5, 3), pos: right-of("x1")),
    text-node("dots", " ... ", title-size: 1em, pos: right-of("x2"), node-size: (2, 3)),
    image-node("xT", title: $X_T$, src: src_dir + "/img/noisy_images/noisy_image_10.0.png", pos: right-of("dots"), cover: true, image-size: (2.5, 3)),
    arrow-node("fwd", title: "Forward diffusion", dir: right, shape: "arrow", label-pos: "bellow", size: (2.5, 0.8), pos: below("x2", by: 0.6)),
    arrow-node("fwd", title: "Reverse diffusion", dir: left, shape: "arrow", label-pos: "above", size: (2.5, 0.8), pos: above("x2", by: 0.7)),
  )

  let edges = (
    ml-edge("x0", "x1", label: "Adding noise", label-sep: 0.2em),
    ml-edge("x1", "x2"),
    ml-edge("x2", "dots"),
    ml-edge("dots", "xT"),
    ml-edge("xT", "dots", bend: -45deg, from-side: "top", to-side: "top", label: "Removing noise", label-sep: 0.2em),
    ml-edge("dots", "x2", bend: -45deg, from-side: "top", to-side: "top"),
    ml-edge("x2", "x1", bend: -45deg, from-side: "top", to-side: "top"),
    ml-edge("x1", "x0", bend: -45deg, from-side: "top", to-side: "top"),
  )

  ml-diagram(nodes, edges: edges, label-size: 0.7em, spacing: 4.1em)
  },
  caption: "Complete diffusion process",
) <fig3>

#pagebreak()

= Binary Diffusion Models

The binary diffusion models are a specific type of diffusion models that are designed to handle binary data, which is a common type of quantized data.
For instance, in the context of computational imaging, we can deal with binary representations of images by a decomposition of the image into bit-planes or by binarization of the image using some neural network-based binarization method.

In binary diffusion models, thanks to the binary nature of the data, the management of the noise addition and removal process is simplified compared to the general case of continuous data.
The forward diffusion process can be defined as a simple XOR operation between the binary data and a binary noise term, like the reverse diffusion process that is also defined as a simple XOR operation between the noisy binary data and a predicted binary noise term.

Thanks to the binary nature of the data, there is no need to tune a complex noise schedule as in the case of continuous data.
This allows for a simpler preprocessing of the data and a more straightforward training of the model, which can lead to faster convergence and improved performance when dealing with binary data.

== Training the model <train>

The training of the binary diffusion model involves optimizing loss functions that encourage the model to learn how to effectively reverse the noise addition process and recover the original binary data from the noisy binary data.

The complete process can be represented as in the @fig4, where the model learns to predict the original binary data $hat(X)_0_t$ and the predicted noise $hat(z)_t$ from the noisy binary data $X_t$ at each time step $t$.
In this way, the model can reverse the noise either by predicting the original binary data $hat(X)_0_t$ ("target" mode), or by directly predicting the noise $hat(z)_t$ and then applying the XOR operation with the noisy binary data $X_t$ to remove the noise and recover the original data ("mask" mode).

#figure(
  {
  let nodes = (
    dataset(0, title: $t_0$),
    module(1, title: $T$, pos: right-of("0")),
    vector(2, title: $X_0$, pos: right-of("1"), dir: "v", stack: 3),
    gate-node(3, pos: right-of("2")),
    vector(6, title: $X_t$, pos: right-of("3"), dir: "v", stack: 3),
    module(7, title: $p_theta (hat(X)_0_t, hat(z)_t | X_t, t, Y_epsilon)$, pos: right-of("6"), size: (5, 2)),
    vector(16, title: $hat(X)_0_t$, pos: right-of("7"), dir: "v", stack: 3),
    module(11, title: $T^(-1)$, pos: right-of("16")),
    dataset(13, title: $hat(t)_0_t$, pos: right-of("11")),

    compare-node(12, title: $cal(L)_z$, size: (1.5, 2), pos: below("11", by: 1.15)),

    text-node(4, $bold(z)_B$, pos: below("0", by: 1.3)),
    module(5, title: $cal(M)_t$, pos: right-of("4")),
    vector(14, title: $z_t$, pos: right-of("5"), dir: "v", stack: 3),

    compare-node(15, title: $cal(L)_x$, size: (1.5, 2), pos: above("11", by: 1.15)),

    dataset(8, title: $Y$, pos: below("4", by: 1.05)),
    module(9, title: $epsilon_y$, pos: right-of("8")),
    vector(10, title: $Y_epsilon$, pos: right-of("9"), dir: "v", stack: 3),

  )

  let edges = (
    ml-edge("0", "1"),
    ml-edge("1", "2"),
    ml-edge("2", "3"),
    ml-edge("3", "6"),

    ml-edge("4", "5"),
    ml-edge("5", "14"),
    ml-edge("14", "3", orthogonal: true),


    ml-edge("6", "7"),
    ml-edge("8", "9"),
    ml-edge("9", "10"),


    ml-edge("7", "11"),
    ml-edge("11", "13"),
    ml-edge("11", "13"),

    ml-edge("7", "12", orthogonal: "v", from-shift: -0.3, to-shift: -0.15, label: $hat(z)_t$, label-pos: 33%, label-side: left),
    ml-edge("14", "12", to-shift: -0.15),

    ml-edge("10", "7", orthogonal: true, to-shift: -0.3, crossing: true),

    ml-edge("2", "15", to-shift: -0.15, orthogonal: "v"),

    ml-edge("16", "15", orthogonal: "v", to-shift: 0.15),

  )

  ml-diagram(nodes, edges: edges, label-size: 0.7em, spacing: 1.3em)
  },
  caption: "Binary diffusion process",
) <fig4>

As shown in @fig4, the model $p_theta$ takes also as input some additional information $Y_epsilon$ that can be used to condition the generation process on some specific attributes of the data.
This can be seen as the prompt of the user that guides the generation process towards specific types of samples.

=== Loss functions

Two loss functions are computed to train the model:
- $cal(L)_z$: A loss function that encourages the model to accurately predict the noise $hat(z)_t$ that was added to the data.
- $cal(L)_x$: A loss function that encourages the model to accurately predict the original binary data $hat(X)_0_t$ from the noisy binary data $X_t$.

In the paper "Tabular Data Generation using Binary Diffusion" by V. Kinakh and S. Voloshynovskiy @tabular, the authors propose the binary cross-entropy as the loss function for both $cal(L)_z$ and $cal(L)_x$, which is a common choice for binary data as it measures how well the predicted probabilities match the true binary labels.
$ cal(L)_z = - sum_(i=1)^N z_(t,i) log(hat(z)_(t,i)) + (1 - z_(t,i)) log(1 - hat(z)_(t,i)) $ <eq9>
$ cal(L)_x = - sum_(i=1)^N X_(0,i) log(hat(X)_(0,i)) + (1 - X_(0,i)) log(1 - hat(X)_(0,i)) $ <eq10>

In the @eq9 and @eq10:
- $N$ is the number of binary features in the data.
- $z_(t,i)$ is the true binary noise for feature $i$ at time step $t$.
- $hat(z)_(t,i)$ is the predicted probability of the noise for feature $i$ at time step $t$.
- $X_(0,i)$ is the true binary value of feature $i$ in the original data.
- $hat(X)_(0,i)$ is the predicted probability of the original binary value for feature $i$.

The overall loss function for training the model is then simply the mean of the two loss functions, as shown in @eq11 where $B$ denotes the batch size used during training and $theta$ the parameters of the model to be optimized.

$ cal(L)(theta) = 1/B sum_(b=1)^B (cal(L)_z^((b)) + cal(L)_x^((b))) $ <eq11>

=== First transformation

The first transformation $T$ can be defined as a simple binary encoding of the original data that must be inversible ($T^( -1 )$) to ensure that the model can recover the original data from the noisy data.

For instance, in the context of computational imaging, we can decompose an image into bit-planes, where each bit-plane represents a binary image that captures a specific bit of the pixel values, as illustrated in @fig5, with the $k$-th bit-plane representing the $k$-th bit of the pixel values.

#figure(
  {
  let nodes = (
    image-node("bitplane8", title: "Bit-plane 8", src: src_dir + "/img/bit_planes/bit_plane_7.png", cover: true, image-size: (3, 3), title-gap: 0.5em),
    image-node("bitplane7", title: "Bit-plane 7", src: src_dir + "/img/bit_planes/bit_plane_6.png", cover: true, image-size: (3, 3), title-gap: 0.5em, pos: right-of("bitplane8")),
    text-node("dots", " ... ", title-size: 1em, pos: right-of("bitplane7"), node-size: (3, 3)),
    image-node("bitplane1", title: "Bit-plane 1", src: src_dir + "/img/bit_planes/bit_plane_0.png", cover: true, image-size: (3, 3), title-gap: 0.5em, pos: right-of("dots")),
    image-node("original", title: "Original image", caption-pos: "top", src: src_dir + "/img/bit_planes/original.png", cover: true, image-size: (3, 3), title-gap: 0.5em, pos: explicit-pos( 2.5, y: -4)),
  )

  let edges = (
    ml-edge("original", "bitplane8", to-side: "top"),
    ml-edge("original", "bitplane7", to-side: "top"),
    ml-edge("original", "dots", to-side: "top"),
    ml-edge("original", "bitplane1", to-side: "top"),
  )

  ml-diagram(nodes, edges: edges, label-size: 0.7em, spacing: 1em)
  },
  caption: "Example of a binary encoding of an image into bit-planes"
) <fig5>

So the first transformation $T$ is important since it allows to work with different types of data such as images @binary, tabular data @tabular etc... by simply applying an appropriate binary encoding to the data.

== Sampling from the model

The sampling process from the binary diffusion model involves starting from pure noise (the noisiest data) and iteratively applying the reverse diffusion process to generate a sample that resembles the original data.

It can be illustrated as in @fig6, where the sampling process starts from pure noise $X_T$ and iteratively applies the reverse diffusion process to generate a sample $hat(X)_0_t$ that resembles the original data.

Note that the sampling process can also be guided by conditioning variables such as $Y_epsilon$ to generate samples that have specific desired properties.


#figure(
  {
  let nodes = (
    text-node(4, $bold(z)_B$),
    module(5, title: $cal(M)_t$, pos: right-of("4")),
    vector(14, title: $z_t$, pos: right-of("5"), dir: "v", stack: 3),
    gate-node(3, pos: right-of("14")),
    vector(6, title: $X_t$, pos: right-of("3"), dir: "v", stack: 3),
    module(7, title: $p_theta (hat(X)_0_t, hat(z)_t | X_t, t, Y_epsilon)$, pos: right-of("6"), size: (5, 2)),
    vector(16, title: $hat(X)_0_t$, pos: right-of("7"), dir: "v", stack: 3),
    module(11, title: $T^(-1)$, pos: right-of("16")),
    dataset(13, title: $hat(t)_0_t$, pos: right-of("11")),

    dataset(8, title: $Y$, pos: below("4", by: 1.4)),
    module(9, title: $epsilon_y$, pos: right-of("8")),
    vector(10, title: $Y_epsilon$, pos: right-of("9"), dir: "v", stack: 3),

    module(15, title: $t = 0 ... T$, pos: above("6", by: 4))

  )

  let edges = (
    ml-edge("3", "6"),

    ml-edge("4", "5"),
    ml-edge("5", "14"),
    ml-edge("14", "3"),


    ml-edge("6", "7"),
    ml-edge("8", "9"),
    ml-edge("9", "10"),


    ml-edge("7", "11"),
    ml-edge("11", "13"),
    ml-edge("11", "13"),


    ml-edge("10", "7", orthogonal: true),

    ml-edge("16", "3", from-side: "top", to-side: "top", via: ("u", "u", "l", "l", "l")),

    ml-edge("15", "7", orthogonal: "h", dash: "dashed"),
    ml-edge("15", "5", orthogonal: "h", dash: "dashed"),



  )

  ml-diagram(nodes, edges: edges, label-size: 0.7em, spacing: 1.3em)
  },
  caption: "Sampling process",
) <fig6>

On the @fig6, the noise depends on the time step $t$ and is computed by a noise scheduler $cal(M)_t$ that determines the amount of noise to be removed to granually denoise the data.

#pagebreak()

= Binary Diffusion Models for Tabular Data Generation

Before working with images which are quite complex data and require more computational resources, we can start by working with tabular data which is simpler and more manageable, and then we can extend the approach to images.

In the paper "Tabular Data Generation using Binary Diffusion" by V. Kinakh and S. Voloshynovskiy @tabular, the authors propose a binary diffusion model for generating synthetic tabular data, which is a common type of quantized data that can be represented in a binary format.

The model is trained to learn the underlying distribution of the tabular data and can be used to generate new synthetic samples that resemble the original data (which can be useful for data augmentation, privacy preservation, etc...).

== Link with image generation

The process of generating tabular data is similar to the process of generating images, the main difference being the first transformation $T$ and the fact that the data are 1-dimensional instead of 2-dimensional as in the case of images.

== First transformation for tabular data

The first transformation $T$ for tabular data can be defined as a simple binary encoding of the original data, where the categorical features are one-hot encoded and the numerical features are binarized using a fixed number of bits to represent the values.

Then, all binary features are concatenated to form a single binary vector that represents the original tabular data, that can be reversed using the inverse transformation $T^(-1)$ to recover the original tabular data from the binary representation.

== Architecture of the model

As described in the paper @tabular, the sampling algorithm for generating synthetic data from the binary diffusion model can be represented as in the @algo.

#pagebreak()

#algorithm-figure(
  "Sampling Algorithm",
  inset: 0.3em,
  {
    import algorithmic: *
        LineComment(
          Assign[$X_t$][random binary tensor],
          [Initialize noisy sample]
        )

        Assign[$Y$][condition/label]

        Assign[$Y_epsilon$][apply condition]
        Assign[$text("threshold")$][threshold value for binarization]


        Assign[$p_theta$][pre-trained denoiser network]



        For(
          $t in {T, ..., 0}$,
          {
            LineComment(
              Assign[$hat(X)_0, hat(z)_t$][$p_theta (hat(X)_0_t, hat(z)_t | X_t, t, Y_epsilon)$],
              [Predict clean sample and latent noise]
            )

            LineComment(
              Assign[$hat(X)_0$][$sigma(hat(X)_0) > text("threshold")$],
              [Apply sigmoid and binarize]
            )
            Assign[$hat(z)_t$][$sigma(hat(z)_t) > text("threshold")$]

            LineComment(
              Assign[$z_t$][
                get_binary_noise(t-1)
              ],
              [Generate random binary noise for time step t-1]
            )

            LineComment(
              Assign[$X_t$][$hat(X)_0 xor z_t$],
              [Update sample using XOR]
            )
          }
        )

        Return[$X_t$]
      },
    ) <algo>


=== Evaluation of the model

To evaluate the performance of the binary diffusion model for tabular data generation, the procedure described in the paper @procedure is followed.

The evaluation procedure consists of the following steps:
+ Train the binary diffusion model on the original dataset.
+ Generate a synthetic dataset by sampling from the trained model.
+ Train some model on the synthetic dataset.
+ Evaluate the performances.

In the paper @tabular, the authors use a Random Forest classifier, a Linear/Logistic Regression classifier, and a Decision Tree classifier to evaluate the performance of the generated synthetic data on a classification and regression tasks, and they report the results in terms of prediction accuracy for classification tasks and mean squared error for regression tasks.

=== Fixed thresholding step

In the paper @tabular, the authors mention that the model doesn't perform well when increasing the number of time steps $T$ in the reverse diffusion process.

This behavior is quite surprising since increasing the number of time steps $T$ should allow the model to generate higher quality samples by gradually removing the noise from the data, which is a common behavior observed in diffusion models for continuous data.

However, note that in the @algo, the binarization step uses a fixed threshold value, which can be the root cause of the performance degradation.

For instance, if the model is in mode "mask", the predicted noise $hat(z)_t$ can have 50% of ones meaning that the model will bring 50% of changes to the data at each step of the reverse diffusion process.
This can degrade the quality of the generated samples since the model will be forced to make a large number of changes to the data at each step instead of gradually removing the noise.

Some tests have been done to try to understand the impact of the fixed thresholding step on the performance of the model

#pagebreak()
= Results of Tabular Data Generation

First, I managed to reproduce the results of the paper @tabular on the Adult dataset @adult_2.

Then, I tried to improve the performance of the model by bringing some modifications:
- Removing the fixed thresholding
- Correctly applying the guidance during the sampling process
- Fixing the noise addition step in the sampling algorithm

== Removing the fixed thresholding step

The first modification was to remove the fixed thresholding step in the sampling algorithm (mode "mask").

Instead of applying a fixed threshold to binarize the predicted clean sample $hat(X)_0$, I applied a dynamic threshold that is computed based on the quantity of noise to be removed at each step of the reverse diffusion process.

This dynamic thresholding approach allows the model to adapt the binarization process to the amount of noise that is being removed at each step, which can lead to a gradual removal of the noise and an improvement in the quality of the generated samples as we increase the number of time steps $T$ in the reverse diffusion process.

The @fig7 shows the results of the evaluation of the model with the dynamic thresholding approach (linear and quadratic schedules) compared to the fixed thresholding approach (constant schedule) for different values of $T$.

We can see that the dynamic thresholding approach doesn't improve at all the performance of the model compared to the fixed thresholding approach.
And between the two dynamic thresholding approaches, the linear schedule performs worse than the quadratic schedule, which is expected since the model was trained with a quadratic noise schedule.

```python
%| echo: false
%| plt-axes.grid: false
%| img-width: 100%
%| label: fig7

import pandas as pd

df = load_results("study_results.csv")[0]

fig1 = plot_heatmaps(
    df,
    value_col     = "mean",
    x_col         = "n_timesteps",
    y_col         = "schedule",
    fixed_filters = {"strategy": "mask", "renoise_factor": 1.0},
    subplot_cols  = ["use_t_next"],
    # ncols         = 2,
    suptitle      = "Schedule comparison for `strategy=mask` (baseline: `schedule=const`)",
    cell_size      = (5, 4),
    std_col       = "std",
)
plt.show()
```

Indeed, the model already predicts the right amount of noise to be removed at each step, as shown by the following results that compare the quantity of noise in the predicted mask with the quantity of noise that should be removed at each step of the reverse diffusion process.
```raw
Step 800: Ones in mask: 33.6364%, Ones required: 33.6031%
Step 600: Ones in mask: 20.4545%, Ones required: 20.3957%
Step 400: Ones in mask: 10.4545%, Ones required: 10.4685%
Step 200: Ones in mask: 4.0909%, Ones required: 3.8215%
Step 0: Ones in mask: 0.4332%, Ones required: 0.4545%
```
So this analysis shows that the fixed thresholding step is not the cause of the performance degradation since the model is already predicting the right amount of noise to be removed at each step, and the dynamic thresholding approach doesn't improve the performance of the model compared to the fixed thresholding approach.

== Correctly applying the guidance during the sampling process

The second modification was to follow the approach described in the paper "Classifier-Free Diffusion Guidance" @guidance to guide correctly the sampling process.
In fact, the guidance was not correctly applied in the current implementation since it was not following the approach proposed in the paper, even though the goal was to implement the classifier-free guidance as described in the paper @guidance.

In the paper, the approach proposed is the following:
$ (1 + w) dot epsilon_c - w dot epsilon_u $
Where $epsilon_c$ is conditioned prediction and $epsilon_u$ is unconditioned prediction, and $w$ is a scalar value that controls the strength of the guidance.

The conditioned prediction can be seen as the conformity to the desired properties, while the unconditioned prediction can be seen as the diversity of the generated samples, and the guidance allows to find a good balance between these two aspects to generate high-quality samples that match the desired properties while still being diverse and representative of the general distribution of the data.

The current implementation was doing the following:
$ epsilon_u + g dot (epsilon_c - epsilon_u) $
which is equivalent to the approach proposed in the paper @guidance but with a different parameterization of the guidance strength: $g = 1 + w$:
$ epsilon_u + (1 + w) dot (epsilon_c - epsilon_u) = (1 + w) dot epsilon_c - w dot epsilon_u $

This modification allows to correctly apply the guidance during the sampling process, but this is only a small fix of -1 in the implementation that doesn't have a significant impact on the performance of the model.

== Fixing the noise addition step in the sampling algorithm

The third modification was a small fix in the implementation of the sampling algorithm.

As described in the @algo at line 11, the sample process adds the noise of the step $t-1$ to the predicted clean sample $hat(X)_0$ to get the new sample $X_t$ for the next step of the reverse diffusion process.

#let src_dir = "../../.."

#let make-image-node = set-image-node-defaults(options: (image-pad: 0, image-width: 3.5, image-height: 4, border: false, title-size: 1em))

#let src_dir = "../../.."

#let make-image-node = set-image-node-defaults(options: (image-pad: 0, image-width: 3.5, image-height: 4, border: false, title-size: 1em))

#figure(
  {
  let size = (2, 3)
  let nodes = (
    image-node("step_t", title: $X_t$, src: src_dir + "/img/noisy_images/noisy_image_0.5.png", cover: true, image-size: size, pos: explicit-pos(0, y: 0)),
    arrow-node("predict", title: "Predict", dir: right, shape: "arrow", label-pos: "inside", size: (3, 0.8), pos: right-of("step_t")),
    gate-node("gate", pos: right-of("predict")),
    image-node("pred", title: $hat(X)_0$, caption-pos: "top", title-gap: 0.4em, src: src_dir + "/img/noisy_images/tangled.png", cover: true, image-size: size, pos: above("gate", by: 0.6)),
    image-node("noise", title: $z_t$, caption-pos: "bottom", title-gap: 0.4em, src: src_dir + "/img/noisy_images/noise_0.5.png", cover: true, image-size: size, pos: below("gate", by: 0.6)),
    arrow-node("update", title: "Renoise", dir: right, shape: "arrow", label-pos: "inside", size: (3.3, 0.8), pos: right-of("gate")),
    module("done", title: "What was done", pos: above("update", by: 1), title-size: 0.8em),

    image-node("pred_2", title: $hat(X)_0$, caption-pos: "top", title-gap: 0.4em, src: src_dir + "/img/noisy_images/tangled.png", cover: true, image-size: size, pos: right-of("pred", by: 2)),
    image-node("noise_2", title: $z_(t)$, caption-pos: "bottom", title-gap: 0.4em, src: src_dir + "/img/noisy_images/noise_0.5.png", cover: true, image-size: size, pos: right-of("noise", by: 2)),
    gate-node("gate_2", pos: right-of("update", by: 3)),
    image-node("updated_sample", title: $X_(t-1)$, src: src_dir + "/img/noisy_images/noisy_image_0.5.png", cover: true, image-size: size, pos: right-of("gate_2", by: 2)),

  )

  let edges = (
    ml-edge("pred_2", "gate_2", from-side: "right", to-side: "top", orthogonal: true),
    ml-edge("noise_2", "gate_2", from-side: "right", to-side: "bottom", orthogonal: true),
    ml-edge("gate_2", "updated_sample", from-side: "right", to-side: "left"),
    ml-edge("updated_sample", "step_t", from-side: "top", to-side: "top", bend: -45deg, dash: "dashed", label: "Same amount of noise", mark: "<|-|>"),
  )

  ml-diagram(nodes, edges: edges, label-size: 1em, spacing: 1.5em)
  },
  caption: "Current implementation of the noise addition step in the sampling algorithm",
) <noise_1>


But in the existent implementation, the noise of the current step $t$ was added instead of the noise of the next step $t-1$ as shown in @noise_1.

The noise of the current step $t$ is not the correct noise to be added since it corresponds to the noise that was removed at the current step, while the noise of the next step $t-1$ corresponds to the noise that will be removed at the next step.
The correct implementation should add the noise of the next step $t-1$ to the predicted clean sample $hat(X)_0$ to get the new sample $X_(t-1)$ for the next step of the reverse diffusion process, as shown in @noise_2.

#let src_dir = "../../.."

#let make-image-node = set-image-node-defaults(options: (image-pad: 0, image-width: 3.5, image-height: 4, border: false, title-size: 1em))

#let src_dir = "../../.."

#let make-image-node = set-image-node-defaults(options: (image-pad: 0, image-width: 3.5, image-height: 4, border: false, title-size: 1em))

#figure(
  {
  let size = (2, 3)
  let nodes = (
    image-node("step_t", title: $X_t$, src: src_dir + "/img/noisy_images/noisy_image_0.5.png", cover: true, image-size: size, pos: explicit-pos(0, y: 0)),
    arrow-node("predict", title: "Predict", dir: right, shape: "arrow", label-pos: "inside", size: (3, 0.8), pos: right-of("step_t")),
    gate-node("gate", pos: right-of("predict")),
    image-node("pred", title: $hat(X)_0$, caption-pos: "top", title-gap: 0.4em, src: src_dir + "/img/noisy_images/tangled.png", cover: true, image-size: size, pos: above("gate", by: 0.6)),
    image-node("noise", title: $z_t$, caption-pos: "bottom", title-gap: 0.4em, src: src_dir + "/img/noisy_images/noise_0.5.png", cover: true, image-size: size, pos: below("gate", by: 0.6)),
    arrow-node("update", title: "Renoise", dir: right, shape: "arrow", label-pos: "inside", size: (3.3, 0.8), pos: right-of("gate")),
    module("done", title: "What is correct", pos: above("update", by: 1), title-size: 0.8em),

    image-node("pred_2", title: $hat(X)_0$, caption-pos: "top", title-gap: 0.4em, src: src_dir + "/img/noisy_images/tangled.png", cover: true, image-size: size, pos: right-of("pred", by: 2)),
    image-node("noise_2", title: $z_(t-1)$, caption-pos: "bottom", title-gap: 0.4em, src: src_dir + "/img/noisy_images/noise_0.3.png", cover: true, image-size: size, pos: right-of("noise", by: 2)),
    gate-node("gate_2", pos: right-of("update", by: 3)),
    image-node("updated_sample", title: $X_(t-1)$, src: src_dir + "/img/noisy_images/noisy_image_0.3.png", cover: true, image-size: size, pos: right-of("gate_2", by: 2)),

  )

  let edges = (
    ml-edge("pred_2", "gate_2", from-side: "right", to-side: "top", orthogonal: true),
    ml-edge("noise_2", "gate_2", from-side: "right", to-side: "bottom", orthogonal: true),
    ml-edge("gate_2", "updated_sample", from-side: "right", to-side: "left"),
    ml-edge("updated_sample", "step_t", from-side: "top", to-side: "top", bend: -45deg, dash: "dashed", label: "Not same amount of noise", label-side: "below", mark: "<|-|>"),
  )

  ml-diagram(nodes, edges: edges, label-size: 1em, spacing: 1.5em)
  },
  caption: "Correct implementation of the noise addition step in the sampling algorithm",
) <noise_2>


By adding the correct noise at each step, the amount of noise is now correctly reduced at each step of the reverse diffusion process, and this fix is called `use_t_next` since it uses the noise of the next step $t-1$ instead of the noise of the current step $t$.

However, as shown in @fig8, this modification allows to generate better samples for small values of $T$ but it doesn't allow to improve the performance of the model for larger values of $T$.

```python
%| echo: false
%| label: fig8
%| plt-axes.grid: false
# plot_sensitivity(PARAM_KEYS, DATASETS, PLOT_CONFIG)
# plot_timestep_lines(DATASETS, PLOT_CONFIG)

import pandas as pd

df = load_results("study_results.csv")[0]

fig1 = plot_heatmaps(
    df,
    value_col     = "mean",
    x_col         = "n_timesteps",
    y_col         = "use_t_next",
    fixed_filters = {"renoise_factor": 1.0, "schedule": "const"},
    subplot_cols  = ["strategy"],
    ncols         = 2,
    suptitle      = "Fixed vs. non-fixed noise addition for different strategies\n(baseline: `use_t_next=False`)",
    cell_size      = (5, 4),
    std_col       = "std",
)
plt.show()
```

=== Increasing the noise in the sampling process

The last modification was to increase the noise in the sampling process by multiplying the predicted noise by a renoise factor that is greater than 1, which allows to add more noise at each step of the reverse diffusion process.

This modification allows to better explore the space of possible samples, which results in an improvement of the performance of the model for larger values of $T$ in the reverse diffusion process, as shown in @fig9.

```python
%| echo: false
%| label: fig9
%| plt-axes.grid: false
# plot_sensitivity(PARAM_KEYS, DATASETS, PLOT_CONFIG)
# plot_timestep_lines(DATASETS, PLOT_CONFIG)

import pandas as pd

df = load_results("study_results.csv")[0]

fig1 = plot_heatmaps(
    df,
    value_col     = "mean",
    x_col         = "n_timesteps",
    y_col         = "renoise_factor",
    fixed_filters = {"use_t_next": True, "schedule": "const"},
    subplot_cols  = ["strategy"],
    ncols         = 2,
    suptitle      = "Increase the noise in the sampling process (`renoise_factor`) for `use_t_next=True`\n(baseline: `renoise_factor=1.0`)",
    cell_size      = (5, 4),
    std_col       = "std",
)
plt.show()
```

The @fig10 shows the best configurations for each strategy compared to the original implementation of the model (baseline) for different values of $T$ in the reverse diffusion process.
We can see that the modifications allow to improve the performance of the model of up to 1% for different configurations, which shows that the modifications have a positive impact on the performance of the model, but there is still room for improvement to further enhance the performance of the model for larger values of $T$ in the reverse diffusion process.

```python
%| echo: false
%| label: fig10
%| fig-kind: table
%| caption: Best configurations vs. original implementation (`baseline`)
plot_study_table()
```

Here we were dealing with tabular data which is simpler than images, but the same modifications can be applied to the case of image generation to try to improve the performance of the model for larger values of $T$ in the reverse diffusion process.

Theses modifications can be much more impactful in the case of images since the data are more complex and the noise addition and removal process is more challenging compared to the case of tabular data.

#pagebreak()

= Results of Image Generation

After working on the tabular data generation, I started to work on the image generation using binary diffusion models, and I tried to apply the same modifications that were applied to the case of tabular data generation to see if they can also improve the performance of the model.

The results of the evaluation of the model were obtained after 5 days of computation on a RTX 4080 (16 Go VRAM), which shows that working with images is much more computationally expensive than working with tabular data, and it requires more time and resources to test and evaluate.

== Evaluation of the model

To evaluate the performance of the binary diffusion model for image generation, I used the CIFAR-10 dataset @cifar that was used for the training of the model.

The evaluation procedure consists of the following steps:
+ Train the binary diffusion model on the CIFAR-10 dataset.
+ Generate a synthetic dataset by sampling from the trained model.
+ Evaluate the quality of the generated images using metrics such as Inception Score (IS) Fréchet Inception Distance (FID) and Kernel Inception Distance (KID) that are commonly used for evaluating the performance of generative models for images.

The IS measures the quality of the generated images by evaluating how well they can be classified by a pre-trained Inception model, while the FID measures the distance between the distribution of the generated images and the distribution of the real images in the feature space of a pre-trained Inception model.


For the evaluation, the IS, FID, FID-clip and KID metrics were used, where FID-clip is a variant of the FID metric that uses a pre-trained CLIP model instead of an Inception model and KID is a variant of the FID metric that uses a kernel-based approach to measure the distance between the distributions of the generated and real images.

=== How to interpret the results of the evaluation

- IS: Higher is better
- FID: Lower is better
- FID-clip: Lower is better
- KID: Lower is better

== Fixing the thresholding step in the sampling algorithm

The same test of fixing the thresholding step to a certain amount of noise to be removed at each step of the reverse diffusion process was applied to the case of image generation, but it doesn't allow to improve the performance of the model, as shown in @fig11.

This is consistent with the results obtained for the case of tabular data generation since in both cases the model already predicts the right amount of noise to be removed at each step.

The only exception is the case of the FID-clip metric where the linear and quadratic schedules perform better than the baseline.

```python
%| echo: false
%| label: fig11

import pandas as pd
df = pd.read_csv("results_img.csv")

fig1 = plot_heatmaps(
    df,
    value_col     = "fid",
    x_col         = "use_t_next",
    y_col         = "schedule",
    fixed_filters = {"renoise_factor": 1.0, "strategy": "mask"},
    subplot_cols  = ["n_timesteps"],
    ncols         = 2,
    suptitle      = "Schedule comparison for `strategy=mask`\n(baseline: `schedule=const`)",
    cell_size      = (5, 4),
    use_kid         = True,
)
plt.show()
```

== Fixing the noise addition step in the sampling algorithm

The same test of fixing the noise addition step to add the correct noise at each step of the reverse diffusion process was applied to the case of image generation, leading to a big improvement of the performance of the model for smaller values of $T$ in the reverse diffusion process, and only a small improvement for larger values of $T$, as shown in @fig12.

For larger values of $T$, the FID-clip metric is the only metric that shows a decrease in the performance of the model when fixing the noise addition step, which is quite surprising since this modification should allow to improve the performance of the model by ensuring that the correct amount of noise is removed at each step of the reverse diffusion process.

```python
%| echo: false
%| label: fig12
%| plt-axes.grid: false

import pandas as pd

df = pd.read_csv("results_img.csv")

fig1 = plot_heatmaps(
    df,
    value_col     = "fid",
    x_col         = "strategy",
    y_col         = "use_t_next",
    fixed_filters = {"renoise_factor": 1.0, "schedule": "const"},
    subplot_cols  = ["n_timesteps"],
    ncols         = 2,
    suptitle      = "Fixed vs. non-fixed noise addition (`use_t_next`) for different metrics and strategies (baseline: `use_t_next=False`)",
    cell_size      = (5, 4),
    use_kid         = True,
)
plt.show()
```

So fixing the noise addition step allows to improve globally the performance of the model, making it more effective at reversing the noise addition process and generating higher quality samples, but it doesn't allow to improve significantly the performance of the model for larger values of $T$ in the reverse diffusion process, which indicates that there are still some issues that need to be addressed to further enhance the performance of the model for larger values of $T$.

Compared to the case of tabular data generation, fixing the noise addition step has a much bigger impact on the performance of the model for image generation, which can be explained by the fact that images are more complex data than tabular data, and the noise addition and removal process is more challenging for images compared to tabular data.

== Increasing the noise in the sampling process

The same test of increasing the noise in the sampling process by multiplying the predicted noise by a renoise factor that is greater than 1 was applied to the case of image generation, leading to a significant decrease in the performance of the model for larger values of $T$ in the reverse diffusion process, as shown in @fig13.

This result is expected since increasing the noise means degradating the quality of the generated samples since the model will not be able to remove the noise effectively.

```python
%| echo: false
%| label: fig13
%| plt-axes.grid: false
import pandas as pd
df = pd.read_csv("results_img.csv")

fig1 = plot_heatmaps(
    df,
    value_col     = "fid",
    x_col         = "strategy",
    y_col         = "renoise_factor",
    fixed_filters = {"use_t_next": True, "schedule": "const"},
    subplot_cols  = ["n_timesteps"],
    ncols         = 2,
    suptitle      = "Increase the noise in the sampling process (`renoise_factor`) for `use_t_next=True`\n(baseline: `renoise_factor=1.0`)",
    cell_size      = (5, 4),
    use_kid         = True,
)
plt.show()
```

== Generated images

The @fig14 shows some examples of generated images for different values of $T$.
We can distinguish horses in both images, but the image generated with $T=100$ shows more details and better quality.

#figure(grid(
  columns: 2,
  figure(image("img/n10.jpg", width: 30%), caption: [$T=10$], kind: "subfigure"),
  figure(image("img/n100.jpg", width: 30%), caption: [$T=100$], kind: "subfigure"),
), caption: [Generated images for different values of $T$ in the reverse diffusion process]
) <fig14>


#pagebreak()

= Conclusion

In this report, we explored the concept of binary diffusion models and their application to both tabular data generation and image generation.

We started by understanding the complete diffusion process, which consists of a forward diffusion process that adds noise to the data and a reverse diffusion process that learns to remove the noise and recover the original data.

Then, we focused on binary diffusion models, which are designed to handle binary data, and we discussed the training of the model, including the loss functions used to optimize the model and the first transformation that allows to work with different types of data.

Next, we evaluated the performance of the binary diffusion model for tabular data generation and image generation, and we applied some modifications to try to improve the performance of the model for larger values of $T$ in the reverse diffusion process.
The modifications included removing the fixed thresholding step, correctly applying the guidance during the sampling process, fixing the noise addition step, and increasing the noise in the sampling process.

Overall, the fixing of the noise addition step was the most impactful modification that allowed to improve the performance of the model for both tabular data generation and image generation, while the other modifications had a smaller impact on the performance of the model.

But there is still room for improvement to further enhance the performance of the model for larger values of $T$ in the reverse diffusion process, and future work can focus on addressing the remaining issues and exploring other modifications that can further improve the performance of the model.

== Acknowledgements

I would like to thank V. Kinakh, the author of the papers "Tabular Data Generation using Binary Diffusion" @tabular and "Binary Diffusion Probabilistic Model" @binary, for his guidance and support throughout this project, and for providing investigations directions that helped to understand the issues with the model and how to address them.

== Usage of AI

- Claude Sonnet 4.6 (Free): code generation, code debugging, code review, graph library generation.
- ChatGPT: explanations of concepts, code debugging.
- Github Copilot autocompletion: code writing and report writing.
