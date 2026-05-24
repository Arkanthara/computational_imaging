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

= Introduction on Diffusion Models

Generative models have revolutionized the field of artificial intelligence, enabling machines to create content that is indistinguishable from human-generated data or to improve the quality of data through various transformations.
Among these models, diffusion models have emerged as a powerful class of generative models that have shown remarkable performance in tasks such as image generation, image super-resolution, and data denoising.

Diffusion models are based on the concept of a diffusion process, which is a stochastic process that describes how data points evolve over time. The core idea behind diffusion models is to model the data generation process as a gradual transformation from noise to structured data. This is achieved by defining a forward diffusion process that adds noise to the data and a reverse diffusion process that learns to remove the noise and recover the original data.

Among the various types of diffusion models, Denoising Diffusion Probabilistic Models (DDPMs) have gained significant attention due to their ability to generate high-quality samples.
DDPMs work by iteratively adding noise to the data and then learning to reverse this process, effectively denoising the data step by step.

But are these models very suitable for all types of data, especially for quantized data?
Quantized data, which is often used in scenarios like in computational imaging, presents unique challenges for generative modeling.
The discrete nature of quantized data can make it difficult for diffusion models to learn the underlying data distribution effectively.

In answer to this problem, researchers have proposed various approaches to adapt diffusion models for quantized data, such as Binary Diffusion Models (BDMs) and Quantized Diffusion Models (QDMs).
These models are designed to handle the discrete nature of quantized data while still leveraging the powerful generative capabilities of diffusion models.

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

First, what are the foundational principles of diffusion models?

Diffusion models are a Markov chain-based generative model that defines a forward diffusion process and a reverse diffusion process.
The forward diffusion process gradually adds noise to the data, while the reverse diffusion process learns to remove the noise and recover the original data.
The training of diffusion models involves optimizing a variational bound on the data likelihood, which allows the model to learn the underlying data distribution effectively.

== Forward Diffusion Process

Suppose we have an original image $X_0$ that follows a data distribution $q$.

In Denoising Diffusion Probabilistic Models (DDPMs), the forward diffusion process is defined as first order Markov chain that gradually adds noise to the data over a series of time steps.

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
  caption: "Illustration of the forward diffusion process",
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
We want to express $N_t$ as a single noise term since a sum of independent Gaussian noise terms is also a Gaussian noise term with variance equal to the sum of the variances of the individual noise terms.
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

So if $mu_theta$ is zero, the hiker don't know which direction to go and can take any path, while if $mu_theta$ is large, the hiker has a clear direction to follow.
And if $Sigma_theta$ is small, the hiker has only one path to follow, while if $Sigma_theta$ is large, the hiker has many different paths to choose from.

For instance, in the case of image generation, $mu_theta$ indicates that the image should be a bear, while $Sigma_theta$ indicates if it should be a black bear, a brown bear, or a polar bear.

So the model aims to minimize the difference between the predicted distribution $p_theta (X_(t-1) | X_t)$ and the true distribution $q(X_(t-1) | X_t)$ to learn how to effectively reverse the noise addition process and recover the original data from the noisy data.

The complete process can be represented as a Markov chain that alternates between the forward diffusion process and the reverse diffusion process, where the model learns to effectively reverse the noise addition process at each step to recover the original data from the noisy data, as illustrated in @fig3.

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
    arrow-node("fwd", title: "Forward diffusion", dir: right, shape: "arrow", label-pos: "bellow", size: (2.5, 1), pos: below("x2", by: 0.6)),
    arrow-node("fwd", title: "Reverse diffusion", dir: left, shape: "arrow", label-pos: "above", pos: above("x2", by: 0.7)),
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

  ml-diagram(nodes, edges: edges, label-size: 0.7em, spacing: 4.5em)
  },
  caption: "Illustration of the complete diffusion process, which consists of a forward diffusion process that adds noise to the data and a reverse diffusion process that learns to remove the noise and recover the original data",
) <fig3>

= Binary Diffusion Models

The binary diffusion models are a specific type of diffusion models that are designed to handle binary data, which is a common type of quantized data.
For instance, in the context of computational imaging, we can deal with binary representations of images by a decomposition of the image into bit-planes, where each bit-plane represents a binary image that captures a specific bit of the pixel values.

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

As shown in @fig4, the model $p_theta$ takes as input the noisy binary data $X_t$, the time step $t$, and an optional conditioning variable $Y_epsilon$ that can be used to guide the generation process, and outputs a prediction of the original binary data $hat(X)_0_t$ and the predicted noise $hat(z)_t$ that was added to the data at time step $t$ in the forward diffusion process.

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

The first transformation $T$ can be defined as a simple binary encoding of the original data that must be inversible ($T^( -1 )$) to ensure that the model can learn to effectively reverse the noise addition process and recover the original data from the noisy data.

For instance, in the context of computational imaging, we can decompose an image into bit-planes, where each bit-plane represents a binary image that captures a specific bit of the pixel values, as illustrated in @fig5, with the $k$-th bit-plane representing the $k$-th bit of the pixel values.

#figure(
  {
  let nodes = (
    image-node("bitplane8", title: "Bit-plane 8", src: src_dir + "/img/bit_planes/bit_plane_7.png", cover: true, image-size: (3, 3)),
    image-node("bitplane7", title: "Bit-plane 7", src: src_dir + "/img/bit_planes/bit_plane_6.png", cover: true, image-size: (3, 3), pos: right-of("bitplane8")),
    text-node("dots", " ... ", title-size: 1em, pos: right-of("bitplane7"), node-size: (3, 3)),
    image-node("bitplane1", title: "Bit-plane 1", src: src_dir + "/img/bit_planes/bit_plane_0.png", cover: true, image-size: (3, 3), pos: right-of("dots")),
    image-node("original", title: "Original image", caption-pos: "top", src: src_dir + "/img/bit_planes/original.png", cover: true, image-size: (3, 3), pos: explicit-pos( 2.5, y: -4)),
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

So the first transformation $T$ is important since it allows to work with different types of data such as images @binary, tabular data @tabular etc... by simply applying an appropriate binary encoding to the data, which can be easily done as long as the encoding is inversible to ensure that the model can learn to effectively reverse the noise addition process and recover the original data from the noisy data.

== Sampling from the model

The sampling process from the binary diffusion model involves starting from pure noise (the noisiest data) and iteratively applying the reverse diffusion process to generate a sample that resembles the original data.

It can be illustrated as in @fig6, where the sampling process starts from pure noise $X_T$ and iteratively applies the reverse diffusion process to generate a sample $hat(X)_0_t$ that resembles the original data.

Note that the sampling process can be guided by conditioning variables such as $Y_epsilon$ to generate samples that have specific desired properties.
For instance, the conditioning variable can be a text description of the desired sample, which can be used to guide the generation process to produce samples that match the given description.


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

On the @fig6, the noise depends on the time step $t$ and is computed by a noise scheduler $cal(M)_t$ that determines the amount of noise to be removed at each step of the reverse diffusion process.
In this way, the noise is gradually removed from the data as we iteratively apply the reverse diffusion process.

#pagebreak()

= Binary Diffusion Models for Tabular Data Generation

Before working with images which are quite complex data and require more computational resources, we can start by working with tabular data which is simpler and more manageable, and then we can extend the approach to images.

In the paper "Tabular Data Generation using Binary Diffusion" by V. Kinakh and S. Voloshynovskiy @tabular, the authors propose a binary diffusion model for generating synthetic tabular data, which is a common type of quantized data that can be represented in a binary format.

The model is trained to learn the underlying distribution of the tabular data and can be used to generate new synthetic samples that resemble the original data, which can be useful for various applications such as data augmentation, privacy preservation, and imputation of missing values.

== Link with image generation

The process of generating tabular data is similar to the process of generating images, the main difference being the first transformation $T$ and the fact that the data are 1-dimensional instead of 2-dimensional as in the case of images.

== First transformation for tabular data

The first transformation $T$ for tabular data can be defined as a simple binary encoding of the original data, where the categorical features are one-hot encoded and the numerical features are binarized using a fixed number of bits to represent the values.

Then, all binary features are concatenated to form a single binary vector that represents the original tabular data, that can be reversed using the inverse transformation $T^(-1)$ to recover the original tabular data from the binary representation.

== Architecture of the model

As described in the paper @tabular, the sampling algorithm for generating synthetic data from the binary diffusion model can be represented as in the @algo.

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

=== Fixed thresholding step

In the paper @tabular, the authors mention that the model doesn't perform well when increasing the number of time steps $T$ in the reverse diffusion process.

This behavior is quite surprising since increasing the number of time steps $T$ should allow the model to generate higher quality samples by gradually removing the noise from the data, which is a common behavior observed in diffusion models for continuous data.

However, note that in the @algo, the binarization step uses a fixed threshold value that is applied to the predicted clean sample $hat(X)_0$ to determine the binary values of the generated sample.
This fixed thresholding step can be the root cause of the performance degradation.

For instance, if the model is in mode "mask", the predicted noise $hat(z)_t$ can have 50% of ones meaning that the model will bring 50% of changes to the data at each step of the reverse diffusion process.
This can degrade the quality of the generated samples since the model will be forced to make a large number of changes to the data at each step instead of gradually removing the noise.

Some tests have been done to try to understand the impact of the fixed thresholding step on the performance of the model

=== Evaluation of the model

To evaluate the performance of the binary diffusion model for tabular data generation, the procedure described in the paper @procedure is followed.

The evaluation procedure consists of the following steps:
+ Train the binary diffusion model on the original dataset.
+ Generate a synthetic dataset by sampling from the trained model.
+ Train a machine learning model on the synthetic dataset.
+ Evaluate the performances.

In the paper @tabular, the authors use a Random Forest classifier, a Linear/Logistic Regression classifier, and a Decision Tree classifier to evaluate the performance of the generated synthetic data on a classification and regression tasks, and they report the results in terms of prediction accuracy for classification tasks and mean squared error for regression tasks.

#pagebreak()
= Results

First, I managed to reproduce the results of the paper @tabular on the Adult dataset @adult_2.

Then, I tried to improve the performance of the model by bringing some modifications:
- Removing the fixed thresholding
- Correctly applying the guidance during the sampling process
- Fixing the noise addition step in the sampling algorithm

== Removing the fixed thresholding step

The first modification was to remove the fixed thresholding step in the sampling algorithm (mode "mask").

Instead of applying a fixed threshold to binarize the predicted clean sample $hat(X)_0$, I applied a dynamic threshold that is computed based on the quantity of noise to be removed at each step of the reverse diffusion process.

This dynamic thresholding approach allows the model to adapt the binarization process to the amount of noise that is being removed at each step, which can lead to a gradual removal of the noise and an improvement in the quality of the generated samples as we increase the number of time steps $T$ in the reverse diffusion process.

This modification doesn't improve the performance of the model.
Indeed, the model already predicts the right amount of noise to be removed at each step, as shown by the following results that compare the quantity of noise in the predicted mask with the quantity of noise that should be removed at each step of the reverse diffusion process.
```raw
Step 800: Ones in mask: 33.6364%, Ones required: 33.6031%
Step 600: Ones in mask: 20.4545%, Ones required: 20.3957%
Step 400: Ones in mask: 10.4545%, Ones required: 10.4685%
Step 200: Ones in mask: 4.0909%, Ones required: 3.8215%
Step 0: Ones in mask: 0.4332%, Ones required: 0.4545%
```
So this analysis shows that the model predicts the right amount of noise to be removed at each step, which indicates that the fixed thresholding step is not the root cause of the performance degradation when increasing the number of time steps $T$ in the reverse diffusion process.

== Correctly applying the guidance during the sampling process

The second modification was to follow the approach described in the paper "Classifier-Free Diffusion Guidance" @guidance to guide correctly the sampling process.
In fact, the guidance was not correctly applied in the current implementation since it was not following the approach proposed in the paper, even though the goal was to implement the classifier-free guidance as described in the paper @guidance.

In the paper, the approach proposed is the following:
$ (1 + w) dot epsilon_c - w dot epsilon_u $
Where $epsilon_c$ is conditioned prediction and $epsilon_u$ is unconditioned prediction, and $w$ is a scalar value that controls the strength of the guidance.

The conditioned prediction can be seen as the conformity to the desired properties, while the unconditioned prediction can be seen as the diversity of the generated samples, and the guidance allows to find a good balance between these two aspects to generate high-quality samples that match the desired properties while still being diverse and representative of the general distribution of the data.

The current implementation was doing the following:
$ epsilon_u + g dot (epsilon_c - epsilon_u) $
which is equivalent to the approach proposed in the paper @guidance but with a different parameterization of the guidance strength, where $g = 1 + w$:
$ epsilon_u + (1 + w) dot (epsilon_c - epsilon_u) = (1 + w) dot epsilon_c - w dot epsilon_u $

This modification allows to correctly apply the guidance during the sampling process, but this is only a small fix of -1 in the implementation that doesn't have a significant impact on the performance of the model.

== Fixing the noise addition step in the sampling algorithm

The third modification was a small fix in the implementation of the sampling algorithm.

As described in the @algo at line 11, the sample process adds the noise of the step $t-1$ to the predicted clean sample $hat(X)_0$ to get the new sample $X_t$ for the next step of the reverse diffusion process.

But in the existent implementation, the noise of the current step $t$ was added instead of the noise of the next step $t-1$, which can lead to a degradation of the performance of the model.

Indeed, the noise of the current step $t$ is not the correct noise to be added since it corresponds to the noise that was removed at the current step, while the noise of the next step $t-1$ corresponds to the noise that will be removed at the next step, which is the correct noise to be added to ensure that the model effectively reverses the noise addition process and generates high-quality samples.

By adding the correct noise at each step, the amount of noise is now correctly reduced at each step of the reverse diffusion process.
