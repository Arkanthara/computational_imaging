// Main report file
#import "template.typ": make-report, report-footnote
#import "metadata.typ": my-report
#import "@preview/theofig:0.1.0": definition
#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node
#import "@preview/cetz:0.4.2": canvas, draw
// #import "graph_utils.typ": *
#import "neural-viz/lib.typ": *
#import emoji: *

// Main content
#show: make-report.with(my-report)

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
    image-node("x0", title: $X_0$, src: src_dir + "/img/tangled.png", cover: true, image-size: (2.5, 3)),
    image-node("x1", title: $X_1$, src: src_dir + "/img/noisy_image_0.3.png", cover: true, image-size: (2.5, 3)),
    image-node("x2", title: $X_2$, src: src_dir + "/img/noisy_image_0.5.png", cover: true, image-size: (2.5, 3)),
    text-node("dots", " ... ", title-size: 1em, node-size: (2, 3)),
    image-node("xT", title: $X_T$, src: src_dir + "/img/noisy_image_10.0.png", cover: true, image-size: (2.5, 3)),
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
    image-node("x0", title: $X_0$, src: src_dir + "/img/tangled.png", cover: true, image-size: (2.5, 3)),
    image-node("x1", title: $X_1$, src: src_dir + "/img/noisy_image_0.3.png", cover: true, image-size: (2.5, 3), pos: right-of("x0")),
    image-node("x2", title: $X_2$, src: src_dir + "/img/noisy_image_0.5.png", cover: true, image-size: (2.5, 3), pos: right-of("x1")),
    text-node("dots", " ... ", title-size: 1em, pos: right-of("x2"), node-size: (2, 3)),
    image-node("xT", title: $X_T$, src: src_dir + "/img/noisy_image_10.0.png", pos: right-of("dots"), cover: true, image-size: (2.5, 3)),
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
    image-node("x0", title: $X_0$, src: src_dir + "/img/tangled.png", cover: true, image-size: (2.5, 3)),
    image-node("x1", title: $X_1$, src: src_dir + "/img/noisy_image_0.3.png", cover: true, image-size: (2.5, 3), pos: right-of("x0")),
    image-node("x2", title: $X_2$, src: src_dir + "/img/noisy_image_0.5.png", cover: true, image-size: (2.5, 3), pos: right-of("x1")),
    text-node("dots", " ... ", title-size: 1em, pos: right-of("x2"), node-size: (2, 3)),
    image-node("xT", title: $X_T$, src: src_dir + "/img/noisy_image_10.0.png", pos: right-of("dots"), cover: true, image-size: (2.5, 3)),
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