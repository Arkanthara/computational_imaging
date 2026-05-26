
#show figure.where(kind: "subfigure"): set figure(supplement: "Figure")

#show figure.where(kind: image): outer => {
  counter(figure.where(kind: "subfigure")).update(0)
  set figure(numbering: (..nums) => {
    let outer-nums = counter(figure.where(kind: image)).at(outer.location())
    std.numbering("1a", ..outer-nums, ..nums)
  })
  show figure.where(kind: "subfigure"): inner => {
    show figure.caption: it => context {
      std.numbering("(a)", it.counter.at(inner.location()).last())
      [ ]
      it.body
    }
    inner
  }
  outer
}
// Main report file

#import "@preview/touying:0.7.3": *
#import themes.university: *
#import themes.stargazer: *
#import "@preview/chronos:0.3.0": *
#import "@preview/pintorita:0.1.4"

#import "@preview/theorion:0.6.0": *
#import cosmos.clouds: *
#show: show-theorion

#show raw.where(lang: "pintora"): it => pintorita.render(it.text)

#import "@preview/numbly:0.1.0": numbly

// #context pdfpc.pdfpc-file(here())

#show: stargazer-theme.with(
  aspect-ratio: "16-9",
  // Fix header logo: box it with explicit height and vertical alignment  
  header-right: self => {  
    box(utils.display-current-heading(level: 1))  
    h(.2em)  
    box(height: 1.5em, baseline: 30%, image("img/unige.svg", height: 1.5em))  
  },  
  config-info(
    title: [Binary Diffusion Probabilistic Models],
    subtitle: [],
    author: [Michel Jean Joseph Donnet],
    date: datetime.today(),
    institution: [Faculty of Science, University of Geneva],
    contact: [],
    logo: image("./img/unige.svg", height: 2em),
  ),
    // transparence du décor de fond
  alpha: 12%,

  // barre de progression discrète
  progress-bar: true,

  // Palette personnalisée
  config-colors(
    // Couleur institutionnelle adoucie
    // primary: rgb("#CF0063"),
    primary: rgb("#0A7A66"),

    // Version sombre pour slides focus / contrastes
    // primary-dark: rgb("#7A0042"),
    primary-dark: rgb("#065345"),

    // Couleur du texte sur fonds colorés
    secondary: rgb("#FFFFFF"),

    // Couleur faculté (vert sarcelle)
    tertiary: rgb("#0A7A66"),

    // Fond général légèrement cassé
    neutral-lightest: rgb("#FAFAFA"),

    // Texte principal
    // neutral-darkest: rgb("#7A0042"),
    neutral-darkest: rgb("#032e26"),
  ),
)


// #set heading(numbering: numbly("{1}.", default: "1.1"))

#show raw.where(block: true): set block(fill: luma(240), inset: 1em, radius: 0.5em, width: 100%)
// #show raw.where(block: false): set block(fill: luma(240), inset: 1em, radius: 0.5em, width: 100%)
// #show raw.where(block: false): box.with(
//   fill: rgb("#e573e927"),
//   inset: (x: 3pt, y: 0pt),
//   outset: (y: 3pt),
//   radius: 2pt,
// )

// #import "graph_utils.typ": *
#import "neural-viz/lib.typ": *
#import emoji: camera

#import "@preview/algorithmic:1.0.7"
#import algorithmic: style-algorithm, algorithm-figure

#show: style-algorithm

#show raw.where(block: true): set block(fill: luma(240), inset: 1em, radius: 0.5em, width: 100%)


#title-slide()

#outline-slide()

```python
%| refresh: true
%| echo: false
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *
```

= Introduction on Diffusion Models

#let src_dir = "../../.."

#figure(
  {
    let nodes = (
      image-node(0, src: src_dir + "/img/tangled_2.png", cover: true, title: "", image-size: (5, 6), pos: explicit-pos(0, y: 0)),
      image-node(1, src: src_dir + "/img/tangled.png", cover: true, title: "", image-size: (5, 6), pos: explicit-pos(-1, y: 0.5)),
      image-node(2, src: src_dir + "/img/tangled_3.png", cover: true, title: "", image-size: (5, 6), pos: explicit-pos(1.3, y: -0.5)),
      image-node(3, src: src_dir + "/img/tangled_4.png", cover: true, title: "", image-size: (8, 6), pos: explicit-pos(-1, y: -0.6)),
      image-node(4, src: src_dir + "/img/tangled_5.png", cover: true, title: "", image-size: (5, 6), pos: explicit-pos(1.5, y: 0.6)),
      image-node(5, src: src_dir + "/img/tangled_6.png", cover: true, title: "", image-size: (5, 6), pos: explicit-pos(0, y: 1)),
      text-node(6, "?", size: 5em, pos: explicit-pos(-2, y: 0)),
      text-node(7, "?", size: 5em, pos: explicit-pos(4, y: 0)),
      text-node(8, "?", size: 5em, pos: explicit-pos(3, y: 1)),
      text-node(9, "?", size: 5em, pos: explicit-pos(-1.7, y: 1)),

    )
    let edges = (
      ml-edge("0", "1"),
      ml-edge("0", "2"),
      ml-edge("0", "3"),
      ml-edge("0", "4"),
      ml-edge("0", "5"),
    )
    ml-diagram(nodes, edges: edges, spacing: 1em, label-size: 0.7em)
  }
)

#pdfpc.speaker-note(
  "Diffusion models:
  - Data augmentation: create new samples (e.g., medical imaging, where data is limited)
  - Data augmentation: DB balancing
  - Image generation: generate new images from noise
  - Image restoration: remove noise from images
  - Image restoration: super-resolution/restore details
  - Computational imaging: reconstruct images from measurements (e.g., MRI reconstruction, CT reconstruction -> less radiation, faster acquisition)
  - Image anonymization: generate synthetic data that preserves statistical properties of the original data while removing personally identifiable information (e.g., in medical imaging, to protect patient privacy)
  - etc."
)

#let src_dir = "../../.."

#figure(
  {
    let nodes = (
      image-node(0, src: src_dir + "/img/tangled_low_quality_10.png", cover: true, title: "", image-size: (12, 14)),
      image-node(1, src: src_dir + "/img/tangled_2.png", cover: true, title: "", image-size: (12, 14), pos: right-of("0", by: 8)),

    )
    let edges = (
      ml-edge("0", "1", label: "?"),

    )
    ml-diagram(nodes, edges: edges, spacing: 1em, label-size: 4em)
  }
)

#pdfpc.speaker-note(
  "Diffusion models:
  - Data augmentation: create new samples (e.g., medical imaging, where data is limited)
  - Data augmentation: DB balancing
  - Image generation: generate new images from noise
  - Image restoration: remove noise from images
  - Image restoration: super-resolution/restore details
  - Computational imaging: reconstruct images from measurements (e.g., MRI reconstruction, CT reconstruction -> less radiation, faster acquisition)
  - Image anonymization: generate synthetic data that preserves statistical properties of the original data while removing personally identifiable information (e.g., in medical imaging, to protect patient privacy)
  - etc."
)

= Methodology

== Markov Chain Based Generative Models

=== Forward Diffusion Process

#let src_dir = "../../.."

#let make-image-node = set-image-node-defaults(options: (image-pad: 0, image-width: 3.5, image-height: 4, border: false, title-size: 1em))

#figure(
  {
  let size = (4, 6)
  let nodes = (
    image-node("x0", title: $X_0$, src: src_dir + "/img/noisy_images/tangled.png", cover: true, image-size: size),
    image-node("x1", title: $X_1$, src: src_dir + "/img/noisy_images/noisy_image_0.3.png", cover: true, image-size: size),
    image-node("x2", title: $X_2$, src: src_dir + "/img/noisy_images/noisy_image_0.5.png", cover: true, image-size: size),
    text-node("dots", " ... ", title-size: 1em, node-size: (2, 3)),
    image-node("xT", title: $X_T$, src: src_dir + "/img/noisy_images/noisy_image_10.0.png", cover: true, image-size: size),
  )

  let edges = (
    ml-edge("x0", "x1", label: "Adding noise", label-sep: 0.2em),
    ml-edge("x1", "x2"),
    ml-edge("x2", "dots"),
    ml-edge("dots", "xT"),
  )

  ml-diagram(nodes, edges: edges, spacing: 4.5em, label-size: 0.7em)
  },
)

$ X_t = sqrt(1-beta_t) X_(t-1) + sqrt(beta_t) epsilon_t = #box(stroke: 1pt, inset: 10pt, baseline: 30%)[$sqrt(overline(alpha)_t) X_0 + sqrt(1 - overline(alpha)_t) epsilon$]^(text("  ")#text(size: 0.8em)[@diffusion]) $
$ q(X_t | X_(t-1)) = cal(N)(X_t; sqrt(1 - beta_t) X_(t-1), beta_t I) $

#pdfpc.speaker-note(
  "
There are several types of generative models, including:
- Generative Adversarial Networks (GANs)
- Variational Autoencoders (VAEs)
- Normalizing Flows
- Diffusion Models

Note:
- Y = sqrt(beta_t) epsilon_t -> Var(Y) = beta_t Var(epsilon_t)
- q(X_T | X_0) = product_(t=1)^(T) q(X_t | X_(t-1))
- overline(alpha)_t = product_(i=1)^(t) alpha_i with alpha_t = 1 - beta_t
")

=== Reverse Diffusion Process

#let src_dir = "../../.."

#let make-image-node = set-image-node-defaults(options: (image-pad: 0, image-width: 3.5, image-height: 4, border: false, title-size: 1em))

#let src_dir = "../../.."

#let make-image-node = set-image-node-defaults(options: (image-pad: 0, image-width: 3.5, image-height: 4, border: false, title-size: 1em))

#figure(
  {
  let size = (4, 6)
  let nodes = (
    image-node("x0", title: $X_0$, src: src_dir + "/img/noisy_images/tangled.png", cover: true, image-size: size),
    image-node("x1", title: $X_1$, src: src_dir + "/img/noisy_images/noisy_image_0.3.png", cover: true, image-size: size, pos: right-of("x0")),
    image-node("x2", title: $X_2$, src: src_dir + "/img/noisy_images/noisy_image_0.5.png", cover: true, image-size: size, pos: right-of("x1")),
    text-node("dots", " ... ", title-size: 1em, pos: right-of("x2"), node-size: size),
    image-node("xT", title: $X_T$, src: src_dir + "/img/noisy_images/noisy_image_10.0.png", pos: right-of("dots"), cover: true, image-size: size),
  )

  let edges = (
    ml-edge("xT", "dots", label: "Removing noise", label-sep: 0.2em),
    ml-edge("dots", "x2"),
    ml-edge("x2", "x1"),
    ml-edge("x1", "x0"),

  )

  ml-diagram(nodes, edges: edges, label-size: 0.6em, spacing: 4.5em)
  },
)

$ p_theta (X_(t-1) | X_t) = cal(N)(X_(t-1); mu_theta (X_t, t), Sigma_theta (X_t, t))  approx q(X_(t-1) | X_t) $

#pdfpc.speaker-note(
  "
  - $ mu_theta (X_t, t) = 1/sqrt(alpha_t)(X_t - (1 - alpha_t) / sqrt(1 - overline(alpha)_t) epsilon_theta (X_t, t)) $
  - $ Sigma_theta (X_t, t) = beta_t I $: how much noise to remove
  "
)

#pagebreak()

=== Intuitive interpretation

#figure(
  {
    let nodes = (
      image-node("0", title: $mu_theta (X_t, t)$, title-size: 1.2em, title-gap: 0.5em,src: src_dir + "/img/direction.jpg", cover: true, image-size: (10, 8)),
      image-node("1", title: $Sigma_theta (X_t, t)$, title-size: 1.2em, title-gap: 0.5em, src: src_dir + "/img/choices.webp", cover: true, image-size: (9, 10), pos: right-of("0", by: 5)),
      image-node("2", title: "", src: src_dir + "/img/climbing.webp", cover: true, image-size: (10., 10), pos: right-of("1", by: 5, dy: 0.1)),
      image-node("3", title: $X_0$, title-size: 1.2em, title-gap: 0.5em, caption-pos: "top", src: src_dir + "/img/tangled_2.png", cover: true, image-size: (4, 6), pos: right-of("1", by: 5.1, dy: -1.)),
    )
      let edges = (
        ml-edge("0", "1"),
        ml-edge("1", "2"),
      )
      ml-diagram(nodes, edges: edges, spacing: 1em, label-size: 3em)
  }
)

#pagebreak()

=== Complete process

#let src_dir = "../../.."

#let make-image-node = set-image-node-defaults(options: (image-pad: 0, image-width: 3.5, image-height: 4, border: false, title-size: 1em))

#let src_dir = "../../.."

#let make-image-node = set-image-node-defaults(options: (image-pad: 0, image-width: 3.5, image-height: 4, border: false, title-size: 1em))

#figure(
  {
  let size = (4, 6)
  let nodes = (
    image-node("x0", title: $X_0$, src: src_dir + "/img/noisy_images/tangled.png", cover: true, image-size: size),
    image-node("x1", title: $X_1$, src: src_dir + "/img/noisy_images/noisy_image_0.3.png", cover: true, image-size: size, pos: right-of("x0")),
    image-node("x2", title: $X_2$, src: src_dir + "/img/noisy_images/noisy_image_0.5.png", cover: true, image-size: size, pos: right-of("x1")),
    text-node("dots", " ... ", title-size: 1em, pos: right-of("x2"), node-size: size),
    image-node("xT", title: $X_T$, src: src_dir + "/img/noisy_images/noisy_image_10.0.png", pos: right-of("dots"), cover: true, image-size: size),
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
)

#pagebreak()

= Binary Diffusion Models

== Training the model

#figure(
  {
  let nodes = (
    dataset(0, title: $t_0$, size: (2, 3)),
    module(1, title: $T$, pos: right-of("0"), size: (2, 2)),
    vector(2, title: $X_0$, pos: right-of("1"), dir: "v", stack: 3, size: (1.5, 3)),
    gate-node(3, pos: right-of("2")),
    vector(6, title: $X_t$, pos: right-of("3"), dir: "v", stack: 3, size: (1.5, 3)),
    module(7, title: $p_theta (hat(X)_0_t, hat(z)_t | X_t, t, Y_epsilon)$, pos: right-of("6"), size: (9, 2)),
    vector(16, title: $hat(X)_0_t$, pos: right-of("7"), dir: "v", stack: 3, size: (1.5, 3)),
    module(11, title: $T^(-1)$, pos: right-of("16"), size: (2, 2)),
    dataset(13, title: $hat(t)_0_t$, pos: right-of("11"), size: (2, 3)),

    compare-node(12, title: $cal(L)_z$, size: (2, 2.5), pos: below("11", by: 1.15)),

    text-node(4, $bold(z)_B$, pos: below("0", by: 1.3)),
    module(5, title: $cal(M)_t$, pos: right-of("4"), size: (2, 2)),
    vector(14, title: $z_t$, pos: right-of("5"), dir: "v", stack: 3, size: (1.5, 3)),

    compare-node(15, title: $cal(L)_x$, size: (2, 2.5), pos: above("11", by: 1.15)),

    dataset(8, title: $Y$, pos: below("4", by: 1.05), size: (2, 3)),
    module(9, title: $epsilon_y$, pos: right-of("8"), size: (2, 2)),
    vector(10, title: $Y_epsilon$, pos: right-of("9"), dir: "v", stack: 3, size: (1.5, 3)),

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
) 

#pdfpc.speaker-note(
  "
  Why binary diffusion models?
  - Manage binary data (e.g., bit-planes of images, tabular data with binary features, etc...)
  - Simpler noise addition and removal process (XOR operation)
  - No need to tune a complex noise schedule as in the case of continuous data
  - Better performance when dealing with binary data due to the simplicity of the model and the noise management process
  
  Loss functions:
  - $cal(L)_z$: encourages the model to accurately predict the noise $hat(z)_t$ that was added to the data.
  - $cal(L)_x$: encourages the model to accurately predict the original binary data $hat(X)_0_t$ from the noisy binary data $X_t$.
  - Cross-entropy is a common choice for binary data as it measures how well the predicted probabilities match the true binary labels.
  - $ cal(L)_z = - sum_(i=1)^N z_(t,i) log(hat(z)_(t,i)) + (1 - z_(t,i)) log(1 - hat(z)_(t,i)) $
  - $ cal(L)_x = - sum_(i=1)^N X_(0,i) log(hat(X)_(0,i)) + (1 - X_(0,i)) log(1 - hat(X)_(0,i)) $
  - $ cal(L)(theta) = 1/B sum_(b=1)^B (cal(L)_z^((b)) + cal(L)_x^((b))) $
  "
)

#pagebreak()


=== First transformation

#figure(
  {
  let size = (4, 5)
  let nodes = (
    image-node("bitplane8", title: "Bit-plane 8", title-gap: 0.5em, src: src_dir + "/img/bit_planes/bit_plane_7.png", cover: true, image-size: size),
    image-node("bitplane7", title: "Bit-plane 7", title-gap: 0.5em, src: src_dir + "/img/bit_planes/bit_plane_6.png", cover: true, image-size: size, pos: right-of("bitplane8")),
    text-node("dots", " ... ", title-size: 1em, pos: right-of("bitplane7"), node-size: size),
    image-node("bitplane1", title: "Bit-plane 1", title-gap: 0.5em, src: src_dir + "/img/bit_planes/bit_plane_0.png", cover: true, image-size: size, pos: right-of("dots")),
    image-node("original", title: "Original image", title-gap: 0.5em, caption-pos: "top", src: src_dir + "/img/bit_planes/original.png", cover: true, image-size: size, pos: explicit-pos( 2.5, y: -4)),
  )

  let edges = (
    ml-edge("original", "bitplane8", to-side: "top"),
    ml-edge("original", "bitplane7", to-side: "top"),
    ml-edge("original", "dots", to-side: "top"),
    ml-edge("original", "bitplane1", to-side: "top"),
  )

  ml-diagram(nodes, edges: edges, label-size: 0.7em, spacing: 1em)
  },

)

#pdfpc.speaker-note(
  "
  - Binarization of a table
    - Categorical features: one-hot encoding
    - Numerical features: binarization using a fixed number of bits to represent the values (e.g., 8 bits for 256 possible values)
  - Bit-planes of an image
  - Binary encoding using a learned transformation (e.g., using an autoencoder to learn a binary representation of the data)
  - etc...
  "
)

#pagebreak()

== Sampling from the model


#figure(
  {
  let nodes = (
    text-node(4, $bold(z)_B$),
    module(5, title: $cal(M)_t$, pos: right-of("4"), size: (2, 2)),
    vector(14, title: $z_t$, pos: right-of("5"), dir: "v", stack: 3, size: (1.5, 3)),
    gate-node(3, pos: right-of("14")),
    vector(6, title: $X_t$, pos: right-of("3"), dir: "v", stack: 3, size: (1.5, 3)),
    module(7, title: $p_theta (hat(X)_0_t, hat(z)_t | X_t, t, Y_epsilon)$, pos: right-of("6"), size: (9, 2)),
    vector(16, title: $hat(X)_0_t$, pos: right-of("7"), dir: "v", stack: 3, size: (1.5, 3)),
    module(11, title: $T^(-1)$, pos: right-of("16"), size: (2, 2)),
    dataset(13, title: $hat(t)_0_t$, pos: right-of("11"), size: (2, 3)),

    dataset(8, title: $Y$, pos: below("4", by: 1.4), size: (2, 3)),
    module(9, title: $epsilon_y$, pos: right-of("8"), size: (2, 2)),
    vector(10, title: $Y_epsilon$, pos: right-of("9"), dir: "v", stack: 3, size: (1.5, 3)),

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
)

#pdfpc.speaker-note(
  "
  - Start from pure noise
  - Iteratively apply the reverse diffusion process to generate a sample that resembles the original data
  - Gradually remove the noise from the data at each step using the learned denoising model $p_theta$
  - Noise scheduler $cal(M)_t$ determines the amount of noise to be removed at each step of the reverse diffusion process
  - Conditioning variables (e.g., text description) can be used to guide the generation process to produce samples that match the given description
  - The sampling process can be computationally expensive due to the iterative nature of the reverse diffusion process
  - Requires multiple steps to generate a high-quality sample
  "
)

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

As described in the paper @tabular, the sampling algorithm for generating synthetic data from the binary diffusion model can be represented as in the .

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
    )
#pagebreak()

=== Fixed thresholding step

In the paper @tabular, the authors mention that the model doesn't perform well when increasing the number of time steps $T$ in the reverse diffusion process.

This behavior is quite surprising since increasing the number of time steps $T$ should allow the model to generate higher quality samples by gradually removing the noise from the data, which is a common behavior observed in diffusion models for continuous data.

However, note that in the , the binarization step uses a fixed threshold value that is applied to the predicted clean sample $hat(X)_0$ to determine the binary values of the generated sample.
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

The  shows the results of the evaluation of the model with the dynamic thresholding approach (linear and quadratic schedules) compared to the fixed thresholding approach (constant schedule) for different values of $T$.

We can see that the dynamic thresholding approach doesn't improve at all the performance of the model compared to the fixed thresholding approach.
And between the two dynamic thresholding approaches, the linear schedule performs worse than the quadratic schedule, which is expected since the model was trained with a quadratic noise schedule.

```python
%| echo: false
%| plt-axes.grid: false
%| img-width: 60%
%| label: fig7
fig1 = plot_heatmaps(
    value_col     = "mean",
    x_col         = "n_timesteps",
    y_col         = "schedule",
    fixed_filters = {"strategy": "mask", "use_t_next": True, "renoise_factor": 1.0},
    # subplot_cols  = ["strategy"],
    # ncols         = 2,
    cmap          = "RdYlGn",
    suptitle      = "Schedule manual vs. auto (`const`) for `strategy=mask` and `use_t_next=True`",
    cell_size      = (5, 4),
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

As described in the  at line 11, the sample process adds the noise of the step $t-1$ to the predicted clean sample $hat(X)_0$ to get the new sample $X_t$ for the next step of the reverse diffusion process.

But in the existent implementation, the noise of the current step $t$ was added instead of the noise of the next step $t-1$, which can lead to a degradation of the performance of the model.

Indeed, the noise of the current step $t$ is not the correct noise to be added since it corresponds to the noise that was removed at the current step, while the noise of the next step $t-1$ corresponds to the noise that will be removed at the next step, which is the correct noise to be added to ensure that the model effectively reverses the noise addition process and generates high-quality samples.

By adding the correct noise at each step, the amount of noise is now correctly reduced at each step of the reverse diffusion process.

However, as shown in , this modification allows to generate better samples for small values of $T$ but it doesn't allow to improve the performance of the model for larger values of $T$.

```python
%| echo: false
%| label: fig8
%| plt-axes.grid: false
%| grid-columns: (1fr, 0.92fr)
# plot_sensitivity(PARAM_KEYS, DATASETS, PLOT_CONFIG)
# plot_timestep_lines(DATASETS, PLOT_CONFIG)

fig1 = plot_heatmaps(
    value_col     = "mean",
    x_col         = "n_timesteps",
    y_col         = "use_t_next",
    fixed_filters = {"renoise_factor": 1.0, "schedule": "const"},
    subplot_cols  = ["strategy"],
    ncols         = 2,
    cmap          = "RdYlGn",
    suptitle      = "Fixed vs. non-fixed noise addition (`use_t_next`) for different strategies",
    cell_size      = (5, 4),
)
plt.show()
```

=== Increasing the noise in the sampling process

The last modification was to increase the noise in the sampling process by multiplying the predicted noise by a renoise factor that is greater than 1, which allows to add more noise at each step of the reverse diffusion process.

This modification allows to better explore the space of possible samples, which results in an improvement of the performance of the model for larger values of $T$ in the reverse diffusion process, as shown in.

```python
%| echo: false
%| label: fig9
%| plt-axes.grid: false
%| grid-columns: (1fr, 0.92fr)
# plot_sensitivity(PARAM_KEYS, DATASETS, PLOT_CONFIG)
# plot_timestep_lines(DATASETS, PLOT_CONFIG)

fig1 = plot_heatmaps(
    value_col     = "mean",
    x_col         = "n_timesteps",
    y_col         = "renoise_factor",
    fixed_filters = {"use_t_next": True, "schedule": "const"},
    subplot_cols  = ["strategy"],
    ncols         = 2,
    cmap          = "RdYlGn",
    suptitle      = "Impact of increasing the noise in the sampling process (`renoise_factor`) for `use_t_next=True`",
    cell_size      = (5, 4),
)
plt.show()
```

The shows the best configurations for each strategy compared to the original implementation of the model (baseline) for different values of $T$ in the reverse diffusion process.
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

#bibliography("bibliography.bib")