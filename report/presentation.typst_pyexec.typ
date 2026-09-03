
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

== Architecture of the model

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

        LineComment(
        Return[$X_t$],
        [Taken from "Tabular Data Generation using Binary Diffusion" @tabular]
         )
      },
    )

#pdfpc.speaker-note(
  "
  - Binarization of table
  "
)

#pagebreak()



=== Evaluation of the model

Procedure described in the paper "Language Models are Realistic Tabular Data Generators" @procedure is followed.

#tblock(title: "Steps")[
+ Train the binary diffusion model on the *original* dataset.
+ Generate a *synthetic* dataset
+ Train a machine learning model on the *synthetic* dataset.
+ Evaluate the performances on a *real test* dataset.
]

#pagebreak()
= Results

#tblock(title: "Configuration")[
- Dataset: Adult dataset @adult_2
- Model: Binary Diffusion Model for Tabular Data Generation @tabular
- Evaluation models
  - Logistic Regression
  - Random Forest
  - Decision Tree
  - Linear Regression (for regression tasks)
- Evaluation metrics: 
  - Accuracy score (for classification tasks)
  - MSE (for regression tasks)
]

== Fixed thresholding step ?

```raw
Step 800: Ones in mask: 33.6364%, Ones required: 33.6031%
Step 600: Ones in mask: 20.4545%, Ones required: 20.3957%
Step 400: Ones in mask: 10.4545%, Ones required: 10.4685%
Step 200: Ones in mask: 4.0909%, Ones required: 3.8215%
Step 0: Ones in mask: 0.4332%, Ones required: 0.4545%
```

#note-block("Model already predict right amount of noise !")

#pagebreak()

#figure(grid(columns: 2, [#figure(image(".typst_pyexec/figures/cell_2_1_1.svg"), kind: "subfigure", caption: [use_t_next=False]) <fig7-a>], [#figure(image(".typst_pyexec/figures/cell_2_1_2.svg"), kind: "subfigure", caption: [use_t_next=True]) <fig7-b>]), caption: [Schedule comparison for `strategy=mask` (baseline: `schedule=const`)], kind: image) <fig7>


== Guidance during sampling process ?

Original formula from the paper "Classifier-Free Diffusion Guidance" @guidance:
$ (1 + w) dot epsilon_c - w dot epsilon_u $

- $epsilon_c$ is conditioned prediction
- $epsilon_u$ is unconditioned prediction
- $w$ is a scalar value that controls the strength of the guidance

Actual implementation:
$ epsilon_u + g dot (epsilon_c - epsilon_u) = (1 + w) dot epsilon_c - w dot epsilon_u $
with $g = (1 + w)$
// #note-block("Small fix of -1 in the implementation that doesn't have a significant impact on the performance of the model")
#pagebreak()

== Wrong noise addition during sampling process ?

#let src_dir = "../../.."

#let make-image-node = set-image-node-defaults(options: (image-pad: 0, image-width: 3.5, image-height: 4, border: false, title-size: 1em))

#let src_dir = "../../.."

#let make-image-node = set-image-node-defaults(options: (image-pad: 0, image-width: 3.5, image-height: 4, border: false, title-size: 1em))

#figure(
  {
  let size = (3, 5)
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

  ml-diagram(nodes, edges: edges, label-size: 1em, spacing: 1.7em)
  },
)

#let src_dir = "../../.."

#let make-image-node = set-image-node-defaults(options: (image-pad: 0, image-width: 3.5, image-height: 4, border: false, title-size: 1em))

#let src_dir = "../../.."

#let make-image-node = set-image-node-defaults(options: (image-pad: 0, image-width: 3.5, image-height: 4, border: false, title-size: 1em))

#figure(
  {
  let size = (3, 5)
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

  ml-diagram(nodes, edges: edges, label-size: 1em, spacing: 1.7em)
  },
)

#pdfpc.speaker-note(
  "
  - Model trained to predict the noise z_t at step t and z_(t-1) at step t-1
  - Noise added to the predicted clean sample hat(X)_0 must be the noise that will be removed at the next step t-1, which is z_(t-1) and not z_t
  "
)

#pagebreak()


#figure(grid(columns: (1fr, 0.92fr), [#figure(image(".typst_pyexec/figures/cell_3_1_1.svg"), kind: "subfigure", caption: [strategy=mask]) <fig8-a>], [#figure(image(".typst_pyexec/figures/cell_3_1_2.svg"), kind: "subfigure", caption: [strategy=target]) <fig8-b>]), caption: [Fixed vs. non-fixed noise addition for different strategies #linebreak() (baseline: `use_t_next=False`)], kind: image) <fig8>


#figure(grid(columns: (1fr, 0.9fr), [#figure(image(".typst_pyexec/figures/cell_4_1_1.svg", width: 80%), kind: "subfigure", caption: [strategy=mask]) <fig9-a>], [#figure(image(".typst_pyexec/figures/cell_4_1_2.svg", width: 80%), kind: "subfigure", caption: [strategy=target]) <fig9-b>]), caption: [Increase the noise in the sampling process (`renoise_factor`) for `use_t_next=True` #linebreak() (baseline: `renoise_factor=1.0`)], kind: image) <fig9>


== Best configurations vs. original implementation

#figure(image(".typst_pyexec/figures/cell_5_1.svg"), caption: [Best configurations vs. original implementation (`baseline`)], kind: table) <fig10>


== What about images ?

#tblock(title: "Configuration")[
- Dataset: CIFAR-10 @cifar
- Model: Binary Diffusion Probabilistic Model @binary
- Denoising model architecture: UNet
- Evaluation metric:
  - FID score
  - FID clip score
  - Inception Score
  - KID score
- Experimental setup: RTX 4080, 16GB of VRAM, running for #text("5 days", fill: red, weight: "bold")
]

#pdfpc.speaker-note(
  "
  FID score:
  - FID: Fréchet Inception Distance
  - Measures the distance between the distribution of generated images and the distribution of real images in the feature space of a pre-trained Inception network
  - Lower FID indicates that the generated images are more similar to the real images in terms of quality and diversity

  FID clip score:
  - Similar to FID but uses features from a CLIP model instead of an Inception model
  - CLIP features capture both visual and semantic information, so FID clip score can providea more comprehensive evaluation of the generated images, especially in terms of their semantic relevance to the real images

  Inception Score:
  - Measures the quality and diversity of generated images based on the predictions of a pre-trained Inception network
  - Higher Inception Score indicates that the generated images are of higher quality and more diverse

  KID score:
  - Kernel Inception Distance
  - Similar to FID but uses a different distance metric (MMD) and a different kernel function to compare the distributions of generated and real images
  - Lower KID indicates that the generated images are more similar to the real images in terms of quality and diversity
  "
)

#pagebreak()


#tblock(title: "How to interpret the results of the evaluation")[
- IS: Higher is better
- FID: Lower is better
- FID-clip: Lower is better
- KID: Lower is better
]

== Fixing the thresholding step in the sampling algorithm

#figure(grid(columns: 2, [#figure(image(".typst_pyexec/figures/cell_6_1_1.svg"), kind: "subfigure", caption: [n_timesteps=10]) <fig11-a>], [#figure(image(".typst_pyexec/figures/cell_6_1_2.svg"), kind: "subfigure", caption: [n_timesteps=100]) <fig11-b>]), caption: [Schedule comparison for `strategy=mask` #linebreak() (baseline: `schedule=const`)], kind: image) <fig11>


== Fixing the noise addition step in the sampling algorithm

#figure(grid(columns: 2, [#figure(image(".typst_pyexec/figures/cell_7_1_1.svg"), kind: "subfigure", caption: [n_timesteps=10]) <fig12-a>], [#figure(image(".typst_pyexec/figures/cell_7_1_2.svg"), kind: "subfigure", caption: [n_timesteps=100]) <fig12-b>]), caption: [Fixed vs. non-fixed noise addition (`use_t_next`) for different metrics and strategies (baseline: `use_t_next=False`)], kind: image) <fig12>


== Increasing the noise in the sampling process

#figure(grid(columns: 2, [#figure(image(".typst_pyexec/figures/cell_8_1_1.svg"), kind: "subfigure", caption: [n_timesteps=10]) <fig13-a>], [#figure(image(".typst_pyexec/figures/cell_8_1_2.svg"), kind: "subfigure", caption: [n_timesteps=100]) <fig13-b>]), caption: [Increase the noise in the sampling process (`renoise_factor`) for `use_t_next=True` #linebreak() (baseline: `renoise_factor=1.0`)], kind: image) <fig13>




#pagebreak()

== Visual comparison of the generated images

#grid(
  columns: (1fr, 1fr),
  figure(image("img/cifar/t_10.jpg"), caption: "10 steps of reverse diffusion process"),
  figure(image("img/cifar/t_100.jpg"), caption: "100 steps of reverse diffusion process")
)


#pagebreak()


= Conclusion

Binary diffusion models are a #text("promising", fill: red, weight: "extrabold") approach for generating *synthetic quantized* data, such as *tabular* data or *images*.

=== Pros:

- Simpler noise management process
- Better performance when dealing with binary data compared to continuous models
- Less parameters to tune (e.g., noise scheduler)
- Smaller model size compared to continuous diffusion models
- Faster compared to continuous diffusion models (XOR operation)

=== Cons:

- Decreased performance when increasing number of steps (e.g., 1000 steps vs. 100 steps)

#pagebreak()

#tblock(title: "Future work")[
  - Fix decreased performance when increasing number of steps 
  - Apply binary diffusion models to other types of data (e.g., text, audio, etc...)

]

=== Thanks for your attention !
=== Acknowledgements
- #text("Vitaliy Kinakh", fill: red, weight: "bold") (the author of the two papers @tabular and @binary) for his guidance and support during this project

#pagebreak()


#bibliography("bibliography.bib")