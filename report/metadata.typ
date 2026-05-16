// Main report file
#import "template.typ": create-report-template

// Configure your report
#let my-report = create-report-template(
  // Required information
  logo: "./img/unige.svg",
  logosize: 6cm,
  university: "University of Geneva",
  title: "Binary Diffusion Probabilistic Models",

  // Structured authors
  authors: (
    (
      name: "Michel Jean Joseph Donnet",
    ),
  ),

  // Optional information
  faculty: "Faculty of Science",
  // subtitle: "Report Subtitle",
  course-name: "Computational Imaging",
  course-id: "14x062",
  illustrations: (
    (
      path: "img/tangled.png",
      width: 5cm,
    ),
    (
      path: "img/noisy_image_0.3.png",
      width: 5cm,
    )
  ),
  project-name: "Computational Imaging",
  date: none,

  // Document options
  toc: true,
  numbering: true,
  bibliography: "./bibliography.bib",
  appendix: false,
)
