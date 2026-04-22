// Main report file
#import "template.typ": create-report-template

// Configure your report
#let my-report = create-report-template(
  // Required information
  logo: "./img/unige.svg",
  logosize: 6cm,
  university: "University of Geneva",
  title: "Assignment 4",

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
      path: ".typst_pyexec/figures/cell_5_1_1.svg",
      width: 7cm,
    ),
    (
      path: ".typst_pyexec/figures/cell_5_1_2.svg",
      width: 7cm,
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
