// Main report file
#import "template.typ": create-report-template

// Configure your report
#let my-report = create-report-template(
  // Required information
  logo: "./img/unige.svg",
  logosize: 6cm,
  university: "University of Geneva",
  title: "Assignment 2: Phase imaging",

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
      path: ".typst_pyimage/b29ca5c3f14469865d3ea6064891b9a93e02f8ef.png",
      width: 7cm,
    ),
  ),
  project-name: "Computational Imaging",
  date: none,

  // Document options
  toc: true,
  numbering: true,
  bibliography: none,
  appendix: false,
)
