// Main report file
#import "template.typ": create-report-template

// Configure your report
#let my-report = create-report-template(
  // Required information
  logo: "./img/unige.svg",
  logosize: 6cm,
  university: "University of Geneva",
  title: "Assignment 3",

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
  // illustrations: (
  //   (
  //     path: ".typst_py/img/b5_f1_a1.png",
  //     width: 7cm,
  //   ),
  //   (
  //     path: ".typst_py/img/b4_f1_a2.png",
  //     width: 7cm,
  //   )
  // ),
  project-name: "Computational Imaging",
  date: none,

  // Document options
  toc: true,
  numbering: true,
  bibliography: none,
  appendix: false,
)
