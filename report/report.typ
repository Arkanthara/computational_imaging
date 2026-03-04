// Main report file
#import "template.typ": make-report, report-footnote
#import "metadata.typ": my-report
#import "@preview/theofig:0.1.0": definition
#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node
#import ".typst_pyimage/pyimage.typ": pyimage, pyinit

// Main content
#show: make-report.with(my-report)

= Introduction

A large part of the operations performed on an image in digital processing are carried out in the Fourier domain.
Working in the Fourier domain speeds up image filtering and expands the range of possibilities by providing a different representation of the image.

The Fourier transform, which enables the transition between the spatial domain and the Fourier domain, transforms real numbers into complex numbers.

According to Euler's formula, a complex number can be represented as a magnitude and a phase: $z = |M| e^(j phi)$, where $|M| in RR$ is the magnitude and $phi in [0, 2pi)$ is the phase.

In image processing, the magnitude of a given image $I$ is given by the absolute value of the Fourier transform $|F(I)|$, while the phase is given by the angle of the Fourier transform $angle F(I)$.

In this work, we will attempt to analyze the impact of the amplitude and phase of the Fourier transform by replacing and mixing the phases of two images.

= Implementation <impl>

For the implementation, the code follows the command line interface with the documentation given bellow:

#set block(fill: luma(240), inset: 10pt, radius: 10pt)
```raw
usage: tp2.py [-h] [-i1 IMAGE1] [-i2 IMAGE2] [-a ALPHA]

TP1 of computational imaging

options:
  -h, --help            show this help message and exit
  -i1, --image1 IMAGE1  Path to the first input image
  -i2, --image2 IMAGE2  Path to the second input image
  -a, --alpha ALPHA     Alpha value for phase mixing
```
#set block(fill: none)

The program returns a figure called ```raw result.jpg``` containing the two original images and the images obtained from mixing the phases.

= Results

#pyinit(
  "
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/../code')
sys.path.append(os.getcwd())
from tp2 import Mixture
mixture = Mixture('img/tangled.jpg', 'img/tangled_2.jpg')
",
)

#figure(
  pyimage("mixture.switchPhase(0)
  mixture.printMixture()"),
  caption: [Phase mixture with $phi = phi_"img_1"$],
) <fig-1>
#figure(
  pyimage("mixture.switchPhase(0.25)
  mixture.printMixture()"),
  caption: [Phase mixture with $phi = 3/4 phi_"img_1" + 1/4 phi_"img_2"$],
) <fig-2>
#figure(
  pyimage("mixture.switchPhase(1.0)
  mixture.printMixture()"),
  caption: [Phase mixture with $phi = phi_"img_2"$],
) <fig-3>


= Discussion

We can see in @fig-1 that when the phase is preserved intact, the image looks like the original.
In fact, the image obtained by reconstructing the amplitude and phase of image 1 is exactly similar to image 1.

However, if we look at image 2, we can see that when the phase of image 2 is replaced by the phase of image 1, the result looks like a damaged image 1.
In fact, even if we try to zoom in on the image, only the structure of image 1 is visible and we cannot see anything from image 2.

The case is similar in @fig-3 when the phase of image 1 is replaced by the phase of image 2: the result appears as a damaged version of image 2.

Nevertheless, we can observe that in @fig-3, the reconstructed image 1 is “brighter” than the original image 2.
This is because the magnitude of image 1 is greater than that of image 2.
Magnitude describes the amount of energy contained in the image, and the more energy an image contains, the brighter it is.
Thus, by combining the magnitude of image 1 with the phase of image 2, we obtain a brighter version of image 2 thanks to the magnitude of image 1.

So the magnitude could be responsible of the brightness and the phase could be responsible of the structure of the image.

The @fig-2 shows a mixture of the two phases: 3/4 from the phase of image 1 and 1/4 from the phase of image 2.
In both cases, changing the magnitude of the image with that of image 1 or image 2 changes approximately nothing.
Thus, only the structure of the strongest phase is visible.
This confirms that the phase is responsible for the structure of the image. 




= Conclusion