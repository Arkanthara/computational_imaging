// Main report file
#import "template.typ": make-report, report-footnote
#import "metadata.typ": my-report
#import "@preview/theofig:0.1.0": definition
#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node

// Main content
#show: make-report.with(my-report)

= Introduction

Humans have always tried to capture the present moment, both in painting and in photography.
However, the photos taken are not always of good quality,
that's why some computational methods, such as image filtering, have been developped to increase the quality of a given picture.

In this project, we will examine some basic image filtering methods and some basic way to reconstruct damaged pictures.

= Methodology <methodology>

The image filtering works in both spatial and Fourier domain.

The basics are the following:

- A filter is created to apply some operation on the image.
  For instance, a laplacian filter gives the gradient of the image.

- The filter is applied to the image in spatial or in Fourier domain.
  - In spatial domain, the convolution between the image and the filter is performed to apply the filter to each pixel of the image.
  - In Fourier domain, a simple multiplication between the image and the filter applies the filter to the image.

    #let tint(c) = (stroke: c, fill: rgb(..c.components().slice(0, 3), 5%), inset: 8pt)

    #figure(
      diagram(
        node-corner-radius: 5pt,
        node-inset: 6pt,
        spacing: 1em,

        node((0, 1), "Original image", ..tint(teal)),
        node((0, 2), "filter", ..tint(teal)),
        node((0, 0), "Spatial domain"),
        node((0, 3), "Filtered image", ..tint(purple), name: <res>),
        node((1, 2), "                                 "),
        node((2, 1), "F", ..tint(orange)),
        node((3, 1), "H", ..tint(orange)),
        node((2.5, 0), "Fourier domain"),
        node((2.5, 3), "F * H", ..tint(orange), name: <mul>),
        edge((2, 1), (2.5, 3), "-|>"),
        edge((3, 1), (2.5, 3), "-|>"),
        node(enclose: ((0, 0), (0, 4)), ..tint(teal), name: <spatial>, inset: 15pt),
        node(enclose: ((0, 1), (0, 2)), ..tint(teal), name: <original>, inset: 4pt),
        node(enclose: ((2, 1), (3, 1)), ..tint(orange), name: <transform>, inset: 5pt),
        node(
          enclose: ((2, 1), (2, 3), (3, 2), (3, 1.5), (2.5, 3), (2.5, 0)),
          ..tint(orange),
          name: <fourier>,
          inset: 10pt,
        ),

        edge(
          <original>,
          <transform>,
          "=>",
          stroke: green + .75pt,
          label: "Fourier Transform",
          label-fill: rgb(..green.components().slice(0, 3), 20%),
          label-size: 8pt,
        ),
        edge(
          <mul>,
          <res>,
          "=>",
          stroke: green + .75pt,
          label: "Inverse Fourier Transform",
          label-fill: rgb(..green.components().slice(0, 3), 20%),
          label-size: 8pt,
        ),
      ),
      caption: "Image filtering in Fourier domain",
    )

== Types of filters

In this project, we will use the following filters:
- Gaussian Blur: a low-pass filter that blurs the image by reducing the high-frequencies components. In this way, the small details (contained in high frequencies) are removed.
- Unsharp Mask: a high-pass filter that sharpens the image by enhancing the high-frequency components.
  As the low-pass filter kill the high frequencies, the unsharp mask can be obtained by substracting a blurred version of the image to the original image. Indeed, by substracting to the original image an image composed of low frequencies, only high frequencies are kept: $ "Unsharpened image" = "Image" - "Blurred image" $

== Image restoration

=== Inverse filtering

Inverse filtering is a technique used to restore an image that has been filtered by a known filter.
As the filtering  operation in Fourier domain is a simple multiplication, the inverse filter consists only to inverse the multiplication to retrieve the original image
For instance, if the image was filtered in Fourier domain by a known filter $H$, the original image can be retrieved by multiplying the filtered image by the inverse filter $1/H$. Note that in general, to avoid division by zero, a small value $K$ is added to stabilize the inverse filter: $ "Inverse filter" = 1/(H + K) $

=== Wiener filtering

The wiener filter is based on the inverse filter method. However, the wiener filter takes care about the noise present in the image. Indeed, the inverse filter amplify the noise, and the wiener filter allows to avoid the noise by adding a term dependent on the noise present in the image. The wiener filter is defined as follows: $ "Wiener filter" = 1/H dot ((|H|^2)/(|H|^2 + 1/"SNR")) $ with $"SNR" = "Average pixel value in noised image" / sigma_"noise"$.
To stabilize the wiener filter, a small value $K$ is added like for the inverse filter.

=== Gradient-based image restoration

We can consider that we have some image $b$ that was filtered by a filter $A$ (Here, we can consider that $A$ is a convolution matrix).
We want to retrieve the original image $x$ such that $A x = b$.

To do that, we can use the gradient descent method to minimize the mean squared error between $A x$ and $b$:

$ arg min_x f(x) = arg min_x 1/2||A x - b||_2^2 $.

So the gradient descent algorithm give us:

$ x^(k+1) = x^k - alpha gradient_x f(x^k) $

The gradient $gradient_x f$ can be defined as follows:

#align(
  left,
  $
    gradient_x f(x) & = gradient_x(1/2||A x - b||_2^2) \
                    & = gradient_x (1/2 (A x - b)(A x - b)^T)) \
                    & = gradient_x (1/2 (x^T A^T A x - x^T A^T b - A x b^T + b b^T )) \
                    & = gradient_x (1/2 (x^T A^T A x - 2x^T A^T b + b b^T )) \
                    & = 1/2 (2 A^T A x - 2 A^T b) \
                    & = A^T A x - A^T b
  $,
)

In our work, the numnber of iterations is used as stopping criterion for the gradient descent algorithm.

Note that as $A$ is a convolution matrix, the transpose of $A$ is the convolution with the flipped version of the filter.

==== Stochastic gradient descent

The stochastic gradient descent is based on the gradient descent method.
However, instead of computing the gradient on all the image, the stochastic gradient descent select a set of rows and columns of the image to compute the gradient on this subset of the image.
This variation is faster than the classical gradient descent thanks to the computation on a subset of the image.
However, as the gradient is not computed for the whole image, the result is less accurate than the classical gradient descent.

= Implementation <impl>

The code can be executed in command line and follows the documentation below:


#set block(fill: luma(240), inset: 1em, radius: 0.5em, width: 100%)
#raw(
  "TP1 of computational imaging

options:
  -h, --help         show this help message and exit
  -t, --task TASK    Enter the number of the task to execute
  -i, --image IMAGE  Path to the input image
",
  lang: "raw",
  block: true,
)
#set block(fill: none)

For the gradient descent and the stochastic gradient descent, to avoid allocation of hudge convolution matrix, I try to filter the image in Fourier domain. Indeed, if the image is of size $64 times 64$, it means that the flat version of the image is of size $4096$ and then the convolution matrix needed is of size $4096 times 4096 = 16777216$. If we consider that each element of the matrix is an integer of 1 octet, it means that the matrix uses $16 "Mo"$. So due to memory usage, I make another implementation:
we know that convolution gives multiplication in Fourier domain.
So instead of working with convolution matrix of the filter with the image, I use multiplication in Fourier domain. And the flipped version of the filter gives the conjugate in Fourier domain.
So for instance the gradient $gradient_x f$ becomes $G(H x - b)$ with $G$ the conjugate of the filter in Fourier domain and $H$ the filter in Fourier domain.


= Results

```python
%| echo: false
import os
import sys
os.chdir(os.path.normpath(os.path.join(os.getcwd(), "../code")))
sys.path.append(os.getcwd())
from tp1 import tasks
```

== Image filtering

```python
%| echo: false
%| caption: Low-Pass filter: Gaussian Blur
%| img-width: 80%
%| grid-align: bottom
%| label: t11
tasks(1, 1)
```

```python
%| echo: false
%| caption: High-Pass filter: Unsharp Mask
%| img-width: 140%
%| grid-align: bottom
%| label: t12
tasks(1, 2)
```

== Image restoration

```python
%| echo: false
%| caption: Original and blured image
%| img-width: 80%
%| label: t210
tasks(2, 1, original=True)
```

```python
%| echo: false
%| caption: Inverse filtering on blured image
%| img-width: 100%
%| label: t21
tasks(2, 1)
```

```python
%| echo: false
%| caption: Wiener filter on blured image
%| img-width: 140%
%| label: t22
tasks(2, 2)
```

== Gradient-based image restoration

```python
%| echo: false
%| caption: Image restoration using gradient descent in Fourier domain
%| grid-align: bottom
%| label: t31
tasks(3, 1)
```

```python
%| echo: false
%| img-width: 120%
%| caption: Image restoration using stochastic gradient descent in Fourier domain
%| grid-align: bottom
%| label: t32
tasks(3, 2)
```

```python
print("COucou")
print("Hello")
print("Coucou")
print("Haha")
print("blabla")
```


= Discussion

In the @t11, we can easily see that bigger is the Gaussian kernel, more blurred is the image.
Indeed, a Gaussian kernel of $0.1$ gives an image close to the original image as shown on @t11b and @t11c whereas a Gaussian kernel of $10$ gives a more blurred image as shown in @t11h and @t11i.
So the size of the Gaussian kernel determines the threshold to cut-off high-frequencies.

As shown in @t11, the spatial and Fourier domain gives similar results.
However, if we compare @t11h and @t11i, the filtered image in spatial domain have a kind of black blurred border.
This is due to the zero padding applied for the convolution of the image with the filter.
Indeed, a padding is required in spatial domain to allows the application of the filter on each pixel of the image because of the convolution.



= Conclusion
