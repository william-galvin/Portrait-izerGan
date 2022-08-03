# Portrait-izer GAN

A GAN that transfers stylistic elements from oil-painted portraits onto photographs. 

## Motivation
I made this project as a way to begin to explore generative adversarial networks (GANs), which I recognize as one the most visually impressive types of neural networks, but which I had no prior experience in.

## Acknowledgements

The datasets used in this project can be found here:
- [Human Faces](https://www.kaggle.com/datasets/ashwingupta3012/human-faces)
- [Portraits](https://www.kaggle.com/datasets/karnikakapoor/art-portraits)

The architecture for the generator half of the GAN is the *pix2pix* U-net proposed in [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004), and is pre-trained with ResNet18.

The architecture for the discriminator is taken from [this tutorial](https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8) by Moein Shariatnia.

Some of the other particularly helpful resources and tutorials I used:
- [Image Colorization with Convolutional Neural Networks](https://lukemelas.github.io/image-colorization.html) from Luke Melas-Kyriazi
- [GAN — CycleGAN (Playing magic with pictures)](https://jonathan-hui.medium.com/gan-cyclegan-6a50e7600d7) from Jonathan Hui
- [Make-A-Monet: Image Style Transfer With Cycle GANs](https://ucladatares.medium.com/make-a-monet-image-style-transfer-with-cycle-gans-5475dcb525b8) from DataRes at UCLA

## Examples:
![](https://github.com/william-galvin/Portrait-izerGan/blob/main/examples/example6.png?raw=true)
![](https://github.com/william-galvin/Portrait-izerGan/blob/main/examples/example2.png?raw=true)
![](https://github.com/william-galvin/Portrait-izerGan/blob/main/examples/example8.png?raw=true)
![](https://github.com/william-galvin/Portrait-izerGan/blob/main/examples/example4.png?raw=true)
![](https://github.com/william-galvin/Portrait-izerGan/blob/main/examples/example5.png?raw=true)
![](https://github.com/william-galvin/Portrait-izerGan/blob/main/examples/example1.png?raw=true)
![](https://github.com/william-galvin/Portrait-izerGan/blob/main/examples/example7.png?raw=true)
![](https://github.com/william-galvin/Portrait-izerGan/blob/main/examples/example3.png?raw=true)
![](https://github.com/william-galvin/Portrait-izerGan/blob/main/examples/example10.png?raw=true)
![](https://github.com/william-galvin/Portrait-izerGan/blob/main/examples/example9.png?raw=true)

## Comments
The goal of this project was to take stylistic elements from oil-painted portraits and transfer them onto photographs—and within that scope, it was largely successful. Characteristics of the training data, like darker and smudgier backgrounds, more severe facial features, and less depth can be seen in the examples.

However, this project is nowhere near outputting flawless, high-definition portraits on demand. Its inconsistency and limitations mean that it’s a better proof of concept and learning experience than a product.

## Areas for improvement
- The focus and cropping of the two training datasets are different—the portraits are more zoomed-out than the photos. Fixing this could improve results.
- Changing architecture to use a “CycleGAN”, as described in the tutorials linked above. Their results all seem much better than these, but it also appears to be a more complicated program to implement.
- Longer training time, more data, or a bigger computer. Each dataset that I used has less than 10,000 images, which makes it easy to work with, but is perhaps limiting. As well, I trained this on my laptop for a few hours—so certainly not a professional-grade pipeline.
