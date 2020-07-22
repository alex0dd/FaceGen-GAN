# FaceGen-GAN
Conditional face generation experiments using GAN models on CelebA dataset.

## Architectures
- **Vanilla DCGAN**: a normal DCGAN as described in [DCGAN paper](https://arxiv.org/abs/1511.06434), has training stability issues.
- **Hinge DCGAN with custom layers**: an improved DCGAN with spectral normalization, self-attention, minibatch std and pixelwise normalization, which allows stable training with better visual results than DCGAN.
