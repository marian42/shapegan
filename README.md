# Generative Adversarial Networks and Autoencoders for 3D Shapes

This repository contains code for the paper "[Adversarial Generation of Continuous Implicit Shape
Representations](TODO)" my master thesis about generative models for 3D shapes.
It contains:

- the networks proposed in the paper (GANs with a DeepSDF network as the generator and a 3D CNN or Pointnet as dicriminator)
- an autoencoder, variational autoencoder and GANs for SDF voxel volumes using [3D CNNs](http://papers.nips.cc/paper/6096-learning-a-probabilistic-latent-space-of-object-shapes-via-3d-generative-adversarial-modeling.pdf)
- an implementation of [the DeepSDF autodecoder](https://arxiv.org/pdf/1901.05103.pdf) that learns implicit function representations of 3D shapes
- a GAN that uses a DeepSDF network as the generator and a 3D CNN as the discriminator ("Hybrid GAN", as proposed in the paper, but without progressive growing and without gradient penalty)
- a data prepration pipeline that can prepare SDF datasets from triangle meshes, such as the [Shapenet dataset](https://www.shapenet.org/) (based on my [mesh_to_sdf](https://github.com/marian42/mesh_to_sdf) project)
- a [ray marching](http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/) renderer to render signed distance fields represented by a neural network, as well as a classic rasterized renderer to render meshes reconstructed with Marching Cubes
- tools to visualize the results

Note that although the code provided here works, most of the scripts need some configuration to work for a specific task.

This project uses two different ways to represent 3D shapes.
These representations are voxel volumes and implicit functions.
Both use [signed distances](https://en.wikipedia.org/wiki/Signed_distance_function).

For both representations, there are networks that learn embeddings and reconstruct objects from latent codes.
These are the *autoencoder* and *variational autoencoder* for voxel volumes and the [*autodecoder* for the DeepSDF network](https://arxiv.org/pdf/1901.05103.pdf).

In addition, for both representations, there are *generative adversarial networks* that learn to generate novel objects from random latent codes.
All GANs come in a classic and a Wasserstein flavor.

## Data preparation

TODO

## Training

Run any of the scripts that start with `train_` to train the networks.
The `train_autoencoder.py` trains the variational autoencoder, unless the `classic` argument is supplied.
All training scripts take these command line arumgnets:
- `continue` to load existing parameters
- `nogui`  to not show the model viewer, which is useful for VMs
- `show_slice` to show a text representation of the learned shape

Progress is saved after each epoch.
There is no stopping criterion.
The longer you train, the better the result.
You should have at least 8GB of GPU RAM available.
Use a datacenter GPU, training on a desktop GPU will take several days to get good results.
The classifiers take the least time to train and the GANs take the most time.