# Generative Adversarial Networks and Autoencoders for 3D Shapes

This is the repository for my master thesis about generative models for 3D shapes.
It contains:

- an autoencoder, variational autoencoder and GANs for SDF voxel volumes using [3D CNNs](http://papers.nips.cc/paper/6096-learning-a-probabilistic-latent-space-of-object-shapes-via-3d-generative-adversarial-modeling.pdf)
- an implementation of [the DeepSDF paper](https://arxiv.org/pdf/1901.05103.pdf) that learns implicit function representations of 3D shapes
- a GAN that uses a DeepSDF network as the generator and a 3D CNN as the discriminator ("Hybrid GAN")
- a data prepration pipeline that can calculate signed distance fields for non-watertight meshes at arbitrary sample points
- a [ray marching](http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/) renderer to render signed distance fields represented by a neural network, as well as a classic rasterized renderer to render meshes reconstructed with Marching Cubes
- tools to visualize the results

This project uses two different ways to represent 3D shapes.
These representations are voxel volumes and implicit functions.
Both use [signed distances](https://en.wikipedia.org/wiki/Signed_distance_function).

For both representations, there are networks that learn embeddings and reconstruct objects from latent codes.
These are the *autoencoder* and *variational autoencoder* for voxel volumes and the [*autodecoder* for the DeepSDF network](https://arxiv.org/pdf/1901.05103.pdf).

In addition, for both representations, there are *generative adversarial networks* that learn to generate novel objects from random latent codes.
All GANs come in a classic and a Wasserstein flavor.

## Data preparation

This process requires ~270 GB of disk space and takes ~40 hours.
If you're interested in using the data preparation pipeline for something else, have a look at the file `demo_data_preparation.py`, which demonstrates some data preparation tasks with an example mesh that is provided with this repository.

1. Download and unpack the [ShapeNet Core v2](https://shapenet.org/) dataset
2. Move the dataset into the `data/shapenet/` directory relative to the project root or create a symbolic link.
For example, this path should exist:

```data/shapenet/03001627/6ae8076b0f9c74199c2009e4fd70d135/models/model_normalized.obj```

3. Run the `python3 prepare_sdf_data.py`.
This is the part that takes long.
You can stop the process at any time and resume by running the script again.
This creates files with SDF data for each model inside the `data/shapenet` directory.
4. Run

```python3 dataset.py init prepare_voxels prepare_sdf```
    
You can omit `prepare_voxels` or `prepare_sdf` if you're only interested in voxels or implicit representations.

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