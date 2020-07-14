# Generative Adversarial Networks and Autoencoders for 3D Shapes

![Shapes generated with our propsed GAN architecture and reconstructed using Marching Cubes](https://raw.githubusercontent.com/marian42/shapegan/master/examples/gan_shapes.png)

This repository provides code for the paper "[Adversarial Generation of Continuous Implicit Shape
Representations](https://arxiv.org/abs/2002.00349)" and for my master thesis about generative machine learning models for 3D shapes.
It contains:

- the networks proposed in the paper (GANs with a DeepSDF network as the generator and a 3D CNN or Pointnet as discriminator)
- an autoencoder, variational autoencoder and GANs for SDF voxel volumes using [3D CNNs](http://papers.nips.cc/paper/6096-learning-a-probabilistic-latent-space-of-object-shapes-via-3d-generative-adversarial-modeling.pdf)
- an implementation of [the DeepSDF autodecoder](https://arxiv.org/pdf/1901.05103.pdf) that learns implicit function representations of 3D shapes
- a GAN that uses a DeepSDF network as the generator and a 3D CNN as the discriminator ("Hybrid GAN", as proposed in the paper, but without progressive growing and without gradient penalty)
- a data prepration pipeline that can prepare SDF datasets from triangle meshes, such as the [Shapenet dataset](https://www.shapenet.org/) (based on my [mesh_to_sdf](https://github.com/marian42/mesh_to_sdf) project)
- a [ray marching](http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/) renderer to render signed distance fields given by a neural network, as well as a classic rasterized renderer to render triangle meshes reconstructed with Marching Cubes
- tools to visualize the results

Note that although the code provided here works, most of the scripts need some configuration to work for a specific task.

This project uses two different ways to represent 3D shapes.
These representations are *voxel volumes* and *implicit functions*.
Both use [signed distances](https://en.wikipedia.org/wiki/Signed_distance_function).

For both representations, there are networks that learn latent embeddings and then reconstruct objects from latent codes.
These are the *autoencoder* and *variational autoencoder* for voxel volumes and the [*autodecoder* for the DeepSDF network](https://arxiv.org/pdf/1901.05103.pdf).

In addition, for both representations, there are *generative adversarial networks* that learn to generate novel objects from random latent codes.
The GANs come in a classic and a Wasserstein flavor.


# Reproducing the paper

This section explains how to reproduce the paper "Generative Adversarial Networks and Autoencoders for 3D Shapes".

## Data preparation

To train the model, the meshes in the Shapenet dataset need to be voxelized for the voxel-based approach and converted to SDF point clouds for the point based approach.

We provide readily prepared datasets for the Chairs, Airplanes and Sofas categories of Shapenet as a [download](https://ls7-data.cs.tu-dortmund.de/shape_net/ShapeNet_SDF.tar.gz).
The size of that dataset is 71 GB.

To prepare the data yourself, follow these steps:

1. install the `mesh_to_sdf` pip module.
2. Download the Shapenet files to the `data/shapenet/` directory or create an equivalent symlink.
3. Review the settings at the top of `prepare_shapenet_dataset.py`.
The default settings are configured for reproducing the GAN paper, so you shouldn't need to change anything.
You can change the dataset category that will be prepared, the default is the chairs category.
You can disable preparation of either the voxel or point datasets if you don't need both.
4. Run `prepare_shapenet_dataset.py`.
You can stop and resume this script and it will continue where it left off.

## Training

### Voxel-based discriminator

To train the GAN with the 3D CNN discriminator, run

    python3 train_hybrid_progressive_gan.py iteration=0
    python3 train_hybrid_progressive_gan.py iteration=1
    python3 train_hybrid_progressive_gan.py iteration=2
    python3 train_hybrid_progressive_gan.py iteration=3

This runs the four steps of progressive growing.
Each iteration will start with the result of the previous iteration or the most recent result of the current iteration if the "continue" parameter is supplied.
Add the `nogui` parameter to disable the model viewer during training.
This parameter should be used when the script is run remotely.

### Point-based discriminator

TODO

Note that the pointnet-based approach currently has a separate implementation of the generator and doesn't work with the visualization scripts provided here.
The two implementations will be merged soon so that the demos work.

## Use pretrained generator models

In the `examples` directory, you find network parameters for the GAN generators trained on chairs, airplanes and sofas with the 3D CNN discriminator.
You can use these by loading the generator from these files, i.e. in `demo_sdf_net.py` you can change `sdf_net.filename` accordingly.

TODO: Examples for the pointnet-based GANs will be added soon.

# Running other 3D deep learning models

## Data preparation

Two data preparation scripts are available, `prepare_shapenet_dataset.py` is configured to work specifically with the Shapenet dataset.
`prepare_data.py` can be used with any folder of 3D meshes.
Both need to be configured depending on what data you want to prepare.
Most of the time, not all types of data need to be prepared.
For the DeepSDF network, you need SDF clouds.
For the remaining networks, you need voxels of resolution 32.
The "uniform" and "surface" datasets, as well as the voxels of other resolutions are only needed for the GAN paper (see the section above).

## Training

Run any of the scripts that start with `train_` to train the networks.
The `train_autoencoder.py` trains the variational autoencoder, unless the `classic` argument is supplied.
All training scripts take these command line arguments:
- `continue` to load existing parameters
- `nogui`  to not show the model viewer, which is useful for VMs
- `show_slice` to show a text representation of the learned shape

Progress is saved after each epoch.
There is no stopping criterion.
The longer you train, the better the result.
You should have at least 8GB of GPU RAM available.
Use a datacenter GPU, training on a desktop GPU will take several days to get good results.
The classifiers take the least time to train and the GANs take the most time.

## Visualization

To visualize the results, run any of the scripts starting with `demo_`.
They might need to be configured depending on the model that was trained and the visualizations needed.
The `create_plot.py` contains code to generate figures for my thesis.

## Using the pretrained DeepSDF model and recreating the latent space traversal animation

This section explains how get a DeepSDF network model that was pre-trained on the Shapenet dataset and how to use it to recreate [this latent space traversal animation](https://twitter.com/marian42_/status/1188969971898048512).

Since the model was trained, some network parameters have changed.
If you're training a new model, you can use the parameters on the master branch and it will work as well.
To be compatible with the pretrained model, you'll need the changes in the [`pretrained-deepsdf-shapenet`](https://github.com/marian42/shapegan/tree/pretrained-deepsdf-shapenet) branch.

To generate the latent space animation, follow these steps:

1. Switch to the the [`pretrained-deepsdf-shapenet`](https://github.com/marian42/shapegan/tree/pretrained-deepsdf-shapenet) branch.

2. Move the contents of the `examples/deepsdf-shapenet-pretrained` directory to the project root directory.
The scripts will look for the .to files in `/models` and `/data` relative to the project root.

3. Run `python3 demo_latent_space.py`.
This takes about 40 minutes on my machine.
To make it faster, you can lower the values of `SAMPLE_COUNT` and `TRANSITION_FRAMES` in `demo_latent_space.py`.

4. To render a video file from the frames, run `ffmpeg -framerate 30 -i images/frame-%05d.png -c:v libx264 -profile:v high -crf 19 -pix_fmt yuv420p video.mp4`.

Note that after completing steps 1 and 2, you can run `python3 demo_sdf_net.py` to show a realtime latent space interpolation.