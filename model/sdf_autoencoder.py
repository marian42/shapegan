from model import *

from model.sdf_net import SDFNet
from model.sdf_gan import SDFDiscriminator

class SDFAutoencoder(SavableModule):
    def __init__(self):
        super(SDFAutoencoder, self).__init__(filename='sdf_autoencoder.to')

        self.encoder = SDFDiscriminator(output_size=LATENT_CODE_SIZE)
        self.decoder = SDFNet()

    def encode(self, points, sdf):
        return(self.encoder.forward(points, sdf))

    def decode(self, latent_code, points):
        z = latent_code.repeat(points.shape[0], 1)
        distances = self.decoder.forward(points, z)
        return distances

    def forward(self, points, sdf):
        z = self.encode(points, sdf)
        return self.decode(z, points)
