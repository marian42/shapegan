from model import *
from inception_score import inception_score
from util import standard_normal_distribution

class Generator(SavableModule):
    def __init__(self):
        super(Generator, self).__init__(filename="generator.to")

        self.layers = nn.Sequential(
            nn.ConvTranspose3d(in_channels = LATENT_CODE_SIZE, out_channels = 256, kernel_size = 4, stride = 1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(negative_slope = 0.2),
            
            nn.ConvTranspose3d(in_channels = 256, out_channels = 128, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(negative_slope = 0.2),

            nn.ConvTranspose3d(in_channels = 128, out_channels = 64, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(negative_slope = 0.2),

            nn.ConvTranspose3d(in_channels = 64, out_channels = 1, kernel_size = 4, stride = 2, padding = 1),
            nn.Tanh()
        )

        self.inception_score_latent_codes = dict()
        self.cuda()

    def forward(self, x):
        return self.layers.forward(x)

    def generate(self, sample_size = 1):
        shape = torch.Size([sample_size, LATENT_CODE_SIZE, 1, 1, 1])
        x = standard_normal_distribution.sample(shape).to(self.device)
        return self.forward(x)

    def copy_autoencoder_weights(self, autoencoder):
        def copy(source, destination):
            destination.load_state_dict(source.state_dict(), strict=False)

        raise Exception("Not implemented.")        

    def get_inception_score(self, sample_size = 1000):
        with torch.no_grad():
            if sample_size not in self.inception_score_latent_codes:
                shape = torch.Size([sample_size, LATENT_CODE_SIZE, 1, 1, 1])
                self.inception_score_latent_codes[sample_size] = standard_normal_distribution.sample(shape).to(self.device)

            sample = self.forward(self.inception_score_latent_codes[sample_size])
            return inception_score(sample)


class Discriminator(SavableModule):
    def __init__(self):
        super(Discriminator, self).__init__(filename="discriminator.to")

        self.use_sigmoid = True
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels = 1, out_channels = 64, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Conv3d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Conv3d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Conv3d(in_channels = 256, out_channels = 1, kernel_size = 4, stride = 1),
            Lambda(lambda x: torch.sigmoid(x) if self.use_sigmoid else x)
        )

        self.cuda()

    def forward(self, x):
        if (len(x.shape) < 5):
            x = x.unsqueeze(dim = 1) # add dimension for channels
            
        return self.layers.forward(x).squeeze()

    def clip_weights(self, value):
        for parameter in self.parameters():
            parameter.data.clamp_(-value, value)
