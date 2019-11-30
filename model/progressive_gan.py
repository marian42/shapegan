from model import *
from util import standard_normal_distribution

RESOLUTIONS = [8, 16, 32, 64]
FEATURE_COUNTS = [128, 64, 32, 1]
FINAL_LAYER_FEATURES = 256

# works like fromRGB in the Progressive GAN paper
def from_SDF(x, iteration):
    resolution = RESOLUTIONS[iteration]
    target_feature_count = FEATURE_COUNTS[iteration]
    
    x = x.reshape((-1, 1, resolution, resolution, resolution))
    batch_size = x.shape[0]
    x = torch.cat((x, torch.zeros((batch_size, target_feature_count - 1, resolution, resolution, resolution), device=x.device)), dim=1)
    return x

class Discriminator(SavableModule):
    def __init__(self):
        self.iteration = 0
        self.filename_base="hybrid_progressive_gan_discriminator_{:d}.to"
        super(Discriminator, self).__init__(filename=self.filename_base.format(self.iteration))

        self.fade_in_progress = 1

        self.head = nn.Sequential(
            Lambda(lambda x: x.reshape(-1, 64*FINAL_LAYER_FEATURES)),
            nn.Linear(64*FINAL_LAYER_FEATURES, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 1)
        )

        self.optional_layers = nn.ModuleList()
        for i in range(len(FEATURE_COUNTS)):
            in_channels = FEATURE_COUNTS[i]
            out_channels = FEATURE_COUNTS[i-1] if i > 0 else FINAL_LAYER_FEATURES
            submodule = nn.Sequential(
                nn.Conv3d(in_channels = in_channels, out_channels = out_channels, kernel_size = 4, stride = 2, padding = 1),
                nn.LeakyReLU(negative_slope=0.2)
            )
            self.optional_layers.append(submodule)
            self.add_module('optional_layer_{:d}'.format(i), submodule)

    def forward(self, x):
        x_in = x
        x = from_SDF(x, self.iteration)
        x = self.optional_layers[self.iteration](x)
        if (self.fade_in_progress < 1.0) and self.iteration > 0:
            x2 = from_SDF(x_in[:, ::2, ::2, ::2], self.iteration - 1)
            x = self.fade_in_progress * x + (1.0 - self.fade_in_progress) * x2

        i = self.iteration - 1
        while i >= 0:
            x = self.optional_layers[i](x)
            i -= 1
            
        return self.head(x).squeeze()

    def set_iteration(self, value):
        self.iteration = value
        self.filename = self.filename_base.format(self.iteration)
