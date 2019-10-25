from model import *
from inception_score import inception_score
from util import standard_normal_distribution

def test(x):
    #print(x.shape)
    return x

class Discriminator(SavableModule):
    def __init__(self):
        super(Discriminator, self).__init__(filename="discriminator.to")

        self.layers = nn.Sequential(
            nn.Conv3d(in_channels = 1, out_channels = 8, kernel_size = 1),
            nn.LeakyReLU(negative_slope=0.2),

            Lambda(test),
            nn.Conv3d(in_channels = 8, out_channels = 16, kernel_size = 3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            Lambda(test),
            nn.Conv3d(in_channels = 16, out_channels = 32, kernel_size = 3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            Lambda(test),
            nn.AvgPool3d(2),

            nn.Conv3d(in_channels = 32, out_channels = 32, kernel_size = 3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            Lambda(test),
            nn.Conv3d(in_channels = 32, out_channels = 32, kernel_size = 3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            Lambda(test),
            nn.AvgPool3d(2),

            Lambda(test),
            Lambda(lambda x: x.reshape(x.shape[0], -1)),
            Lambda(test),
            nn.Linear(256, 1),
            Lambda(test),
            nn.Sigmoid()
        )

        self.cuda()

    def forward(self, x):
        if (len(x.shape) < 5):
            x = x.unsqueeze(dim = 1) # add dimension for channels
            
        return self.layers.forward(x).squeeze()

    def clip_weights(self, value):
        for parameter in self.parameters():
            parameter.data.clamp_(-value, value)
