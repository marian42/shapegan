from model import *

class Classifier(SavableModule):
    def __init__(self, label_count):
        super(Classifier, self).__init__(filename="classifier.to")
        
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels = 1, out_channels = 12, kernel_size = 5),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(in_channels = 12, out_channels = 16, kernel_size = 5),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(in_channels = 16, out_channels = 32, kernel_size = 5),
            nn.ReLU(inplace=True),

            Lambda(lambda x: x.view(x.shape[0], -1)),

            nn.Linear(in_features = 32, out_features = label_count),
            nn.Softmax(dim=1)
        )

        self.cuda()

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(dim = 0)  # add dimension for batch
        if len(x.shape) == 4:
            x = x.unsqueeze(dim = 1)  # add dimension for channels
        
        return self.layers(x)