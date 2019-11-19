import torch
from model.classifier import Classifier
from dataset import dataset

# Inception score of a sample from the dataset
REFERENCE_INCEPTION_SCORE_VOXEL = 4.430443
REFERENCE_INCEPTION_SCORE_POINTS = 4.283844

class InceptionScore():
    def __init__(self):
        self.classifier = Classifier()
        self.classifier.load()

    def __call__(self, input):
        with torch.no_grad():
            label_distribution = self.classifier(input)
            marginal_distribution = torch.mean(label_distribution, dim = 0)
            
            kld = -torch.sum(label_distribution * torch.log(marginal_distribution / label_distribution), dim = 1)
            
            score = torch.exp(torch.mean(kld[torch.isfinite(kld)]))
        return score.item() / REFERENCE_INCEPTION_SCORE_VOXEL

class PointcloudInceptionScore():
    def __init__(self):
        from model.pointcloud_classifier import PointcloudClassifier
        self.classifier = PointcloudClassifier()
        self.classifier.load()

    def __call__(self, points, object_count):
        pointcloud_size = points.shape[0] // object_count

        with torch.no_grad():
            label_distribution = torch.zeros(object_count, dataset.label_count)
            for i in range(object_count):
                label_distribution[i, :] = self.classifier(points[i * pointcloud_size:(i + 1) * pointcloud_size, :])
            
            marginal_distribution = torch.mean(label_distribution, dim = 0)            
            kld = -torch.sum(label_distribution * torch.log(marginal_distribution / label_distribution), dim = 1)            
            score = torch.exp(torch.mean(kld[torch.isfinite(kld)]))
        return score.item() / REFERENCE_INCEPTION_SCORE_POINTS

try:
    inception_score = InceptionScore()
    available = True
except FileNotFoundError:
    print("Warning: No classifier was found, disabling inception score.")
    available = False
    inception_score = lambda _: 0

try:    
    inception_score_points = PointcloudInceptionScore()
    available_for_points = True
except FileNotFoundError:
    print("Warning: No point cloud classifier was found, disabling point cloud inception score.")
    available_for_points = False
    inception_score_points = lambda _0, _1: 0
except ImportError as error:
    if error.name != 'torch_geometric':
        raise    
    print("Warning: PyTorch Geometric is not available, disabling point inception score.")
    available_for_points = False
    inception_score_points = lambda _0, _1: 0

if __name__ == '__main__':
    from dataset import dataset as dataset
    from util import device
    import torch
    dataset.load_voxels(device=device)
    score = inception_score(dataset.voxels[:3000, :, :, :])
    print("Raw inception score of the voxel dataset:", score * REFERENCE_INCEPTION_SCORE_VOXEL)
    
    dataset.load_surface_clouds(device)
    SURFACE_POINTCLOUD_SIZE = dataset.surface_pointclouds.shape[0] // dataset.size
    SAMPLE_SIZE = 1000
    TEST_POINTCLOUD_SIZE = 1000
    del dataset.voxels
    torch.cuda.empty_cache()

    points = torch.zeros((SAMPLE_SIZE * TEST_POINTCLOUD_SIZE, 3), device=device)
    for i in range(SAMPLE_SIZE):
        indices = torch.LongTensor(TEST_POINTCLOUD_SIZE).random_(0, SURFACE_POINTCLOUD_SIZE - 1)
        indices += i * SURFACE_POINTCLOUD_SIZE
        points[i * TEST_POINTCLOUD_SIZE:(i+1)*TEST_POINTCLOUD_SIZE, :] = dataset.surface_pointclouds[indices, :]
    del dataset.surface_pointclouds
    torch.cuda.empty_cache()
    score = inception_score_points(points, SAMPLE_SIZE)
    print("Raw inception score of the surface point dataset:", score * REFERENCE_INCEPTION_SCORE_POINTS)
