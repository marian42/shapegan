import os
import json
import numpy as np
from tqdm import tqdm
import torch
import random
import sys

DATASET_DIRECTORY = "data/shapenet/"
MIN_SAMPLES_PER_CATEGORY = 2000
VOXEL_RESOLUTION = 32

VOXELS_SDF_FILENAME = "data/voxels-{:d}.to".format(VOXEL_RESOLUTION)
SDF_POINTS_FILENAME = "data/sdf-points-{:d}.to"
SDF_VALUES_FILENAME = "data/sdf-values-{:d}.to"
SURFACE_POINTCLOUDS_FILENAME = "data/surface-pointclouds.to"
LABELS_FILENAME = "data/labels.to"

VOXEL_FILENAME = "sdf-{:d}.npy".format(VOXEL_RESOLUTION)
SDF_CLOUD_FILENAME = "sdf-pointcloud.npy"
SURFACE_POINTCLOUD_FILENAME = "surface-pointcloud.npy"

DIRECTORIES_FILE = 'data/models.txt'

SDF_CLIPPING = 0.1
POINTCLOUD_SIZE = 200000
SURFACE_POINTCLOUD_SIZE = 50000

SDF_PARTS = 8

from util import device

class Category():
    def __init__(self, name, id, count):
        self.name = name
        self.id = id
        self.is_root = True
        self.children = []
        self.count = count
        self.label = None

    def print(self, depth = 0):
        if self.count < MIN_SIZE:
            return
        print('  ' * depth + self.name + '({:d})'.format(self.count))
        for child in self.children:
            child.print(depth = depth + 1)

    def get_directory(self):
        return os.path.join(DATASET_DIRECTORY, str(self.id).rjust(8, '0'))

class SDFPart():
    def __init__(self, index, dataset):
        self.index = index
        self.points_filename = SDF_POINTS_FILENAME.format(index)
        self.values_filename = SDF_VALUES_FILENAME.format(index)
        self.points = None
        self.values = None
        self.dataset = dataset

    def load(self):
        self.points = torch.load(self.points_filename, map_location=device)
        self.values = torch.load(self.values_filename, map_location=device)
        
        if self.dataset.clip_sdf:
            torch.clamp_(self.values, -SDF_CLIPPING, SDF_CLIPPING)
            if self.dataset.rescale_sdf:
                self.values /= SDF_CLIPPING

    def unload(self):
        del self.points
        del self.values

class Dataset():
    def __init__(self):
        self.clip_sdf = True
        self.rescale_sdf = True

        self.load_categories()
        self.labels = None

        self.sdf_part_count = SDF_PARTS
        self.last_part_loaded = None
        self.sdf_parts = [SDFPart(i, self) for i in range(self.sdf_part_count)]

    def load_categories(self):
        taxonomy_filename = os.path.join(DATASET_DIRECTORY, "taxonomy.json")
        if not os.path.isfile(taxonomy_filename):
            taxonomy_filename = 'examples/shapenet_taxonomy.json'
        file_content = open(taxonomy_filename).read()
        taxonomy = json.loads(file_content)
        categories = dict()
        for item in taxonomy:
            id = int(item['synsetId'])
            category = Category(item['name'], id, item['numInstances'])
            categories[id] = category

        for item in taxonomy:
            id = int(item['synsetId'])
            category = categories[id]
            for str_id in item["children"]:
                child_id = int(str_id)
                category.children.append(categories[child_id])
                categories[child_id].is_root = False
        
        self.categories = [item for item in categories.values() if item.is_root and item.count >= MIN_SAMPLES_PER_CATEGORY]
        self.categories = sorted(self.categories, key=lambda item: item.id)
        self.categories_by_id = {item.id : item for item in self.categories}
        self.label_count = len(self.categories)
        for i in range(len(self.categories)):
            self.categories[i].label = i

    def prepare_models_file(self):
        directories = []
        
        for category in tqdm(self.categories):
            category_directory = category.get_directory()
            for subdirectory in os.listdir(category_directory):
                model_directory = os.path.join(category_directory, subdirectory, "models")

                if all(os.path.isfile(os.path.join(model_directory, n)) for n in (VOXEL_FILENAME, SDF_CLOUD_FILENAME, SURFACE_POINTCLOUD_FILENAME)):
                    directories.append(model_directory)

        random.shuffle(directories)
        
        with open(DIRECTORIES_FILE, 'w') as file:
            file.write('\n'.join(directories))

    def get_models(self):
        if not os.path.isfile(DIRECTORIES_FILE):
            self.prepare_models_file()
        
        with open(DIRECTORIES_FILE, 'r') as file:
            return [line.strip() for line in file.readlines()]

    def prepare_voxels(self):
        models = []
        print("Loading models...")
        for directory in tqdm(self.get_models()):
            voxels = np.load(os.path.join(directory, VOXEL_FILENAME))
            voxels = torch.tensor(voxels)
            models.append(voxels)
        
        print("Stacking...")
        tensor = torch.stack(models)
        print("Saving...")
        torch.save(tensor, VOXELS_SDF_FILENAME)        
        print("Done.")

    def prepare_labels(self):
        directories = self.get_models()
        labels = torch.zeros(len(directories), dtype=torch.int64)

        for i in range(len(directories)):
            category_id = int(directories[i].split('/')[2])
            label = self.categories_by_id[category_id].label
            labels[i] = label
        
        torch.save(labels, LABELS_FILENAME)

    def prepare_sdf_clouds(self):
        directories = self.get_models()
        part_size = POINTCLOUD_SIZE // SDF_PARTS

        for part in range(SDF_PARTS):
            print("Preparing part {:d} / {:d}...".format(part + 1, SDF_PARTS))

            points = torch.zeros((part_size * len(directories), 3))
            sdf = torch.zeros((part_size * len(directories)))            
            position = 0

            print("Loading models...")
            for directory in tqdm(directories):
                cloud = np.load(os.path.join(directory, SDF_CLOUD_FILENAME))
                cloud = cloud[part::SDF_PARTS, :]
                if cloud.shape[0] != part_size:
                    raise Exception("Bad pointcloud shape: ", cloud.shape)

                cloud = torch.tensor(cloud)
                points[position * part_size:(position + 1) * part_size, :] = cloud[:, :3]
                sdf[position * part_size:(position + 1) * part_size] = cloud[:, 3]
                position += 1
            
            print("Saving...")
            torch.save(points, SDF_POINTS_FILENAME.format(part))
            torch.save(sdf, SDF_VALUES_FILENAME.format(part))
            del points
            del sdf
        
        print("Done.")

    def prepare_surface_clouds(self):
        SKIP_POINTS = 2
        directories = self.get_models()

        points = torch.zeros((SURFACE_POINTCLOUD_SIZE * len(directories) // SKIP_POINTS, 3))
        position = 0

        print("Loading models...")
        for directory in tqdm(directories):
            cloud = np.load(os.path.join(directory, SURFACE_POINTCLOUD_FILENAME))
            if cloud.shape[0] != SURFACE_POINTCLOUD_SIZE:
                raise Exception("Bad pointcloud shape: ", cloud.shape)

            cloud = torch.tensor(cloud[::SKIP_POINTS, :3])
            points[position * SURFACE_POINTCLOUD_SIZE // SKIP_POINTS:(position + 1) * (SURFACE_POINTCLOUD_SIZE // SKIP_POINTS), :] = cloud
            position += 1
        
        print("Saving...")
        torch.save(points, SURFACE_POINTCLOUDS_FILENAME)
        
        print("Done.")

    def load_voxels(self, device):
        print("Loading dataset...")
        self.voxels = torch.load(VOXELS_SDF_FILENAME).to(device).float()

        if self.clip_sdf:
            torch.clamp_(self.voxels, -SDF_CLIPPING, SDF_CLIPPING)
            if self.rescale_sdf:
                self.voxels /= SDF_CLIPPING

        self.load_labels(device=device)        
        
    def load_labels(self, device = None):
        if self.labels is None:
            labels = torch.load(LABELS_FILENAME)
            if device is not None:
                labels = labels.to(device)
            self.labels = labels
            self.size = self.labels.shape[0]

    def load_surface_clouds(self, device):
        self.surface_pointclouds = torch.load(SURFACE_POINTCLOUDS_FILENAME).to(device)
        self.load_labels(device)
        return self.surface_pointclouds

    def get_labels_onehot(self, device):
        labels_onehot = torch.nn.functional.one_hot(self.labels, self.label_count).to(torch.float32)
        return labels_onehot.type(torch.float32).to(device)
    
    def get_color(self, label):
        if label == 2:
            return (0.9, 0.1, 0.14) # red
        elif label == 1:
            return (0.8, 0.7, 0.1) # yellow
        elif label == 6:
            return (0.05, 0.5, 0.05) # green
        elif label == 5:
            return (0.1, 0.2, 0.9) # blue
        elif label == 4:
            return (0.46, 0.1, 0.9) # purple
        elif label == 3:
            return (0.9, 0.1, 0.673) # purple
        elif label == 0:
            return (0.01, 0.6, 0.9) # cyan
        else:
            return (0.7, 0.7, 0.7)

    def load_sdf_part(self, index):
        if self.last_part_loaded is not None and self.last_part_loaded.index != index:
            self.last_part_loaded.unload()
        part = self.sdf_parts[index]
        part.load()
        self.last_part_loaded = part
        return part

dataset = Dataset()

if __name__ == "__main__":
    if "init" in sys.argv:
        dataset.get_models()
        dataset.prepare_labels()
    if "prepare_voxels" in sys.argv:
        dataset.prepare_voxels()
    if "prepare_sdf" in sys.argv:
        dataset.prepare_sdf_clouds()
    if "prepare_surface" in sys.argv:
        dataset.prepare_surface_clouds()
    if "stats" in sys.argv:
        dataset.load_labels()
        label_count = torch.sum(dataset.get_labels_onehot('cpu'), dim=0)
        for category in sorted(dataset.categories, key=lambda c: -c.count):
            print('{:d}: {:s} - used {:d} / {:d}'.format(
                category.label,
                category.name,
                int(label_count[category.label]),
                category.count))
    
    if "show_meshes" in sys.argv:
        import trimesh
        import os
        from rendering import MeshRenderer
        import time
        viewer = MeshRenderer()

        for directory in dataset.get_models():
            model_filename = os.path.join(directory, 'model_normalized.obj')        
            mesh = trimesh.load(model_filename)
            viewer.set_mesh(mesh, center_and_scale=True)
            time.sleep(0.5)
    if "show_voxels" in sys.argv:
        from rendering import MeshRenderer
        import time
        viewer = MeshRenderer()
        dataset.load_voxels('cpu')
        for i in tqdm(list(range(dataset.voxels.shape[0]))):
            try:
                viewer.set_voxels(dataset.voxels[i, :, :, :].squeeze().detach().cpu().numpy())
                time.sleep(0.5)
            except KeyboardInterrupt:
                viewer.stop()
                break