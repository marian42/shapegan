import os
import json
import numpy as np
from tqdm import tqdm
import torch
import random
import sys

DATASET_DIRECTORY = "data/shapenet/"
MIN_SAMPLES_PER_CATEGORY = 2000
VOXEL_SIZE = 32

VOXELS_SDF_FILENAME = "data/voxels-{:d}.to".format(VOXEL_SIZE)
SDF_POINTS_FILENAME = "data/sdf-points-{:d}.to"
SDF_VALUES_FILENAME = "data/sdf-values-{:d}.to"
SURFACE_POINTCLOUDS_FILENAME = "data/surface-pointclouds.to"
LABELS_FILENAME = "data/labels.to"

VOXEL_FILENAME = "sdf-{:d}.npy".format(VOXEL_SIZE)
SDF_CLOUD_FILENAME = "sdf-pointcloud.npy"
SURFACE_POINTCLOUD_FILENAME = "surface-pointcloud.npy"

DIRECTORIES_FILE = 'data/models.txt'

SDF_CLIPPING = 0.1
POINTCLOUD_SIZE = 200000

SDF_PARTS = 8

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

class Dataset():
    def __init__(self):
        self.clip_sdf = True
        self.rescale_sdf = True

        self.load_categories()

    def load_categories(self):
        taxonomy_filename = os.path.join(DATASET_DIRECTORY, "taxonomy.json")
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
        tensor = torch.transpose(tensor, 1, 2)
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

    def load_voxels(self, device):
        print("Loading dataset...")
        self.voxels = torch.load(VOXELS_SDF_FILENAME).to(device).float()

        if self.clip_sdf:
            torch.clamp_(self.voxels, -SDF_CLIPPING, SDF_CLIPPING)
            if self.rescale_sdf:
                self.voxels /= SDF_CLIPPING
        self.size = self.voxels.shape[0]
        
        self.labels = torch.load(LABELS_FILENAME).to(device)

    def get_labels_onehot(self, device):
        labels_onehot = torch.nn.functional.one_hot(self.labels, self.label_count).to(torch.float32)
        return labels_onehot.type(torch.float32).to(device)

dataset = Dataset()

if __name__ == "__main__":
    if "init" in sys.argv:
        dataset.get_models()
        dataset.prepare_labels()
    if "prepare_voxels" in sys.argv:
        dataset.prepare_voxels()
    if "prepare_sdf" in sys.argv:
        dataset.prepare_sdf_clouds()
