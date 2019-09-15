import os
import json
import numpy as np
from tqdm import tqdm
import torch
import random

DATASET_DIRECTORY = "data/shapenet/"
MIN_SAMPLES_PER_CATEGORY = 2000
VOXEL_SIZE = 32
MODELS_FILENAME = "data/dataset-{:d}-{:d}.to".format(VOXEL_SIZE, MIN_SAMPLES_PER_CATEGORY)
MODELS_SDF_FILENAME = "data/dataset-sdf-{:d}.to".format(VOXEL_SIZE)
CLOUDS_SDF_FILENAME = "data/dataset-sdf-clouds.to"
SURFACE_POINTCLOUDS_FILENAME = "data/dataset-surface-pointclouds.to"
LABELS_FILENAME = "data/labels-{:d}-{:d}.to".format(VOXEL_SIZE, MIN_SAMPLES_PER_CATEGORY)

SDF_CLIPPING = 0.1
MIN_OCCUPIED_VOXELS = 550

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

class Dataset():
    def __init__(self):
        self.prepare_categories()

    def prepare_categories(self):
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

    def find_model_files(self, model_filename):
        labels = []
        filenames = []

        print("Scanning directory...")        
        for label in range(len(self.categories)):
            current_category = self.categories[label]
            items_in_category = 0
            category_directory = os.path.join(DATASET_DIRECTORY, str(current_category.id).rjust(8, '0'))
            for subdirectory in os.listdir(category_directory):
                filename = os.path.join(category_directory, subdirectory, "models", model_filename)
                if os.path.isfile(filename):
                    filenames.append(filename)
                    category_directory += 1

            labels.append(torch.ones(category_directory) * label)

        return filenames, labels

    def prepare_binary(self):
        from voxel.binvox_rw import read_as_3d_array

        filenames, labels = self.find_model_files("model_normalized.solid.binvox")        
        
        indices = list(range(len(filenames)))
        random.shuffle(indices)
        if LIMIT_SIZE > 0:
            indices = indices[:LIMIT_SIZE]
        filenames = [filenames[i] for i in indices]        

        models = []
        pool = torch.nn.MaxPool3d(4)
        print("Loading models...")
        for filename in tqdm(filenames):
            voxels = torch.tensor(read_as_3d_array(open(filename, 'rb')).data.astype(np.float32)) * -2 + 1
            voxels = torch.unsqueeze(voxels, 0)
            voxels = pool(voxels).squeeze()
            models.append(voxels.to(torch.int8))
        
        print("Saving...")
        tensor = torch.stack(models).to(torch.int8)
        torch.save(tensor, MODELS_FILENAME)

        labels = torch.cat(labels).to(torch.int8)[indices]
        torch.save(labels, LABELS_FILENAME)

        print("Done.")

    def prepare_sdf(self):
        filenames, _ = self.find_model_files("sdf-{:d}.npy".format(VOXEL_SIZE))
        
        random.shuffle(filenames)

        models = []
        print("Loading models...")
        for filename in tqdm(filenames):
            voxels = np.load(filename)
            voxels = torch.tensor(voxels)
            models.append(voxels)
        
        print("Saving...")
        tensor = torch.stack(models)
        tensor = torch.transpose(tensor, 1, 2)
        torch.save(tensor, MODELS_SDF_FILENAME)
        
        print("Done.")

    def prepare_sdf_clouds(self):
        filenames, _ = self.find_model_files("sdf-pointcloud.npy")
        used_filenames = []
        
        POINTCLOUD_SIZE = 100000

        random.shuffle(filenames)
        result = torch.zeros((POINTCLOUD_SIZE * len(filenames), 4))
        position = 0

        print("Loading models...")
        for filename in tqdm(filenames):
            cloud = np.load(filename)
            if cloud.shape[0] != POINTCLOUD_SIZE:
                print("Bad pointcloud shape: ", cloud.shape)
                continue
            
            model_size = np.count_nonzero(cloud[-20000:, 3] < 0) / 20000
            if model_size < 0.015:
                continue

            cloud = torch.tensor(cloud)
            result[position * POINTCLOUD_SIZE:(position + 1) * POINTCLOUD_SIZE, :] = cloud
            position += 1
            used_filenames.append(filename)
        
        print("Saving...")
        result = result[:position * POINTCLOUD_SIZE, :].clone()
        torch.save(result, CLOUDS_SDF_FILENAME)

        with open('data/sdf-clouds.txt', 'w') as file:
            file.write('\n'.join(used_filenames))
        
        print("Used {:d}/{:d} pointclouds.".format(len(used_filenames), len(filenames)))

    def prepare_surface_clouds(self, limit_models_number=None):
        filenames, _ = self.find_model_files("surface-pointcloud.npy")
        
        POINTCLOUD_SIZE = 50000
        if limit_models_number is not None:
            filenames = filenames[:limit_models_number]

        random.shuffle(filenames)
        result = torch.zeros((POINTCLOUD_SIZE * len(filenames), 6))
        position = 0

        print("Loading models...")
        for filename in tqdm(filenames):
            cloud = np.load(filename)
            if cloud.shape[0] != POINTCLOUD_SIZE:
                print("Bad pointcloud shape: ", cloud.shape)
                continue
            cloud = torch.tensor(cloud)
            result[position * POINTCLOUD_SIZE:(position + 1) * POINTCLOUD_SIZE, :] = cloud
            position += 1
        
        print("Saving...")
        torch.save(result, SURFACE_POINTCLOUDS_FILENAME)
        
        print("Done.")

    def load_sdf(self, device):
        print("Loading dataset...")
        self.voxels = torch.load(MODELS_SDF_FILENAME).to(device).float()

        if MIN_OCCUPIED_VOXELS != 0:
            occupied = torch.sum(self.voxels < 0, dim=[1, 2, 3])
            mask = occupied > MIN_OCCUPIED_VOXELS
            self.voxels = self.voxels[mask, :, :, :]

        torch.clamp_(self.voxels, -SDF_CLIPPING, SDF_CLIPPING)
        self.voxels /= SDF_CLIPPING
        self.size = self.voxels.shape[0]
        self.label_indices = torch.zeros(self.size).to(torch.int64).to(device)
        self.labels = torch.zeros((self.size, self.label_count))
        self.labels[torch.arange(0, self.size, dtype=torch.long, device=device), self.label_indices] = 1
        self.labels = self.labels.to(device)


dataset = Dataset()

if __name__ == "__main__":
    dataset.prepare_sdf_clouds()
else:
    from util import device
    dataset.load_sdf(device)
