import os
import json

DATASET_DIRECTORY = "data/shapenet/"
MIN_SAMPLES_PER_CATEGORY = 2000

from util import device

class ShapenetCategory():
    def __init__(self, name, id, count):
        self.name = name
        self.id = id
        self.is_root = True
        self.children = []
        self.count = count
        self.label = None

    def print(self, depth = 0):
        print('  ' * depth + self.name + '({:d})'.format(self.count))
        for child in self.children:
            child.print(depth = depth + 1)

    def get_directory(self):
        return os.path.join(DATASET_DIRECTORY, str(self.id).rjust(8, '0'))

class ShapenetMetadata():
    def __init__(self):
        self.clip_sdf = True
        self.rescale_sdf = True

        self.load_categories()
        self.labels = None

    def load_categories(self):
        taxonomy_filename = os.path.join(DATASET_DIRECTORY, "taxonomy.json")
        if not os.path.isfile(taxonomy_filename):
            taxonomy_filename = 'examples/shapenet_taxonomy.json'
        file_content = open(taxonomy_filename).read()
        taxonomy = json.loads(file_content)
        categories = dict()
        for item in taxonomy:
            id = int(item['synsetId'])
            category = ShapenetCategory(item['name'], id, item['numInstances'])
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

shapenet = ShapenetMetadata()

if __name__ == "__main__":
    for category in sorted(shapenet.categories, key=lambda c: -c.count):
        print('{:d}: {:s} - {:d}'.format(
            category.label,
            category.name,
            category.count))