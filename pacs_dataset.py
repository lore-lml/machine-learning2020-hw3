from torchvision.datasets import VisionDataset
from PIL import Image

import os


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Pacs(VisionDataset):

    def __init__(self, root, transform=None, source='photo'):
        super(Pacs, self).__init__(root, transform=transform)
        self.sources = ['art_painting', 'cartoon', 'photo', 'sketch']

        self.root = root
        self.transform = transform

        if source not in self.sources:
            raise ValueError(f"source must be a string in the following list: {self.sources}")
        self.source_path = os.path.join(root, source)

        self.class_names = {class_name: i for i, class_name in enumerate(sorted(os.listdir(self.source_path)))}
        self.images = []
        self.labels = []
        for c, label in self.class_names.items():
            current_path = os.path.join(self.source_path, c)
            for img_path in os.listdir(current_path):
                self.images.append(pil_loader(os.path.join(current_path, img_path)))
                self.labels.append(label)

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.labels)

    def get_img_with_labels(self):
        return self.images, self.labels


if __name__ == '__main__':
    pacs = [Pacs("PACS", source=s) for s in ['art_painting', 'cartoon', 'photo', 'sketch']]
    print(sum([len(p) for p in pacs]))
