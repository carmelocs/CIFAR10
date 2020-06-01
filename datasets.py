import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms, utils
import numpy as np
import json
import matplotlib.pyplot as plt

class CIFAR10_IMG(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(CIFAR10_IMG, self).__init__()
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if self.train:
            file_annotation = root + '/annotations/cifar10_train.json'
            img_folder = root + '/train_cifar10/'
        else:
            file_annotation = root + '/annotations/cifar10_test.json'
            img_folder = root + '/test_cifar10/'
        fp = open(file_annotation, 'r')
        data_dict = json.load(fp)

        assert len(data_dict['images'])==len(data_dict['categories'])
        num_data = len(data_dict['images'])

        self.filenames = []
        self.labels = []
        self.img_folder = img_folder
        for i in range(num_data):
            self.filenames.append(data_dict['images'][i])
            self.labels.append(data_dict['categories'][i])

    def __getitem__(self, idx):
        img_name = self.img_folder + self.filenames[idx]
        label = self.labels[idx]

        img = plt.imread(img_name)
        if self.transform is not None:
            img = self.transform(img)

        return img, label
        
    def __len__(self):
        return len(self.filenames)

    def load_label_names(self, idx):
        label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        return label_names[idx]


if __name__ == '__main__':
    train_dataset = CIFAR10_IMG('./datasets', train=True, transform=torchvision.transforms.ToTensor())
    test_dataset = CIFAR10_IMG('./datasets', train=False, transform=torchvision.transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=6, shuffle=True)

    for step, (b_train, b_label) in enumerate(train_loader):
        if step < 1:
            # b_train:[64, 3, 32, 32] (batch, colour, height, width)
            #print(b_train.shape)
            imgs = torchvision.utils.make_grid(b_train)
            # make_grid: make a grid of images (combine images)
            # imgs: [3, 274, 274] (colour, height, width)
            #print(imgs.shape)
            imgs = np.transpose(imgs, (1,2,0))
            plt.imshow(imgs)
            plt.show()