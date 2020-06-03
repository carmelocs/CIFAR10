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
        self.label_names = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


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

    def num_classes(self):
        return len(set(self.labels))

    def load_label_names(self, label_idx):
        label_names = self.label_names
        return label_names[label_idx]

    def count_items(self):
        labels = self.labels
        count = {}
        label_names = self.label_names
        for item in labels:
            count[label_names[item]] = count.get(label_names[item], 0) + 1
        return count


if __name__ == '__main__':
    train_dataset = CIFAR10_IMG('./datasets', train=True, transform=torchvision.transforms.ToTensor())
    test_dataset = CIFAR10_IMG('./datasets', train=False, transform=torchvision.transforms.ToTensor())
    count_train = train_dataset.count_items()
    count_test = test_dataset.count_items()
    print(count_train)
    print(count_test)
    print('Number of classes: ', train_dataset.num_classes())
    '''
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
    '''