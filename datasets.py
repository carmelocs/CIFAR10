import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
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
        
        self.num_classes = len(set(self.labels))

    def __getitem__(self, idx):
        img_name = self.img_folder + self.filenames[idx]
        label = self.labels[idx]

        img = plt.imread(img_name)
        if self.transform is not None:
            img = self.transform(img)

        return img, label
        
    def __len__(self):
        return len(self.filenames)

    def count_items(self):
        labels = self.labels
        count = {}
        label_names = self.label_names
        for item in labels:
            count[label_names[item]] = count.get(label_names[item], 0) + 1
        return count

def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1,2,0))) # transposed to (height, width, channel)
    plt.show()

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_dataset = CIFAR10_IMG('./datasets', train=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

    test_dataset = CIFAR10_IMG('./datasets', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

    label_list = []
    image_list = []
    for i in range(4):
        images, labels = train_dataset[i]
        image_list.append(images)
        label_list.append(labels)

    imshow(torchvision.utils.make_grid(image_list))
    print('-'.join('%5s' % train_dataset.label_names[label_list[j]] for j in range(4)))

    '''
    print(train_dataset[0][0]) # print out tensor of the first image 
    # print out number of samples for each category
    print(train_dataset.count_items())
    print(test_dataset.count_items())
    
    print('Number of classes: ', train_dataset.num_classes())
    '''
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