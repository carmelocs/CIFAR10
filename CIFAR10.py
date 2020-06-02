import torch
import torchvision
import numpy as np
from cv2 import cv2
import os
import json

DOWNLOAD = False

cifar10 = torchvision.datasets.CIFAR10(
    root='datasets',
    train=True,
    download=DOWNLOAD
)
cifar10_test = torchvision.datasets.CIFAR10(
    root='datasets',
    train=False,
    download=DOWNLOAD
)
# print out dataset info
print(cifar10)
print(cifar10_test)

train_filenames = []
train_annotations = []
test_filenames = []
test_annotations= []

# CIFAR10 official uncompression function
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

anno_loc = './datasets/annotations/'
loc_1 = './datasets/train_cifar10/'
loc_2 = './datasets/test_cifar10/'

#判断文件夹是否存在，不存在的话创建文件夹
if os.path.exists(loc_1) == False:
    os.mkdir(loc_1)
if os.path.exists(loc_2) == False:
    os.mkdir(loc_2)
if os.path.exists(anno_loc) == False:
    os.mkdir(anno_loc)

def cifar10_img(file_dir):
    for i in range(1,6):
        data_name = file_dir + '/'+'data_batch_'+ str(i)
        data_dict = unpickle(data_name)
        print(data_name + ' is processing')

        for j in range(10000):
            img = np.reshape(data_dict[b'data'][j],(3,32,32))
            img = np.transpose(img,(1,2,0))
            # channel in RGB order
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            # change suffix (.jpg) to save in different file format
            img_name = loc_1 + str(data_dict[b'labels'][j]) + str((i)*10000 + j) + '.jpg'
            annot_img_name = str(data_dict[b'labels'][j]) + str((i)*10000 + j) + '.jpg'
            img_annotations = data_dict[b'labels'][j]
            train_filenames.append(annot_img_name)
            train_annotations.append(img_annotations)

            cv2.imwrite(img_name,img)

        print(data_name + ' is done')


    test_data_name = file_dir + '/test_batch'
    print(test_data_name + ' is processing')
    test_dict = unpickle(test_data_name)

    for m in range(10000):
        img = np.reshape(test_dict[b'data'][m], (3, 32, 32))
        img = np.transpose(img, (1, 2, 0))
        # channel in RGB order
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # change suffix (.jpg) to save in different file format
        img_name = loc_2 + str(test_dict[b'labels'][m]) + str(10000 + m) + '.jpg'
        annot_img_name = str(test_dict[b'labels'][m]) + str(10000 + m) + '.jpg'
        img_annotations = test_dict[b'labels'][m]
        test_filenames.append(annot_img_name)
        test_annotations.append(img_annotations)
        cv2.imwrite(img_name, img)
    print(test_data_name + ' is done')
    print('Finish transforming to image')

if __name__ == '__main__':

    file_dir = './datasets/cifar-10-batches-py'
    cifar10_img(file_dir)

    train_annot_dict = {
        'images': train_filenames,
        'categories': train_annotations
    }
    test_annot_dict = {
        'images':test_filenames,
        'categories':test_annotations
    }

    train_json = json.dumps(train_annot_dict)
    train_file = open('./datasets/annotations/cifar10_train.json', 'w')
    train_file.write(train_json)
    train_file.close()

    test_json = json.dumps(test_annot_dict)
    test_file = open('./datasets/annotations/cifar10_test.json','w')
    test_file.write(test_json)
    test_file.close()
    print('annotations have writen to json file')