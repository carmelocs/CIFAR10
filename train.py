import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import CIFAR10_IMG
from model import ConvNet

def load_train_data():
    root = './datasets'
    train_dataset = CIFAR10_IMG(root, train=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    return train_loader

def load_test_data():
    root = './datasets'
    test_dataset = CIFAR10_IMG(root, train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=64)

    return test_loader

train_loader = load_train_data()
net = ConvNet()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()


for epoch in range(1):
    for step, (inputs, labels) in enumerate(train_loader):
        outputs = net(inputs)
        loss = loss_fn(outputs, labels) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            correct = 0.0
            pred = outputs.argmax(dim=1)
            correct += torch.eq(pred, labels).sum().float().item()
            accuracy = correct / len(inputs)
            print('Epoch {}: step: {} loss: {:.4f} accuracy: {}'.format(epoch, step, loss, accuracy ))

print('Finish training.')

test_dataset = CIFAR10_IMG(root='./datasets', train=False, transform=transforms.ToTensor())

pred_list = []
label_list = []

for i in range(10):
    test_inputs, test_labels = test_dataset[i]
    test_inputs = torch.unsqueeze(test_inputs, 0)
    test_outputs = net(test_inputs)
    pred = test_outputs.argmax(1)
    pred_list.append(int(pred.detach().numpy()))
    label_list.append(test_labels)

print(pred_list, 'prediction labels')
print(label_list, 'test labels')

'''
cifar10 = CIFAR10_IMG(root='./datasets', train=True, transform=transforms.ToTensor())
print('Length of cifar10: {}'.format(len(cifar10))) # 50000 samples

for i in range(len(cifar10)):
    print('Image {} in cifar10: {}'.format(i+1, cifar10.load_label_names(cifar10[i][-1])))
'''
'''
train_loader = DataLoader(cifar10, batch_size=64, shuffle=True)
for epoch, (images, labels) in enumerate(train_loader, 0):
    for i in range(len(labels)):
        print('Epoch{} {}th input {} is: {}'.format(epoch, i+1, images[i].shape, cifar10.load_label_names(labels[i])))
    if epoch == 0:
        break
'''