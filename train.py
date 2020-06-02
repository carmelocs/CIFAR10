import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import CIFAR10_IMG
from model import ConvNet


ROOT = './datasets'

train_dataset = CIFAR10_IMG(root=ROOT, train=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = CIFAR10_IMG(root=ROOT, train=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=64)

net = ConvNet(num_classes=train_dataset.num_classes())
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
            pred = outputs.argmax(dim=-1)
            correct += torch.eq(pred, labels).sum().float().item()
            accuracy = correct / len(inputs)
            print('Epoch {}: step: {} loss: {:.4f} accuracy: {}'.format(epoch, step, loss, accuracy ))

print('Finish training.')

# compare predictions and labels of first 10 test samples
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


print('Length of train dataset: {}'.format(len(train_dataset))) # 50000 train samples
print('Length of test dataset: {}'.format(len(test_dataset))) # 10000 test samples

# print first 10 samples: (file_name, label_name)
for i in range(10):
    print('Image {} in train dataset: {}'.format(train_dataset.filenames[i], train_dataset.load_label_names(train_dataset.labels[i])))