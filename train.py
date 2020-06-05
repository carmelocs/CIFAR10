import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import CIFAR10_IMG, imshow
from model import Net, NetGPU
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=64, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--epoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--dataset', type=str, default='./datasets', help='dataset path')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

opt = parser.parse_args()

ROOT = opt.dataset
BATCH_SIZE = opt.batchSize
WORKERS = opt.workers
EPOCH = opt.epoch
LR = opt.lr
MOMENTUM = opt.momentum

transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]
    )

train_dataset = CIFAR10_IMG(root=ROOT, train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)

test_dataset = CIFAR10_IMG(root=ROOT, train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
'''
if torch.cuda.is_available():
    net = NetGPU(num_classes=train_dataset.num_classes)
else:
    net = Net(num_classes=train_dataset.num_classes)
'''
net = models.resnet18()
net.to(device)
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM)

for epoch in range(EPOCH):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        # clear the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item() # accumulate 100 mini-batches' loss
        if i % 100 == 99: # print every 100 mini-batches
            print('[{}, {}] loss: {:.3f}'.format(epoch+1, i+1, running_loss / 100))
            running_loss = 0.0

print('Finish Training.')

'''
dataiter = iter(test_loader)
images, labels = dataiter.next()

# show images
#imshow(torchvision.utils.make_grid(images))
print('Ground Truth: ', '-'.join('%5s' % test_dataset.label_names[labels[j]] for j in range(4)))

# print predictions
outputs = net(images)
_, predictions = outputs.max(-1)
print('Prediction: ', '-'.join('%5s' % test_dataset.label_names[predictions[j]] for j in range(4)))
'''
# print the accuracy on the 10000 test images
correct = 0
total = 0
with torch.no_grad():
    for (images, labels) in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predctions = outputs.max(-1)
        total += labels.size(0)
        correct += (predctions == labels).sum().item()

print('Accuracy of the network on the 10000 test images: {}%'.format(100*correct/total))

# print the accuracies for each categories
class_correct = list(0.0 for i in range(test_dataset.num_classes))
class_total = list(0.0 for i in range(test_dataset.num_classes))
with torch.no_grad():
    for (images, labels) in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predictions = outputs.max(-1)
        c = (predictions == labels)
        for i in range(16):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of {} : {:.2f}%'.format(test_dataset.label_names[i], 100*class_correct[i]/class_total[i]))