'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small, MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights

import os
import argparse

from PIL import Image
import numpy as np

from sklearn.metrics import f1_score, precision_score, recall_score

parser = argparse.ArgumentParser(description='PyTorch TrashBox Training')
parser.add_argument('--dataroot', default='/data/kien/TrashBox', type=str, help='data root folder')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--epochs', default=10, type=int, help='num epochs')
parser.add_argument('--milestones', default=[20, 40, 50, 60, 70])
parser.add_argument('--resume', '-r', type=str, help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
MIN_VALID_IMG_DIM = 100
IMG_CROP_SIZE = 224


def rgb_loader(path):
    img = Image.open(path)
    if img.getbands() != ('R', 'G', 'B'):
        img = img.convert('RGB')
    return img

def is_valid_file(path):
    try:
        img = Image.open(path)
        img.verify()
    except:
        return False
    
    if not(img.height >= MIN_VALID_IMG_DIM and img.width >= MIN_VALID_IMG_DIM):
        return False

    return True


transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

trainset = torchvision.datasets.ImageFolder(root=os.path.join('/data/kien/TrashBox', 'train'), transform=transform_train,
                                            loader=rgb_loader, 
                                            is_valid_file=is_valid_file)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=4)

valset = torchvision.datasets.ImageFolder(root=os.path.join('/data/kien/TrashBox', 'val'), transform=transform_test,
                                            loader=rgb_loader, 
                                            is_valid_file=is_valid_file)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=100, shuffle=False, num_workers=4)

testset = torchvision.datasets.ImageFolder(root=os.path.join('/data/kien/TrashBox', 'test'), transform=transform_test,
                                            loader=rgb_loader, 
                                            is_valid_file=is_valid_file)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=4)

# Model
print('==> Building model..')

# net = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
# net.classifier._modules['3'] = nn.Linear(1280, 7) # large

net = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
net.classifier._modules['3'] = nn.Linear(1024, 7) # small
net = net.to(device)

# import sys
# sys.exit(0)

# if device == 'cuda':
#     net = torch.nn.DataParallel(net, [0, 1])
#     cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones)

# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('Epoch: %d - Loss: %.3f | Acc: %.3f%% (%d/%d)' 
          % ((epoch + 1), train_loss/(batch_idx + 1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print('Epoch: %d - Loss: %.3f | Acc: %.3f%% (%d/%d)' 
          % ((epoch + 1), test_loss/(batch_idx + 1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(args.epochs):
    train(epoch)
    test(epoch)
    scheduler.step()

# inference
ckpt = torch.load('./checkpoint/ckpt.pth')
net.load_state_dict(ckpt['net'])
net.eval()
gt = []
preds = []
correct = 0
total = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)

        _, predicted = outputs.max(1)

        gt.append(targets.detach().cpu().numpy())
        preds.append(predicted.detach().cpu().numpy())

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

gt = np.concatenate(gt, axis=0)
preds = np.concatenate(preds, axis=0)

print('Inference - Acc: %.3f%% (%d/%d) - F1: %.3f - Precision: %.3f - Recall: %.3f'
        % (100.*correct/total, correct, total, 
           f1_score(gt, preds, average='macro'), 
           precision_score(gt, preds, average='macro'), 
           recall_score(gt, preds, average='macro')))