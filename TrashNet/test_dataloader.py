import torchvision.transforms as transforms
import torchvision
import torch
import os
from PIL import Image

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

print(len(trainloader), len(valloader), len(testloader))
for (x, y) in trainloader:    
    print(x.size(), y.size())

for (x, y) in valloader:    
    print(x.size(), y.size())

for (x, y) in testloader:    
    print(x.size(), y.size())