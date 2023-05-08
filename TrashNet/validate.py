import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

class_names = ['cardboard', 'e-waste', 'glass', 'medical', 'metal', 'paper', 'plastic']

parser = argparse.ArgumentParser(description='PyTorch TrashBox Validation with Image')
parser.add_argument('--image', type=str, help='path to image')
parser.add_argument('--ckpt', type=str, help='checkpoint of the model')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

raw_img = Image.open(args.image)
img = transform_test(raw_img)

x = img.unsqueeze(0).to(device)
net = mobilenet_v3_large()
net.classifier._modules['3'] = nn.Linear(1280, 7) # large
net = net.to(device)

if args.ckpt:
    # Load checkpoint
    print('==> Loading from checkpoint..')
    checkpoint = torch.load(args.ckpt)
    net.load_state_dict(checkpoint['net'])

output = torch.softmax(net(x), dim=-1).detach().cpu()
pred = torch.argmax(output).item()
print(f'Prediction: {class_names[pred]}')
