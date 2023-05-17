from flask import Flask, jsonify, request
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision.models import mobilenet_v3_large
import torch.nn as nn
import torchvision.transforms as transforms

import base64
from io import BytesIO
import os

app = Flask(__name__)
class_names = ['cardboard', 'e-waste', 'glass', 'medical', 'metal', 'paper', 'plastic']

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

net = mobilenet_v3_large()
net.classifier._modules['3'] = nn.Linear(1280, 7) # large

# load ckpt
print("=> Loading network checkpoint...")
checkpoint = torch.load('checkpoint/mobilenetv3_large.pth', map_location=torch.device("cpu"))
net.load_state_dict(checkpoint['net'])
net.eval()

#index
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    im_bytes = base64.b64decode(data['image'])
    im_file = BytesIO(im_bytes)
    img = Image.open(im_file)   
    img = np.array(img)
    img = cv2.resize(img,(224,224))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    img = transform(img) # apply transformation
    img = img.unsqueeze(0)
    logits = torch.softmax(net(img), dim=-1)
    pred_idx = torch.argmax(logits).item()
    return jsonify({"class": class_names[pred_idx]})
    

if __name__ == '__main__':
    app.secret_key = 'randomfkingkeyboiz'
    app.run(debug=True)
