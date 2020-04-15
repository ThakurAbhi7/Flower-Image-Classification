import argparse, json
import torch
import torchvision.models as models
from collections import OrderedDict
from torch import nn
from torch import optim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_model(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'vgg13':
        model = getattr(models, arch)(pretrained=True)
    elif checkpoint['arch'] == 'alexnet':
        model = getattr(models, arch)(pretrained=True)
    layer_size = checkpoint['layer_size']
    model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(layer_size[0], layer_size[1])),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(layer_size[1], layer_size[2])),
                          ('relu', nn.ReLU()),
                          ('fc3', nn.Linear(layer_size[2], layer_size[3])),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    im_resized = im.resize((256, 256))
    im_crop = im_resized.crop((16, 16, 240, 240))
    np_image = np.asarray(im_crop)
    np_image = np_image/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = np.subtract(np_image, mean)
    np_image = np.divide(np_image, std)
    np_image = np.transpose(np_image, (2, 0, 1))
    return (np_image)

def predict(image_path, model, cat_to_name, topk=5, gpu = True):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img = process_image(image_path)
    img = torch.FloatTensor(img)
    if gpu == True:
        img = img.cuda()
    logps = model.forward(img.unsqueeze(0))
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(topk, dim=1)
    top_class = top_class.cpu()
    predict_class = [None]*topk 
    predict_prob = [None]*topk 
    class_to_idx = dict(map(reversed, model.class_to_idx.items()))
    for i, cat in enumerate(top_class[0]):
        predict_class[i] = cat_to_name[class_to_idx[cat.item()]]
        predict_prob[i] = top_p[0][i].item()
    s = pd.Series(predict_prob, index = predict_class)
    print(s,top_p, top_class)

def parse_args():
    """
    Parse input arguments
    """    
    parser = argparse.ArgumentParser(description = 'Image classification')
    parser.add_argument('--top_k', dest = 'top_k', help = 'No of entries in output', default = 5, type = int)
    parser.add_argument('--category_names', dest = 'category_names',  help = 'convert index to name category', default = 'cat_to_name.json', type = str)
    parser.add_argument('--gpu', dest = 'gpu',  help = 'want to use gpu or not', default = True, type = bool)
    parser.add_argument('--filepath', dest = 'filepath',  help = 'path of flower image', type = str)
    parser.add_argument('--checkpoint', dest = 'checkpoint',  help = 'saved model path', type = str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True

    model = load_model(args.checkpoint)  

    if args.gpu == True:
        model = model.cuda()
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    predict(args.filepath, model, cat_to_name, args.top_k, args.gpu)
    