import torch
from torchvision import datasets,models
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
from collections import OrderedDict
import PIL
import argparse
import warnings
import io


import json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

predict_on_gpu = torch.cuda.is_available()
warnings.filterwarnings("ignore")
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_compose = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    image = io.BytesIO(image)
    pil_img = PIL.Image.open(image)
    img = img_compose(pil_img)
    
    
    return img

# def imshow(image, ax=None, title=None):
#     """Imshow for Tensor."""
#     if ax is None:
#         fig, ax = plt.subplots()
    
#     # PyTorch tensors assume the color channel is the first dimension
#     # but matplotlib assumes is the third dimension
#     image = image.transpose((1, 2, 0))
    
#     # Undo preprocessing
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     image = std * image + mean
    
#     # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
#     image = np.clip(image, 0, 1)
    
#     ax.imshow(image)
    
#     return ax


def load_model():
    loaded_model = 'model_flowerComplete.pt'
    
    if not predict_on_gpu:
        state = torch.load(loaded_model,map_location='cpu')
    else:
        state = torch.load(loaded_model)

    hidden_size = 1024

    model = models.resnet101(pretrained=True)  

    n_features = model.fc.in_features
    classifier = nn.Sequential(OrderedDict([('fc1',nn.Linear(n_features,hidden_size)),
                                            ('relu',nn.ReLU()),
                                            ('dropout',nn.Dropout(p=0.3)),
                                            ('fc2',nn.Linear(hidden_size,hidden_size)),
                                            ('relu',nn.ReLU()),
                                            ('dropout',nn.Dropout(p=0.3)),
                                            ('fc3',nn.Linear(hidden_size,102)),
                                            ('output', nn.LogSoftmax(dim=1))]))
    model.fc = classifier
    model.load_state_dict(state['state_dict'])
    
    

    model.class_to_idx = state['class_to_idx']

    return model

def predict(img):
    
    # if args.gpu and predict_on_gpu:
    #     model = model.cuda()
    #     print("Predict using GPU...")
    # else:
    #     print("Predict using CPU, this will take a while")
    model = load_model()   

    model.eval()

    image = process_image(img)

    image = image.unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        top_probs, top_labels =  output.topk(1)
        top_probs = top_probs.exp()
    
    class_to_idx_inv = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mapped_classes = list()
    
    for label in top_labels.numpy()[0]:
        mapped_classes.append(class_to_idx_inv[label])
        
    return top_probs.numpy()[0], cat_to_name[mapped_classes[0]]

# def show_prediction(image,top_probs,top_classes,cat_to_name,topk):
#     image = PIL.Image.open(image)
#     label = top_classes[0]

#     fig = plt.figure(figsize=(6,6))
#     subplot_img = plt.subplot2grid((15,9), (0,0), colspan=9, rowspan=9)
#     subplot_preds = plt.subplot2grid((15,9), (9,2), colspan=5, rowspan=5)

#     subplot_img.axis('off')
#     subplot_img.set_title('{}'.format(cat_to_name[label]))

#     subplot_img.imshow(image)

#     labels = []
#     for class_idx in top_classes:
#         labels.append(cat_to_name[class_idx])

#     yp = np.arange(topk)
#     subplot_preds.set_yticks(yp)
#     subplot_preds.set_yticklabels(labels)
#     subplot_preds.set_xlabel('Probability')
#     subplot_preds.invert_yaxis()
#     subplot_preds.barh(yp, top_probs, xerr=0, align='center', color='blue')

#     plt.show()

# def predict_flower(img):
#     # parser = argparse.ArgumentParser(description='Flower Classification Predict')
#     # parser.add_argument('--loaded_model',type = str,default='model_flowerComplete.pt',
#     #                     help='Path of saved model')
#     # parser.add_argument('--image_path', type = str, help='Path of image to predict')
#     # parser.add_argument('--hidden_size', type=int, default=1024, 
#     #                     help='Size of hidden unit (default 100)')
#     # parser.add_argument('--map_json', type=str, default='cat_to_name.json', 
#     #                     help='Mapper of category to name in json extension')
#     # parser.add_argument('--gpu',type=str,default=False,help='Use GPU or not?')
#     # parser.add_argument('--topk',type=int,default=5, help='top K probabilities')

#     # args = parser.parse_args()


    
#     model = load_model()

#     top_probs,top_class = predict(model,cat_to_name)

#     # show_prediction(args.image_path,top_probs,top_class,cat_to_name,args.topk)

# if __name__ == "__main__":
#     main()