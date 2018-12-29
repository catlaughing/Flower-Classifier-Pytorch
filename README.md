# Flower-Classifier-Pytorch

## Description
Flower Classification using Pytorch and ResNet101 Architecture

## Requirement
Forgot to make that requirement.txt so in the meantime i will list it here:
* Numpy
* Matplotlib
* Torch 0.4.1
* Torchvision
* Argparse
* Pillow 5.0.0

## How to use it
1. if you want to use without training, download the already trained model [here](http://gg.gg/flower_classifier_model)
2. install the requirement
3. clone/download this repo

### Predict a image with GPU and taking the top 3 probability
```
python3 predict.py --image_path "image_path" --gpu True --topk 3
```
### Train a model with your own dataset
Coming Soon !
