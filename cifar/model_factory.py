"""
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): ReLU(inplace=True)
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace=True)
    (8): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace=True)
    (10): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (11): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (12): ReLU(inplace=True)
    (13): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (14): ReLU(inplace=True)
    (15): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (16): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (17): ReLU(inplace=True)
    (18): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (19): ReLU(inplace=True)
    (20): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
"""
from itertools import chain

import torch as t


def vgg_tune_2(n_classes):
    model = t.hub.load("pytorch/vision", "vgg11", pretrained=True)

    # First freeze all the parameters.
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze parameters of classifier[3], which is the second last linear layer.
    for param in model.classifier[3].parameters():
        param.requires_grad = True

    # Replace the last linear layer. This will automatically unfreeze it.
    in_features = model.classifier[6].in_features
    model.classifier[6] = t.nn.Linear(in_features, n_classes)
    tunable_params = chain(model.classifier[3].parameters(), model.classifier[6].parameters())

    return model, tunable_params


def vgg_tune_1(n_classes):
    model = t.hub.load("pytorch/vision", "vgg11", pretrained=True)

    # First freeze all the parameters.
    for param in model.parameters():
        param.requires_grad = False

    # Replace the last linear layer. This will automatically unfreeze it.
    in_features = model.classifier[6].in_features
    model.classifier[6] = t.nn.Linear(in_features, n_classes)

    return model, model.classifier[6].parameters()


def vgg_tune_classifier(n_classes):
    model = t.hub.load("pytorch/vision", "vgg11", pretrained=True)

    # Freeze features layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the last linear layer.
    in_features = model.classifier[6].in_features
    model.classifier[6] = t.nn.Linear(in_features, n_classes)

    return model, model.classifier.parameters()


def build_cifar_model(model_type):
    if model_type == "tune-1":
        return vgg_tune_1(10)
    elif model_type == "tune-2":
        return vgg_tune_2(10)
    elif model_type == "tune-classifier":
        return vgg_tune_classifier(10)
    else:
        raise ValueError(f"Unknown model type {model_type}")
