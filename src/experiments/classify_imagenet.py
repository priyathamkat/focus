from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as D
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv

np.set_printoptions(precision=4, linewidth=150)

test_transform = T.Compose(
    [
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def classify(model, classifiers, dataloader, dataset):
    aggregates = {
        "time": np.zeros(3),
        "weather": np.zeros(7),
        "location": np.zeros(9),
    }
    with torch.no_grad():
        with open("val-imagenet-attributes.csv", "w", newline="\n") as csvfile:
            field_names = ["image", "time", "weather", "location"]
            csvwriter = csv.DictWriter(csvfile, fieldnames=field_names)
            csvwriter.writeheader()
            idx = 0
            for images, _ in tqdm(dataloader):
                images = images.cuda()

                _ = model(images)
                predicted = {}
                for attribute_name, classifier in classifiers.items():
                    outputs = classifier(features["features"].squeeze())
                    
                    if attribute_name != "location":
                        _, predicted[attribute_name] = torch.max(outputs, dim=1)
                    else:
                        predicted[attribute_name] = F.softmax(outputs, dim=1)
                for time, weather, location in zip(predicted["time"], predicted["weather"], predicted["location"]):
                    time = time.cpu().item()
                    weather = weather.cpu().item()
                    location = location.cpu().numpy()
                    csvwriter.writerow({"image": dataset.imgs[idx][0], "time": time, "weather": weather, "location": str(location)})
                    idx += 1
                    aggregates["time"][time] += 1
                    aggregates["weather"][weather] += 1
                    aggregates["location"] += location
    print(aggregates)



features = {}


def hook(model, input, output):
    features["features"] = output


model = models.resnet50(pretrained=True)
for parameter in model.parameters():
    parameter.requires_grad_(False)

attribute_classifiers = {
    "time": nn.Linear(2048, 3),
    "weather": nn.Linear(2048, 7),
    "location": nn.Linear(2048, 9),
}

for _, classifier in attribute_classifiers.items():
    classifier.cuda()

model.eval()
model.cuda()
model.avgpool.register_forward_hook(hook)

imagenet = D.ImageNet("/cmlscratch/pkattaki/datasets/ILSVRC2012/", split="val", transform=test_transform)
dataloader = DataLoader(
        imagenet,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
)

torch.backends.cudnn.benchmark = True
ckpt = torch.load("attribute_classifiers-0.pth")
for k, v in attribute_classifiers.items():
    v.load_state_dict(ckpt[k])
classify(model, attribute_classifiers, dataloader, imagenet)