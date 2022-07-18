#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.dataset import Focus, split_dataset


# In[3]:


categories = list(Focus.categories.keys())
locations = list(Focus.locations.keys())
focus_root = Path("/cmlscratch/pkattaki/datasets/focus")


# In[4]:

train_transform = T.Compose(
    [
        T.Resize(224),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transform = T.Compose(
    [
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# In[5]:


def train(model, classifiers, dataloader, optimizers):
    for images, _, time_labels, weather_labels, locations_labels in tqdm(
        dataloader, leave=False
    ):
        images, time_labels, weather_labels, locations_labels = (
            images.cuda(),
            time_labels.cuda(),
            weather_labels.cuda(),
            locations_labels.cuda(),
        )
        locations_labels = locations_labels / torch.sum(
            locations_labels, dim=1, keepdim=True
        )
        attributes = [time_labels, weather_labels, locations_labels]

        _ = model(images)
        for (_, classifier), (_, optimizer), attribute in zip(
            classifiers.items(), optimizers.items(), attributes
        ):
            optimizer.zero_grad()
            outputs = classifier(features["features"].squeeze())
            loss = F.cross_entropy(outputs, attribute)
            loss.backward()
            optimizer.step()


def test(model, classifiers, dataloader):

    num_corrects = {
        "time": 0,
        "weather": 0,
        "locations": 0,
    }
    confusion_matrices = {
        "time": np.zeros((3, 3)),
        "weather": np.zeros((7, 7)),
    }
    total = 0.0
    avg_loss = 0
    with torch.no_grad():
        for images, _, time_labels, weather_labels, locations_labels in dataloader:
            images, time_labels, weather_labels, locations_labels = (
                images.cuda(),
                time_labels.cuda(),
                weather_labels.cuda(),
                locations_labels.cuda(),
            )
            locations_labels = locations_labels / torch.sum(
                locations_labels, dim=1, keepdim=True
            )
            attributes = [time_labels, weather_labels, locations_labels]

            _ = model(images)
            total += images.shape[0]
            for (_, classifier), attribute, attribute_name in zip(
                classifiers.items(), attributes, num_corrects.keys()
            ):
                outputs = classifier(features["features"].squeeze())
                loss = F.cross_entropy(outputs, attribute)
                _, predicted = torch.max(outputs, dim=1)
                if attribute_name != "locations":
                    num_corrects[attribute_name] += (predicted == attribute).sum().item()
                    for i in range(images.shape[0]):
                        confusion_matrices[attribute_name][attribute[i]][predicted[i]] += 1
                else:
                    avg_loss += loss
        for attribute in num_corrects:
            num_corrects[attribute] /= total
    return num_corrects, avg_loss.cpu().item() / len(dataloader.dataset), confusion_matrices


# In[ ]:


locations_dataset = Focus(
    focus_root,
    categories=categories,
    times=None,
    weathers=None,
    locations=None,
    transform=train_transform,
)
print(len(locations_dataset))

train_set, test_set = split_dataset(locations_dataset, train_fraction=0.85)

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

train_loader = DataLoader(
    train_set, batch_size=32, num_workers=4, pin_memory=True, shuffle=True
)
test_loader = DataLoader(
    test_set, batch_size=32, num_workers=4, pin_memory=True, shuffle=False
)

num_epochs = 50
optimizers = {
    k: torch.optim.Adam(v.parameters(), weight_decay=5e-3) for k, v in attribute_classifiers.items()
}
test_accuracies = np.zeros(num_epochs)
for epoch in range(num_epochs):
    train(model, attribute_classifiers, train_loader, optimizers)

ckpt = {k: v.state_dict() for k, v in attribute_classifiers.items()}
torch.save(ckpt, "attribute_classifiers.pth")
num_corrects, location_loss, confusion_matrices = test(model, attribute_classifiers, test_loader)
print(f"Test loss: {num_corrects, location_loss}")
print(confusion_matrices)


# In[ ]:
