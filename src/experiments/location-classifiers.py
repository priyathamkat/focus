#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from experiments.dataset import Focus


# In[3]:


categories = list(Focus.categories.keys())
locations = list(Focus.locations.keys())
focus_root = Path("/cmlscratch/pkattaki/datasets/focus")
full_dataset = Focus(
    focus_root,
    categories=categories,
    times=None,
    weathers=None,
    locations=None,
    transform=None
)
all_images = {f[0][1:] for f in full_dataset.image_files}


# In[4]:


class AttDataset(Dataset):
    def __init__(self, image_list, transform=None, target_transform=None):
        super().__init__()
        self._image_list = image_list
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self._image_list)
    
    def __getitem__(self, idx):
        image_path, label = self._image_list[idx]
        image_path = focus_root / image_path
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

test_transform = T.Compose(
    [
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# In[5]:


def train(model, attribute_classifier, dataloader, optimizer):
    for images, labels in dataloader:
        images, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()
        _ = model(images)
        outputs = attribute_classifier(features["features"].squeeze())
        loss = F.binary_cross_entropy_with_logits(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        
def test(model, attribute_classifier, dataloader):

    num_correct = total = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.cuda(), labels.cuda()

            _ = model(images)
            outputs = attribute_classifier(features["features"].squeeze())
            predictions = (outputs >= 0).type(torch.cuda.FloatTensor)
            for label, prediction in zip(labels, predictions):
                total += 1
                num_correct += (label == prediction).cpu().item()
    return num_correct / total


# In[ ]:


for idx in range(len(locations)):
    attribute_label = locations[idx]
    attribute_dataset = Focus(
        focus_root,
        categories=categories,
        times=None,
        weathers=None,
        locations=[attribute_label],
        transform=None
    )
    attribute_images = {f[0][1:] for f in attribute_dataset.image_files}
    non_attribute_images = all_images.difference(attribute_images)

    attribute_images = list(attribute_images)
    non_attribute_images = list(non_attribute_images)

    num_attribute_images = len(attribute_images)
    num_non_attribute_images = len(non_attribute_images)
    
    test_size = 2 * int(0.3 * min(num_attribute_images, num_non_attribute_images))
    train_size = 2 * min(num_attribute_images, num_non_attribute_images) - test_size
    train_images = [(f, 0.0) for f in non_attribute_images[:train_size // 2]] 
    train_images = train_images + [(f, 1.0) for f in attribute_images[:train_size // 2]]
    test_images = [(f, 0.0) for f in non_attribute_images[train_size // 2:train_size // 2 + test_size]]
    test_images = test_images + [(f, 1.0) for f in attribute_images[train_size // 2:train_size // 2 + test_size]]
    
    train_set = AttDataset(train_images, transform=test_transform)
    test_set = AttDataset(test_images, transform=test_transform)
    
    features = {}

    def hook(model, input, output):
        features["features"] = output

    model = models.resnet50(pretrained=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    attribute_classifier = nn.Linear(2048, 1)
    attribute_classifier.cuda()

    model.eval()
    model.cuda()
    model.avgpool.register_forward_hook(hook)
    
    train_loader = DataLoader(train_set, batch_size=32, num_workers=1, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, num_workers=1, pin_memory=True, shuffle=False)
    
    num_epochs = 30
    optimizer = torch.optim.SGD(attribute_classifier.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-2)
    test_accuracies = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        train(model, attribute_classifier, train_loader, optimizer)
        
    torch.save(attribute_classifier, f"{attribute_label}.pth")
    print(f"{attribute_label}: {test(model, attribute_classifier, test_loader):.3f}")



# In[ ]:




