import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import torchvision
import torchvision.models as models
import torchvision.transforms as T
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Tests a model on ImageNet 2012 Validation set"
)

parser.add_argument(
    "--checkpoint", type=str, default="", help="Path to model checkpoint"
)

parser.add_argument(
    "--imagenet",
    type=str,
    default="/cmlscratch/pkattaki/datasets/ILSVRC2012",
    help="Path to ImageNet",
)

if __name__ == "__main__":
    args = parser.parse_args()

    model = models.resnet50(pretrained=True)
    model.cuda()
    if args.checkpoint != "":
        ckpt = torch.load(args.checkpoint)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_transform = T.Compose(
        [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = torchvision.datasets.ImageNet(
        args.imagenet,
        split="val",
        transform=test_transform,
    )

    test_dataloader = D.DataLoader(
        dataset, batch_size=32, num_workers=1, pin_memory=True
    )

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, leave=False):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += images.shape[0]
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total:.2f}")
