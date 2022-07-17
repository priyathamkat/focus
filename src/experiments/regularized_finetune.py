import argparse
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import torchvision
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from experiments.evaluate_model import categories, label_to_correct_idxes, test, generate_heatmap
from experiments.dataset import BGVarDataset, split_dataset

parser = argparse.ArgumentParser(
    description="Finetune a model on FOCUS"
)


parser.add_argument(
    "--bg_var_root",
    type=str,
    default="/cmlscratch/pkattaki/datasets/focus",
    help="Path to FOCUS",
)
parser.add_argument(
    "--imagenet",
    type=str,
    default="/cmlscratch/pkattaki/datasets/ILSVRC2012",
    help="Path to ImageNet",
)

parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--num_epochs", type=int, default=15, help="Number of epochs")
parser.add_argument("--alpha", type=float, help="regularization strength")

parser.add_argument(
    "--model_arch", type=str, default="resnet", help="Model architecture"
)

parser.add_argument(
    "--log_folder",
    type=str, 
    default="../../logs/regularized_finetuning",
    help="Path to the log folder",
)

if __name__ == "__main__":
    args = parser.parse_args()

    args.log_folder = Path(args.log_folder)

    locations = list(BGVarDataset.locations.keys())

    train_transform = T.Compose(
        [
            T.RandomResizedCrop(224),
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

    dataset = BGVarDataset(
        args.bg_var_root,
        categories=BGVarDataset.categories,
        times=None,
        weathers=None,
        locations=None,
        transform=test_transform,
    )

    # train_set, test_set = split_dataset(dataset, train_fraction=0.7)
    _, test_set = split_dataset(dataset, train_fraction=0.7)
    # train_dataloader = DataLoader(train_set, batch_size=args.batch_size, num_workers=1, pin_memory=True)
    train_set = torchvision.datasets.ImageNet(
        args.imagenet,
        split="val",
        transform=train_transform,
    )

    train_dataloader = D.DataLoader(
        train_set, shuffle=True, batch_size=32, num_workers=1, pin_memory=True
    )
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, num_workers=1, pin_memory=True)


    model = models.resnet50(pretrained=True)
    model.cuda()

    for name, parameter in model.named_parameters():
        if "fc" in name:
            parameter.requires_grad_(True)
        else:
            parameter.requires_grad_(False)

    location_classifiers = [torch.load(f"{l}.pth") for l in locations[:-1]]
    for lc in location_classifiers:
        for parameter in lc.parameters():
            parameter.requires_grad_(False)

    optimizer = optim.SGD(model.fc.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(f"../../logs/regularized_finetuning/{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}-{args.alpha}")

    # _, _, _, categories_matrix, _, _ = test(model, test_dataloader, args.batch_size, cocaus=True)

    # generate_heatmap(
    #     categories_matrix[0],
    #     categories_matrix[2],
    #     args.log_folder / "pre_on_bg_var_top1.pdf",
    #     xticks=["0", "1", "2", "3", "all"],
    #     yticks=categories + ["total"],
    # )

    # generate_heatmap(
    #     categories_matrix[1],
    #     categories_matrix[2],
    #     args.log_folder / "pre_on_bg_var_top5.pdf",
    #     xticks=["0", "1", "2", "3", "all"],
    #     yticks=categories + ["total"],
    # )

    step = 0
    for epoch in range(args.num_epochs):
        model.eval()
        with tqdm(train_dataloader, leave=False) as pbar:
            for images, labels in pbar:
                # print(images.shape, labels.shape, torch.min(labels), torch.max(labels))
                optimizer.zero_grad()
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                # print(outputs.shape)
                ce_loss = criterion(outputs, labels)
                # print(loss)
                writer.add_scalar("ce_loss", ce_loss, step)
                inner_products = 0
                for lc in location_classifiers:
                    inner_products = inner_products + torch.mean((model.fc.weight @ lc.weight.T) ** 2)
                writer.add_scalar("inner_products", inner_products, step)
                loss = ce_loss + args.alpha * inner_products
                writer.add_scalar("loss", loss, step)
                loss.backward()
                optimizer.step()
                step += 1 
                pbar.set_postfix(ce_loss=ce_loss.item(), inner_products=inner_products.item(), loss=loss.item())

    _, _, _, categories_matrix, _, _ = test(model, test_dataloader, args.batch_size, cocaus=True)

    generate_heatmap(
        categories_matrix[0],
        categories_matrix[2],
        args.log_folder / "post_on_bg_var_top1.pdf",
        xticks=["0", "1", "2", "3", "all"],
        yticks=categories + ["total"],
    )

    generate_heatmap(
        categories_matrix[1],
        categories_matrix[2],
        args.log_folder / "post_on_bg_var_top5.pdf",
        xticks=["0", "1", "2", "3", "all"],
        yticks=categories + ["total"],
    )
    
    torch.save({
        "model_state_dict": model.state_dict(),
    }, args.log_folder / "ckpt.pth", )

    writer.close()
    



