import argparse
from pathlib import Path

from evaluate_model import CLIP
import timm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

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

parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")

parser.add_argument(
    "--model_arch", type=str, default="resnet", help="Model architecture"
)

parser.add_argument(
    "--log_folder",
    type=str, 
    default="../../logs/finetune",
    help="Path to the log folder",
)

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args = parser.parse_args()

    args.log_folder = Path(args.log_folder)

    if args.model_arch == "efficientnet-b4":
        input_size = 380
    elif args.model_arch == "efficientnet-b7":
        input_size = 600
    else:
        input_size = 224
    train_transform = T.Compose(
        [
            T.RandomResizedCrop(input_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    
    test_transform = T.Compose(
        [
            T.Resize(input_size),
            T.CenterCrop(input_size),
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

    train_set, test_set = split_dataset(dataset, train_fraction=0.7)
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, num_workers=4, pin_memory=True)


    if args.model_arch == "resnet":
        model = models.resnet50(pretrained=True)
        model.cuda()
        optimizer = optim.SGD(model.fc.parameters(), lr=args.lr, momentum=0.9)
    elif args.model_arch == "mobilenet":
        model = models.mobilenet_v3_large(pretrained=True)
        model.cuda()
        optimizer = optim.SGD(model.classifier.parameters(), lr=args.lr, momentum=0.9)
    elif args.model_arch == "wide_resnet":
        model = models.wide_resnet50_2(pretrained=True)
        model.cuda()
        optimizer = optim.SGD(model.fc.parameters(), lr=args.lr, momentum=0.9)
    elif args.model_arch == "efficientnet-b4":
        from efficientnet_pytorch import EfficientNet

        model = EfficientNet.from_pretrained("efficientnet-b4")
        model.cuda()
        for name, parameter in model.named_parameters():
            if "_fc" in name:
                parameter.requires_grad_(True)
            else:
                parameter.requires_grad_(False)
        optimizer = optim.SGD(model._fc.parameters(), lr=args.lr, momentum=0.9)
    elif args.model_arch == "efficientnet-b7":
        from efficientnet_pytorch import EfficientNet

        model = EfficientNet.from_pretrained("efficientnet-b7")
        model.cuda()
        for name, parameter in model.named_parameters():
            if "_fc" in name:
                parameter.requires_grad_(True)
            else:
                parameter.requires_grad_(False)
        optimizer = optim.SGD(model._fc.parameters(), lr=args.lr, momentum=0.9)
    elif args.model_arch == "clip":
        model = CLIP()
        test_transform = model.preprocess
    elif args.model_arch == "vit":
        model = timm.create_model("vit_base_patch16_224", pretrained=True)
        model.cuda()
        for name, parameter in model.named_parameters():
            if "head" in name:
                parameter.requires_grad_(True)
            else:
                parameter.requires_grad_(False)
        optimizer = optim.SGD(model.head.parameters(), lr=args.lr, momentum=0.9)
    elif args.model_arch == "resnext":
        model = timm.create_model("resnext50_32x4d", pretrained=True)
        model.cuda()
        for name, parameter in model.named_parameters():
            if "fc" in name:
                parameter.requires_grad_(True)
            else:
                parameter.requires_grad_(False)
        optimizer = optim.SGD(model.fc.parameters(), lr=args.lr, momentum=0.9)
    else:
        raise ValueError("Unknown model architecture")

    criterion = nn.NLLLoss()

    _, _, _, categories_matrix, _, _ = test(model, test_dataloader, args.batch_size, cocaus=True)

    generate_heatmap(
        categories_matrix[0],
        categories_matrix[2],
        args.log_folder / "pre_on_bg_var_top1.pdf",
        xticks=["0", "1", "2", "3", "all"],
        yticks=categories + ["total"],
    )

    generate_heatmap(
        categories_matrix[1],
        categories_matrix[2],
        args.log_folder / "pre_on_bg_var_top5.pdf",
        xticks=["0", "1", "2", "3", "all"],
        yticks=categories + ["total"],
    )

    for epoch in range(args.num_epochs):
        model.train()
        for images, labels, _, _, _ in tqdm(train_dataloader, leave=False):
            optimizer.zero_grad()
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            outputs = F.softmax(outputs, dim=1)
            collated_logprobs = torch.zeros(images.shape[0], len(BGVarDataset.categories), device=torch.device("cuda"))
            for idx in range(len(BGVarDataset.categories)):
                collated_logprobs[:, idx] = torch.log(torch.sum(outputs[:, list(label_to_correct_idxes[idx])], 1))
            loss = criterion(collated_logprobs, labels)
            loss.backward()
            optimizer.step()

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


    



