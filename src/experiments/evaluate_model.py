import argparse
from pathlib import Path

import clip
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import timm
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.dataset import DCR, BGVarDataset

parser = argparse.ArgumentParser(
    description="Evaluate a model on a subset of ImageNet and BG-Var"
)

parser.add_argument(
    "--dcr_root",
    type=str,
    default="/cmlscratch/yogesh22/data/DCR/DCRImagenet",
    help="Path to ImageNet",
)
parser.add_argument(
    "--bg_var_root",
    type=str,
    default="/cmlscratch/pkattaki/datasets/focus",
    help="Path to ImageNet",
)

parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument(
    "--model_arch", type=str, default="resnet", help="Model architecture"
)
parser.add_argument(
    "--model_checkpoint", type=str, default="", help="Path to the model checkpoint"
)
parser.add_argument(
    "--log_folder",
    type=str,
    default="logs/pretrained_dcr",
    help="Path to the log folder",
)

categories = list(BGVarDataset.categories.keys())
times = list(BGVarDataset.times.keys())
weathers = list(BGVarDataset.weathers.keys())
locations = list(BGVarDataset.locations.keys())

uncommon = {
    0: {  # truck
        "time": {1},
        "weather": {1, 3, 4},
        "locations": {2, 3, 6, 7},
    },
    1: {  # car
        "time": {1},
        "weather": {1, 3, 4},
        "locations": {2, 3, 6, 7},
    },
    2: {  # plane
        "time": {1},
        "weather": {1, 3, 4},
        "locations": {0, 2, 3, 4, 6, 7},
    },
    3: {  # ship
        "time": {1},
        "weather": {1, 3, 4},
        "locations": {0, 1, 2, 3, 4, 5, 6},
    },
    4: {  # cat
        "time": {1},
        "weather": {1, 3, 4},
        "locations": {0, 3, 4, 6, 7},
    },
    5: {  # dog
        "time": {1},
        "weather": {1, 3, 4},
        "locations": {0, 3, 6},
    },
    6: {  # horse
        "time": {1},
        "weather": {1, 3, 4},
        "locations": {2, 3, 5, 6, 7},
    },
    7: {  # deer
        "time": {1},
        "weather": {1, 3, 4},
        "locations": {2, 3, 4, 5, 6, 7},
    },
    8: {  # frog
        "time": {},
        "weather": {1, 3, 4},
        "locations": {2, 5, 6},
    },
    9: {  # bird
        "time": {1},
        "weather": {1, 3, 4},
        "locations": {2, 5, 6},
    },
}


def make_uncommon_masks(mask, attribute):
    for category in range(len(categories)):
        mask[category, list(uncommon[category][attribute])] = 1

    return mask


label_to_correct_idxes = {
    0: {  # truck
        407,
        555,
        569,
        654,
        675,
        717,
        757,
        779,
        864,
        867,
    },
    1: {  # car
        436,
        468,
        511,
        609,
        627,
        656,
        705,
        734,
        751,
        817,
    },
    2: {  # plane
        404,
        895,
    },
    3: {403, 472, 510, 554, 625, 628, 693, 724, 780, 814, 914},  # ship
    4: {281, 282, 283, 284, 285},  # cat
    5: {  # dog
        151,
        152,
        153,
        154,
        155,
        156,
        157,
        158,
        159,
        160,
        161,
        162,
        163,
        164,
        165,
        166,
        167,
        168,
        169,
        170,
        171,
        172,
        173,
        174,
        175,
        176,
        177,
        178,
        179,
        180,
        181,
        182,
        183,
        184,
        185,
        186,
        187,
        188,
        189,
        190,
        191,
        192,
        193,
        194,
        195,
        196,
        197,
        198,
        199,
        200,
        201,
        202,
        203,
        204,
        205,
        206,
        207,
        208,
        209,
        210,
        211,
        212,
        213,
        214,
        215,
        216,
        217,
        218,
        219,
        220,
        221,
        222,
        223,
        224,
        225,
        226,
        227,
        228,
        229,
        230,
        231,
        232,
        233,
        234,
        235,
        236,
        237,
        238,
        239,
        240,
        241,
        242,
        243,
        244,
        245,
        246,
        247,
        248,
        249,
        250,
        251,
        252,
        253,
        254,
        255,
        256,
        257,
        258,
        259,
        260,
        261,
        262,
        263,
        264,
        265,
        266,
        267,
        268,
        273,
        274,
        275,
    },
    6: {  # "horse"
        339,
        340,
    },
    7: {  # "deer"
        351,
        352,
        353,
    },
    8: {  # "frog"
        30,
        31,
        32,
    },
    9: {  # "bird"
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        127,
        128,
        129,
        130,
        131,
        132,
        133,
        134,
        135,
        136,
        137,
        138,
        139,
        140,
        141,
        142,
        143,
        144,
        145,
        146,
    },
}


def check_label(top5_prediction, ground_truth, use_strict=False):
    """Returns "is top1 correct?", "is top5 correct?"""
    if use_strict:
        return top5_prediction[0] == ground_truth, ground_truth in top5_prediction
    else:
        return (
            top5_prediction[0].item() in label_to_correct_idxes[ground_truth.item()],
            not set(top5_prediction.tolist()).isdisjoint(
                label_to_correct_idxes[ground_truth.item()]
            ),
        )


def is_time_uncommon(category_label, time_label):
    uncommon_settings = uncommon[category_label.item()]
    return time_label.item() in uncommon_settings["time"]


def is_weather_uncommon(category_label, weather_label):
    uncommon_settings = uncommon[category_label.item()]
    return weather_label.item() in uncommon_settings["weather"]


def is_locations_uncommon(category_label, locations_label):
    uncommon_settings = uncommon[category_label.item()]
    return not set(np.nonzero(locations_label.tolist())[0]).isdisjoint(
        uncommon_settings["locations"]
    )


def count_uncommon_attributes(
    category_label, time_label, weather_label, locations_label
):
    return sum(
        [
            is_time_uncommon(category_label, time_label),
            is_weather_uncommon(category_label, weather_label),
            is_locations_uncommon(category_label, locations_label),
        ]
    )


def test(model, dataloader, batch_size, cocaus=False):
    model.eval()

    if cocaus:
        categories_matrix = np.zeros(
            (3, len(categories), 4)
        )  # last dim is for number of uncommon, 0 is common
        # In the first dim, 0 -> top1, 1 -> top5, 2 -> total
        times_matrix = np.zeros((3, len(categories), len(times)))
        weathers_matrix = np.zeros((3, len(categories), len(weathers)))
        locations_matrix = np.zeros((3, len(categories), len(locations)))
        prediction_array = np.zeros((len(dataloader.dataset), 15), dtype=np.float64)
        incorrect_samples = []
    else:
        correct_predictions = np.zeros((2, len(categories)), dtype=np.float64)
        total_predictions = np.zeros(len(categories), dtype=np.float64)

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc=f"Testing: ", leave=False)
        ):
            if cocaus:
                (
                    images,
                    category_labels,
                    time_labels,
                    weather_labels,
                    locations_labels,
                ) = batch
            else:
                images, category_labels = batch
            images = images.cuda()

            outputs = model(images)
            _, predictions = torch.sort(outputs, dim=1, descending=True)
            top5_predictions = predictions[:, :5].cpu()

            if cocaus:
                for (
                    idx_in_batch,
                    category_label,
                    time_label,
                    weather_label,
                    locations_label,
                    top5_prediction,
                ) in zip(
                    range(batch_size),
                    category_labels,
                    time_labels,
                    weather_labels,
                    locations_labels,
                    top5_predictions,
                ):
                    is_top1_correct, is_top5_correct = check_label(
                        top5_prediction, category_label
                    )

                    num_uncommon_attributes = count_uncommon_attributes(
                        category_label, time_label, weather_label, locations_label
                    )

                    row_idx = batch_idx * batch_size + idx_in_batch
                    prediction_array[row_idx, category_label] = 1
                    prediction_array[row_idx, 10] = is_time_uncommon(
                        category_label, time_label
                    )
                    prediction_array[row_idx, 11] = is_weather_uncommon(
                        category_label, weather_label
                    )
                    prediction_array[row_idx, 12] = is_locations_uncommon(
                        category_label, locations_label
                    )
                    if is_top1_correct:
                        times_matrix[0, category_label, time_label] += 1
                        weathers_matrix[0, category_label, weather_label] += 1
                        locations_matrix[
                            0, category_label, np.nonzero(locations_label)
                        ] += 1
                        categories_matrix[
                            0, category_label, num_uncommon_attributes
                        ] += 1

                        prediction_array[row_idx, 13] = 1
                    else:
                        incorrect_samples.append(
                            batch_idx * batch_size + idx_in_batch
                        )
                    if is_top5_correct:
                        times_matrix[1, category_label, time_label] += 1
                        weathers_matrix[1, category_label, weather_label] += 1
                        locations_matrix[
                            1, category_label, np.nonzero(locations_label)
                        ] += 1
                        categories_matrix[
                            1, category_label, num_uncommon_attributes
                        ] += 1

                        prediction_array[row_idx, 14] = 1
                    times_matrix[2, category_label, time_label] += 1
                    weathers_matrix[2, category_label, weather_label] += 1
                    locations_matrix[
                        2, category_label, np.nonzero(locations_label)
                    ] += 1
                    categories_matrix[2, category_label, num_uncommon_attributes] += 1
            else:
                for category_label, top5_prediction in zip(
                    category_labels, top5_predictions
                ):
                    is_top1_correct, is_top5_correct = check_label(
                        top5_prediction, category_label
                    )
                    if is_top1_correct:
                        correct_predictions[0, category_label] += 1
                    if is_top5_correct:
                        correct_predictions[1, category_label] += 1
                    total_predictions[category_label] += 1
        if cocaus:
            return (
                times_matrix,
                weathers_matrix,
                locations_matrix,
                categories_matrix,
                prediction_array,
                incorrect_samples,
            )
        else:
            return correct_predictions, total_predictions


def add_cumulatives(mat, add_across_columns=False):
    if mat.ndim == 1:
        mat = np.expand_dims(mat, axis=1)
    extended_mat = np.vstack([mat, np.sum(mat, axis=0, keepdims=True)])
    if add_across_columns:
        extended_mat = np.hstack(
            [extended_mat, np.sum(extended_mat, axis=1, keepdims=True)]
        )
    return extended_mat


def generate_heatmap(
    correct_predictions, total_predictions, save_path, xticks=False, yticks=False
):
    correct_predictions = add_cumulatives(correct_predictions, add_across_columns=True)
    total_predictions = add_cumulatives(total_predictions, add_across_columns=True)

    heatmap = sns.heatmap(
        correct_predictions / (total_predictions + 1e-6),
        vmin=0,
        vmax=1,
        cmap="viridis",
        annot=True,
        linewidths=0.5,
        xticklabels=xticks,
        yticklabels=yticks,
        mask=(total_predictions == 0),
    )
    heatmap.vlines([4], *heatmap.get_ylim(), colors=["white"], lw=10)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
    heatmap.get_figure().savefig(save_path)
    plt.close()


class CLIP(nn.Module):
    # Adapted from https://github.com/openai/CLIP
    def __init__(self):
        super().__init__()

        device = "cuda"
        self._clip_model, self.preprocess = clip.load("ViT-B/32", device=device)

        from imagenet_labels import imagenet_labels

        imagenet_labels = [v.split(", ")[0] for v in imagenet_labels.values()]
        self._text_inputs = torch.cat(
            [clip.tokenize(f"a photo of a {c}") for c in imagenet_labels]
        ).to(device)

    def forward(self, x):
        logits_per_image, _ = self._clip_model(x, self._text_inputs)
        return logits_per_image


def generate_gap_heatmap(
    correct_predictions,
    total_predictions,
    mask,
    save_path,
    xticks=False,
    yticks=False,
):
    correct_predictions = add_cumulatives(correct_predictions)
    total_predictions = add_cumulatives(total_predictions)
    accuracies = np.hstack(
        (
            correct_predictions / (total_predictions + 1e-6),
            np.zeros((len(categories) + 1, 1)),
        )
    )
    common_accuracies_mask = np.zeros_like(accuracies, dtype=bool)
    common_accuracies_mask[:-1, :-1] = mask == 1
    common_accuracies_mask[:, -1] = True
    common_accuracies_mask[-1, :] = True

    uncommon_accuracies_mask = np.zeros_like(accuracies, dtype=bool)
    uncommon_accuracies_mask[:-1, :-1] = mask == 0
    uncommon_accuracies_mask[:, -1] = True
    uncommon_accuracies_mask[-1, :] = True
    gap = np.zeros_like(accuracies)
    gap_mask = gap == 0
    gap_mask[:, -1] = False
    gap_mask[-1, :] = False
    gap[:-1, -1] = np.sum(correct_predictions[:-1, :] * (1 - mask), axis=1) / (
        1e-6 + np.sum(total_predictions[:-1, :] * (1 - mask), axis=1)
    ) - np.sum(correct_predictions[:-1, :] * mask, axis=1) / (
        1e-6 + np.sum(total_predictions[:-1, :] * mask, axis=1)
    )
    gap[-1, :-1] = (correct_predictions[-1]) / (total_predictions[-1])
    gap[-1, -1] = np.sum(correct_predictions[:-1, :] * (1 - mask)) / (
        1e-6 + np.sum(total_predictions[:-1, :] * (1 - mask))
    ) - np.sum(correct_predictions[:-1, :] * mask) / (
        1e-6 + np.sum(total_predictions[:-1, :] * mask)
    )
    sns.heatmap(
        accuracies,
        vmin=0,
        vmax=1,
        cmap="Blues",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        xticklabels=False,
        yticklabels=False,
        mask=common_accuracies_mask,
        cbar=False,
    )
    sns.heatmap(
        accuracies,
        vmin=0,
        vmax=1,
        cmap="Oranges",
        fmt=".2f",
        linewidths=0.5,
        annot=True,
        xticklabels=False,
        yticklabels=False,
        mask=uncommon_accuracies_mask,
        cbar=False,
    )
    heatmap = sns.heatmap(
        gap,
        center=0,
        cmap=sns.diverging_palette(145, 300, s=50, as_cmap=True),
        fmt=".2f",
        linewidths=0.5,
        annot=True,
        xticklabels=xticks,
        yticklabels=yticks,
        mask=gap_mask,
        cbar=False,
    )
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
    heatmap.get_figure().savefig(save_path)
    plt.close()


if __name__ == "__main__":
    args = parser.parse_args()

    args.log_folder = Path(args.log_folder)

    if args.model_arch == "efficientnet-b4":
        input_size = 380
    elif args.model_arch == "efficientnet-b7":
        input_size = 600
    else:
        input_size = 224
    
    test_transform = T.Compose(
        [
            T.Resize(input_size),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if args.model_arch == "resnet":
        if args.model_checkpoint:
            model = models.resnet50()
            model.cuda()
            ckpt = torch.load(args.model_checkpoint)
            # modified_ckpt = {}
            # for key in ckpt["state_dict"]:
            #     modified_ckpt[".".join((key.split(".")[1:]))] = ckpt["state_dict"][key]
            # model.load_state_dict(modified_ckpt)
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model = models.resnet50(pretrained=True)
            model.cuda()
    elif args.model_arch == "mobilenet":
        model = models.mobilenet_v3_large(pretrained=True)
        model.cuda()
    elif args.model_arch == "wide_resnet":
        model = models.wide_resnet50_2(pretrained=True)
        model.cuda()
    elif args.model_arch == "efficientnet-b4":
        from efficientnet_pytorch import EfficientNet

        model = EfficientNet.from_pretrained("efficientnet-b4")
        model.cuda()
    elif args.model_arch == "efficientnet-b7":
        from efficientnet_pytorch import EfficientNet

        model = EfficientNet.from_pretrained("efficientnet-b7")
        model.cuda()
    elif args.model_arch == "clip":
        model = CLIP()
        test_transform = model.preprocess
    elif args.model_arch == "vit":
        model = timm.create_model("vit_base_patch16_224", pretrained=True)
        model.cuda()
    elif args.model_arch == "resnext":
        model = timm.create_model("resnext50_32x4d", pretrained=True)
        model.cuda()
    else:
        raise ValueError("Unknown model architecture")

    bg_var_test_dataset = BGVarDataset(
        args.bg_var_root,
        categories=BGVarDataset.categories,
        times=None,
        weathers=None,
        locations=None,
        transform=test_transform,
    )

    bg_var_test_dataloader = DataLoader(
        bg_var_test_dataset,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
    )

    (
        times_matrix,
        weathers_matrix,
        locations_matrix,
        categories_matrix,
        prediction_array,
        incorrect_samples,
    ) = test(model, bg_var_test_dataloader, args.batch_size, cocaus=True)

    uncommon_times_mask = make_uncommon_masks(
        np.zeros((len(categories), len(times) - 1)), "time"
    )
    uncommon_weathers_mask = make_uncommon_masks(
        np.zeros((len(categories), len(weathers) - 1)), "weather"
    )
    uncommon_locations_mask = make_uncommon_masks(
        np.zeros((len(categories), len(locations) - 1)), "locations"
    )

    generate_heatmap(
        categories_matrix[0],
        categories_matrix[2],
        args.log_folder / "on_bg_var_top1.pdf",
        xticks=["0", "1", "2", "3", "all"],
        yticks=categories + ["total"],
    )

    generate_heatmap(
        categories_matrix[1],
        categories_matrix[2],
        args.log_folder / "on_bg_var_top5.pdf",
        xticks=["0", "1", "2", "3", "all"],
        yticks=categories + ["total"],
    )

    generate_gap_heatmap(
        times_matrix[0][:, :-1],
        times_matrix[2][:, :-1],
        uncommon_times_mask,
        args.log_folder / "category-vs-time.pdf",
        xticks=times[:-1] + ["gap"],
        yticks=categories + ["total"],
    )

    generate_gap_heatmap(
        weathers_matrix[0][:, :-1],
        weathers_matrix[2][:, :-1],
        uncommon_weathers_mask,
        args.log_folder / "category-vs-weather.pdf",
        xticks=weathers[0:2] + ["partly\ncloudy"] + weathers[3:-1] + ["gap"],
        yticks=categories + ["total"],
    )

    generate_gap_heatmap(
        locations_matrix[0][:, :-1],
        locations_matrix[2][:, :-1],
        uncommon_locations_mask,
        args.log_folder / "category-vs-location.pdf",
        xticks=locations[:-1] + ["gap"],
        yticks=categories + ["total"],
    )

    with open(args.log_folder / "evaluation.txt", "w") as f:
        f.write(f"Total number of images: {len(bg_var_test_dataset)}\n")
        f.write(f"Incorrect samples:\n {incorrect_samples}")
    np.save(args.log_folder / "predictions.npy", prediction_array)
    np.save(args.log_folder / "categories.npy", categories_matrix)
