import json
from pathlib import Path
from unicodedata import category

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, random_split

from experiments.database import BGVarDB

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Focus(Dataset):

    categories = {
        "truck": 0,
        "car": 1,
        "plane": 2,
        "ship": 3,
        "cat": 4,
        "dog": 5,
        "horse": 6,
        "deer": 7,
        "frog": 8,
        "bird": 9,
    }

    times = {
        "day": 0,
        "night": 1,
        "none": 2,
    }

    weathers = {
        "cloudy": 0,
        "foggy": 1,
        "partly cloudy": 2,
        "raining": 3,
        "snowing": 4,
        "sunny": 5,
        "none": 6,
    }

    locations = {
        "forest": 0,
        "grass": 1,
        "indoors": 2,
        "rocks": 3,
        "sand": 4,
        "street": 5,
        "snow": 6,
        "water": 7,
        "none": 8,
    }

    def __init__(
        self,
        root,
        database=None,
        categories=None,
        times=None,
        weathers=None,
        locations=None,
        humans="yes",  # TODO: change this later
        transform=None,
        target_transform=None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform

        if database is None:
            database = BGVarDB(self.root / "annotations.db")

        self.image_files = list(
            database.read_entries(
                categories=categories,
                times=times,
                weathers=weathers,
                locations=locations,
                humans=humans,
            )
        )

    def __getitem__(self, idx):
        image_path = self.root / (self.image_files[idx][0][1:])
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        category = Focus.categories[self.image_files[idx][1]]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            category = self.target_transform(category)
        time = Focus.times[self.image_files[idx][2]]
        weather = Focus.weathers[self.image_files[idx][3]]
        locations = torch.zeros(len(Focus.locations), dtype=torch.long)
        for location in (self.image_files[idx][4]).split(", "):
            locations[Focus.locations[location]] = 1

        return image, category, time, weather, locations

    def __len__(self) -> int:
        return len(self.image_files)


def split_dataset(dataset, train_fraction=0.9) -> None:
    SEED = 7342984

    train_size = int(len(dataset) * train_fraction)
    test_size = len(dataset) - train_size

    return random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(SEED)
    )


class DCR(Dataset):
    classes = [
        "truck",
        "car",
        "plane",
        "ship",
        "cat",
        "dog",
        "equine",
        "deer",
        "frog",
        "bird",
    ]

    def __init__(self, root, split, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.image_files = list((Path(root) / split).glob("*.JPEG"))

    def __getitem__(self, idx):

        image_file = self.image_files[idx]
        label_file = image_file.parent / (image_file.stem + ".json")
        image = Image.open(image_file)
        if image.mode != "RGB":
            image = image.convert("RGB")

        with open(label_file, "r") as f:
            label = json.load(f)["label"]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.image_files)


if __name__ == "__main__":
    database = BGVarDB("./temp.db")
    database.populate_temp_table()
    dataset = Focus(
        ".", database=database, categories=["cat", "car"], locations=["forest"]
    )
    train_dataset, test_dataset = split_dataset(dataset, train_fraction=0.5)
    print(train_dataset[0], len(train_dataset))
    print(test_dataset[0], len(test_dataset))
    database.clear_annotations()
