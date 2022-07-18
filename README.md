# FOCUS: Familiar Objects in Common and Uncommon Settings

Repository for the ICML 2022 paper of the same name [[link](https://proceedings.mlr.press/v162/kattakinda22a.html)].

## Instructions

### Dataset

1. Download the dataset from [here](https://umd.box.com/s/w7tvxer0wur7vtsoqcemfopgshn6zklv). Or alternately, use the following command:
```
curl -L -o focus.zip https://umd.box.com/s/w7tvxer0wur7vtsoqcemfopgshn6zklv
```

2. Unzip the dataset:
```
unzip focus.zip
```

### Setup

1. Clone this repository:
```
git clone https://github.com/priyathamkat/focus.git
```

2. Add `src` to your `PYTHONPATH`:
```
export PYTHONPATH="$PYTHONPATH:/path/to/focus/src"
```

3. Use FOCUS as follows:

```
from experiments.dataset import Focus
focus = Focus(
    path_to_focus,
    categories=[
        "truck",
        "car",
        "plane",
        "ship",
        "cat",
        "horse",
        "horse",
        "deer",
        "frog",
        "bird",
    ],
    times=["day"],
    weathers=["sunny"],
    locations=["grass", "street"],
    transform=None
)

```

See [src/experiments/dataset.py](src/experiments/dataset.py) for the valid arguments that can be passed to `times`, `weathers` and `locations`. Also, checkout [src/experiments/focus-dataset.ipynb](src/experiments/focus-dataset.ipynb) for more details about how to use the dataset.

## What else is in this repo?

1. [src/experiments/evaluate_model.py](src/experiments/evaluate_model.py) - For evaluating a pretrained model on FOCUS (Section 4.2 in the paper).

2. [src/experiments/finetune.py](src/experiments/finetune.py) - For finetuning a model on FOCUS (Section 4.3 in the paper).

3. [src/experiments/grad_cam_visualizations.ipynb](src/experiments/grad_cam_visualizations.ipynb) - GradCAM visualizations (Figure 4 in the paper).

## ImageNet Attributes

Machine generated environmental attributes for images in ImageNet can be downloaded from [here](https://umd.box.com/s/ntswzbasgcbcmulodqcons226ccwjytz) (See Section 4.4 in our paper for more details).

## Citation

If you use FOCUS, please consider citing our work:

```

@InProceedings{pmlr-v162-kattakinda22a,
  title = 	 {{FOCUS}: Familiar Objects in Common and Uncommon Settings},
  author =       {Kattakinda, Priyatham and Feizi, Soheil},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {10825--10847},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/kattakinda22a/kattakinda22a.pdf},
  url = 	 {https://proceedings.mlr.press/v162/kattakinda22a.html},
}


```