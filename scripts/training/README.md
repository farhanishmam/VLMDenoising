## Folder structure for csvd_training.py:
```
project_root/
│
├── clean_dir/
│   ├── img1.jpg
│   ├── img2.jpg
│   ├── img3.jpg
│   └── ...
│
└── noisy_base_dir/ (for each visual corruption)
    ├── L1/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    │
    ├── L2/
    ├── L3/
    ├── L4/
    └── L5/
```

## Folder structure for vcrn_training.py:
```
root_dir/
│
├── clean_dir/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
│
├── Brightness/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
│
├── Gaussian-noise/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
```
For VCRN training, we take a 70:15:15 train-val-test split of the augmented VQAv2 dataset (3000 x 18 x 5 = 270,000
images) created from the 3000 VQAv2 image subset
