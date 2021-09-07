# Brain-Tumor-Segmentation-And-Classification

### Brain Tumor Segmentation And Classification using artificial intelligence.

A library for classifying and segmentaing Brain tumors in brain MRI's.\
It can classify an image as tumorus or non-tumorus and is able to localize the tumor if there is one.

## Installation and Quick Start
**Note: Pretrained models available [here](https://dontasktoask.com/)**
```
git clone https://github.com/Rohith04MVK/Brain-Tumor-Segmentation-And-Classification
cd Brain-Tumor-Segmentation-And-Classification 
mkdir models
python data/segmentation/download_segmentation_data.py
python src/train_seg.py
python src/train_clf.py
python example.py
```

### Project structure

This project has 3 main sections.

- `src/` Contains the python scripts for training the ML Models.
- `notebooks/` contains the jupyter notebooks with explanations.
- `models/` pretrained models.
- `data/` datasets for training the model.
