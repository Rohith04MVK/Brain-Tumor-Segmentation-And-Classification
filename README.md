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

This project has 4 main sections.

- `src/` Contains the python scripts for training the ML Models.
- `notebooks/` contains the jupyter notebooks with explanations.
- `models/` pretrained models.
- `data/` datasets for training the model.
## Data
We are using the [lgg-mri-dataset](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation) which has the images and masks as .tif files.\
This data will be split into 
- 3006 Train images
- 590 Testing images
- 333 Validating images

An example of the mask and the original image\
![tumor and brain](images/brain_and_tumor.png)

This is the mask applied on the MIR
![mask on mri](images/tumor_on_brain.png)

## Predictions
The model returns a pandas dataframe as its out put
![predictions](images/predictions.PNG)
