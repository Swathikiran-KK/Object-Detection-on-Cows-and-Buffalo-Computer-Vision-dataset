# Object Detection Using Cows and Buffalo Computer Vision Dataset

## Project Objective

The primary objective of this project is to develop and train an object detection model capable of accurately identifying and localizing cows and buffalo within images. This has practical applications in livestock monitoring, population counting, and automated farming.

## Problem Statement

The dataset contains images of cows and buffalo with bounding box annotations in YOLO format. The dataset exhibits class imbalance, with some classes having significantly fewer instances than others. Classes with very few instances (specifically classes 9 and 10) have been removed. The task is to train a robust object detection model that can handle this imbalance and perform well on unseen data.

## Dataset

The dataset used in this project is the [Cows and Buffalo Computer Vision Dataset](https://www.kaggle.com/datasets/raghavdharwal/cows-and-buffalo-computer-vision-dataset/data). It contains images annotated with bounding boxes in YOLO format. The dataset was split into training, validation, and test sets using a 70/15/15 ratio after removing the underrepresented classes. Data augmentation was applied to the training set to mitigate class imbalance.

## Methodology

1. **Data Loading and Exploration:** Loaded image and label data; analyzed class distribution, bounding box sizes, and aspect ratios.  
2. **Data Preprocessing and Splitting:** Cleaned the dataset by removing classes with few instances and split the data into training, validation, and test sets.  
3. **Data Augmentation:** Applied augmentation techniques to increase instances for underrepresented classes, combining augmented and original data for training.  
4. **Model Training:** Trained YOLOv8 object detection models (YOLOv8s, YOLOv8m, YOLOv8l) on combined training data.  
5. **Model Evaluation:** Assessed models on the validation set using Precision, Recall, mAP50, and mAP50-95 metrics.  
6. **Inference:** Applied trained models on test set images for prediction.

## Models Used

- **Feature Extraction:** ResNet50 pretrained on ImageNet for image feature extraction and clustering.  
- **Object Detection:** YOLOv8 family models including YOLOv8s, YOLOv8m, and YOLOv8l.  
- **Dimensionality Reduction:** UMAP for visualization of image embeddings in lower-dimensional space.

## Results

The models’ performance on the validation set is summarized below:

| Model    | Precision | Recall | mAP50 | mAP50-95 | Fitness |
| -------- | --------- | ------ | ------| -------- | ------- |
| YOLOv8s  | 0.759     | 0.764  | 0.725 | 0.504    | 0.504   |
| YOLOv8m  | 0.834     | 0.660  | 0.732 | 0.507    | 0.507   |
| YOLOv8l  | 0.706     | 0.749  | 0.705 | 0.486    | 0.486   |

Based on the mAP50-95 metric, the **YOLOv8m** model showed the best overall performance.

 
