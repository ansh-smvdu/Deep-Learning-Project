# Skin Lesion Classification using DenseNet169

## Project Overview
This project implements a deep learning based skin lesion classification system using the DenseNet169 architecture in PyTorch.

The model classifies skin lesion images into multiple disease categories using transfer learning on the MILK10k medical image dataset.

The system is designed to support medical image analysis by identifying visual patterns in skin lesion images and predicting the corresponding lesion class.

## Team Members
1. Ansh Singh (23bcs018)
2. Madhav Sharn (22bcs050)
3. Laxman Singh Kirar (23bcs048)

## Dataset
Dataset Used:
MILK10k Skin Lesion Dataset

Dataset Structure:
- MILK10k_Training_Input
- MILK10k_Training_GroundTruth.csv
- MILK10k_Training_Metadata.csv
- MILK10k_Training_Supplement.csv
- MILK10k_Test_Input
- MILK10k_Test_Metadata.csv

The images are stored inside nested folders. The ground truth CSV contains one-hot encoded labels, which were converted into integer class labels for model training.

Classes:
- AKIEC
- BCC
- BEN_OTH
- BKL
- DF
- INF
- MAL_OTH
- MEL
- NV
- SCCKA
- VASC

## Technologies Used
- Python
- PyTorch
- Torchvision
- Google Colab
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- PIL

## Dataset Split
The labeled dataset contained 5,240 usable labeled images.

The dataset was split using stratified sampling:
- Training: 70%
- Validation: 15%
- Testing: 15%

Final split:
- Training Images: 3,668
- Validation Images: 786
- Testing Images: 786

Stratified splitting was used to preserve class distribution across training, validation, and testing sets.

## Model
- DenseNet169
- Transfer Learning
- Pretrained ImageNet Weights

The pretrained DenseNet169 model was used as a feature extractor. Its original classifier layer was replaced with a custom classifier suitable for the 11 skin lesion classes in the MILK10k dataset.

The feature extraction layers were frozen initially, and only the final classifier was trained.

## Image Preprocessing
The input images were resized and normalized before being passed to the model.

Training images were augmented using:
- Random horizontal flip
- Random vertical flip
- Random rotation
- Color jitter
- ImageNet normalization

Validation and testing images were only resized and normalized.

## Training Details
The model was trained in Google Colab using GPU acceleration.

GPU Used:
- Tesla T4

Loss Function:
- CrossEntropyLoss

Optimizer:
- Adam Optimizer

Scheduler:
- ReduceLROnPlateau

Number of Epochs:
- 15

Batch Size:
- 32

## Evaluation Metrics
The model was evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Loss Curve
- Accuracy Curve

## Results
Best Validation Accuracy Achieved:
62.60%

Testing Accuracy Achieved:
61.32%

Macro Precision:
26.85%

Macro Recall:
21.57%

Macro F1-Score:
21.15%

Weighted Precision:
54.73%

Weighted Recall:
61.32%

Weighted F1-Score:
54.43%

ROC-AUC Macro:
82.75%

ROC-AUC Weighted:
84.00%

## Result Analysis
The model achieved moderate performance on the MILK10k dataset.

The model performed better on majority classes such as BCC and NV, but struggled with minority classes such as BEN_OTH, DF, INF, MAL_OTH, and VASC.

This happened mainly because the dataset was imbalanced. Some classes had many images, while some classes had very few images in the test set.

Although the overall accuracy was 61.32%, the low macro F1-score shows that the model did not perform equally well across all classes.

## Files Included
- Skin_Lesion_Classification.ipynb
- best_densenet169_milk10k.pth
- loss_curve.png
- accuracy_curve.png
- confusion_matrix.png
- metrics_report.txt
- SkinLesion_DenseNet169_Full_Explanation.pdf
- README.md

## How to Run
1. Open the notebook in Google Colab
2. Mount Google Drive
3. Set the correct dataset paths
4. Install/import required dependencies
5. Load the MILK10k dataset and ground-truth CSV
6. Convert one-hot encoded labels into integer labels
7. Perform stratified train-validation-test split
8. Apply image preprocessing and augmentation
9. Load pretrained DenseNet169
10. Replace the classifier layer
11. Train the model on GPU
12. Evaluate the model on the test set
13. Generate metrics, graphs, and confusion matrix
14. Save the trained model weights

## Future Work
- Improve accuracy using class-weighted loss
- Use WeightedRandomSampler for class imbalance
- Fine-tune deeper DenseNet169 layers
- Train for more epochs
- Use larger and more balanced datasets
- Combine image features with metadata features
- Apply explainable AI techniques such as Grad-CAM
- Deploy the model as a web application

## Conclusion
This project successfully demonstrates a complete deep learning pipeline for multiclass skin lesion classification using DenseNet169 and transfer learning.

The model was trained and evaluated on the labeled MILK10k dataset using PyTorch in Google Colab. The system generated the trained model weights, evaluation metrics, accuracy/loss curves, and confusion matrix.

The results show that the model learned meaningful features from the dataset, but performance was affected by class imbalance. Future improvements can focus on class balancing, deeper fine-tuning, and metadata-based learning.
