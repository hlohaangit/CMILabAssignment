# Image Classification of Sclerotic and Non-Sclerotic Glomeruli

## Project Overview
This repository is dedicated to the development and evaluation of machine learning models for the binary classification of sclerotic and non-sclerotic glomeruli using microscopic images. This complex task is tackled using three different modeling approaches: logistic regression, a custom CNN, and a pretrained ResNet-50. The project not only aims to achieve high accuracy in classification but also to explore the strengths and limitations of each modeling approach in handling medical image data.

## Models Overview

Here is a quick summary of the models and their performances:

| Model               | Test Set Accuracy |
|---------------------|-------------------|
| Logistic Regression | 90.5%             |
| Simple CNN          | 98.3%             |
| ResNet-50           | 99%               |

## Files in the Repository

- **`LogisticRegression.ipynb`**: Explores the use of logistic regression for initial model benchmarking.
- **`CNNClassifier.ipynb`**: Develops a custom CNN tailored to the specific needs of glomeruli image classification.
- **`resnet50classifier.ipynb`**: Adapts a pretrained ResNet-50 model to the task, leveraging deep learning advancements for improved accuracy.
- **`evaluation.py`**: Provides utilities for model evaluation and created evaluation.csv

## Detailed Workflow and Model Insights

### Step 1: Logistic Regression (LogisticRegression.ipynb)

#### Purpose
To establish a baseline for classification performance using a simple logistic regression model, which helps identify key features and dataset characteristics.

#### Challenges
- **Image Size Variability**: The initial dataset contained images of varying sizes, which complicates the creation of a uniform feature set for logistic regression.
- **Class Imbalance**: A significant imbalance between the classes which can bias the model toward the majority class.

#### Solutions
- **Image Preprocessing**: Implemented image resizing and padding to standardize the input size for all images.
- **Rebalancing Techniques**: Applied data augmentation techniques such as image rotations and flips to enhance the dataset's diversity and balance.

#### Results
The logistic regression model achieved a respectable accuracy of 90.5%, providing valuable insights into the data's characteristics and the feasibility of using simpler models for preliminary analysis. The model can be downloaded from [here](https://uflorida-my.sharepoint.com/:u:/g/personal/h_lohaan_ufl_edu/EW5vDob0XNlNph6NJRwG-fgBska1ZZKdm6gh5OiwJtwOHw?e=DQRJkT) (407 kb).

### Step 2: Custom CNN Model (CNNClassifier.ipynb)

#### Purpose
To significantly enhance model performance by leveraging a CNN's capability to capture spatial hierarchies in image data.

#### Model Architecture
- **Layers**: Consists of several convolutional layers, each followed by max pooling. Batch normalization is included to maintain stability and speed up the network's training. The network ends with dense layers for classification.
- Following is the model summary:
<img width="644" alt="Screenshot 2024-05-03 at 8 49 24 PM" src="https://github.com/hlohaangit/CMILabAssignment/assets/54259754/ac477637-31ba-4086-992a-97f1908b4f15">


#### Challenges
- **Computational Resources**: Initially faced issues with large image sizes overwhelming the available computational resources (curernt workstaion m1 macbook- 8gb ram).

#### Solutions
- **Optimized Image Size**: Reduced the image dimensions to 128x128 pixels to accommodate the hardware limitations without substantially sacrificing model accuracy.

#### Results
This tailored CNN architecture improved accuracy to 98.3%, validating the effectiveness of convolutional networks in handling image classification tasks, especially with well-preprocessed input data. The model can be downloaded from [here](https://uflorida-my.sharepoint.com/:u:/g/personal/h_lohaan_ufl_edu/EQCuDdQ6hcJMvQkinFpIEsEBiFFIsB6Y__YgymEP86f-Mg?e=XG3l5Z) (9.8 mb)

Following was the confustion matrix without data augmentation. (test set is 20% of the given dataset chosen at random)

<img width="540" alt ="without augmentation cnn" src="https://github.com/hlohaangit/CMILabAssignment/assets/54259754/df5b60c7-efee-45b3-abda-f75b54442506">

Following is the confusion matrix of the model currently after data augmentation (test set is 20% of the augmented dataset chosen at random)

<img width="539" alt="Screenshot 2024-05-03 at 8 41 30 PM" src="https://github.com/hlohaangit/CMILabAssignment/assets/54259754/b32f77e7-1f4f-46a5-b030-111b8f0b3617">

### Step 3: Using Pretrained Model - ResNet-50 (resnet50classifier.ipynb)

#### Purpose
To utilize the powerful, pretrained ResNet-50 model to push the boundaries of accuracy and performance in our classification task.

#### Model Adaptation
- **Fine-tuning**: Modified the last few layers of the ResNet-50 to better suit our binary classification needs.

#### Challenges
- **Overfitting**: Managing the model's complexity to prevent overfitting while maintaining high accuracy on unseen data.

#### Solutions
- **Data Augmentation**: Enhanced the dataset with more diverse image transformations to improve the model's ability to generalize.

#### Results
Achieved an exceptional accuracy of 99%, demonstrating the potential of using advanced pretrained models in specialized areas such as medical image analysis. The model can be downloaded from [here](https://uflorida-my.sharepoint.com/:u:/g/personal/h_lohaan_ufl_edu/EX8jR98yALBAlSRGcRVUHskBCB2bUTqXFvWYJnFtpFdmgA?e=cauoyU)

Following is the confustion matrix of the current model. (test set is 20% of the augmented data chosen at random)

<img width="538" alt="Screenshot 2024-05-03 at 8 45 38 PM" src="https://github.com/hlohaangit/CMILabAssignment/assets/54259754/9e327f54-140f-4cca-b242-54f8463a98f6">




### Evaluation and Metrics (evaluation.py)

#### Functionality
This script is crucial for assessing the performance of different models. It processes test images, loads the trained models, and predicts outcomes, providing a standardized evaluation framework.

## Usage

1. Download this repo, it's probably easier if you download the zip and extract the files.
2. Make sure you have pipenv installed on your system. If not you have use the following command to install it:
```bash
pip install pipenv
```
3. On your terminal navigate to where you've donloaded this repository.
4. Install all the dependencies with the following command:
```bash
pipenv install
```
5. download any of the models mentioned above (or from the links below) and run the following command
```bash
pipenv run python evaluation.py --test_data_path [path_to_test_data] --model_path [path_to_saved_model]
```
- [Logistic Regression](https://uflorida-my.sharepoint.com/:u:/g/personal/h_lohaan_ufl_edu/EW5vDob0XNlNph6NJRwG-fgBska1ZZKdm6gh5OiwJtwOHw?e=DQRJkT) (407 kb)

- [Simple CNN](https://uflorida-my.sharepoint.com/:u:/g/personal/h_lohaan_ufl_edu/EQCuDdQ6hcJMvQkinFpIEsEBiFFIsB6Y__YgymEP86f-Mg?e=XG3l5Z) (9.8 mb)
  
- [ResNet-50 Classifier](https://uflorida-my.sharepoint.com/:u:/g/personal/h_lohaan_ufl_edu/EX8jR98yALBAlSRGcRVUHskBCB2bUTqXFvWYJnFtpFdmgA?e=cauoyU) (1.76 gb)
