# Melanoma Ski Cancer Detection

## Problem Statement
To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.


## Table of Contents
* General Information
* Model Architecture
* Model Summary
* Model Evaluation
* Technology Used
* Conclusion


## General Information
- The dataset consists of 2,357 images representing both malignant and benign oncological conditions, sourced from the International Skin Imaging Collaboration (ISIC). These images have been classified according to ISIC's classification system, ensuring that each category is balanced with an equal number of images.

![image](https://github.com/user-attachments/assets/768f4237-10c0-440c-8e56-bfdbc727525f)


## Model Architecture

1. Data Augmentation:
The augmentation_data variable defines the data augmentation techniques applied to the training images. Augmentation artificially expands the dataset by applying random transformations, such as rotation, scaling, and flipping. This helps the model generalize better by exposing it to more varied training examples.

2. Normalization:
A Rescaling(1./255) layer normalizes the input image pixel values to a range of 0 to 1. This normalization stabilizes the training process and accelerates convergence by scaling the pixel values to a consistent range.

3. Convolutional Layers:
The model consists of three sequential convolutional layers implemented with the Conv2D function. Each layer uses the ReLU activation function, which introduces non-linearity.
The padding='same' parameter ensures that the spatial dimensions of the feature maps remain unchanged after convolution.
The number of filters (16, 32, 64) in each convolutional layer determines the depth of the feature maps and allows the model to capture increasingly complex features.

4. Pooling Layers:
Following each convolutional layer is a MaxPooling2D layer, which reduces the spatial dimensions of the feature maps by downsampling. This helps retain important features while reducing computational complexity and mitigating the risk of overfitting.

5. Dropout Layer:
A Dropout layer with a rate of 0.2 is added after the last pooling layer. Dropout is a regularization technique that prevents overfitting by randomly setting a fraction of neurons to zero during training, which forces the model to learn more robust features.

6. Flatten Layer:
The Flatten layer reshapes the 2D feature maps into a 1D vector, making it ready for input into the fully connected layers.

7. Fully Connected Layers:
Two fully connected (dense) layers follow the flattening process:
The first dense layer has 128 neurons and uses the ReLU activation function.
The second dense layer outputs the final classification probabilities.

8. Output Layer:
The output layer’s number of neurons is determined by the target_labels variable, which corresponds to the number of classes in the classification task. This layer doesn’t use an activation function, as the output is processed by the loss function during training.

9. Model Compilation:
The model is compiled with the Adam optimizer, which adapts the learning rate during training. The loss function used is Sparse Categorical Crossentropy, which is suitable for multi-class classification tasks.
Accuracy is chosen as the evaluation metric to assess the model’s performance.

10. Training:
The model is trained using the fit method with 50 epochs. To ensure the model converges effectively and avoids overfitting, the following callbacks are utilized:
ModelCheckpoint: Saves the model whenever the validation accuracy improves.
EarlyStopping: Stops training if the validation accuracy doesn't improve after 5 consecutive epochs (patience=5).


## Conclusions

![image](https://github.com/user-attachments/assets/0649f605-b5be-4f5b-a5e4-33d61afc642d)


![image](https://github.com/user-attachments/assets/8203e4f6-552f-441d-ba14-8809ffe79ae4)


 ## Observation 
- The final model demonstrates balanced performance, with no signs of underfitting or overfitting.

- Implementing class rebalancing has significantly enhanced the model's performance on both the training and validation datasets.

- After 37 epochs, the model achieves an accuracy of **84%** on the training set and approximately **79%** on the validation set.

- The minimal gap between training and validation accuracies indicates the model's strong ability to generalize.

- However, the introduction of batch normalization did not result in any noticeable improvements in either training or validation accuracy.


## Technologies Used
- Python
- Matplotlib
- Numpy
- Pandas
- Seaborn
- Tensorflow

## Contact
Created by Selva Sankari
