# Diabetic Retinopathy Detection

Diabetic retinopathy (DR) is an illness which causes damages to retina.
The goal of this project is to build a generic framework to classify nonreferable (NRDR) and referable (RDR) diabetic retinopathy based on fundus images with deep learning algorithms.


## Dependancy
Project is created with:
- Python 3.6
- Tensorflow 2.0.0

## Dataset
Indian Diabetic Retinopathy Image Dataset (IDRID)

## Input Pipeline
### Preprocessing
* Redefine labels for binary classification
* Use the left and right border of retina to crop the images and resize them to 256 * 256
* Oversampling to balance the dataset
### Serialization
* Use TFRecord to load data efficiently
* Reducing the disk space usage 212.2MB --> 21.9MB 
### Data augmentation
* Rotation, Shift, Zoom, Brightness, Flipping, Shearing
### Model
ResNet is a kind of convolutional neural network with skip-connection, which has shown high accuracy for classification tasks.   
Our network is a simplified version of ResNetV2.
![accuracy](https://github.com/LEGO999/Diabetic-Retinopathy-Detection/blob/master/resnetunit.png) 

|Architecture|
|:-----:|
|  7 * 7 Conv(strides=2), BN, ReLU  |
|  3 * 3 Max pooling(strides=2)|
|ResBlock(strides=2) ch=16 | 
| ResBlock(strides=2) ch=32| 
| ResBlock(strides=2) ch=64|
| ReLU, Global average pooling|
|Dense(unit=2), Softmax|

Loss function: sparse cross-entropy
### Metrics
* Accuracy
* Confusion Matrix

### Features
* Logging with abseil
* Configuration with gin instead of hard coding
* AutoGraph, average epoch time: 11.92s --> 7.18s
* Deep visualization(Grad-CAM)
* Grid search for hyperparameter tuning(TensorBoard)

### Training and evaluation
* Use Adam optimizer to train our network for 300 epochs
* Save checkpoint for every 5 epochs
* Hyperparameter tuning - grid search
  * learning rate: 1e-2, 1e-3, 1e-4 and 1e-5
  * batch size: 2 and 4   
![accuracy](https://github.com/LEGO999/Diabetic-Retinopathy-Detection/blob/master/hp2.png) 
* Based on above best hyperparameters, it's still underfitting , use small learning rate=1e-5 and stronger data augmentation for further training(50 epochs)

### Results
#### Confusion matrix
![accuracy](https://github.com/LEGO999/Diabetic-Retinopathy-Detection/blob/master/confusionmatrix2.png)  
Balance accuracy on test set: 82.2%
#### Explainability  
Grad-CAM Result
![deepv](https://github.com/LEGO999/Diabetic-Retinopathy-Detection/blob/master/dpv.png)  
E.g., for RDR images, our network mainly focuses on hard exudates on retinal fundus, which is reasonable. 

