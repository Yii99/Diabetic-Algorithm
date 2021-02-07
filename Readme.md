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

<img width="450" height="450" src="https://github.com/Yii99/Diabetic-Algorithm/blob/main/restnet.png"/> 

Loss function: sparse cross-entropy
### Metrics
* Accuracy
* Confusion Matrix

### Training and evaluation
* Use Adam optimizer to train our network for 300 epochs
* Save checkpoint for every 5 epochs
* batch_size=8

### Results
#### Confusion matrix
<img width="450" height="450" src="https://github.com/Yii99/Diabetic-Algorithm/blob/main/cm.png"/> 

Accuracy on test set: 91% (RDR), 84%(NRDR)
#### Explainability  
Grad-CAM Result
![deepv](https://github.com/Yii99/Diabetic-Algorithm/blob/main/gc.png)  
E.g., for RDR images, our network mainly focuses on hard exudates on retinal fundus, which is reasonable. 

