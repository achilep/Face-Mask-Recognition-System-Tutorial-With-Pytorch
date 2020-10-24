# Face Mask Recognition System Tutorial With Pytorch

## Overview 
In this tutorial, you will make the first steps towards developing an algorithm that could be used as part of a mobile or web app. You will learn and use the concept of transfer learning to build a face mask recognition system with pytorch.
Sample face mask recognition  Output

<img src="https://github.com/achilep/Face-Mask-Recognition-System-Tutorial-With-Pytorch/blob/main/Resource/readme_image/test result1.png" alt="output"/>


### features coverd by the tutorial 

1. [transform](https://pytorch.org/docs/stable/torchvision/transforms.html?highlight=transform)
Transforms are common image transformations.
Data transformation is the process in which you take data from its raw, siloed and normalized source state and transform it into data that's joined together, dimensionally modelled, de-normalized, and ready for analysis.

5. [models](https://pytorch.org/docs/stable/torchvision/models.html) 
The model's subpackage contains definitions of models for addressing different tasks, including image classification, pixel-wise semantic segmentation, object detection, instance segmentation, person keypoint detection and video classification.

2. [datasets](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)

3. [DataLoader](https://pytorch.org/docs/stable/torchvision/datasets.html)

3. [nn](https://pytorch.org/docs/stable/nn.html)

4. [optim](https://pytorch.org/docs/stable/optim.html#:~:text=optim-,torch.,easily%20integrated%20in%20the%20future.):torch.optim is a package implementing various optimization algorithms.

      
### requirement 
- Computer with GPU
- Good knowledge of python.
- Basic knowledge of deep learning (neural network, convolutional neural network(CNN), etc. ) 

### Setting up the working environment :
- Local computer: you can follow the instruction [here](https://pytorch.org/get-started/locally/) to set up PyTorch in the computer. 

- platform as a service: Kaggle Kernels is a free platform to run Jupyter notebooks in the browser. kaggle provide free GPU to train your model.
you can Sign in [here](https://www.kaggle.com/)

## Building the App step by step  

### Step 0: Import Datasets
Make sure that you've downloaded the required dataset.

Download the [dataset](https://www.kaggle.com/achilep/covid19-face-mask-data/download) to train our model, For testing we are using a different  [dataset](https://www.kaggle.com/achilep/covid19-face-mask-recognition-test-data).
### Step 1: Specify Data Loaders for the covid19-face-mask-data dataset

- Loading Image Data

The easiest way to load image data is with datasets.ImageFolder from torchvision [documentation](http://pytorch.org/docs/master/torchvision/datasets.html#imagefolder). In general you'll use ImageFolder like so:

```dataset = datasets.ImageFolder('path/to/data', transform=transform)```
where 'path/to/data' is the file path to the data directory and transform is a sequence of processing steps built with the transforms module from torchvision. ImageFolder expects the files and directories to be constructed like so:

```
root/covid19-face-mask-data/face-mask-dataset/train/faceWithMask/xxx.png
root/covid19-face-mask-data/face-mask-dataset/train/faceWithMask/xxy.png
root/covid19-face-mask-data/face-mask-dataset/train/faceWithMask/xxz.png

root/covid19-face-mask-data/face-mask-dataset/train/faceWithoutMask/123.png
root/covid19-face-mask-data/face-mask-dataset/train/faceWithoutMask/nsdf3.png
root/covid19-face-mask-data/face-mask-dataset/train/faceWithoutMask/asd932_.png
```
where each class has it's own directory (faceWithMask and faceWithoutMask) for the images. The images are then labeled with the class taken from the directory name. So here, the image 123.png would be loaded with the class label faceWithoutMask. 

- Data Loaders

With the ImageFolder loaded, you have to pass it to a DataLoader. The DataLoader takes a dataset (such as you would get from ImageFolder) and returns batches of images and the corresponding labels. You can set various parameters like the batch size and if the data is shuffled after each epoch.

```dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)```
Here dataloader is a [generator](https://jeffknupp.com/blog/2013/04/07/improve-your-python-yield-and-generators-explained/). To get data out of it, you need to loop through it or convert it to an iterator and call next().


```aidl
# Looping through it, get a batch on each loop 
for images, labels in dataloader:
    pass

# Get one batch
images, labels = next(iter(dataloader))
```

Dataloader is able to spit out random samples of our data, so our model won’t have to deal with the entire dataset every time. This makes training more efficient.
We specify how many images we want at once as our batch_size (so 32 means we want to get 32 images at one time). We also want to shuffle our images so it gets inputted randomly into our AI model.

##### Data Augmentation and Transforms
- transforms

When you load in the data with ImageFolder, you'll need to define some transforms. For example, the images are different sizes but we'll need them to all be the same size for training. You can either resize them with ```transforms.Resize() ``` or crop with ```transforms.CenterCrop()```, ```transforms.RandomResizedCrop()```, etc. We'll also need to convert the images to PyTorch tensors with ```transforms.ToTensor()```. Typically you'll combine these transforms into a pipeline with ```transforms.Compose()```, which accepts a list of transforms and runs them in sequence.
 
- ```transforms.Compose``` just clubs all the transforms provided to it. So, all the transforms in the ```transforms.Compose``` are applied to the input one by one.


- ```transforms.RandomResizedCrop(224)```: This will extract a patch of size (224, 224) from your input image randomly. So, it might pick this path from top-left, bottom right or anywhere in between. So, you are doing data augmentation in this part. Also, changing this value won't play nice with the fully-connected layers in your model, so not advised to change this.

- ```transforms.RandomHorizontalFlip()```: Once we have our image of size (224, 224), we can choose to flip it. This is another part of data augmentation.

- ```transforms.ToTensor()```: This just converts your input image to PyTorch tensor.

- ```transforms.Resize(256)```: First your input image is resized to be of size (256, 256)

- ```transforms.CentreCrop(224)```: Crops the center part of the image of shape (224, 224)


- ```transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])```: This is just input data scaling and these values (mean and std) must have been precomputed for your dataset. Changing these values is also not advised.

It looks something like this to scale, then crops, then converts to a tensor:
```aidl
transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])
```
There are plenty of transforms available, I'll cover more in a bit and you can read through the [documentation](http://pytorch.org/docs/master/torchvision/transforms.html).

- Data Augmentaion

A common strategy for training neural networks is to introduce randomness in the input data itself. For example, you can randomly rotate, mirror, scale, and/or crop your images during training. This will help your network generalize as it's seeing the same images but in different locations, with different sizes, in different orientations, etc.

To randomly rotate, scale and crop, then flip your images you would define your transforms like this:

```aidl
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], 
                                                            [0.5, 0.5, 0.5])])
```

You'll also typically want to normalize images with transforms. Normalize. You pass in a list of means and list of standard deviations, then the colour channels are normalized like so

```input[channel] = (input[channel] - mean[channel]) / std[channel]```

Subtracting mean centres the data around zero and dividing by std squishes the values to be between -1 and 1. Normalizing helps keep the network weights near zero which in turn makes backpropagation more stable. Without normalization, networks will tend to fail to learn.

You can find a list of all [the available transforms here](http://pytorch.org/docs/0.3.0/torchvision/transforms.html). When you're testing, however, you'll want to use images that aren't altered other than normalizing. So, for validation/test images, you'll typically just resize and crop.





*The code cell below write three separate data loaders for the training, validation, and test datasets of humans images (located at covid19-face-mask-data/face-mask-dataset/train, covid19-face-mask-data/face-mask-dataset/valid, and covid19-face-mask-data/face-mask-dataset/test, respectively). You may find this [documentation on custom datasets](http://pytorch.org/docs/stable/torchvision/datasets.html) to be a useful resource. If you are interested in augmenting your training and/or validation data, check out the wide variety of [transforms](http://pytorch.org/docs/stable/torchvision/transforms.html?highlight=transform)!*

```
from PIL import Image
import torchvision.transforms as transforms
import torch
import os
from torchvision import datasets

datadir = {

    'train': '../input/covid19-face-mask-data/face-mask-dataset/train/',
    'valid': '../input/covid19-face-mask-data/face-mask-dataset/validation/',
    'test': '../input/covid19-face-mask-data/face-mask-dataset/test'
}

trns_normalize = transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])

transform_transfer = {}
transform_transfer['train'] = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        trns_normalize
    ])
transform_transfer['valid'] = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        trns_normalize
    ])
transform_transfer['test'] = transform_transfer['valid']

# Trying out an idiom found in the pytorch docs
datafolder_transfer = {x: datasets.ImageFolder(datadir[x], transform=transform_transfer[x]) for x in ['train', 'valid', 'test']}

batch_size = 20
num_workers = 0

# Trying out an idiom found in the pytorch docs
loaders_transfer = {x: torch.utils.data.DataLoader(datafolder_transfer[x], batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True) 
for x in ['train', 'valid', 'test']}
```
### Step 2: Define the Model Architecture

#### Transfer Learning

In this section, you'll learn how to use pre-trained networks to solved challenging problems in computer vision. Specifically, you'll use networks trained on [ImageNet available from torchvision](http://pytorch.org/docs/0.3.0/torchvision/models.html).

ImageNet is a massive dataset with over 1 million labelled images in 1000 categories. It's used to train deep neural networks using an architecture called convolutional layers. I'm not going to get into the details of convolutional networks here, but if you want to learn more about them, please [watch this](https://www.youtube.com/watch?v=2-Ol7ZB0MmU).

Once trained, these models work astonishingly well as feature detectors for images they weren't trained on. Using a pre-trained network on images, not in the training set is called transfer learning. Here we'll use transfer learning to train a network that can classify our face with mask and face without mask photos with near-perfect accuracy.

With torchvision.models you can download these pre-trained networks and use them in your applications. We'll include models in our imports now.

```aidl
from torchvision import datasets, transforms, models
```
Most of the pretrained models require the input to be 224x224 images. Also, we'll need to match the normalization used when the models were trained. Each color channel was normalized separately, the means are [0.485, 0.456, 0.406] and the standard deviations are [0.229, 0.224, 0.225].

Transfer Learning

Most of the time you won't want to train a whole convolutional network yourself. Modern ConvNets training on huge datasets like ImageNet take weeks on multiple GPUs.

Instead, most people use a pre-trained network either as a fixed feature extractor or as an initial network to fine-tune.

In this notebook, you'll be using [VGGNet](https://arxiv.org/pdf/1409.1556.pdf) trained on the [ImageNet](http://www.image-net.org/) dataset as a feature extractor. Below is a diagram of the VGGNet architecture, with a series of convolutional and max-pooling layers, then three fully-connected layers at the end that classify the 1000 classes found in the ImageNet database.
 
<img src="https://github.com/achilep/Face-Mask-Recognition-System-Tutorial-With-Pytorch/blob/main/Resource/readme_image/vgg_16_architecture.png" alt="Load the Model"/>


VGGNet is great because it's simple and has great performance, coming in second in the ImageNet competition. The idea here is that we keep all the convolutional layers, but replace the final fully-connected layer with our own classifier. This way we can use VGGNet as a fixed feature extractor for our images then easily train a simple classifier on top of that.

- Use all but the last fully-connected layer as a fixed feature extractor.
- Define a new, final classification layer and apply it to the task of our choice!
You can read more about transfer learning from the [CS231n Stanford course notes](http://cs231n.github.io/transfer-learning/).


```
import torchvision.models as models
import torch.nn as nn


##  Specify model architecture 
model_transfer = models.vgg16(pretrained=True)

for param in model_transfer.features.parameters():
    param.requires_grad = False
    
### make changes to final fully collected layers
n_inputs = model_transfer.classifier[6].in_features
last_layer = nn.Linear(n_inputs, 133)
model_transfer.classifier[6] = last_layer
# check if CUDA is available
use_cuda = torch.cuda.is_available()

if use_cuda:
    model_transfer = model_transfer.cuda()
  ```
<img src="https://github.com/achilep/Face-Mask-Recognition-System-Tutorial-With-Pytorch/blob/main/Resource/readme_image/pretrainmodel.png" alt="Load the Model"/>

```
print(model_transfer)
```
<img src="https://github.com/achilep/Face-Mask-Recognition-System-Tutorial-With-Pytorch/blob/main/Resource/readme_image/print-modeltrasnfert.png" alt="Load the Model"/>

### Step 3: Specify Loss Function and Optimizer
Error and Loss Function: In most learning networks, the error is calculated as the difference between the actual output and the predicted output.
The function that is used to compute this error is known as Loss Function.

- Loss function
loss functions are mathematical algorithms that help measure how close a neural net learns to get the actual result. In machine learning, a loss function is a mathematical algorithm that evaluates the performance of an ML algorithm with respect to its desired result. There are various loss functions for various problems. You are aware that machine learning problem can (in basic terms) be either a classification problem or a regression problem. This implies that we do have optimized loss functions for classification and others for regression. To mention a few, we do have the following loss functions as classification based (binary cross-entropy, categorical cross-entropy, cosine similarity and others). We also have, mean squared error (MSE), mean absolute percentage error (MAPE), mean absolute error (MAE), just to mention a few, used for regression-based problems.


- An optimizer
In simple sentences, an optimizer can basically be referred to as an algorithm that helps another algorithm to reach its peak performance without delay. With respect to machine learning (neural network), we can say an optimizer is a mathematical algorithm that helps our loss function reach its convergence point with minimum delay (and most importantly, reduce the possibility of gradient explosion). Examples include adam, stochastic gradient descent (SGD), adadelta, rmsprop, adamax, adagrad, nadam etc.

Use the next code cell to specify a loss [function](http://pytorch.org/docs/master/nn.html#loss-functions) and [optimizer](http://pytorch.org/docs/master/optim.html). Save the chosen loss function as criterion_transfer, and the optimizer as optimizer_transfer below.

Below we'll use cross-entropy loss and stochastic gradient descent with a small learning rate. Note that the optimizer accepts as input only the trainable parameters vgg.classifier.parameters().
```
import torch.optim as optim
criterion_transfer = nn.CrossEntropyLoss()

optimizer_transfer = optim.SGD(model_transfer.classifier.parameters(), lr=0.001)
```
### Step 4: Train and Validate the Model
Train and validate your model in the code cell below. [Save the final model parameters](http://pytorch.org/docs/master/notes/serialization.html) at filepath ```'model_transfer.pt'```.
```
import numpy as np
from glob import glob
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        torch.enable_grad()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            # train_loss += loss.item()*data.size(0) 
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
        
        ######################
        # validate the model #
        ######################
        model.eval()
        torch.no_grad()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            output = model(data)
            loss = criterion(output, target)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
            ))
        
        ##  save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
            
    # return trained model
    return model
    ```
    
    ```
    # train the model
n_epochs =4 #25
model_transfer = train(n_epochs, loaders_transfer, model_transfer, optimizer_transfer, criterion_transfer, use_cuda, 'model_transfer.pt')

# load the model that got the best validation accuracy (uncomment the line below)
model_transfer.load_state_dict(torch.load('model_transfer.pt'))
```
<img src="https://github.com/achilep/Face-Mask-Recognition-System-Tutorial-With-Pytorch/blob/main/Resource/readme_image/train.png" alt="Load the Model"/>

```
print(model_transfer)
```
<img src="https://github.com/achilep/Face-Mask-Recognition-System-Tutorial-With-Pytorch/blob/main/Resource/readme_image/after_train.png" alt="Load the Model"/>

### Step 5: Test the Model
Try out your model on the test dataset . Use the code cell below to calculate and print the test loss and accuracy. Ensure that your test accuracy is greater than 60%.

```
def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

# call test function
```

```
test(loaders_transfer, model_transfer, criterion_transfer, use_cuda)
```
<img src="https://github.com/achilep/Face-Mask-Recognition-System-Tutorial-With-Pytorch/blob/main/Resource/readme_image/test_accuraty.png" alt="Test Accuracy"/>

### Step 6: Predict if a human is wearing a face mask or not  with the Model
Write a function that takes an image path as input and returns the mask if the man present on the image is wearing a face mask or not base on the prediction of the model.
```
###  Write a function that takes a path to an image as input
### and returns the prediction of the model.

def predict_transfer(img_path):
    # load the image and return the predicted result
    img = Image.open(img_path).convert('RGB')
    size = (224, 224) # ResNet image size requirements
    transform_chain = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))
    ])

    img = transform_chain(img).unsqueeze(0)

    if use_cuda:
        img = img.cuda()

    model_out = model_transfer(img)

    if use_cuda:
        model_out = model_out.cpu()
    
    prediction = torch.argmax(model_out)
    
    return class_names[prediction]  # predicted class label
  ```
 ### Step 7: Write your Algorithm 
 Write the run_app that an image of a human an print ```This person is responsible, he wears his face mask!!!!``` when a that person is wearing a face 
 and print ``` This person is irresponsible, he does not wear his face mask!!!!!``` when a that does not have a face mask.
 

```$xslt
import matplotlib.pyplot as plt 
def run_app(img_path):
   
    result = predict_transfer(img_path)
    #print(result )
    # display the image, along with bounding box
    if result == " mask" :
        print('This person is responsible, he wears his face mask!!!!!')
    else :
        print('This person is irresponsible, he does not wear his face mask!!!!!')
    img = plt.imread(img_path, 3)
    plt.imshow(img)
    plt.show()
```
### Step 8: test our function run_app 
We can now use how test dataset to test our system.

```$xslt
## Execute your algorithm from Step 6 on
## at least 6 images on your computer.
## Feel free to use as you want.

for file in np.array(glob("../input/covid19-face-mask-recognition-test-data/Covid19-face-mask-recognition-test-data/*")):
    run_app(file)
```
<img src="https://github.com/achilep/Face-Mask-Recognition-System-Tutorial-With-Pytorch/blob/main/Resource/readme_image/test result1.png" alt="result of the predition"/>
<img src="https://github.com/achilep/Face-Mask-Recognition-System-Tutorial-With-Pytorch/blob/main/Resource/readme_image/test result1.png" alt="result of the predition"/>
<img src="https://github.com/achilep/Face-Mask-Recognition-System-Tutorial-With-Pytorch/blob/main/Resource/readme_image/test result1.png" alt="result of the predition"/>

### Step 9: optional integrate opencv to the project
Write the run_app_with_opencv method that an image of a human an print ```This person is responsible, he wears his face mask!!!!``` when a that person is wearing a face 
 and print ``` This person is irresponsible, he does not wear his face mask!!!!!``` when a that does not have a face mask. and in addition located the highlight the face of a person .
 ```
import matplotlib.pyplot as plt 
import cv2                                       
%matplotlib inline 
def run_app_with_opencv(img_path):
    # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier('../input/xmldoc/haarcascade_frontalface_default.xml')

    # load color (BGR) image
    img = cv2.imread(img_path)
    # convert BGR image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find faces in image
    faces = face_cascade.detectMultiScale(gray)

    # get bounding box for each detected face
    for (x,y,w,h) in faces:
        # add bounding box to color image
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    # convert BGR image to RGB for plotting
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = predict_transfer(img_path)
    # display the image, along with bounding box
    if result == " mask" :
        print("This person is responsible, he wears his face mask!!!!!" )
    else :
        print('This person is irresponsible, he does not wear his face mask!!!!!')

    plt.imshow(cv_rgb)
    plt.show()
 ```
 ### Step 10: Test Your Algorithm
 you can use one image to test ```run_app_with_opencv```
 ```
## Execute your algorithm from Step 6 
## on 1 images on your computer.
## Feel free to use as many you want.
for file in np.array(glob("../input/covid19-face-mask-recognition-test-data/Covid19-face-mask-recognition-test-data/4.jpeg")):
    run_app_with_opencv(file)
  ```
<img src="https://github.com/achilep/Face-Mask-Recognition-System-Tutorial-With-Pytorch/blob/main/Resource/readme_image/test result with opencv.png" alt="result of the predition"/>

## summary 
we have learn how to use pre- train model to speed up the training of our model. 

## future work 
the project can be used in the public service to control people that are entered, the make sure that they have they face mask.
1. we can integrate our model with a webcam or video camera using OpenCV.
2. we can integrate a notification system.
2. we can integrate our model in automation door open, in such a way that the door will open only when a person is wearing a face mask.
3. we can use it in school to make sure that the student always wears the face mask. 
 
 ## Resources 
 The notebook code can be found [here](https://github.com/achilep/Face-Mask-Recognition-System-Tutorial-With-Pytorch/blob/main/face-mask-detection.ipynb)
 The French version of the tutorial can be found [here](https://github.com/achilep/Tutoriel-Sur-Un-System-De-Reconnaissance-Du-Cache-Nez-Avec-pytorch) 
