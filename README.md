# State-Farm-Distracted-Driver-Detect-Pytorch
This project is from a kaggle competition in 2016, we use mobileNet V2 to achieve classification recognition based on pytorch.  
***  

## 1、Brief  
---  
### 1、1 project source  
The implementation of the [State Farm Distracted Driver Detection] competition in kaggle.  
### 1、2 Task content  
Classify a picture of driver behavior, a total of 10 categories  
The 10 classes to predict are:  


c0: safe driving  
c1: texting - right  
c2: talking on the phone - right  
c3: texting - left  
c4: talking on the phone - left  
c5: operating the radio  
c6: drinking  
c7: reaching behind  
c8: hair and makeup  
c9: talking to passenger  
  
  
![](https://raw.githubusercontent.com/TachibanaYoshino/State-Farm-Distracted-Driver-Detect-Pytorch/master/test_output/img_13.jpg?token=AIZ4GWOSUUQ64FREA4XKCCS4YUXVM "")  

### 1、3 Method used  
Training and identification with MobileNet V2  
### 1、4 Dataset download address:  
*https://www.kaggle.com/c/state-farm-distracted-driver-detection/data*
***
## 2、Project configuration  
---  
### 2、1 Operating environment  
1: Download the repository to the local;  
2: In this directory, create a folder as follows, and download the train dataset to the data folder:  
>---data  
>------train
  
### 2、2 Python dependency package  
> pytorch1.0.1  
> easydict  
> torchvision  
> PIL  
> python-opencv  
> pip install --upgrade git+https://github.com/Lyken17/pytorch-OpCounter.git    
  
### 2、3 Running  
2、3、1 training  
+ *python train.py* 

2、3、2 testing  
+ Select the model to load, write it to test.py, then,  then  *python test.py*
