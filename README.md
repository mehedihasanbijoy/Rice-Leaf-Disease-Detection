# 
<h1 align="center">Towards Sustainable Agriculture: A Novel Approach for Rice Leaf Disease Detection using dCNN and Enhanced Dataset</h1>
<p align="center">
  Paper — <a href="https://scholar.google.com/" target="_blank">link</a>
</p>

## Get Started
```
git clone https://github.com/mehedihasanbijoy/Rice-Leaf-Disease-Detection/
```
or manually **download** and **extract** the github repository of the paper.

<br>

## Download the Enhanced Rice Leaf Disease Dataset
```
gdown https://drive.google.com/drive/folders/1ZYldfJSbqCEEJfmvqFMvn37WRlik9-1Z?usp=sharing -O ./Dataset --folder
```
<p>
or manually <b>download</b> the folder from <a href="https://drive.google.com/drive/folders/1ZYldfJSbqCEEJfmvqFMvn37WRlik9-1Z?usp=sharing" target="_blank">here</a> and keep the extracted files into <b>./Dataset/</b>
</p>

## Train, Validate, and Test the proposed model
```
python main.py
```

## Train, Validate, and Test the benchmark models
* Go to **./Rice-Leaf-Disease-Detection/comparison/**
* Open the desired model, **ResNet50.ipynb** for example, with google colab
* Enable GPU acceleration: **Runtime > Change Runtime Type > T4 GPU** 
* Run all the cells: **Runtime > Run All** 
