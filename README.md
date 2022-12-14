# EEG Emotion Recognition

## Introduction

This is a project of "Neurotechnology and Affective Computing" course for the third year in university.  
Project Report: [Comparison of Effectiveness of Several Classifiers in EEG
Emotion Recognition](https://denisandgzh.github.io/EEG-Emotion-Recognition/)  

## Setup And Run

In order to run feature extraction you need to request the [Seed Dataset](https://bcmi.sjtu.edu.cn/home/seed/seed.html), you need to apply for it, download the dataset and copy the SEED folder to the root directory.  
Or you can also use the features already extracted in TrainData.  

To run Model:  

```shell
cd EmotionClassifierModel
python Models.py
```

To run Application:

```shell
cd EmotionClassifierApplication
python App.py
```

## Project Objectives

In this project we will use EEG signals for emotion recognition. Through different kinds of classifiers, analyze their accuracy and performance and draw conclusions.  
After that, We will use the trained model to build the application.

## Experimental Equipment and Datasets

EEG Equipment: Open BCI 16 channels  
Datasets: Seed

## Library Required

1. [NumPy](https://numpy.org/)  
2. [SciPy](https://scipy.org/)  
3. [scikit-learn](https://scikit-learn.org/stable/)
4. [matplotlib](https://matplotlib.org/)

For running Application, you may need to install extra package:  

1. For Window you need to check in the Python install the optional feature "tcl/tk and IDLE", while installing python.  
2. For mac you need to install python-tk:

    ```shell
    brew install python-tk@(Your python version)
    ```  

## Participants

Suvorov Denis Vitalievich / Суворов Денис Витальевич  
![VK](https://img.shields.io/badge/VK-denissvvv-green)
![Email](https://img.shields.io/badge/mail-erkobraxx%40gmail.com-blue)  
Guo ZiHan / Го Цзыхань  
![VK](https://img.shields.io/badge/VK-zjjhgzh-green)
![Email](https://img.shields.io/badge/mail-zjjhgzh%40gmail.com-blue)  

## Notes

If you find any mistakes or areas that can be improved, we are welcome to receive issue from you.
