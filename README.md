# Credit_Risk_Analysis
## 1. Overview
FastLending, a peer to peer lending services company wants to use machine learning to predict credit risk. By using machine learning, the management believes it will provide a quicker and more reliable loan experience, and will lead to a more accurate identification of good candidates for loans which will lead to lower default rates. In this project, we will build and evaluate several machine learning models to predict credit risk. 

The following tasks will be performed for the analysis:

1.Oversample the data using the RandomOverSampler and SMOTE algorithms

2.Undersample the data using the ClusterCentroids algorithm

3.Use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm.

4.Compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier

## 2.Resources
Data Source: LoanStats_2019Q1.csv

Software: Python, Jupyter Notebook


## 3. Result

### Naive Random Oversampling

<p align="center">
    <img src="https://user-images.githubusercontent.com/88597187/146665836-13929d4f-08e3-4781-badc-a0f798489178.png"  width="500" height="100"/>
</p>

<p align="center">
  <sub>Figure 1 Imbalance Classification Report Naive Random Oversampling </sub>
</p>

### SMOTE Oversampling

<p align="center">
    <img src="https://user-images.githubusercontent.com/88597187/146665876-2eb7f4bf-1c27-4475-873a-ba893f4081e1.png"  width="500" height="100"/>
</p>

<p align="center">
  <sub>Figure 2 Imbalance Classification Report SMOTE Oversampling </sub>
</p>


### Cluster Centroids Undersampling

<p align="center">
    <img src="https://user-images.githubusercontent.com/88597187/146665902-bde19196-1f7e-4e44-a164-8b4029e95d08.png"  width="500" height="100"/>
</p>

<p align="center">
  <sub>Figure 3 Imbalance Classification Report Cluster Centroids Undersampling </sub>
</p>

### SMOTEENN Under and Over Sampling

<p align="center">
    <img src="https://user-images.githubusercontent.com/88597187/146665939-d4b2aaa9-c808-4d06-ae69-c6a0d92f35a1.png"  width="500" height="100"/>
</p>

<p align="center">
  <sub>Figure 4 Imbalance Classification Report SMOTEENN Under and Over Sampling </sub>
</p>


### Balanced Random Forest Classifier

<p align="center">
    <img src="https://user-images.githubusercontent.com/88597187/146665949-46d0ea32-60ef-4026-8d4a-cf06b0e80f3c.png"  width="500" height="100"/>
</p>

<p align="center">
  <sub>Figure 5 Imbalance Classification Report Balanced Random Forest Classifier </sub>
</p>


### Easy Ensemble AdaBoost Classifier

<p align="center">
    <img src="https://user-images.githubusercontent.com/88597187/146665996-cc37bd88-b1a4-457f-8102-3990a566acc0.png"  width="500" height="100"/>
</p>

<p align="center">
  <sub>Figure 6 Imbalance Classification Report Easy Ensemble AdaBoost Classifier</sub>
</p>




## 4. Summary
