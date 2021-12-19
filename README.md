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

In random oversampling, instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced.

<p align="center">
    <img src="https://user-images.githubusercontent.com/88597187/146665836-13929d4f-08e3-4781-badc-a0f798489178.png"  width="400" height="100"/>
        <img src="https://user-images.githubusercontent.com/88597187/146666378-1af17795-0fc4-41ef-8311-8a39ce3bd77f.png"  width="400" height="100"/>
</p>

<p align="center">
  <sub>Figure 1 Imbalance Classification Report & Balanced Accuracy Score Naive Random Oversampling </sub>
</p>

* Balanced accuracy score:0.66
* Precision: 

    High risk: 0.01
    
    Low risk: 1.00
    
    Average/total :0.99

* Recall:

    High risk: 0.71
    
    Low risk: 0.60
    
    Average/total: 0.60



### SMOTE Oversampling

The synthetic minority oversampling technique (SMOTE) is another oversampling approach to deal with unbalanced datasets. In SMOTE, like random oversampling, the size of the minority is increased. The key difference between the two lies in how the minority class is increased in size. In SMOTE, new instances are interpolated. That is, for an instance from the minority class, a number of its closest neighbors is chosen. Based on the values of these neighbors, new values are created.

<p align="center">
    <img src="https://user-images.githubusercontent.com/88597187/146665876-2eb7f4bf-1c27-4475-873a-ba893f4081e1.png"  width="400" height="100"/>
     <img src="https://user-images.githubusercontent.com/88597187/146666481-5c2ac806-439b-4ad5-89ca-8478c9fc4fd0.png"  width="400" height="100"/>
</p>

<p align="center">
  <sub>Figure 2 Imbalance Classification Report & Balanced Accuracy Score SMOTE Oversampling </sub>
</p>

* Balanced accuracy score:0.66
* Precision: 

    High risk: 0.01
    
    Low risk: 1.00
    
    Average/total: 0.99

* Recall:

    High risk: 0.63
    
    Low risk: 0.69
    
    Average/total: 0.69




### Cluster Centroids Undersampling

Cluster centroid undersampling is akin to SMOTE. The algorithm identifies clusters of the majority class, then generates synthetic data points, called centroids, that are representative of the clusters. The majority class is then undersampled down to the size of the minority class.

<p align="center">
    <img src="https://user-images.githubusercontent.com/88597187/146665902-bde19196-1f7e-4e44-a164-8b4029e95d08.png"  width="400" height="100"/>
    <img src="https://user-images.githubusercontent.com/88597187/146666557-24f9116f-cf2b-481e-a2f1-167da0ec23f6.png"  width="400" height="100"/>
</p>

<p align="center">
  <sub>Figure 3 Imbalance Classification Report & Balanced Accuracy Score Cluster Centroids Undersampling </sub>
</p>

* Balanced accuracy score:0.54
* Precision: 

    High risk: 0.01
    
    Low risk: 1.00
    
    Average/total: 0.99

* Recall:

    High risk: 0.69
    
    Low risk: 0.40
    
    Average/total:0.40



### SMOTEENN Under and Over Sampling
SMOTEENN combines the SMOTE and Edited Nearest Neighbors (ENN) algorithms. SMOTEENN is a two-step process:
* Oversample the minority class with SMOTE.
* Clean the resulting data with an undersampling strategy. If the two nearest neighbors of a data point belong to two different classes, that data point is dropped.


<p align="center">
    <img src="https://user-images.githubusercontent.com/88597187/146665939-d4b2aaa9-c808-4d06-ae69-c6a0d92f35a1.png"  width="400" height="100"/>
    <img src="https://user-images.githubusercontent.com/88597187/146666601-da531832-449d-4f2f-8211-147c2713cef0.png"  width="400" height="100"/>
</p>

<p align="center">
  <sub>Figure 4 Imbalance Classification Report & Balanced Accuracy Score SMOTEENN Under and Over Sampling </sub>
</p>

* Balanced accuracy score:0.69
* Precision: 

    High risk: 0.01
    
    Low risk: 1.00
    
    Average/total: 0.99

* Recall:

    High risk: 0.80
    
    Low risk: 0.57
    
    Average/total:0.57






### Balanced Random Forest Classifier

A balanced random forest perform data resampling on the bootstrap sample in order to explicitly change the class distribution.


<p align="center">
    <img src="https://user-images.githubusercontent.com/88597187/146665949-46d0ea32-60ef-4026-8d4a-cf06b0e80f3c.png"  width="400" height="100"/>
     <img src="https://user-images.githubusercontent.com/88597187/146666653-8b0551b6-0d65-49a5-bcfc-0902c9cc88d7.png"  width="400" height="100"/>
</p>
</p>

<p align="center">
  <sub>Figure 5 Imbalance Classification Report & Balanced Accuracy Score Balanced Random Forest Classifier </sub>
</p>


* Balanced accuracy score:0.79
* Precision: 

    High risk: 0.03 
    
    Low risk: 1.00
    
    Average/total: 0.99

* Recall:

    High risk: 0.70
    
    Low risk: 0.87
    
    Average/total:0.87


### Easy Ensemble AdaBoost Classifier

In AdaBoost, a model is trained then evaluated. After evaluating the errors of the first model, another model is trained. This time, however, the model gives extra weight to the errors from the previous model. The purpose of this weighting is to minimize similar errors in subsequent models. Then, the errors from the second model are given extra weight for the third model

<p align="center">
    <img src="https://user-images.githubusercontent.com/88597187/146665996-cc37bd88-b1a4-457f-8102-3990a566acc0.png"  width="400" height="100"/>
    <img src="https://user-images.githubusercontent.com/88597187/146666694-6973a92a-a696-44ed-942e-98c93e5642c2.png"  width="400" height="100"/>
</p>
</p>

<p align="center">
  <sub>Figure 6 Imbalance Classification Report & Balanced Accuracy Score Easy Ensemble AdaBoost Classifier</sub>
</p>

* Balanced accuracy score:0.93
* Precision: 

    High risk: 0.09  
    
    Low risk: 1.00
    
    Average/total: 0.99

* Recall:

    High risk: 0.92 
    
    Low risk: 0.94
    
    Average/total:0.94



## 4. Summary
