# Data Science Portfolio

This portfolio has a list of my current projects up to date; They're all related to real world problems. In this portfolio, you'll see solutions using the following models: Linear Regression, Logistic Regression, Bagging & Boosting, convolution  deep learning, natural language processing with Neuro Network, Random Forest, Clustering, etc. Most of these projects used supervised learning situations. Nevertheless an unsupervised learning method was used on a real world situation dealing with policy improvement for employee's.

***

[Examining the effect of environmental factors and weather on Bike rentals](https://github.com/shyheim9294/ShyHeim-Gray.github.io/blob/master/Linear_Regression_Project.ipynb)

<img src="images/seoul-bikes.jpeg?raw=true"/>

<b>Skills Used</b>: Python, Numpy, Pandas, Matplotlib, Seaborn, Sklearn, Linear Regression

<b>Project Objective</b>: Used Linear regression to predict the number of bikes rented in the city of Seoul

<b>Quantifiable Results</b>: Based on the linear Regression model with high prediction accuracy through iteration 
  - Mean Absolute Error of linear regression = 322.49952202535525
  - Mean Square Error of linear regresssion = 187563.13400959878
  - #1589F0[R_Squard Score of linear regression = 0.18611407115471357](https://github.com/shyheim9294/ShyHeim-Gray.github.io/blob/master/Linear_Regression_Project.ipynb)

    - The data had quite a few categorical variables which were encoded for use in the model
    - Encoded categorical variables to numeric using Sklearn due to the presence of many string columns
    - Fit a multiple linear regression model with high prediction accuracy through iteration


***

[Predicting customer subscription for term deposit](https://github.com/shyheim9294/ShyHeim-Gray.github.io/blob/master/Logistic%20Regression%20Projecttt.ipynb)

<img src="images/bank-logo.jpg?raw=true"/>

<b>Skills Used</b>: Python, Numpy, Pandas, Matplotlib, Seaborn, Sklearn, Logistic Regression

<b>Project Objective</b>: Used logistic regression to predict customer subscriptions for there term deposit (Used real worked data from UCI, machine learning repository)

<b>Quantifiable Results</b>
  - Accuracy Score = 90.68%
  - Loaded the data
  - Used Logistic regression classifier & optimized the accuracy by using the ROC curve
  - Explored a machine learning approach to financal needs
      
***

[Identifying symptoms of orthopedic patients as normal or abnormal](/https://github.com/shyheim9294/ShyHeim-Gray.github.io/blob/master/Knn_Nb_Project.ipynb)

<img src="images/knee-brace-ortho.png?raw=true"/>

<b>Skills Used</b>: Python, Numpy, Pandas, Matplotlib, Seaborn, Sklearn, Logistic Regresssion

<b>Project Objective</b>: Used the K Nearest Neighnours algorithm to classify a patients condition as normal or abnormal based on various orthopedic parameters

<b>Quantifiable Results</b>
  - Correct Predictions = 63
  - False Predictions = 15
  - Accuracy of Navie Bayes Clasification is 80.77%
  - Compared predictive performance by fitting a Naive Bayes model to the data
  - Selected best model based on train and test performance

***

[Predicting the Proability of false clicks](https://github.com/shyheim9294/ShyHeim-Gray.github.io/blob/master/Bagging_Boosting_Projectt.ipynb)

<img src="images/Click-Mouse.jpg?raw=true"/>

<b>Skills Used</b>: Pandas, Numpy, Seaborn, Matplotlib

<b>Project Objective</b>: Used Bagged and Boosted algorithims to predict the proability of false clicks

<b>Quantifiable Results</b>
  - Accuracy of the Decision Tree is 99.8%
  - Explored the dataset anomolies and missing values
  - Used the necessary techniques to get rid of the apps that very rare (comprised of less than 20% clicks), plotted the rest
  - Divided the data into training and testing subsets (80-20), and checked the average download rates for those subsets
  - Applied XGBoostClassifier on training data to make prediction on test data
  - Applied BaggingClassifier Logistic Regression to compute the ROC/AUC score
  - Finally determined the accuracy of the XGBoostClassifier and BaggingClassifier


***

[Clustering Model helping show the policy makers how they can improve their policies for the employee's benefit](https://github.com/shyheim9294/ShyHeim-Gray.github.io/blob/master/K_Means_Projectt.ipynb)

<img src="images/Kmeans-logo.jpg?raw=true"/>

<b>Skills Used</b>: Numpy, Pandas, Seaborn, matplotlib

<b>Project Objective</b>: Based on the models was able to give some insight on what could be improved 

<b>Quantifiable Results</b>
  - K-value = 3
  - Silhousette = 0.53164
  - Reduced the necessary columns
  - Cleaned up the data
  - Detected the parameters for the clustering algorithm  


***

[Predicting the price of houses](https://github.com/shyheim9294/ShyHeim-Gray.github.io/blob/master/PCA_Projectt.ipynb)

<img src="images/PCA-logo.jpg?raw=true"/>

<b>Skills Used</b>: Numpy, Pandas, Sklearn, Seaborn

<b>Project Objective</b>: Applying PCA to the cleaned up data to increase intrepretability; while minimizing information lost

<b>Quantifiable Results</b>
  - Linear Regrwession proved to be the better model for the orginal dataset, afterwards transfomred to data to smaller features using PCA
  - Orginal - 81 Features - R2 = 0.06444
  - PCA - 2 Features - R2 = 0.07144
  - Extract numerical columns
  - Removed the unnecssary columns
  - Checked for missing values
  - Scaled the data
    

      

***

[Being able to determine wheather a review is postive or negative](https://github.com/shyheim9294/ShyHeim-Gray.github.io/blob/master/NLP%20Projecttt.ipynb)

<img src="images/NLP_Logo.jpg?raw=true"/>

<b>Skills Used</b>: Pandas, Numpy, Sklearn, SQ lite, Seaborn, Matplotlib, Natural Learning

<b>Projedct Objective</b>: Being able to determine wheatheer a review is positive or negative while using Natural Leraning methods

<b>Quantifiable Results</b>
  - Acurracy Score - BoW = 94.1%, Tf-Idf = 92.5%
  - BoW - Best Alpha - 0.005, AUC Score - 0.9412
  - Tf-Idf - Best Alpha - 0.01, AUC Score - 0.9254 
  - Tokenize Test
  - Syntactic Analysis
  - Semantic Analysis
  - Discourse Integration
  - Pragmatic Analysis

***

[Implementing Deep Neutral Network with Keras for handwritting classification and recognition](https://github.com/shyheim9294/ShyHeim-Gray.github.io/blob/master/DNN%20Handwriting%20Recognition.ipynb)

<img src="images/Dnn_logo.jpeg?raw=true"/>

<b>Skills Used</b>: Matplotlib, numpy, Seaborn, Sklearn, Python, Neural Networking 

<b>Project Objective</b>: Implementing Deep Neural Network with Keras for handwritting classification and recognition

<b>Quantifiable Results</b>
  - Loss = 0.037 - 0.9909 = Final Model Eval = 99.0899%
  - Load the data
  - Initial data indagation
  - Initial EDA
  - Data Prepping
  - Model contruction
  - Model eval
  - Model eval metrics
  - Improved the model
  - Improved the depth of model
        
