# Data Science Portfolio

This portfolio has a list of my current projects up to date; They're all related to real world problems. In this portfolio, you'll see solutions using the following models: Linear Regression, Logistic Regression, Bagging & Boosting, convolution  deep learning, natural language processing with Neuro Network, Random Forest, Clustering, etc. Most of these projects used supervised learning situations. Nevertheless an unsupervised learning method was used on a real world situation dealing with policy improvement for employee's.

***

[Examining the effect of environmental factors and weather on Bike rentals](https://github.com/shyheim9294/ShyHeim-Gray.github.io/blob/master/Linear_Regression_Project.ipynb)

<img src="images/seoul-bikes.jpeg?raw=true"/>

- Used Linear Regression to predict the number of bikes rented in the city of Seoul
- The data had quite a few categorical variables which were encoded for use in the model
- Encoded categorical variables to numeric using Sklearn due to the presence of many string columns
- Fit a multiple linear regression model with high prediction accuracy through iteration
  - Results : 
    - Mean Absolute Error of linear regression = 322.49952202535525
    - Mean Square Error of linear regression = 187563.13400959878
    - R_Squard Score of linear regression = 0.18611407115471357


***

[Predicting customer subscription for term deposit](https://github.com/shyheim9294/ShyHeim-Gray.github.io/blob/master/Logistic%20Regression%20Projecttt.ipynb)

<img src="images/bank-logo.jpg?raw=true"/>

- Used logistic regression to predict customer subscriptions for there term deposit (Used real worked data from UCI, machine learning repository)
- Loaded the data
- Used Logistic regression classifier & optimized the accuracy by using the ROC curve
- Explored a machine learning approach to financal needs
    - Result : 
        - Accuracy Score = 90.68%
      
***

[Identifying symptoms of orthopedic patients as normal or abnormal](/https://github.com/shyheim9294/ShyHeim-Gray.github.io/blob/master/Knn_Nb_Project.ipynb)

<img src="images/knee-brace-ortho.png?raw=true"/>

- Used the K Nearest Neighbours algorithm to classify a patient's condition as normal or abnormal based on various orthopedic parameters
- Compared predictive performance by fitting a Naive Bayes model to the data
- Selected best model based on train and test performance
    - Result :
       - Correct Predictions = 63
       - False Predictions = 15
       - Accuracy of Naive Bayes Clasification is 80.77%

***

[Predicting the Proability of false clicks](https://github.com/shyheim9294/ShyHeim-Gray.github.io/blob/master/Bagging_Boosting_Projectt.ipynb)

<img src="images/Click-Mouse.jpg?raw=true"/>

- Used Bagged and Boosted algorithims to predict the proability of false clicks
- Explored the dataset anomolies and missing values
- Used the necessary techniques to get rid of the apps that very rare (comprised of less than 20% clicks), plotted the rest
- Divided the data into training and testing subsets (80-20), and checked the average download rates for those subsets
- Applied XGBoostClassifier on training data to make prediction on test data
- Applied BaggingClassifier Logistic Regression to compute the ROC/AUC score
- Finally determined the accuracy of the XGBoostClassifier and BaggingClassifier
    - Result : 
       - Accuracy of thhe decision Tree is 99.8%

***

[Clustering Model helping show the policy makers how they can improve their policies for the employee's benefit](https://github.com/shyheim9294/ShyHeim-Gray.github.io/blob/master/K_Means_Projectt.ipynb)

<img src="images/Kmeans-logo.jpg?raw=true"/>

- Reduced the necessary columns
- Cleaned up the data
- Detected the parameters for the clustering algorithm
- Based on the models gave some insight on what could be improved  
    - Results :
       - K-value = 3
       - Silhousette = 0.53164

***

[Predicting the price of houses](https://github.com/shyheim9294/ShyHeim-Gray.github.io/blob/master/PCA_Projectt.ipynb)

<img src="images/PCA-logo.jpg?raw=true"/>

- Extract numerical columns
- Removed the unnecssary columns
- Checked for missing values
- Scaled the data
- Apply PCA to the cleaned up data to increase intrepretability; while minimizing information lost
    
    - Result :
        - Linear Regression proved to be better model on the orginal datset, afterwards transformed the data to smaller features using PCA
        - Original - 81 Features - R2 = 0.06444
        - PCA - 2 Features - R2 = 0.07144
      

***

[Being able to determine wheather a review is postive or negative](https://github.com/shyheim9294/ShyHeim-Gray.github.io/blob/master/NLP%20Projecttt.ipynb)

<img src="images/NLP_Logo.jpg?raw=true"/>

- Tokenize Test
- Syntactic Analysis
- Semantic Analysis
- Discourse Integration
- Pragmatic Analysis
     - Result : 
        - BoW - Best Alpha - 0.005, AUC Score - 0.9412
        - Tf-Idf - Best Alhpa - 0.01, AUC Score - 0.9254
     - Accuracy Score :
        - BoW = 94.1%
        - Tf-Idf = 92.5%

***

[Implementing Deep Neutral Network with Keras for handwritting classification and recognition](https://github.com/shyheim9294/ShyHeim-Gray.github.io/blob/master/DNN%20Handwriting%20Recognition.ipynb)

<img src="images/Dnn_logo.jpeg?raw=true"/>

- Load the data
- Initial data indagation
- Initial EDA
- Data Prepping
- Model contruction
- Model eval
- Model eval metrics
- Improved the model
- Improved the depth of model
  - Result: 
    - Loss - 0.037 - 0.9909 = Final Model Evaluation - 99.0899%
        
