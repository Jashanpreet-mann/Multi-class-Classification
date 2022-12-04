# Multiclass Classification Tutorial (MCLF101) - Level Beginner
#Created using: PyCaret 2.0
 **Date Updated: November 30, 2022


# Tutorial Objective
Welcome to the Multiclass Classification Tutorial (MCLF101) - Level Beginner. This tutorial assumes that you are new to PyCaret and looking to get started with Multiclass Classification using the pycaret.classification Module.

In this tutorial we will learn:

Getting Data: How to import data from outise of the PyCaret repository
Setting up Environment: How to setup an experiment in PyCaret and get started with building multiclass models
Create Model: How to create a model, perform stratified cross validation and evaluate classification metrics
Tune Model: How to automatically tune the hyper-parameters of a multiclass model
Plot Model: How to analyze model performance using various plots
Finalize Model: How to finalize the best model at the end of the experiment
Predict Model: How to make predictions on new / unseen data
Save / Load Model: How to save / load a model for future use

**Read Time : Approx. 30 Minutes

# Installing PyCaret on Google Colab or Azure Notebooks
*!pip install numba==0.53
*!pip install pycaret

# Abstract 
Finding the ML model which is accurate on unseen data is quite long procedure and engineers have to go through numerous steps to create it.  This project will show how to find the best model using the pycaret library. It is the multi-class classification where based on the different features of the wine we can predict the quality of the wine. 

The whole machine learning pipeline has been covered in this lesson, including data import, preprocessing, model training, hyperparameter tweaking, prediction, and model storing for future use. With commands like create_model(), tune_model(), and compare_models that are organically built and easy to remember, we were able to perform all of these procedures in under ten commands. Without PyCaret, it would have required far more than 100 lines of code to recreate the entire experiment.But as a reminder only the fundamentals of pycaret.classification have been discussed. 


# Feature Engineering Done
I have used the red wine dataset, the dataset was simple and has numerical feature and based on these features I need to predict the quality of the wine which is categorical feature but has score from 0-10.The dataset was easy to use and do-not need to do any preprocessing and other step.

# youtube Video:
https://youtu.be/k8A-FZzIbMk

# Performance Matrices
Although it really depends on the situation what matrix for comparing the I models for example in detection of covid is serious so we should use recall. capture as many postives as we want to prevent the risk

But in general I am using F1 score to compare models because F1 score is the harmonic mean of Recall and Precision, therefore it balances out the strengths of each. and we knwo recall and Precision are modified version of accuracy so indirectly I am comparing three different matrices. 

F1 = 2*((precision*recall)/(precision+recall)) 



Although my best model for the training dataset was random forest classifier but on unseen data the best model is Logistic Regression with the accuracy of 0.6, recall = 0.27 precision 0.56, f1= 0.577 whereas Random Forest classifier gave accuracy of 0.5125, recall = 0.2593 precision 0.5188 f1= 0.511. All other models were behind them 

Because of the consistent performance of Logistic regression in both Training and Testing/unseen data I will consider this model as best. 


# Literature Review
Literature Review :
Introduction:
Many customers nowadays are appreciating wine to ever-greater degrees. In order to support this advancement, the wine industry is investigating new developments in wine production and providing structures.I have pasted the link for the papers that used the same dataset for their research. Performance matrix used by the papers are Accuracy, Precision, Recall, Specificity, F1 Score and Misclassification Error. 

Techniques Involved  in Research :
a. Naive Bayes Algorithm: Naive Bayes algorithm relies upon  bayes  speculation. To  find  whether  a particular part  has  a  spot  with  a  particular  class  it  utilizes  the possibility  of  likelihood

b. Support  Vector  Machine:  This  technique  was  taken from  factual  learning  theory  by  Vapnik  and Chervonenkis. It  was  first  exhibited  in 1992  by Boser Guyon and  Vapnik. This  technique is  utilized for  the characterization  of  both  nonlinear  and  linear information. 

c. Random Forest: This technique utilizes a blend of tree indicators;  each  individual  tree  depends  upon  an random  vector.  This  arbitrary  vector  has indistinguishable and  a similar circulation for  all trees 

d. Gaussian Process: Gaussian processes are resilient to overfitting, so we include all predictors, quadratics, and their interaction terms as covariates. Due to the very large
amount of observations in the Training set (n = 4331), the fitting procedure for the
Gaussian process took several hours to run due to inverting the n × n covariance
matrix in Equation (3.12) in finding maximum likelihood parameter estimates.


Performance : 
Training set of red wine using Naive Bayes Algorithm 
Accuracy : 0.559158
Error : 0.44

Testing set of red wine using Naive Bayes Algorithm 
Accuracy : 0.558952
Error : 0.441048

Training set of red wine using Support Vector Machine algorithm. 
Accuracy : 0.6725821
Error : 0.3274179

Testing set of red wine using Support Vector Machine algorithm. 
Accuracy : 0.6864407
Error : 0.3135593


Training set of red wine using Random Forest algorithm.
Accuracy : 0.6583851
Error : 0.3416149

Testing set of red wine using Random Forest algorithm.
Accuracy : 0.654661
Error : 0.3415339

 Research 2 Results: 

Usage of Linear Model for Wine Dataset for Prediction : 
a. LM1: this model performs best subset selection using only linear coefficients
on the original p = 12 predictors.

b.  LMBEST: this model performs best subset selection using all original predictors, the quadratic terms mentioned above, and all possible interaction terms.

Predictions and Results :
All methods clearly have issues when Training set Quality is abnormally large or small.For Quality less than 6, the predictions yˆi are mainly above the observed Quality yi. For Quality greater than 6, the predictions yˆi are mainly below the observed Quality yi.

Model-------MSEtr-------------- R2tr------- 5−Fold CV
LM1 -------0.5259 -------0.2935------- 0.5291
LMBEST -------0.5012 -------0.3267 -------0.5149
LMLASSO------- 0.5114 -------0.3129 -------0.5055
GP------- 0.3527 -------0.5261------- 0.4475

Model Prediction on Training set





Model-------MSEte------- R2te------ 
LM1 -------1.052 -------0.2186
LMBEST -------0.6239 -------0.2263
LMLASSO------- 0.6117 -------0.2986
GP------- 0.4706 -------0.4107

Model Prediction on Test set




Research 3 Results: 

Training Results :
SVM Model Prediction on Train Set :
Accuracy : 0.6372 
95% CI : (0.6133, 0.6606)
No Information Rate : 0.4494 
P-Value [Acc > NIR] : < 2.2e-16       
Kappa : 0.4157      

Random Forest Model Prediction on Train Set :
Accuracy : 0.6839           
95% CI : (0.6607, 0.7064)
No Information Rate : 0.4494       
P-Value [Acc > NIR] : < 2.2e-16      
Kappa : 0.4985 


KNN Model Prediction on Train Set :
 Accuracy : 0.6053         
95% CI : (0.5811, 0.6291)
No Information Rate : 0.4494
P-Value [Acc > NIR] : < 2.2e-16       
Kappa : 0.4023



Testing Results :
SVM Model Prediction on Test Set :
Accuracy : 0.646           
95% CI : (0.6036, 0.6867)
No Information Rate : 0.4275          
P-Value [Acc > NIR] : < 2.2e-16       
Kappa : 0.4284        

Random Forest Model Prediction on Test Set :
Accuracy : 0.6874         
95% CI : (0.646, 0.7266)
No Information Rate : 0.4275         
P-Value [Acc > NIR] : < 2.2e-16      
Kappa : 0.4955 


KNN Model Prediction on Test Set :
 Accuracy : 0.629           
95% CI : (0.5863, 0.6702)
No Information Rate : 0.4275          
P-Value [Acc > NIR] : < 2.2e-16       
Kappa : 0.4124  






References:

1.  https://scholarworks.calstate.edu/downloads/mg74qp13p 
2. https://www.researchgate.net/publication/341812162_Red_Wine_Quality_Prediction_Using_Machine_Learning_Techniques
3. https://rstudio-pubs-static.s3.amazonaws.com/175762_83cf2d7b322c4c63bf9ba2487b79e77e.html 

