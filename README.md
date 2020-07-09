# Python-DS-ML-Bootcamp
The projects in this repo are based on Jose Portilla (Pierian Data)'s class "Python for Data Science and Machine Learning Bootcamp." Many of the data sets used and instruction for the code is provided by Jose Portilla for the class. The projects uploaded here were written by me, with instruction based on what we were learning in class.

Class here: https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/ Pierian Data: https://www.pieriandata.com

# Support Vector Machines Project
Based on the series of class lectures and course materials, I created an analysis of the Iris flower data set. This data set was used in order to train a model and evaluate this dataset by building a Support Vector Machine Classifier. By doing this grid search and optimizing the parameters ('C': 10, 'gamma': 0.01) I was able to improve the precision, recall, and f1-score.

              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        13
  versicolor       1.00      0.95      0.97        20
   virginica       0.92      1.00      0.96        12

    accuracy                           0.98        45
   macro avg       0.97      0.98      0.98        45
weighted avg       0.98      0.98      0.98        45



              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        13
  versicolor       1.00      1.00      1.00        20
   virginica       1.00      1.00      1.00        12

    accuracy                           1.00        45
   macro avg       1.00      1.00      1.00        45
weighted avg       1.00      1.00      1.00        45
