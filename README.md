# Python-DS-ML-Bootcamp
The projects in this repo are based on Jose Portilla (Pierian Data)'s class "Python for Data Science and Machine Learning Bootcamp." Many of the data sets used and instruction for the code is provided by Jose Portilla for the class. The projects uploaded here were written by me, with instruction based on what we were learning in class.

Class here: https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/ 
Pierian Data: https://www.pieriandata.com

# Keras LendingClub Project

Based on our classwork up until this point with neural networks and machine learning, I used LendingClub's historical data on loans given out with information on whether or not the borrower defaulted (charge-off) in order to build a model that could predict whether or not a new customer would pay back their loan. I used the dataset provided by the instructor (which had been cleaned in advance) in order to build this model, and I heavily referenced course materials and notebooks for this project. Ultimately this turned out to be an imbalanced model, which I discovered during the initial data exploration, with more "fully paid" than "charged off" loans. After running an analysis, I realized that 80% of points were already predicted as "loan repaid"... so the accuracy wasn't great. I could have added more layers to the neural network in order to analyze the data and possibly come up with a more accurate outcome, so in future iterations I hope to make it more complex as I learn more.
