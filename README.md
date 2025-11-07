# Dissertation

This project focused on developing and evaluating machine learning models for geomagnetic storm forecasting (Dst index) 1 and 5 hours into the future.

The models developed were a baseline Linear Regression model and a double-branch Convolutional Neural Network (CNN). Input data used for training included solar wind and sunspot data, with the target output being the Disturbance Storm Time (Dst) index. A double-branch CNN was developed due to the fact that after performing the initial hyperparameter optimization, it was found that two different kernel sizes performed really well. Therefore a double-branch model was explored and was found to perform more accurately.

The attached files show the final iteration of the project, which includes the preprocessing performed on the training and test data, the two stages of hyperparameter tuning, and the final training and testing stages.

This project was based on the [NOAA Magnetic Forecasting Challenge hosted on the DrivenData platform](https://www.drivendata.org/competitions/73/noaa-magnetic-forecasting/)

[Additional link for further information on the competition](https://doi.org/10.1029/2023SW003514)
