# Predicting the Severity of Geomagnetic Storms using Machine Learning

## Project summary

This project develops and evaluates machine learning models for short term forecasting of the severity of geomagnetic storms using the Disturbance Storm Time (Dst) index. A baseline linear regression model and a double branch convolutional neural network (CNN) are trained on solar wind and sunspot time series data to predict the Dst index one and five hours into the future. The final CNN model outperforms NOAA’s benchmark LSTM model.

---

## Problem definition

Geomagnetic storms pose a risk to satellites, power grids, and communication systems. Accurate short term forecasting of storm intensity, commonly measured using the Dst index, is therefore an important space weather prediction task. This project focuses on predicting future Dst values using upstream solar wind and sunspot observations.

---

## Data sources

Solar wind measurements are collected upstream of Earth at the Sun–Earth L1 Lagrange point, providing advance information on solar wind conditions before they interact with the magnetosphere. These upstream observations are used as model inputs.

The target variable, the Disturbance Storm Time (Dst) index, is derived from ground based measurements and represents the global state of geomagnetic activity. Sunspot data is included as an additional input feature to capture longer term solar activity.

---

## Models

Two modelling approaches are explored.

### Baseline model

A linear regression model is implemented as a baseline to establish reference performance for short term Dst prediction.

### Double branch convolutional neural network

A convolutional neural network is used to model temporal dependencies in the input time series. Initial hyperparameter optimisation revealed that two different convolutional kernel sizes performed particularly well. To exploit this observation, a double branch CNN architecture was designed where each branch learns features using different kernel sizes. The outputs of both branches are concatenated before the final prediction layer.

---

## Training and evaluation

Model development follows a structured pipeline:

* Preprocessing of training and test datasets
* Two stage hyperparameter optimisation for the CNN
* Final model training on the entire training dataset using the optimal hyperparameters
* Evaluation on test data

Performance is assessed using root mean squared error (RMSE) on one hour and five hour ahead predictions.

---

## Results

The final linear and CNN models achieve the following performance on one hour ahead Dst predictions:

* Linear RMSE: **17.48 nT**
* CNN RMSE: **13.71 nT**
* NOAA benchmark LSTM RMSE: **15.2 nT**

This CNN model represents a clear improvement over the linear and benchmark models.

---

## Repository structure

* `Preprocess_Train.py`
  Preprocessing for the training data

* `Preprocess_Test.py`
  Preprocessing for the test data

* `Tune_1st_Stage.py`
  Initial hyperparameter optimisation for the CNN model

* `Tune_2nd_Stage.py`
  Refined hyperparameter optimisation based on first stage results

* `Final_Train.py`
  Final model training and validation

* `Test.py`
  Evaluation on test data

---

## How to run

To reproduce the final trained model and evaluation results, only the final training script needs to be executed.

### Data placement

Ensure the input data is organised as follows:

* Place all training data files inside a folder named `train`

* Place all test data files inside a folder named `test`

Both the train and test folders must be located in the same directory as the Python scripts.

### Execution steps

1. Run `Preprocess_Train.py` and `Preprocess_Test.py` to generate processed datasets

2. Run `Final_Train.py` to train the linear and final CNN model and save the resulting `.keras` model files

3. Run `Test.py` to evaluate model performance using the saved linear and CNN models

The `Tune_1st_Stage.py` and `Tune_2nd_Stage.py` scripts are provided for reference only. They document the hyperparameter exploration process that informed the final CNN model configuration and are not required to reproduce the reported results.

---

## Tools and techniques

* Python
* Pandas and Numpy
* Keras and KerasTuner
* Time series forecasting
* Convolutional neural networks
* Hyperparameter optimisation

---

## Data and challenge sources

* [MagNet: Model the Geomagnetic Field](https://www.drivendata.org/competitions/73/noaa-magnetic-forecasting/)
* [MagNet—A Data-Science Competition to Predict Disturbance Storm-Time Index (Dst) From Solar Wind Data](https://doi.org/10.1029/2023SW003514)

---

## Background reading
* [Chollet F *Deep Learning with Python*, Third Edition, Manning Publications](https://www.manning.com/books/deep-learning-with-python-third-edition)
