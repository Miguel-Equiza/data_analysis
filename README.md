# This repo contains 3 projects

# Olist Profit Improvement Analysis
![image](https://github.com/user-attachments/assets/51d5e822-db16-4213-bcf2-204f6a82cc09)

## Overview

This project aims to analyze how Olist, a leading e-commerce service in Brazil, can increase its profits. Olist connects merchants to major marketplaces and offers various services, including inventory management, handling customer reviews, and logistics. Olist charges sellers a progressive monthly fee based on their volume of orders.

By exploring key factors such as seller performance, product catalog, and customer feedback, this analysis uncovers actionable insights to enhance profitability.


## Objective

The primary objective of this analysis is to understand the factors influencing Olist's profits and to provide recommendations for increasing profitability. This involves exploring the seller and product dynamics and identifying key drivers that affect revenue and profit margins.


## Dataset

The dataset consists of approximately 100,000 orders placed between 2016 and 2018 on Olist’s platform. It was derived from 9 initial CSV files, which were combined into 3 datasets that capture various aspects of the business, including:

1. **Order details** (seller information, product details, logistics performance, etc.).
2. **Seller performance**.
3. **Product details**.



## Methodology

### 1. **Data Preprocessing**
   - **Merged datasets**: Combined the 9 raw CSV files into 3 coherent datasets for analysis.
   - **Data cleaning**: Handled missing data, removed duplicates, and performed data normalization.
   
### 2. **Exploratory Data Analysis (EDA)**
   - **Descriptive statistics**: Generated summaries to understand the distribution of key metrics like profit, orders, and review scores.
   - **Visual analysis**: Created plots (histograms, scatterplots, heatmaps) to examine relationships between variables.
   
### 3. **Correlation Analysis**
   - Computed correlation matrices to identify relationships between numeric variables (e.g., order volume, seller reviews, delivery delays, and profit).
   - Focused on understanding which seller behaviors and product characteristics were positively or negatively correlated with profit.

### 4. **Predictive Modeling**
   - **Linear Regression**: Used to predict the impact of specific features (e.g., number of orders, delay to carrier, and review scores) on profit.
   - **Logistic Regression**: Performed classification analysis to understand the likelihood of a seller achieving high or low profitability.
   
### 5. **Profit Impact Analysis**
   - **Seller analysis**: Identified which sellers were underperforming in terms of profit.
   - **Product analysis**: Determined how specific products and categories contributed to profit margins.
   - **Impact of seller/product removal**: Simulated scenarios where low-performing sellers or low-rated products were removed from the platform and analyzed the resulting impact on overall profit.
# Project README: Time Series Forecasting Models for Madrid Temperature Data
![Captura de pantalla 2024-09-25 094758](https://github.com/user-attachments/assets/068c4d55-2d7f-47d6-b3b5-55c8ba993ebf)

## Project Overview

In this project, we will build and compare three different forecasting models to predict the temperatures of Madrid using the dataset from [Global Climate Change Data](https://data.world/data-society/global-climate-change-data), specifically the file `GlobalLandTemperatures_GlobalLandTemperaturesByMajorCity.csv`. 

### Models to be built:
1. **Baseline Model**: Using a simple forecasting technique (such as moving average or naive forecasting) covered in the practical sessions.
2. **Machine Learning Model**: Implementing a machine learning algorithm (such as Random Forest, XGBoost, or similar) as discussed in theory and practice.
3. **Deep Learning Model**: Using a neural network architecture (such as LSTM or GRU) for time series forecasting.

## Dataset Information

The dataset used is `GlobalLandTemperatures_GlobalLandTemperaturesByMajorCity.csv`, which can be downloaded for free from the [data.world link](https://data.world/data-society/global-climate-change-data). 

We will filter the data for **Madrid** and use it to model and forecast the temperatures for this city only.

## Guidelines for the Project

### Data Preparation

1. **Load the data**: The data is loaded from the CSV file and converted into a Pandas dataframe.
2. **Filter**: We will filter the dataset to only include data for the city of Madrid.
3. **Imputation of missing values**: Handle any missing or incomplete data (if applicable).
4. **Outlier Detection**: Detect and handle any outliers present in the dataset.
5. **Train-test split**: We will split the dataset into training (80%) and validation/testing (20%) sets.

### Model Training

For each model, we will:

1. Split the data into training and validation sets.
2. Import the necessary libraries for each model.
3. Train the model using the 80% training data.
4. Apply hyperparameter tuning using techniques such as grid search (for Machine Learning and Deep Learning models).
5. Perform forecasting on the 20% validation data. The results will be stored and analyzed for comparison.

### Model Evaluation

The performance of each model will be evaluated using relevant metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and others. The final output will include:

- A **ranking** of the models based on the chosen metrics.
- A **graph** comparing the predicted values to the actual values in the validation data.

### Conclusions

A final section will provide insights and conclusions based on the results of each model. We will analyze:

- Which model performed best.
- The practical implications of the results.
- Possible improvements and further experimentation.

# Data Analysis and linear statistical modelling with R
![Captura de pantalla 2024-09-25 092556](https://github.com/user-attachments/assets/69175d62-85de-44ae-84ae-63e2083eaad3)

## Overview
This is a task done during my Masters Degree in Data Science, where I develop several linear models using R and also perform advanced data analysis analyzing residuals, statistics and errors

## Basic tasks
These include:
* Exploratory data analysis
* Analysis of existing linear relationship between variables
* Estimation of a simple linear regresion model between variables
* Interpretation of the outcome of the model

## Mid tasks
These include:
* Fit the best possible multiple linear regression model with that set of variables.
* Compare its metrics and perform the diagnosis of the chosen linear regression model.
* Interpret the coefficients of the chosen linear regression model.
* Decompose the fitted values and residuals of the chosen linear regression model.
* Evaluate and interpret the goodness of fit of the chosen linear regression model ($R^2$ and $R^2_{adj}$).
* Evaluate and interpret the individual significance test of the chosen linear regression model.
* Evaluate and interpret the global significance test of the chosen linear regression model.
* Evaluate and interpret multicollinearity (existence of a linear relationship between the independent variables in the model) using the correlation matrix.
