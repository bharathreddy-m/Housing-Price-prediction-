
# Project 1 - House Price Prediction - Ames Dataset using Machine Learning

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Objectives](#objectives)
- [Project Steps](#project-steps)
  - [Data Loading](#data-loading)
  - [Data Preprocessing](#data-preprocessing)
  - [Data Splitting](#data-splitting)
  - [Model Fitting](#model-fitting)
  - [Model Implementations](#model-implementations)
  - [Gradient Descent](#gradient-descent)
- [Results Summary](#results-summary)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview
The primary goal of this project is to predict house prices based on various features using the Ames Housing dataset. This project explores several regression techniques, evaluates their performance, and compares the results to identify the most effective model for accurate predictions.

## Dataset
The Ames Housing dataset contains detailed information about residential properties sold in Ames, Iowa. It includes a range of features that influence house prices, such as:
- Lot Area
- Year Built
- Total Rooms
- Neighborhood
- Garage Type
- Sale Price (target variable)

The dataset is publicly accessible on Kaggle.

## Objectives
- Implement various regression techniques to predict house prices.
- Evaluate model performance using appropriate metrics.
- Identify the best model for house price prediction.

## Project Steps

### Data Loading
Utilized the Pandas library to load the dataset into the Python environment for analysis.

### Data Preprocessing
- **Handling Missing Values:** Identified and filled missing values using the mean, median, or mode as appropriate for each feature.
- **Encoding Categorical Data:** Employed One-Hot Encoding to convert categorical features into a numerical format suitable for model training.
- **Outlier Detection and Removal:** Detected outliers using the IQR method and removed them to prevent skewed model performance.
- **Normal Distribution Check:** Assessed the distribution of features to ensure they met normality assumptions, utilizing visualization techniques such as histograms and Q-Q plots.
- **Feature Scaling:** Applied Min-Max scaling to normalize feature values, ensuring that all features contributed equally to the model training.

### Data Splitting
Divided the dataset into features (X) and target (y), followed by splitting into training (X_train, y_train) and testing sets for model evaluation.

### Model Fitting
Fitted a linear regression model to the training data and calculated the training/testing errors, followed by visualizing the model fit line.

### Model Implementations
Implemented and evaluated the following regression models, recording their performance metrics:

- **Linear Regression**
  - Training: MSE: 0.0603, MAE: 0.1691, RMSE: 0.2455, R²: 93.61%
  - Testing: MSE: 0.0866, MAE: 0.1855, RMSE: 0.2943, R²: 92.86%
  
- **Polynomial Regression**
  - Training: MSE: 1.6880e-29, MAE: 2.8078e-15, RMSE: 4.1086e-15, R²: 100.0%
  - Testing: MSE: 0.2554, MAE: 0.3611, RMSE: 0.5054, R²: 78.95%
  
- **K-Nearest Neighbors (K-NN)**
  - Training: MSE: 0.0957, MAE: 0.2031, RMSE: 0.3094, R²: 89.85%
  - Testing: MSE: 0.1378, MAE: 0.2498, RMSE: 0.3712, R²: 88.64%
  
- **Support Vector Regression (SVR)**
  - Training: MSE: 0.0297, MAE: 0.1087, RMSE: 0.1724, R²: 96.85%
  - Testing: MSE: 0.0691, MAE: 0.1750, RMSE: 0.2629, R²: 94.31%
  
- **Decision Tree**
  - Training: MSE: 0.0000, MAE: 0.0000, RMSE: 0.0000, R²: 100.0%
  - Testing: MSE: 0.2464, MAE: 0.3269, RMSE: 0.4964, R²: 79.69%
  
- **Random Forest**
  - Training: MSE: 0.0138, MAE: 0.0763, RMSE: 0.1177, R²: 98.53%
  - Testing: MSE: 0.0916, MAE: 0.1980, RMSE: 0.3027, R²: 92.45%

### Gradient Descent
Implemented gradient descent as an optimization technique for the regression models.
- Training: MSE: 0.0091, MAE: 0.0756, RMSE: 0.0956, R²: 99.03%
- Testing: MSE: 0.0592, MAE: 0.1630, RMSE: 0.2434, R²: 95.12%

## Results Summary
The results indicate the performance of each model, showcasing their strengths and weaknesses. Key takeaways include:
- Polynomial Regression demonstrated exceptional training performance, achieving a perfect R² score of 100% but a lower score on the testing set.
- Gradient Descent produced high R² scores for both training and testing sets, indicating robust generalization capabilities.
- Random Forest performed well, balancing training and testing metrics effectively.

## Conclusion
This project effectively utilized multiple regression techniques to predict house prices, providing insights into model performance through thorough evaluation. The analysis highlights the importance of selecting appropriate features and modeling techniques for optimal predictive performance.

## Future Work
Future iterations may explore the following:
- **Hyperparameter Tuning:** Implement advanced techniques like Grid Search and Random Search for model optimization.
- **Feature Engineering:** Investigate new features or transformations to enhance model accuracy.
- **Ensemble Methods:** Combine predictions from different models to improve overall prediction performance.
- **Deployment:** Develop a web application or API to enable real-time predictions using the best-performing model.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
The Ames Housing dataset is provided by Kaggle. Thanks to the contributors of libraries used in this project, including Pandas, NumPy, and Scikit-learn.
