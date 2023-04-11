# House Price Prediction

    This repository contains the code for a machine learning model that predicts house prices based on various features of the property. The dataset used in this project contains information about houses, and the goal is to create a regression model to predict the sale price of a house based on its characteristics.

## Table of Contents

    Data Exploration and Preprocessing
    Feature Engineering
    Feature Selection
    Model Training and Evaluation

## Data Exploration and Preprocessing

    The initial steps in the code involve loading the dataset, exploring its features, handling missing values, and visualizing relationships between the features and the target variable, SalePrice. The dataset contains both numerical and categorical features, which are processed separately. The code also takes care of scaling the numerical features using MinMaxScaler and converting categorical features into numerical values.

## Feature Engineering

    The code contains several feature engineering steps to improve the performance of the model:

        Handling missing values in both numerical and categorical features by filling in missing values with the median (for numerical) or a new category "missing" (for categorical).
        Converting some numerical features into a log-normal distribution.
        Handling rare categorical features by grouping them under a new category "Rare_var".
        Mapping categorical features to numerical values based on their relationship with the target variable.

## Feature Selection

    Feature selection is performed using Lasso regression with a cross-validated alpha value. The selected features are then used to train the final model.
    Model Training and Evaluation

    The code uses Lasso regression as the machine learning model to predict house prices. The model is trained on a training dataset and evaluated using a validation dataset. The performance of the model is measured using the root mean squared error (RMSE) metric. The optimal alpha value for Lasso regression is determined using grid search with cross-validation.

    Once the best alpha value is found, the final model is trained on the entire training set, and its performance is evaluated on a test dataset. The final RMSE value on the test dataset is reported as the model's performance metric.

## Usage

    Clone this repository.
    Download the dataset and place it in the appropriate directory.
    Run the provided code in a Jupyter Notebook or any Python environment.
    Train the model and evaluate its performance on the test dataset.
    Use the trained model to predict house prices based on the input features.

## Dependencies

    Python 3.x
    NumPy
    pandas
    Matplotlib
    seaborn
    scikit-learn

### Contributing

    Contributions are welcome. Please open an issue or submit a pull request to suggest changes or improvements.


### Credits

    Mete Turkan
    linkedIn : linkedin.com/in/mete-turkan
    Inst : m_trkn46
