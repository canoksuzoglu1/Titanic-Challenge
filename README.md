# Titanic Survival Prediction

This project aims to predict the survival of passengers aboard the Titanic using various machine learning models. The dataset contains passenger details such as age, sex, class, and fare, which are analyzed and used to train models for survival prediction. Through thorough data exploration, preprocessing, and feature engineering, we were able to build predictive models with high accuracy.

## Project Overview

The main goal of this project was to predict whether a passenger on the Titanic survived or not, based on features such as age, class, and fare. The project involved several key steps:

1. **Data Exploration**: We first explored the dataset to understand its structure, check for missing values, and observe the distribution of key features like age and fare. This helped identify areas where data cleaning and preprocessing were required.

2. **Handling Missing Data**: Significant portions of the data were missing, especially in the `Age`, `Cabin`, and `Embarked` columns. We handled missing values using various techniques:
   - The **Age** column was filled using regression-based imputation, which predicted missing ages based on other relevant features.
   - The **Cabin** column, with over 77% missing values, was dropped entirely. We also created a new feature, `CabinMissing`, to retain information about whether cabin data was missing.
   - The few missing values in the **Embarked** column were filled using the most common value (mode).

3. **Feature Engineering**: We created new features such as `Ticket Group Size` to capture insights about passengers traveling together, which could affect survival probabilities. We also encoded categorical variables like `Sex`, `Pclass`, and `Embarked` using one-hot encoding, allowing them to be used in machine learning models.

4. **Feature Scaling**: To ensure the models could process the data effectively, we standardized features like `Age` and normalized features like `Fare`. This ensured that features with different scales wouldn’t disproportionately impact the model's learning process.

5. **Feature Selection**: Using **Random Forest Classifier** and **Recursive Feature Elimination (RFE)**, we identified the most important features influencing survival. This helped simplify the model and focus on the most impactful variables.

## Machine Learning Models

We trained several models to predict survival:

1. **Logistic Regression**: A classic linear model often used in classification tasks. This model achieved a cross-validation accuracy of **81.37%**, performing reliably well on this dataset.
   
2. **Random Forest**: An ensemble method that combines multiple decision trees. The model achieved **81.26%** cross-validation accuracy, showing slightly less performance compared to Logistic Regression.

3. **Multilayer Perceptron (MLPClassifier)**: A simple neural network model, which provided **81.37%** cross-validation accuracy, equal to the Logistic Regression model.

4. **Custom Neural Network (Keras)**: A deeper neural network built using Keras, which achieved a validation accuracy of **81.01%**. Although slightly lower than the other models, it demonstrated the potential of more complex architectures.

Each model was evaluated using 5-fold cross-validation, and the best-performing models were saved for future use.

## Test Data Predictions

After training the models, we applied the same preprocessing steps to the test dataset, which included filling missing values, encoding categorical variables, and scaling numerical features. Predictions were made using the trained models, and the results were saved in a `submission.csv` file, ready for competition submission.

## Results Summary

- **Logistic Regression**: 81.37% accuracy
- **Random Forest**: 81.26% accuracy
- **MLPClassifier**: 81.37% accuracy
- **Custom Neural Network**: 81.01% validation accuracy

All models performed similarly, with Logistic Regression and MLPClassifier showing the best performance during cross-validation.

## Conclusion

Through thorough data exploration, preprocessing, feature engineering, and model selection, we achieved highly accurate predictions for Titanic survival. The balance between simplicity (Logistic Regression) and complexity (Neural Networks) allowed us to compare different approaches and ensure robust results. This project demonstrates the entire pipeline from raw data to predictive modeling and submission preparation.
