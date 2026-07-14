# Ford Car Price Prediction

## 📌 Overview
This project builds a machine learning pipeline to predict the prices of used Ford vehicles. It explores the dataset through comprehensive visual Exploratory Data Analysis (EDA), preprocesses the data using two different categorical encoding strategies, and trains Linear Regression models to evaluate the impact of feature engineering on model performance.

## 📊 Exploratory Data Analysis (EDA)
The initial phase of the project involves understanding the data distribution and identifying key relationships between features and the target variable (`price`). The following visualizations were utilized:

* **Target Distribution:** A histogram with a Kernel Density Estimate (KDE) curve was plotted to observe the distribution and skewness of car prices.
* **Feature Correlation:** A correlation heatmap was generated for all numeric features to identify multicollinearity and baseline linear relationships with the target variable.
* **Categorical & Discrete Impact:** Boxplots were created to visualize how categorical and discrete features influence the price. This included comparing prices across different `models`, `years`, `engineSizes`, `transmissions`, and `fuelTypes`.
* **Continuous Feature Impact:** Scatterplots were used to examine the trends between continuous numerical variables (`mileage`, `mpg`, `tax`) and `price`.

## ⚙️ Data Preprocessing
Before modeling, the data was split into independent variables (`X`) and the dependent target (`y`). Numeric columns (`year`, `tax`, `mpg`, `mileage`, `engineSize`) were normalized using `StandardScaler`. 

To handle the categorical features (`model`, `transmission`, `fuelType`), two distinct pipelines were created to compare their effectiveness with a linear model:
1. **One-Hot Encoding:** Applied using `pd.get_dummies(drop_first=True)`. This expanded the dataset horizontally by creating binary columns for each category.
2. **Label Encoding:** Applied using `LabelEncoder`. This kept the dataset compact by assigning a unique integer to each category.

The data for both pipelines was then split into training and testing sets (80/20 split, `random_state=42`).

## 🚀 Modeling & Results
A **Linear Regression** model was trained and evaluated on both preprocessed datasets. The performance was measured using the $R^2$ Score (Accuracy) and the Adjusted $R^2$ Score to account for the number of predictors.

### Model 1: One-Hot Encoding
* **Total Features Used:** 34
* **Accuracy ($R^2$ Score):** 84.64%
* **Adjusted $R^2$ Score:** 84.49%

### Model 2: Label Encoding
* **Total Features Used:** 8
* **Accuracy ($R^2$ Score):** 73.65%
* **Adjusted $R^2$ Score:** 73.60%

## 💡 Conclusion
The **One-Hot Encoding** strategy significantly outperformed the Label Encoding strategy, yielding an accuracy increase of roughly 11%. 

Because Linear Regression applies mathematical weights to feature values, Label Encoding can inadvertently trick the model into assuming an ordinal relationship (e.g., believing that a `fuelType` mapped to 3 is mathematically "greater" or "worth more" than one mapped to 1). One-Hot Encoding resolves this by treating every category as an independent feature, which allows the Linear Regression model to assign appropriate, independent coefficients to each specific car model, fuel type, and transmission.
