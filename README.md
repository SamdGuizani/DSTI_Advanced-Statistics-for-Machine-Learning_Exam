# DSTI Exam - Advanced Statistics for Machine Learning

This examination tests both theoretical understanding and practical application of Statistics & Machine Learning techniques, with a focus on regression and classification problems. 

## Exercise 1: Constrained Multiple Linear Regression (MLR)

### Theoretical Analysis:

The solution for constrained least squares in multiple linear regression is derived, showing that the constrained estimator can be expressed in terms of the unconstrained estimator 
and the constraint matrix

### Practical Application on Ozone Dataset:

**Unconstrained Model**: A multiple linear regression model was developed using all explanatory variables, with coefficients calculated and validated against lm() function in R.

**Constrained Model**: A constraint was applied, and the resulting model's coefficients were compared to the unconstrained model.

**Comparison**: The constrained model showed slightly lower $R^2$ and Adjusted $R^2$ and higher residual standard error compared to the unconstrained model, indicating a slight reduction in performance due to the constraint.


## Exercise 2: Model Selection and Evaluation

### Dataset 1: Classification Problem

**Exploratory Data Analysis**: Identified the most correlated variables with the binary response variable.

**Model Development**:
* Generalized Linear Model (Logistic Regression): Achieved 0% test set classification error using LASSO regularization.
* CART Decision Trees: Pruned tree with 8.7% error.
* Random Forest: Achieved 8.7% error with insights into variable importance.
* VSURF + CART: Combined variable selection based on Random Forest with CART, achieving 4.3% error.

**Conclusion**: Logistic Regression was the best model with 0% error, followed by VSURF + CART with 4.3% error.

### Dataset 2: Regression Problem (PM10 Pollution)

**Exploratory Data Analysis**: Identified correlations between PM10 and explanatory variables, noting the importance of station-specific data.

**Model Development**:
* Linear Models: Three models with varying adjustments, achieving adjusted $R^2$ around 0.54-0.58.
* Non-linear Models:
  - CART Decision Trees: Handled missing values, identified key variables.
  - Random Forest: Best performance with RMSE of 5.25.
  - VSURF + CART: Variable selection through Random Forest followed by CART, with RMSE of 6.85.

**Conclusion**: Random Forest was the best model for combined station data, while station-specific models also showed promising results, particularly for stations with missing data.

**General Conclusion**

**Classification**: Logistic Regression excelled for the binary classification task.

**Regression**: Random Forest provided robust predictions for PM10 levels across different stations, balancing performance and interpretability.

**Model Selection**: These examples highlight the importance of exploratory data analysis and model validation in selecting appropriate statistical models for different datasets and tasks.
