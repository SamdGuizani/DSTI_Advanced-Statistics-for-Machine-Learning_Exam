#######################################################
### DSTI - Advanced Statistics for Machine Learning ###
### Exam 2024                                       ###
### Author: Samd Guizani (SPOC, A23)                ###
#######################################################


# Exercise 1 ----

setwd("C:/Users/SamdGuizani/OneDrive - Data ScienceTech Institute/Documents/DSTI_MSc DS and AI/02-Foundation/05-ASML/Exam")

## Global variables ----
ALPHA = .05 # statistical significance level

## Functions ----
MLR_predict = function(X, beta) 
{
  #' Predicts response values using a multiple linear regression model.
  #'
  #' @param X A numeric matrix of predictors, first column as intercept (all 1s).
  #' @param beta A numeric vector of regression coefficients, including intercept.
  #'
  #' @return A numeric vector of predicted response values.
  
  # Predicted response is X * beta
  y_pred = X %*% beta
  return(y_pred)
}

MLR_residuals = function(X, beta, y_act) 
{
  #' Calculates the residuals from a multiple linear regression model.
  #'
  #' @param X A numeric matrix of predictors, where first column is the intercept (all 1s).
  #' @param beta A numeric vector of regression coefficients, including intercept.
  #' @param y_act A numeric vector of actual response values.
  #'
  #' @return A numeric vector of residuals (actual - predicted values).

  # Calculate residuals (actual - predicted)
  residuals = y_act - MLR_predict(X, beta)
  
  return(residuals)
}

MLR_residual_std_error = function(X, beta, y_act) 
{
  #' Calculates residual standard error (RSE) for a multiple linear regression model.
  #'
  #' @param X A numeric matrix of predictors, with first column as intercept (all 1's).
  #' @param beta A numeric vector of regression coefficients, including intercept.
  #' @param Y_act A numeric vector of actual response values.
  #'
  #' @return A numeric value representing the residual standard error.
  
  # Calculate residuals
  residuals = MLR_residuals(X, beta, y_act)
  
  # Number of observations (n) and number of predictors (p)
  n = length(y_act)
  p = length(beta) - 1  # Subtract 1 for the intercept
  
  # Calculate the Residual Standard Error (RSE)
  rse = sqrt(sum(residuals^2) / (n - p - 1))
  
  return(rse)
}

MLR_coef_ci = function(X, beta, y_act, alpha=0.05) 
{
  #' Calculates confidence intervals for the coefficients of a multiple linear regression model.
  #'
  #' @param X A numeric matrix of predictors, where first column is the intercept (all 1s).
  #' @param beta A numeric vector of regression coefficients, including intercept.
  #' @param y_act A numeric vector of actual response values.
  #' @param alpha A numeric value representing significance level (default 0.05 for a 95% confidence interval).
  #'
  #' @return A numeric matrix with 3 columns: coefficients, lower bound and upper confidence interval limits.

  # Number of observations (n) and number of predictors (p)
  n = nrow(X)
  p = length(beta) - 1 # Subtract 1 for the intercept
  
  # Residual variance
  residual_var = MLR_residual_std_error(X, beta, y_act)^2
  
  # Inverse of (X'X)
  XtX_inv = solve(t(X) %*% X)
  
  # Standard errors of coefficients
  se_beta = sqrt(diag(XtX_inv) * residual_var)
  
  # Critical value from t-distribution
  t_value = qt(1 - alpha / 2, df = n - p - 1)
  
  # Confidence intervals
  lower_bound = beta - t_value * se_beta
  upper_bound = beta + t_value * se_beta
  
  # Create a matrix with coefficients and their confidence intervals
  ci_matrix = cbind(beta, se_beta, lower_bound, upper_bound)
  colnames(ci_matrix) = c("Est. Coef.", "Std. Error", "Lower Bound", "Upper Bound")
  
  return(ci_matrix)
}

MLR_r2 = function(X, beta, y_act) 
{
  #' Calculates R-squared value for a multiple linear regression model.
  #'
  #' @param X A numeric matrix of predictors, where first column is the intercept (all 1s).
  #' @param beta A numeric vector of regression coefficients, including intercept.
  #' @param y_act A numeric vector of actual response values.
  #'
  #' @return A numeric value representing the R-squared value.

  # Total sum of squares (TSS)
  tss = sum((y_act - mean(y_act))^2)
  
  # Residuals sum of squares (RSS)
  rss = sum((MLR_residuals(X, beta, y_act))^2)
  
  # Calculate R-squared
  r_squared = 1 - rss / tss
  
  return(r_squared)
}

MLR_r2_adj = function(X, beta, y_act) 
{
  #' @param X A numeric matrix of predictors, where first column is intercept (all 1s).
  #' @param beta A numeric vector of regression coefficients, including intercept.
  #' @param y_act A numeric vector of actual response values.
  #'
  #' @return A numeric value representing the adjusted R-squared value.

  # Number of observations (n) and number of predictors (p)
  n = nrow(X)
  p = length(beta) - 1  # Subtract 1 to exclude intercept

  # Calculate R-squared
  r_squared = MLR_r2(X, beta, y_act)
  
  # Calculate adjusted R-squared
  r_squared_adj = 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))
  
  return(r_squared_adj)
}


## Import dataset and preprocessing ----
Dataset_ozone = read.csv2("Dataset_ozone.txt", row.names=1)

# Convert variables "vent" and "pluie" to categorical variables
Dataset_ozone$vent = as.factor(Dataset_ozone$vent)
Dataset_ozone$pluie = as.factor(Dataset_ozone$pluie)

# list of categorical and numeric explanatory variables
cat_vars = names(Dataset_ozone[2:ncol(Dataset_ozone)])[sapply(Dataset_ozone[2:ncol(Dataset_ozone)], is.factor)]
num_vars = names(Dataset_ozone[2:ncol(Dataset_ozone)])[sapply(Dataset_ozone[2:ncol(Dataset_ozone)], is.numeric)]

# Define matrix X (explanatory variables) and vector Y (response)
# N.B.: Categorical variables are encoded using one-hot key encoding
X = data.matrix(cbind(rep(1, nrow(Dataset_ozone)), # a column of 1's is added to account for intercept
                      Dataset_ozone[, 2:(ncol(Dataset_ozone)-2)], # take all columns except last 2 (categorical variables)
                      model.matrix(~vent-1, data = Dataset_ozone)[,-1], # apply one-hot key encoding to variable "vent" 
                      model.matrix(~pluie-1, data = Dataset_ozone)[,-1] # apply one-hot key encoding to variable "pluie"
                      )
                ) 
colnames(X)[1] = ("(Intercept)") # tidy up variable names
colnames(X)[ncol(X)] = ("pluieSec") # tidy up variable names

Y = data.matrix(Dataset_ozone$maxO3)

n = nrow(X) # nb observations
p = ncol(X) - 1 # nb explanatory variables, including one-hot key encoded categorical variables

## Exercise 1, Question 2a ----
# Linear model predicting maxO3 as function of all the explanatory variables

LM = lm(formula = maxO3 ~ ., data = Dataset_ozone)
summary(LM)

# "manually" calculate least squares linear model beta coefficients
hbeta = solve(t(X) %*% X) %*% t(X) %*% Y
hbeta_ci = MLR_coef_ci(X, hbeta, Y, ALPHA) # hbeta with confidence intervals

## Exercise 1, Question 2b ----
# Linear model with constraints on beta coefficients of explanatory variables T9, T12 and T15

# Constraints are linear functions on beta coefficients such that R.beta = r
# i.e. each constraint is a weighted sum applied on beta coefficients of explanatory variables
# R matrix holds the weights to apply to constrained beta coefficients
# r vector holds the weighted sums to consider for each constraint
q = 1 # number of constraints to consider. q < p+1. q is rank of R.
R = matrix(0, nrow = q, ncol = p + 1) # initialize matrix of 0's, size q * (p+1)
r = matrix(0, nrow = q, ncol = 1) # initialize vector of 0's, size q

# assign constraint weights on beta coefficients
R[1, 2:4] = 1

# assign constraint weighted sums
r[1, 1] = 0

# Computing beta coefficients of constrained linear model
hbeta_c = 
  hbeta + 
  solve(t(X) %*% X) %*% 
  t(R) %*% 
  solve(R %*% solve(t(X) %*% X) %*% t(R)) %*%
  (r - R %*% hbeta)
hbeta_c_ci = MLR_coef_ci(X, hbeta_c, Y, ALPHA) # hbeta_c with confidence intervals

## Exercise 1, Question 2c ----
# Compare the 2 linear models

# Models' comparison: coefficients and performance metrics
print("### MLR w/o constraints ###")
print("Coefficients")
print(hbeta_ci)
print(paste("R squared =", MLR_r2(X, hbeta, Y)))
print(paste("R squared adjusted =", MLR_r2_adj(X, hbeta, Y)))
print(paste("Residuals Std. Error =", MLR_residual_std_error(X, hbeta, Y)))

print("### MLR with constraints ###")
print("Coefficients")
print(hbeta_c_ci)
print(paste("R squared =", MLR_r2(X, hbeta_c, Y)))
print(paste("R squared adjusted =", MLR_r2_adj(X, hbeta_c, Y)))
print(paste("Residuals Std. Error =", MLR_residual_std_error(X, hbeta_c, Y)))

# Plots

# Predictions from both models
Y_prd1 = MLR_predict(X, hbeta) # equivalent to LM$fitted.values
Y_prd2 = MLR_predict(X, hbeta_c)

# Create scatter plot of Y_prd1 and Y_prd2 vs Y
plot(Y, Y_prd1, main="Model comparison", xlab="Y actual", ylab="Y predicted", pch=19, col="blue")
abline(lm(Y_prd1~Y), col="blue", lwd=2)
points(Y, Y_prd2, pch=19, col="red")
abline(lm(Y_prd2~Y), col="red", lwd=2, lty=3)

# Target line y = x spanning range of Y, Y_prd1 and Y_prd2
Y_range = range(c(Y, Y_prd1, Y_prd2))
abline(a = 0, b = 1, col = "black", lwd = 2, lty = 4, xlim = Y_range)

# Legend
legend("topleft", 
       legend=c("MLR w/o constraints (data1)",
                "Line (data1)",
                "MLR with constraints (data2)",
                "Line (data2)",
                "Target line (act. = pred.)"),
       col=c("blue", "blue", "red", "red", "black"), 
       pch=c(19, NA, 19, NA, NA), 
       lwd=c(NA, 2, NA, 2, 2),
       lty=c(NA, 1, NA, 2, 4),
       box.lwd=1)

# Correlation matrix of explanatory variables
print(round(cor(X[,(1:p)+1]), 2))


# Exercise 2 ----


## Packages installation and imports ----
library(glmnet)
library(caret)
library(rpart)
library(randomForest)
library(VSURF)
library(dplyr)
library(MASS)

## Global variables ----

test_fraction = 0.3 # fraction of datasets to assign to test set

## Functions ----

train_test_split = function(X, Y = NULL, test_size = 0.25)
{
  #' Splits datasets in train and test sets. 
  #' @param X Matrix or dataframe, holding explanatory variables, may also hold response variable(s)
  #' @param Y Vector of a single response (optional, use when explanatory and response variables are not in the same matrix or dataframe) 
  #' @param test_size Float between 0.0 and 1.0. Proportion of datasets to include in test set.
  #' 
  #' @returns List of train and test splits
  
  test_idx = sample(1:nrow(X), floor(test_size * nrow(X)))
  
  X_train = X[-test_idx,]
  X_test = X[test_idx,]
  
  if (!is.null(Y))
  {
    if (length(Y) == nrow(X))
    {
      Y_train = Y[-test_idx]
      Y_test = Y[test_idx]
      return(list(X_train = X_train, X_test = X_test, Y_train = Y_train, Y_test = Y_test))
    }
    else
    {
      print("Can't split Y: Length of Y must be equal number of rows in X.")
    }
  }
  return(list(X_train = X_train, X_test = X_test))
}

CART_prune <- function(X, Y)
{
  #' Tune cp parameter and get pruned CART tree accordingly
  #' @param X Dataframe holding explanatory variables
  #' @param Y Vector, categorical response variable
  #' 
  #' @returns List, containing pruned CART tree and its cp parameter 
  
  Tree = rpart(Y~., data = X, minsplit=2, cp=1e-15) # build maximal tree
  A = printcp(Tree)
  cv_err = A[,4] # Extract cross-validation error
  a = which(cv_err == min(cv_err)) # min value may not be unique
  s = min(cv_err[a] + 1 * A[a,5]) # define the new threshold to apply on cross-validation
  
  b = which(cv_err <= s)
  b1 = b[1]
  cp1 = A[b1, 1]
  return(list(CART_pruned = prune(Tree, cp=cp1), tuned_cp = cp1))
}




## Exercise 2, Question 1 ----

setwd("C:/Users/SamdGuizani/OneDrive - Data ScienceTech Institute/Documents/DSTI_MSc DS and AI/02-Foundation/05-ASML/Exam")
set.seed(310778)

### Import dataset and preprocessing ----

load("data_advanced.RData")

X = A$X
n = nrow(X) # nb obs.
p = ncol(X) # nb explanatory variables
Y = A$Y

# split dataset in train and test sets
split = train_test_split(X, Y, test_size = test_fraction)
X_train = split$X_train; n_train = nrow(X_train)
Y_train = split$Y_train
X_test = split$X_test ; n_test = nrow(X_test)
Y_test = split$Y_test

### Exploratory data analysis ----

# Correlation coefficient of each explanatory variable against response variable level (-1, 1)
corr_coef_X_Y_train = cor(X_train, as.numeric(Y_train))
barplot.default(corr_coef_X_Y_train[,1], xlab = 'Explanatory varaiables', ylab = 'Cor. Coef. (explanatory variable vs. response)')

# Boxplot of 10 first explanatory variable vs. response variable levels (-1, 1)
for (i in 1:10)
{
  boxplot(X[,i]~Y, ylab = paste0("V", i))
}

# Correlation of pair-wise explanatory variables (limited to 10 first)
corr_coef_X_train = round(cor(X_train[,1:10]), 2)
print(corr_coef_X_train)

### Models development ----

models.Q1 = list() # list to collect model candidates

#### Logistic regression ----

# Apply LASSO variable selection in context of Logistic regression
# Reference = https://glmnet.stanford.edu/articles/glmnet.html#quick-start

# Apply cross-validation to find best lambda parameter
cv_logistic_regression = cv.glmnet(as.matrix(X_train), Y_train, family = "binomial", type.measure = "class", nfolds = 10)
print(cv_logistic_regression)
plot(cv_logistic_regression)

logistic_regression = glmnet(X_train, Y_train, family = "binomial", type.measure = "class", nfolds = 10, lambda = cv_logistic_regression$lambda.1se)
models.Q1$logistic_regression = logistic_regression
print(logistic_regression)
as.matrix(coef(logistic_regression)[which(coef(logistic_regression) != 0),]) # identify selected variables

#### CART classifier ----
CART_pruning = CART_prune(X_train, Y_train)
CART_classif = CART_pruning$CART_pruned
models.Q1$CART_classif = CART_classif

CART_classif
summary(CART_classif)
printcp(CART_classif)
plot(CART_classif, uniform = TRUE, compress = TRUE, margin = 0.1)
text(CART_classif, use.n = TRUE, all = TRUE, cex = 0.6)

#### Random Forest ----
RF_classif = randomForest(X_train, Y_train)
models.Q1$RF_classif = RF_classif

varImpPlot(RF_classif)

#### Random Forest for variable selection + CART classifier ----

RF.vsurf = VSURF(X_train, Y_train)
plot(RF.vsurf)

colnames(X)[RF.vsurf$varselect.thres] # Variables kept after Elimination step
colnames(X)[RF.vsurf$varselect.interp] # Variables kept after Interpretation step
colnames(X)[RF.vsurf$varselect.pred] # Variables kept after Prediction step

RF_CART_classif1 = rpart(Y_train~., data=X_train[, RF.vsurf$varselect.interp]) # CART with variables selected after Interpretation step
models.Q1$RF_CART_classif1 = RF_CART_classif1
plot(RF_CART_classif1, uniform = TRUE, compress = TRUE, margin = 0.1)
text(RF_CART_classif1, use.n = TRUE, all = TRUE, cex = 0.6)

RF_CART_classif2 = rpart(Y_train~., data=X_train[, RF.vsurf$varselect.pred]) # CART with variables selected after Prediction step
models.Q1$RF_CART_classif2 = RF_CART_classif2
plot(RF_CART_classif2, uniform = TRUE, compress = TRUE, margin = 0.1)
text(RF_CART_classif2, use.n = TRUE, all = TRUE, cex = 0.6)

### Models evaluation ----

for (model in models.Q1)
{
  print('####################################################################')
  print(model$call)
  print(model)
  
  # Predict class of test set observations 
  Y_prd = as.factor(predict(model, 
                            newx = as.matrix(X_test), 
                            newdata = X_test,
                            type = "class"))
  
  # Test set classification error
  err_test = sum(as.numeric(Y_test) != as.numeric(Y_prd)) / n_test
  print(paste('Test set classification error =', err_test))
  
  # Test set confusion matrix
  # Reference = https://developer.ibm.com/tutorials/awb-confusion-matrix-r/
  cm = confusionMatrix(Y_prd, Y_test)
  print(cm)
  
}
  


## Exercise 2, Question 2 ----

setwd("C:/Users/SamdGuizani/OneDrive - Data ScienceTech Institute/Documents/DSTI_MSc DS and AI/02-Foundation/05-ASML/Exam")
set.seed(310778)

#Import dataset and preprocessing ----

# Description of PM10 dataset can be found in this document:
# Reference: https://cran.r-project.org/web/packages/VSURF/VSURF.pdf### 
dfs = list(jus = VSURF::jus,
           gui = VSURF::gui,
           gcm = VSURF::gcm,
           rep = VSURF::rep,
           hri = VSURF::hri,
           ail = VSURF::ail)

# Concatenate dataframes and create a new 'station' column
combined_df = bind_rows(dfs, .id = "station")
combined_df$station = as.factor(combined_df$station)

print(paste("Number of observations with missing response value (PM10):", sum(is.na(combined_df$PM10))))
combined_df = combined_df[!is.na(combined_df$PM10),] # observations with missing response excluded 

n = nrow(combined_df) # nb obs
p = ncol(combined_df) - 1 # nb explanatory variables (including station)

# Split dataset in train and test sets
train_index = createDataPartition(combined_df$station, p = 1 - test_fraction, list = FALSE) # Create a partition by 'station'
train_df <- combined_df[train_index, ] # Split data into training and testing sets
test_df <- combined_df[-train_index, ]

# Check the distribution of 'station' in both sets
print(table(train_df$station))
print(table(test_df$station))

### Exploratory data analysis ----

# PM10 vs. station
boxplot(PM10 ~ station, data = train_df)
summary(aov(PM10 ~ station, data = train_df))

# Correlation of response and explanatory variables
cor_matrix <- cor(train_df[,2:ncol(train_df)], use = "complete.obs")
print(round(cor_matrix, 2))
heatmap(cor_matrix, symm = TRUE)

### Models development ----

models.Q2 = list() # empty list to collect candidate models

#### Linear models

# Linear regression, only numeric explanatory variables 
# (ATTENTION: SO2, NO, NO2 missing for some stations)
linear_regression.1 = lm(PM10 ~ ., data = train_df[,2:ncol(train_df)])
models.Q2$linear_regression.1 = linear_regression.1
summary(linear_regression.1)
plot(linear_regression.1)

# Linear regression, including 'station' and its interactions with numeric explanatory variables 
# (ATTENTION: SO2, NO, NO2 excluded, because missing for some stations)
linear_regression.2 = lm(PM10 ~ station * (T.min+T.max+T.moy+DV.maxvv+DV.dom+VV.max+VV.moy+PL.som+HR.min+HR.max+HR.moy+PA.moy+GTrouen+GTlehavre),
                                           data = train_df)
models.Q2$linear_regression.2 = linear_regression.2

foo = stepAIC(linear_regression.2, lm(PM10~1, data = train_df), direction = 'both', test='F') # test model reduction

summary(foo)
plot(foo)

#### CART regression ----
CART_pruning = CART_prune(train_df[,-2], train_df[,2])
CART_reg = CART_pruning$CART_pruned
models.Q2$CART_reg = CART_reg

CART_reg
summary(CART_reg)
printcp(CART_reg)
plot(CART_reg, uniform = TRUE, compress = TRUE, margin = 0.1)
text(CART_reg, use.n = TRUE, all = TRUE, cex = 0.6)

#### Random Forest ----
RF_reg = randomForest(PM10 ~ ., data = train_df, na.action = na.roughfix, ncores = detectCores()- 1) # problem with NA in predictor --> error
models.Q2$RF_reg = RF_reg

varImpPlot(RF_reg)

#### Random Forest for variable selection + CART classifier ----

RF.vsurf = VSURF(PM10 ~ ., data = train_df, na.action = na.roughfix, ncores = detectCores()- 1)
plot(RF.vsurf)

colnames(X)[RF.vsurf$varselect.thres] # Variables kept after Elimination step
colnames(X)[RF.vsurf$varselect.interp] # Variables kept after Interpretation step
colnames(X)[RF.vsurf$varselect.pred] # Variables kept after Prediction step

RF_CART_reg1 = rpart(Y_train~., data=X_train[, RF.vsurf$varselect.interp], ncores = detectCores()- 1) # CART with variables selected after Interpretation step
models.Q1$RF_CART_classif1 = RF_CART_classif1

RF_CART_reg2 = rpart(Y_train~., data=X_train[, RF.vsurf$varselect.pred]) # CART with variables selected after Prediction step
models.Q1$RF_CART_classif2 = RF_CART_classif2

