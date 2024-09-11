#######################################################
### DSTI - Advanced Statistics for Machine Learning ###
### Exam 2024                                       ###
### Author: Samd Guizani (SPOC, A23)                ###
#######################################################

### Exercise 1

setwd("C:/Users/SamdGuizani/OneDrive - Data ScienceTech Institute/Documents/DSTI_MSc DS and AI/02-Foundation/05-ASML/Exam")

## Global variables
ALPHA = .05

## User defined functions
MLR_predict <- function(X, beta) 
{
  #' Predicts response values using a multiple linear regression model.
  #'
  #' @param X A numeric matrix of predictors, first column as intercept (all 1s).
  #' @param beta A numeric vector of regression coefficients, including intercept.
  #'
  #' @return A numeric vector of predicted response values.
  
  # Predicted response is X * beta
  y_pred <- X %*% beta
  return(y_pred)
}

MLR_residuals <- function(X, beta, y_act) 
{
  #' Calculates the residuals from a multiple linear regression model.
  #'
  #' @param X A numeric matrix of predictors, where first column is the intercept (all 1s).
  #' @param beta A numeric vector of regression coefficients, including intercept.
  #' @param y_act A numeric vector of actual response values.
  #'
  #' @return A numeric vector of residuals (actual - predicted values).

  # Calculate residuals (actual - predicted)
  residuals <- y_act - MLR_predict(X, beta)
  
  return(residuals)
}

MLR_residual_std_error <- function(X, beta, y_act) 
{
  #' Calculates residual standard error (RSE) for a multiple linear regression model.
  #'
  #' @param X A numeric matrix of predictors, with first column as intercept (all 1's).
  #' @param beta A numeric vector of regression coefficients, including intercept.
  #' @param Y_act A numeric vector of actual response values.
  #'
  #' @return A numeric value representing the residual standard error.
  
  # Calculate residuals
  residuals <- MLR_residuals(X, beta, y_act)
  
  # Number of observations (n) and number of predictors (p)
  n <- length(y_act)
  p <- length(beta) - 1  # Subtract 1 for the intercept
  
  # Calculate the Residual Standard Error (RSE)
  rse <- sqrt(sum(residuals^2) / (n - p - 1))
  
  return(rse)
}

MLR_coef_ci <- function(X, beta, y_act, alpha) 
{
  #' Calculates confidence intervals for the coefficients of a multiple linear regression model.
  #'
  #' @param X A numeric matrix of predictors, where first column is the intercept (all 1s).
  #' @param beta A numeric vector of regression coefficients, including intercept.
  #' @param y_act A numeric vector of actual response values.
  #' @param alpha A numeric value representing significance level (e.g., 0.05 for a 95% confidence interval).
  #'
  #' @return A numeric matrix with 3 columns: coefficients, lower bound and upper confidence interval limits.

  # Number of observations (n) and number of predictors (p)
  n <- nrow(X)
  p <- length(beta)
  
  # # Calculate predicted response
  # y_pred <- X %*% beta
  # 
  # # Calculate residuals and residual sum of squares
  # residuals <- y_act - y_pred
  # rss <- sum(residuals^2)
  
  # Residual variance
  residual_var <- MLR_residual_std_error(X, beta, y_act)^2
  
  # Inverse of (X'X)
  XtX_inv <- solve(t(X) %*% X)
  
  # Standard errors of the coefficients
  se_beta <- sqrt(diag(XtX_inv) * residual_var)
  
  # Critical value from t-distribution
  t_value <- qt(1 - alpha / 2, df = n - p)
  
  # Confidence intervals
  lower_bound <- beta - t_value * se_beta
  upper_bound <- beta + t_value * se_beta
  
  # Create a matrix with coefficients and their confidence intervals
  ci_matrix <- cbind(beta, lower_bound, upper_bound)
  colnames(ci_matrix) <- c("Est. Coef.", "Lower Bound", "Upper Bound")
  
  return(ci_matrix)
}

MLR_r2 <- function(X, beta, y_act) 
{
  #' Calculates R-squared value for a multiple linear regression model.
  #'
  #' @param X A numeric matrix of predictors, where first column is the intercept (all 1s).
  #' @param beta A numeric vector of regression coefficients, including intercept.
  #' @param y_act A numeric vector of actual response values.
  #'
  #' @return A numeric value representing the R-squared value.

  # Total sum of squares (TSS)
  tss <- sum((y_act - mean(y_act))^2)
  
  # Residual sum of squares (RSS)
  rss <- sum(MLR_residuals(X, beta, y_act)^2)
  
  # Calculate R-squared
  r_squared <- 1 - (rss / tss)
  
  return(r_squared)
}

MLR_r2_adj <- function(X, beta, y_act) 
{
  #' @param X A numeric matrix of predictors, where first column is intercept (all 1s).
  #' @param beta A numeric vector of regression coefficients, including intercept.
  #' @param y_act A numeric vector of actual response values.
  #'
  #' @return A numeric value representing the adjusted R-squared value.

  # Number of observations (n) and number of predictors (p)
  n <- nrow(X)
  p <- ncol(X) - 1  # Subtract 1 to exclude intercept
  
  # # Calculate predicted response
  # y_pred <- X %*% beta
  # 
  # # Total sum of squares (TSS)
  # tss <- sum((y_act - mean(y_act))^2)
  # 
  # # Residual sum of squares (RSS)
  # rss <- sum((y_act - y_pred)^2)
  
  # Calculate R-squared
  r_squared <- MLR_r2(X, beta, y_act)
  
  # Calculate adjusted R-squared
  r_squared_adj <- 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))
  
  return(r_squared_adj)
}


## Import dataset
Dataset_ozone <- read.csv2("Dataset_ozone.txt", row.names=1)

# Convert variables "vent" en "pluie" to categorical variables
Dataset_ozone$vent = as.factor(Dataset_ozone$vent)
Dataset_ozone$pluie = as.factor(Dataset_ozone$pluie)

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

## Exercise 1 - Question 2a: Linear model predicting maxO3 as function of all the explanatory variables
LM = lm(formula = maxO3 ~ ., data = Dataset_ozone)
summary(LM)

# manually calculate least squares linear model beta coefficients
hbeta = solve(t(X) %*% X) %*% t(X) %*% Y 
hbeta_ci = MLR_coef_ci(X, hbeta, Y, ALPHA) # hbeta with confidence intervals

## Exercise 1 - Question 2b: Linear model with constraints on beta coefficients of explanatory variables T9, T12 and T15

# Constraints are linear functions on beta coefficients such that R.beta = r
# i.e., each value in r is a weighted sum applied on beta coefficients of selected explanatory variables

# R matrix holds the weights to apply to target beta coefficients
q = 1 # number of constraints to consider. q < p+1. q is rank of R.
R = matrix(0, nrow = q, ncol = p + 1) # initialize matrix of 0's, size q * (p+1)
# assign constraint's weights on beta coefficients
R[1, 2:4] = 1

# r vector holds the weighted sums to consider for the constraints
r = matrix(0, nrow = q, ncol = 1)
# assign constraint's weighted sums
r[1, 1] = 0

# Computing beta coefficients of constrained linear model
hbeta_c = 
  hbeta + 
  solve(t(X) %*% X) %*% 
  t(R) %*% 
  solve(R %*% solve(t(X) %*% X) %*% t(R)) %*%
  (r - R %*% hbeta)
hbeta_c_ci = MLR_coef_ci(X, hbeta_c, Y, ALPHA) # hbeta_c with confidence intervals

## Exercise 1 - Question 2c: Compare the 2 linear models

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
Y_prd1 = LM$fitted.values
Y_prd2 = MLR_predict(X, hbeta_c)

# Create scatter plot of Y_prd1 and Y_prd2 vs Y
plot(Y, Y_prd1, main="Model comparison", xlab="Y actual", ylab="Y predicted", pch=19, col="blue")
abline(lm(Y~Y_prd1), col="blue", lwd=2)
points(Y, Y_prd2, pch=19, col="red")
abline(lm(Y~Y_prd2), col="red", lwd=2, lty=3)

# Target line y = x spanning range of Y, Y_prd1 and Y_prd2
Y_range <- range(c(Y, Y_prd1, Y_prd2))
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
