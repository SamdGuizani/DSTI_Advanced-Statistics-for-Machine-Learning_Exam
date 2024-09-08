#######################################################
### DSTI - Advanced Statistics for Machine Learning ###
### Exam 2024                                       ###
### Author: Samd Guizani (SPOC, A23)                ###
#######################################################

### Exercise 1

setwd("C:/Users/SamdGuizani/OneDrive - Data ScienceTech Institute/Documents/DSTI_MSc DS and AI/02-Foundation/05-ASML/Exam")

# Import dataset
Dataset_ozone <- read.csv2("Dataset_ozone.txt", row.names=1)

# Convert variables "vent" en "pluie" to categorical variables
Dataset_ozone$vent = as.factor(Dataset_ozone$vent)
Dataset_ozone$pluie = as.factor(Dataset_ozone$pluie)

# # Convert variables "Ne9", "Ne12" and "Ne15" to categorical variables
# data$Ne9 = as.factor(data$Ne9)
# data$Ne12 = as.factor(data$Ne12)
# data$Ne15 = as.factor(data$Ne15)

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
print(hbeta)

## Exercise 1 - Question 2b: Linear model with constraints on beta coefficients of explanatory variables T9, T12 and T15

# Constraints are linear relationships on the beta coefficients such that R.beta = r
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
print(hbeta_c)

# Some idea about question 1 : change LS cost function by adding norm(r-R.Beta)
hbeta_c2 = solve(t(X) %*% X + t(R) %*% R) %*% (t(X) %*% Y + t(R) %*% r)
print(hbeta_c2)

# ###### Playing with mock data:
# 
# set.seed(19780731)
# n = 20 # n obs
# 
# # inputs
# X1 = rnorm(20)
# X2 = runif(20)
# X3 = rexp(20)
# 
# # response
# Y = 42 + 2*X1 + 3*X2 - 5*X3 + rnorm(n)
# 
# df = data.frame(Y = Y,
#                 X1 = X1,
#                 X2 = X2,
#                 X3 = X3)
# 
# X = data.matrix(cbind(rep(1, n), df[, 2:ncol(df)]))
# Y = data.matrix(df$Y)
# 
# # "standard" linear regression
# LM = lm(Y~., data = df)
# 
# beta_hat = solve(t(X) %*% X) %*% t(X) %*% Y
# 
# # "constrained" linear regression s.t. R.beta = r
# R = rbind(c(0, 1, 1, 0),
#           c(0, 1, 0, 1))
# 
# r = rbind(0,
#           0)
# 
# beta_c_hat = beta_hat + solve(t(X) %*% X) %*% t(R) %*% solve(R %*% solve(t(X) %*% X) %*% t(R)) %*% (r - R %*% beta_hat)
