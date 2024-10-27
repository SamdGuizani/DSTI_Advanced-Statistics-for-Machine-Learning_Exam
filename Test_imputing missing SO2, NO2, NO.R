# IDEA: Try to impute SO2, NO2, NO missing values.
# NO2 and NO imputed from the other variables (including PM10, excluding SO2, mutually excluding NO and NO2)
# SO2 imputed from the other variables (including PM10, NO, NO2)

# CONCLUSION: Models not improved. NO/NO2/SO2 cannot be well predicted based on other variables --> limited benefit from imputation

LM_NO2 = lm(sqrt(NO2) ~ (.)^2, data = df_subset[, -c(1, 3, 5)]) ; summary_and_plots(LM_NO2)
LM_NO2_reduced = stepAIC(LM_NO2, direction = "back", test="F") ; summary_and_plots(LM_NO2_reduced)

LM_NO = lm(sqrt(NO) ~ (.)^2, data = df_subset[, -c(1, 4, 5)]) ; summary_and_plots(LM_NO)
LM_NO_reduced = stepAIC(LM_NO, direction = "back", test="F") ; summary_and_plots(LM_NO_reduced)

LM_SO2 = lm(log(SO2+.5) ~ (.)^2, data = df_subset[, -c(1)]) ; summary_and_plots(LM_SO2)
LM_SO2_reduced = stepAIC(LM_SO2, direction = "back", test="F") ; summary_and_plots(LM_SO2_reduced)

# Copy train_df and impute missing variables NO, NO2, PM10

train_df_with_imputation = train_df

sum(is.na(train_df_with_imputation$NO2))
missing_NO2 = is.na(train_df$NO2)
imputed_NO2 = predict(LM_NO2_reduced, newdata = train_df[missing_NO2,])^2
train_df_with_imputation$NO2[missing_NO2] = imputed_NO2
sum(is.na(train_df_with_imputation$NO2))

sum(is.na(train_df_with_imputation$NO))
missing_NO = is.na(train_df$NO)
imputed_NO = predict(LM_NO_reduced, newdata = train_df[missing_NO,])^2
train_df_with_imputation$NO[missing_NO] = imputed_NO
sum(is.na(train_df_with_imputation$NO))

sum(is.na(train_df_with_imputation$SO2))
missing_SO2 = is.na(train_df$SO2)
imputed_SO2 = exp(predict(LM_SO2_reduced, newdata = train_df_with_imputation[missing_SO2,])) - 0.5
train_df_with_imputation$SO2[missing_SO2] = imputed_SO2
sum(is.na(train_df_with_imputation$SO2))

# Build models and test them

# Linear regression, only numeric explanatory variables 
# (ATTENTION: SO2, NO, NO2 missing for some stations)
linear_regression.1 = lm(log(PM10) ~ ., data = train_df_with_imputation)
# linear_regression.1 = lm(PM10 ~ ., data = train_df_with_imputation)
models.Q2$linear_regression.1 = linear_regression.1
summary_and_plots(linear_regression.1)

# Linear regression, including 'station' and its interactions with numeric explanatory variables 
# (ATTENTION: SO2, NO, NO2 excluded, because missing for some stations)
linear_regression.2 = lm(log(PM10) ~ station * (NO2+NO+SO2+T.min+T.max+T.moy+DV.maxvv+DV.dom+VV.max+VV.moy+PL.som+HR.min+HR.max+HR.moy+PA.moy+GTrouen+GTlehavre),
                         data = train_df_with_imputation)
# linear_regression.2 = lm(PM10 ~ station * (T.min+T.max+T.moy+DV.maxvv+DV.dom+VV.max+VV.moy+PL.som+HR.min+HR.max+HR.moy+PA.moy+GTrouen+GTlehavre),
#                          data = train_df_with_imputation)
models.Q2$linear_regression.2 = linear_regression.2
summary_and_plots(linear_regression.2)

linear_regression.3 = stepAIC(linear_regression.2, lm(log(PM10)~1, data = train_df_with_imputation), direction = 'back', test='F') # test model reduction
# linear_regression.3 = stepAIC(linear_regression.2, lm(PM10~1, data = train_df_with_imputation), direction = 'back', test='F')
models.Q2$linear_regression.3 = linear_regression.3
summary_and_plots(linear_regression.3)

#### CART regression ----
CART_reg = CART_prune(train_df_with_imputation[,-2], log(train_df_with_imputation[,2]))
# CART_reg = CART_prune(train_df_with_imputation[,-2], train_df_with_imputation[,2])
models.Q2$CART_reg = CART_reg$CART_pruned
summary_and_plots(CART_reg$CART_pruned)

#### Random Forest ----
RF_reg = randomForest(log(PM10) ~ ., data = train_df_with_imputation, na.action = na.roughfix, ncores = detectCores()- 1) # problem with NA in predictor --> added na.action = na.roughfix
# RF_reg = randomForest(PM10 ~ ., data = train_df_with_imputation, na.action = na.roughfix, ncores = detectCores()- 1)
models.Q2$RF_reg = RF_reg
summary_and_plots(RF_reg)

### Models evaluation ----

for (model in models.Q2)
{
  print('####################################################################')
  print(model$call)
  # print(model)
  
  # Predict result of test set observations 
  Y_prd = predict(model, newdata = test_df)
  
  # Calculate Mean Squared Error (MSE)
  MSE = mse(test_df$PM10, exp(Y_prd))
  print(paste0('Root Mean Squared Error = ', sqrt(MSE)))
  
  # # Test set classification error
  # err_test = sum(as.numeric(Y_test) != as.numeric(Y_prd)) / n_test
  # print(paste('Test set classification error =', err_test))
  # 
  # # Test set confusion matrix
  # # Reference = https://developer.ibm.com/tutorials/awb-confusion-matrix-r/
  # cm = confusionMatrix(Y_prd, Y_test)
  # print(cm)
  
  plot(test_df$PM10, exp(Y_prd))
  abline(b=1, a=0, col="black")
}