# Load required packages
library(caTools); library(glmnet); library(glmmLasso); library(MASS);library(haven)
library(cv.glmmLasso);library(caret); library(lme4)

# Load data
data <- read.csv(file = "data.csv", header = TRUE)

# Data generation
## Generate a vector of non-zero coefficients (betas)
coef <- c(513.5282,-20.7033,-11.4100,15.5133,10.3119,-10.8415,8.5774,9.4806,-2.9026,-2.6038,5.7120,4.3972,-3.3299)
coef <- as.matrix(coef, nrow = length(coef), ncol = 1)

## Generate fixed-effects part
IV <- data[,c(2,53,73,80,135:138,140,142,189,237)] # locations of predictors with non-zero coefficients
intercept <- rep(1,dim(data)[1]) # create an intercept vector for all observations 

IV <- cbind(intercept, IV)
IV <- as.matrix(IV, ncol = length(coef) + 1)

fix <- IV%*%coef

## Save the number of students for each school
sch_num <- as.numeric(table(data$IDSCHOOL))

## Specify the model formula
data2 <- data[,-c(1,238)] # exclude schoolID and dependent variable
n <- colnames(data2)
f <- as.formula(paste("MAT ~" , paste(n[!n %in% "MAT"], collapse = " + ")))
f2 <- as.formula(paste("MAT ~" , paste(n[!n %in% "MAT"], collapse = " + "), " + (1+BSBM26BA|IDSCHOOL)"))

## Number of replication
NN <- 100 

for (k in 1:NN){
  
  print(paste("I ", k,sep=""))
  
  ## Generate random-effects part 
  Rij <- rnorm(8116, mean = 0, sd = 60.06) # residuals
  
  ### School-level random effects
  cov_matrix <- matrix(c(37.11^2, 37.11*10.86*0.24,
                         37.11*10.86*0.24, 10.86^2), 
                       nrow = 2, ncol = 2)
  mean_vector = c(0,0)
  random_effects <- mvrnorm(230, mu = mean_vector, Sigma = cov_matrix)
  b_0j <- random_effects[,1] # random intercept
  b_1j <- random_effects[,2] # random slope
  
  ## Repeat each school-level random effect for the number of students in the corresponding school
  b_0 <- c(); b_1 <- c()
  for (j in 1:length(sch_num)){
    b_0 <- c(b_0, rep(b_0j[j], sch_num[j]))
    b_1 <- c(b_1, rep(b_1j[j], sch_num[j]))
  }
  
  ## Create new values of dependent variables
  MAT <- fix + b_0 + b_1*IV[,12] + Rij
  MAT[which(MAT<200)] <- 200 # truncate the dependent variable for values below 200
  MAT[which(MAT>800)] <- 800 # truncate the dependent variable for values above 800
  
  ## Drop the original dependent variable and replace it with newly generated values
  data <- data[,-238] 
  data$MAT <- MAT
  
  ## Data split into training data and test data
  data$IDSCHOOL <- as.factor(data$IDSCHOOL)
  
  spl <- sample.split(data$IDSCHOOL, SplitRatio = 0.7)
  train <- subset(data, spl == TRUE)
  test <- subset(data, spl == FALSE)
  
  ## Train LassoML model on the training data
  x <- model.matrix(f, data=train)[,-1]
  y <- train$MAT

  fit <- cv.glmnet(x, y, alpha = 1, nfolds=5)

  ### The optimal lambda values selected through cross-validation  
  lambda.min <- fit$lambda.min # based on lambda.min
  lambda.1se <- fit$lambda.1se # based on lambda.1se
  
  ### Coefficients from LassoML model 
  coef(fit, s="lambda.min") # based on lambda.min
  coef(fit, s="lambda.1se") # based on lambda.1se

  ## Test LassoML on the test data
  xx <- model.matrix(f, data = test)[,-1]
  yy <- test$MAT
  
  fit.pred.min <- predict(fit, s= "lambda.min", newx=xx)
  fit.pred.1se <- predict(fit, s= "lambda.1se", newx=xx)
  
  ## Calculate root mean squared error (RMSE)
  sqrt(mean((fit.pred.min-yy)^2)) # RMSE of the LassoML model at lambda.min
  sqrt(mean((fit.pred.1se-yy)^2)) # RMSE of the LassoML model at lambda.1se

  
  ## Train GlmmLML model on the training data - fit linear mixed model
  glmm_model <- lmer(f2, data = train)
  
  ### Extract random effects (estimated random intercept and random slope)
  random_effects <- ranef(glmm_model)
  level2.residuals.intercept <- as.numeric(random_effects$IDSCHOOL[,"(Intercept)"])
  level2.residuals.slope <- as.numeric(random_effects$IDSCHOOL[,"BSBM26BA"])
  
  ### Save the number of students for each school in the training data
  sch_num2 <- as.numeric(table(train$IDSCHOOL))
  
  ### Repeat each school-level random effect for the number of students in the corresponding school
  level2.residuals_slope <- c()
  for (j in 1:length(sch_num2)){
    level2.residuals_slope <- c(level2.residuals_slope, rep(level2.residuals.slope[j], sch_num2[j]))
  }
  
  level2.residuals_slope_2 <- c()
  for (q in 1:dim(train)[1]){
    level2.residuals_slope_2 <- c(level2.residuals_slope_2, level2.residuals_slope[q] * train$BSBM26BA[q])
  }
  
  random <- level2.residuals_intercept + level2.residuals_slope_2
  
  ### Extract independent and dependent variables
  x <- model.matrix(f, data=train)[,-1]
  y <- train$MAT- random # extract the random intercept from the original dependent variable
  
  glmmnet_fit <- cv.glmnet(x, y, alpha = 1, nfolds = 5) # fit LassoML
  
  ### The optimal lambda values selected through cross-validation  
  lambda.min <- glmmnet_fit$lambda.min # based on lambda.min
  lambda.1se <- glmmnet_fit$lambda.1se # based on lambda.1se
  
  ### Coefficients from LassoML model 
  coef(glmmnet_fit, s="lambda.min") # based on lambda.min
  coef(glmmnet_fit, s="lambda.1se") # based on lambda.1se
  
  ### Save the number of students for each school in the test data
  sch_num3 <- as.numeric(table(test$IDSCHOOL))
  
  ### Repeat each school-level random effect for the number of students in the corresponding school
  level2.residuals_intercept <- c()
  for (j in 1:length(sch_num3)){
    level2.residuals_intercept <- c(level2.residuals_intercept, rep(level2.residuals.intercept[j], sch_num3[j]))
  }
  
  #level2 random effects for slope
  level2.residuals_slope <- c()
  for (j in 1:length(sch_num3)){
    level2.residuals_slope <- c(level2.residuals_slope, rep(level2.residuals.slope[j], sch_num3[j]))
  }
  
  level2.residuals_slope_2 <- c()
  for (q in 1:dim(test)[1]){
    level2.residuals_slope_2 <- c(level2.residuals_slope_2, level2.residuals_slope[q] * test$BSBM26BA[q])
  }
  
  random <- level2.residuals_intercept + level2.residuals_slope_2
  
  ## Test LassoML on the test data
  xx <- model.matrix(f,data=test)[,-1]
  yy <- test$MAT - random # extract the random intercept from the original dependent variable
  
  fit.pred1 <- predict(glmmnet_fit, s=glmmnet_fit$lambda.min, newx = xx)
  fit.pred2 <- predict(glmmnet_fit, s=glmmnet_fit$lambda.1se, newx = xx)
  
  ## Calculate root mean squared error (RMSE)
  sqrt(mean((fit.pred1-yy)^2)) # RMSE of the LassoML model at lambda.min
  sqrt(mean((fit.pred2-yy)^2)) # RMSE of the LassoML model at lambda.1se

  ## Train glmmLassoML model on the training data

  ### Set the lambda range
  lambdas <- seq(5000,250000,length.out = 100)
  
  ### The optimal lambda values selected through cross-validation
  kfold <- 5 # number of folds
  
  folds <- createFolds(train$MAT, k = kfold, list = TRUE, returnTrain = FALSE)
  
  cv_errors <- numeric(length(lambdas))
  cv_errors_sd <- numeric(length(lambdas)) 
  
  fold_rmse_matrix <- matrix(NA, nrow = kfold, ncol = length(lambdas)) 
  
  for (l in seq_along(lambdas)) {
    lambda <- lambdas[l]
    fold_errors <- numeric(kfold)
    
    for (i in seq_along(folds)) {
      val_idx <- folds[[i]]
      train_idx <- setdiff(seq_len(nrow(train)), val_idx)
      train_data_cv <- train[train_idx, ]
      val_data_cv <- train[val_idx, ]
      
      fit <- tryCatch({
        glmmLasso(fix = f, rnd = list(IDSCHOOL=~1+BSBM26BA), data = train_data_cv, 
                    family = gaussian(link="identity"), lambda = lambda)
      }, error = function(e) NULL)
      
      if (!is.null(fit)) {
        preds <- tryCatch({
          predict(fit, newdata = val_data_cv)
        }, error = function(e) rep(NA, nrow(val_data_cv)))
        actual <- val_data_cv$MAT
        
        if (length(actual) == length(preds)) {
          fold_errors[i] <- sqrt(mean((actual - preds)^2))
        } else {
          fold_errors[i] <- NA
        }
      } else {
        fold_errors[i] <- NA
      }
    }
    cv_errors[l] <- mean(fold_errors, na.rm = TRUE)
    cv_errors_sd[l] <- sd(fold_errors, na.rm = TRUE)
    fold_rmse_matrix[, l] <- fold_errors            
  }
  
  lambda.min <- lambdas[which.min(cv_errors)] # based on lambda.min
  
  min_error <- min(cv_errors, na.rm = TRUE)  
  min_error_se <- cv_errors_sd[which.min(cv_errors)] / sqrt(kfold) 
  
  lambda.1se.idx <- max(which(cv_errors <= min_error + min_error_se))
  lambda.1se <- lambdas[lambda.1se.idx] # based on lambda.1se

  glmm.fit1 <- glmmLasso(fix = f, rnd = list(IDSCHOOL=~1+BSBM26BA), data = train, family=gaussian(link="identity"), lambda = lambda.1se)
  glmm.fit2 <- glmmLasso(fix = f, rnd = list(IDSCHOOL=~1+BSBM26BA), data = train, family=gaussian(link="identity"), lambda = lambda.min)
  
  ### Coefficients from glmmLassoML model 
  coef(glmm.fit1) # based on lambda.min
  coef(glmm.fit2) # based on lambda.1se
  
  ### Save the original dependent variable in the test data
  yy <- test$MAT
  
  ## Test glmmLassoML on the test data
  glmm.pred1 <- predict(glmm.fit1, test)
  glmm.pred2 <- predict(glmm.fit2, test)
  
  ## Calculate root mean squared error (RMSE)
  sqrt(mean((glmm.pred1-yy)^2)) # RMSE of the LassoML model at lambda.min
  sqrt(mean((glmm.pred2-yy)^2)) # RMSE of the LassoML model at lambda.1se 

    
  ## Train mod.glmmLassoML (modified glmmLassoML) model on the training data
  lambdas <- seq(5000, 250000, length.out = 100)
  kfold <- 5
  
  ### The optimal lambda values selected through cross-validation  
  folds <- createFolds(train$MAT, k = kfold, list = TRUE, returnTrain = FALSE)
  
  cv_errors <- numeric(length(lambdas))
  cv_errors_sd <- numeric(length(lambdas)) 
  
  fold_rmse_matrix <- matrix(NA, nrow = kfold, ncol = length(lambdas)) 
  
  for (l in seq_along(lambdas)) {
    lambda <- lambdas[l]
    fold_errors <- numeric(kfold)
    
    for (i in seq_along(folds)) {
      val_idx <- folds[[i]]
      train_idx <- setdiff(seq_len(nrow(train)), val_idx)
      train_data_cv <- train[train_idx, ]
      val_data_cv <- train[val_idx, ]
      
      fit <- tryCatch({
        glmmLasso(fix = f, rnd = NULL, data = train_data_cv, family = gaussian(link="identity"), lambda = lambda)
      }, error = function(e) NULL)
      
      if (!is.null(fit)) {
        X_val <- model.matrix(f, data = val_data_cv)
        coefs <- coef(fit)
        preds <- as.vector(X_val %*% coefs)
        actual <- val_data_cv$MAT
        fold_errors[i] <- sqrt(mean((actual - preds)^2))
      } else {
        fold_errors[i] <- NA
      }
    }
    cv_errors[l] <- mean(fold_errors, na.rm = TRUE)
    cv_errors_sd[l] <- sd(fold_errors, na.rm = TRUE)
    fold_rmse_matrix[, l] <- fold_errors            
  }
  
  lambda.min <- lambdas[which.min(cv_errors)] # based on lambda.min

  min_error <- min(cv_errors, na.rm = TRUE)  
  min_error_se <- cv_errors_sd[which.min(cv_errors)] / sqrt(kfold) 
  
  lambda.1se.idx <- max(which(cv_errors <= min_error + min_error_se))
  lambda.1se <- lambdas[lambda.1se.idx] # based on lambda.1se

  ### Coefficients from mod.glmmLassoML model 
  mglmm.fit <- glmmLasso(fix = f, rnd = NULL, data = train, family=gaussian(link="identity"), lambda = lambda.min) # based on lambda.min
  mglmm.fit2 <- glmmLasso(fix = f, rnd = NULL, data = train, family=gaussian(link="identity"), lambda = lambda.1se) # based on lambda.1se

  mglmm.fit$coefficients
  mglmm.fit2$coefficients
  
  ## Test mod.glmmLassoML on the test data
  mglmm.pred1 <- predict(mglmm.fit, test)
  mglmm.pred2 <- predict(mglmm.fit2, test)
  
  ## Calculate root mean squared error (RMSE)
  mglmm_RMSE_min <- as.numeric(sqrt(mean((mglmm.pred1-yy)^2))) # RMSE of the mod.glmmLassoML model at lambda.min
  mglmm_RMSE_1se <- as.numeric(sqrt(mean((mglmm.pred2-yy)^2))) # RMSE of the mod.glmmLassoML model at lambda.1se
  
  
  ## Train glmmPQLML model on the training data
  PQL.fit <- glmmPQL(f, random = ~1+BSBM26BA|IDSCHOOL, family = gaussian, data = train)
  
  ### Coefficients from glmmPQLML model 
  fixed <- coef(summary(PQL.fit))
  ifelse(fixed[,5]<0.05, fixed[,1], 0) # statistically significant predictors only
  
  sig.predictors <- which(as.numeric(ifelse(fixed[,5]<0.05, fixed[,1], 0))!=0)-1
  
  ### Updated model formula that only includes significant predictors
  n2 <- colnames(data2)[sig.predictors]
  pql.form <- as.formula(paste("MAT ~" , paste(n2[!n2 %in% "MAT"], collapse = " + ")))
  
  ## Test glmmPQLML on the test data
  PQL_significant <- glmmPQL(pql.form, random = ~1+BSBM26BA|IDSCHOOL,
                             family = gaussian(link = "identity"), data = train)
  
  fit.pred.pql <- predict(PQL_significant, newdata = test, type="response")
  
  ## Calculate root mean squared error (RMSE)
  sqrt(mean((fit.pred.pql-yy)^2))
}