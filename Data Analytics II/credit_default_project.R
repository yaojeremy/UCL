library('ggplot2') # visualization
library('ggthemes') # visualization
library('scales') # visualization
library('dplyr') # data manipulation
library('mice') # imputation
library('randomForest') # classification algorithm
library(gtools) # for discretisation
library(corrplot)
library(Hmisc)
library(devtools)
library(PerformanceAnalytics)
library(FactoMineR)
library(Metrics)
library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)
library(Ckmeans.1d.dp)
library(rpart)
library(rpart.plot)
library('corrgram') 

setwd("/Users/Jeremy/Downloads")
default.df  <- read.csv('default of credit card clients.csv', sep = ",") # or please use sep:";" if you have a French csv format.

#descriptive analytics
corrgram(default.df[,c(2:25)], cex.labels=0.5, upper.panel=panel.pie, lower.panel=NULL, text.panel=panel.txt, cor.method = "pearson")



## 75% of the sample size
smp_size <- floor(0.75 * nrow(default.df))

## set the seed to make your partition reproductible
set.seed(123)
train_ind <- sample(seq_len(nrow(default.df)), size = smp_size)

train <- default.df[train_ind, ]
test <- default.df[-train_ind, ]

nTrain <- nrow(train)
nSmallTrain <- round(nTrain*0.75)

print(nSmallTrain)

# Set the seed for a random sample of the row indices in the smaller training set.
set.seed(16875)
# Sample the row indices in the smaller training set
rowIndicesSmallTrain <- sample(1:nTrain, size = nSmallTrain, replace = FALSE)

# Split the training set into the smaller training set and the validation set using these indices. 
default_smallTrain <- train[rowIndicesSmallTrain, ]
default_validation <- train[-rowIndicesSmallTrain, ]


#BRIER

# Write a function to calculate the average Brier score.  
brier.score <- function(pred, real) {
  return(mean((pred - real)^2))   
}

# Establish a reference brier score for a naive forecast
# Naive forecast: use the propotion of default loans in the traning set as the prediction 
# for all the loans in the testing set.

train.default.rate <- mean(default_smallTrain$default.payment.next.month)
test.realizations <- default_validation$default.payment.next.month
naive.pred <- rep(train.default.rate, length(test.realizations))
brier.ref <- brier.score(naive.pred, test.realizations)

# Write a function to calculate the Brier skill score
skill.score <- function(pred, real, brier.ref) {
  brier.score <- brier.score(pred, real) # calculate the Brier score for your predictions.
  return(1 - brier.score/ brier.ref)
}

#LogLoss score function
LogLoss <- function(pred, real) {
  eps <- 1e-15
  pred <- pmin(pmax(pred, eps), 1 - eps)
  -mean(real * log(pred) + (1 - real) * log(1 - pred))
}
###

#logistic regression all
logit.reg <- glm(train$default.payment.next.month ~ LIMIT_BAL+SEX+MARRIAGE+AGE+PAY_0+PAY_2+PAY_3+PAY_4+PAY_5+
                   PAY_6+BILL_AMT1+BILL_AMT2+BILL_AMT3+BILL_AMT4+BILL_AMT5+BILL_AMT6+PAY_AMT1+PAY_AMT2+PAY_AMT3+
                   PAY_AMT4+PAY_AMT5+PAY_AMT6
                 , data = train, family = "binomial")

summary(logit.reg)
pred.logit.reg <- predict(logit.reg, test, type = "response")

# Calculate the Brier Skill Score and Logloss
skill.score(pred.logit.reg, test$default.payment.next.month, brier.ref)
LogLoss(pred.logit.reg, test$default.payment.next.month)
#logistic regression 2
logit2.reg <- glm(train$default.payment.next.month ~ PAY_0+PAY_2+PAY_3+PAY_4+PAY_5+
                    PAY_6
                  , data = train, family = "binomial")

summary(logit2.reg)
pred.logit.reg2 <- predict(logit2.reg, test, type = "response")
#Brier skill score and Logloss
skill.score(pred.logit.reg2, test$default.payment.next.month, brier.ref)
LogLoss(pred.logit.reg2, test$default.payment.next.month)
#logistic regression 3
logit3.reg <- glm(train$default.payment.next.month ~ LIMIT_BAL+SEX+MARRIAGE+AGE+PAY_0+PAY_AMT6
                  +PAY_6
                  , data = train, family = "binomial")

summary(logit3.reg)
pred.logit.reg3 <- predict(logit3.reg, test, type = "response")
#Brier skill score and Logloss
skill.score(pred.logit.reg3, test$default.payment.next.month, brier.ref)
LogLoss(pred.logit.reg3, test$default.payment.next.month)

#Regression Tree
reg.tree <- rpart(default.payment.next.month ~ ., data = train)

reg.tree$variable.importance

reg.tree.pred <- predict(reg.tree, test)
#Brier skill score and Logloss
skill.score(reg.tree.pred, test$default.payment.next.month, brier.ref)
LogLoss(reg.tree.pred, test$default.payment.next.month)

#boosted rpart to get an optimal cp
reg.tree2 <- rpart(default.payment.next.month ~ ., data = train, control = rpart.control(cp = 0.0005 ) )

reg.tree.pred2 <- predict(reg.tree2, test)
#Brier skill score and Logloss
skill.score(reg.tree.pred2, test$default.payment.next.month, brier.ref)
LogLoss(reg.tree.pred2, test$default.payment.next.month)

plotcp(reg.tree2,upper = "splits")
#boosted rpart 
reg.tree3 <- rpart(default.payment.next.month ~ ., data = train, control = rpart.control(cp = 0.00069 ) )

reg.tree.pred3 <- predict(reg.tree3, test)
skill.score(reg.tree.pred3, test$default.payment.next.month, brier.ref)
LogLoss(reg.tree.pred3, test$default.payment.next.month)
rpart.plot(reg.tree3)

# Create Random Forest models

#First random forest model

rf.1 <- randomForest(default_smallTrain[,c(2:24)], y = as.factor(default_smallTrain[,25]),
                     ntrees = 10, importance = TRUE)
rf.predict <- predict(rf.1, default_validation[,c(2:24)], type = "prob")[,2]

#Score the predictions
skill.score(rf.predict, default_validation$default.payment.next.month, brier.ref) 
LogLoss(rf.predict, default_validation$default.payment.next.month)

#Second random forest model
rf.2 <- randomForest(default_smallTrain[,c(2, 8:11, 12, 13, 15, 21, 22)], y = as.factor(default_smallTrain[,25]),
                     ntrees = 10, importance = TRUE)
rf.predict2 <- predict(rf.2, default_validation[,c(2, 8:11, 12, 13, 15, 21, 22)], type = "prob")[,2]

#Score the predictions
LogLoss(rf.predict2, default_validation$default.payment.next.month)
skill.score(rf.predict2, default_validation$default.payment.next.month, brier.ref) 


#xgboost

matrix_train <- apply(train, 2, function(x) as.numeric(as.character(x)))
#matrix_train <- subset(matrix_train, select = -c(25,26,27, 28, 29, 30) )

outcome_yes_train <- ifelse(train$default.payment.next.month == "1", 1, 0)

matrix_test <- apply(test, 2, function(x) as.numeric(as.character(x)))
#matrix_test <- subset(matrix_test, select = -c(25,26,27, 28, 29, 30) )

outcome_yes_test <- ifelse(test$default.payment.next.month == "1", 1, 0)
outcome_yes_train
outcome_yes_test
xgb_train_matrix <- xgboost::xgb.DMatrix(data = as.matrix(matrix_train), label = outcome_yes_train)
xgb_test_matrix <- xgboost::xgb.DMatrix(data = as.matrix(matrix_test), label = outcome_yes_test)
vector_test.xgb <- as.vector(test$default.payment.next.month)
watchlist <- list(train = xgb_train_matrix, test = xgb_test_matrix)
label <- getinfo(xgb_test_matrix, "label")

params <- list("objective" = "binary:logistic",    # binary classification 
               "eval_metric" = "logloss",    # evaluation metric 
               "nthread" = 6,   # number of threads to be used 
               "max_depth" = 2,    # maximum depth of tree 
               "eta" = 0.08,    # step size shrinkage 
               "subsample" = 0.9,    # part of data instances to grow tree 
               "colsample_bytree" = 0.9)  # subsample ratio of columns when constructin

#The cross validation function of xgboost
xgb.cv(param = params, 
       data = xgb_train_matrix, 
       nfold = 3,
       label = getinfo(xgb_train_matrix, "label"),
       nrounds = 1000,
       early_stopping_rounds = 3, maximize = FALSE, prediction = TRUE) 
#setting the parameter early_stopping 
#xgboost will terminate the training process if the performance is getting worse in the iteration.

## we found that the best iteration is nrounds=130
xgb.trees <-xgb.train(param = params, 
                      data = xgb_train_matrix, 
                      nfold = 3,
                      label = getinfo(xgb_train_matrix, "label"),
                      nrounds = 130,
                      prediction = TRUE)
xgb.trees.pred <- predict(xgb.trees, xgb_test_matrix)
skill.score(xgb.trees.pred, vector_test.xgb, brier.ref)
LogLoss(xgb.trees.pred,vector_test.xgb)
#Feature importance
# Gain indicates the contribution of each feature. 
# The higher the percentage, the greater the contribution.
importance_matrix <-xgb.importance(colnames(xgb_train_matrix), model = xgb.trees)  
xgb.plot.importance(importance_matrix)
#Stacking model 

# Split the training set into two parts:
set.seed(22500)
nS1 <- nrow(train)/2
Sa <- train[1 : nS1,]
Sb <- train[-c(1:nS1),]

#Train the base models using sample a and generate predictions for sample b
M1_a <- glm(default.payment.next.month ~  . + (.)^2, data = Sa, family = "binomial")
M1_a.predict <- predict.glm(M1_a, Sb, type = "response")

M2_a <-  rpart(default.payment.next.month ~ ., data = Sa, control = rpart.control(cp = 0.003 ) )
M2_a.predict <-  predict(M2_a, Sb)

#Train the base models using sample b and generate predictions for sample a

M1_b <- glm(default.payment.next.month ~  . + (.)^2, data = Sb, family = "binomial")
M1_b.predict <- predict.glm(M1_b, Sa, type = "response")

M2_b <-  rpart(default.payment.next.month ~ ., data = Sb, control = rpart.control(cp = 0.003 ) )
M2_b.predict <-  predict(M2_b, Sa)

#Fit a regression tree to the predictions generated
#Create a data frame as the input for the stacking model.
stacker.df <- data.frame(default.payment.next.month = train$default.payment.next.month, 
                         M1.predict = c(M1_b.predict, M1_a.predict), 
                         M2.predict = c(M2_b.predict, M2_a.predict))
#Fit a stacker model to the predictions
stackerModel <- rpart(default.payment.next.month ~ ., data = stacker.df, control = rpart.control(cp = 0.003))

#Re-train the base models using the entire training set and generate predictions for the validation set
#Re-train the base models to the entire training set (train).
M1.trainAll <- glm(default.payment.next.month ~ . + .^2, 
                   data = train, family = "binomial")

M2.trainAll <- rpart(default.payment.next.month ~ ., data = train, control = rpart.control(cp = 0.003))

#Generate predictions for the testing set.
M1.predict.test <- predict(M1.trainAll, test, type = "response")
M2.predict.test <- predict(M2.trainAll, test)

#Use M1.predict.test and M2.predict.test as inputs to the stacker model.

predict.variables <- data.frame('M1.predict' = M1.predict.test, 
                                'M2.predict' = M2.predict.test)
stacker.predict <- predict(stackerModel, predict.variables)

#Score the stacker model's prediction
skill.score(stacker.predict, test$default.payment.next.month, brier.ref) 
LogLoss(stacker.predict, test$default.payment.next.month)

#Let's try another stacker model with Random Forest

#Train the base models using sample a and generate predictions for sample b
M3_a <- glm(default.payment.next.month ~  . + (.)^2, data = Sa, family = "binomial")
M3_a.predict <- predict.glm(M3_a, Sb, type = "response")

M4_a <-  randomForest(Sa[,c(2:24)], y = as.factor(Sa[,25]),
             ntrees = 10, importance = TRUE)
M4_a.predict <-  predict(M4_a, Sb)

#Train the base models using sample b and generate predictions for sample a

M3_b <- glm(default.payment.next.month ~  . + (.)^2, data = Sb, family = "binomial")
M3_b.predict <- predict.glm(M3_b, Sa, type = "response")

M4_b <-randomForest(Sb[,c(2:24)], y = as.factor(Sb[,25]),
                      ntrees = 10, importance = TRUE)
M4_b.predict <-  predict(M4_b, Sa)

#Fit a regression tree to the predictions generated
#Create a data frame as the input for the stacking model.
stacker2.df <- data.frame(default.payment.next.month = train$default.payment.next.month, 
                         M3.predict = c(M3_b.predict, M3_a.predict), 
                         M4.predict = c(M4_b.predict, M4_a.predict))
#Fit a stacker model to the predictions
stackerModel2 <- glm(default.payment.next.month ~ . + .^2, 
                     data = stacker2.df, family = "binomial")


#Re-train the base models using the entire training set and generate predictions for the validation set
#Re-train the base models to the entire training set (train).
M3.trainAll <- glm(default.payment.next.month ~ . + .^2, 
                   data = train, family = "binomial")

M4.trainAll <-randomForest(train[,c(2:24)], y = as.factor(train[,25]),
                                      ntrees = 10, importance = TRUE)


#Generate predictions for the testing set.
M3.predict.test <- predict(M3.trainAll, test, type = "response")
M4.predict.test <- predict(M4.trainAll, test)

#Use M1.predict.test and M2.predict.test as inputs to the stacker model.

predict.variables2 <- data.frame('M3.predict' = M3.predict.test, 
                                'M4.predict' = M4.predict.test)
stacker.predict2 <- predict(stackerModel2, predict.variables2)

#Score the stacker model's prediction
skill.score(stacker.predict2, test$default.payment.next.month, brier.ref) 
LogLoss(stacker.predict2, test$default.payment.next.month)
