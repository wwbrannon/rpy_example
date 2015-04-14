### R vs pandas + scikit-learn testing script
# ==================================================================================
library(ROCR)
setwd("~/Downloads/rpy_example")

## Read in the data
train <- read.csv("data/adult.data.txt")
test <- read.csv("data/adult.test.txt")

all.equal(names(train), names(test))
names(train)

## Explore it and investigate variable coding
summary(train)
with(train, table(education, education_num))

hist(train$age)
table(train$sex)

hist(train$capital_loss)
hist(train$capital_gain)
hist(train$hours_per_week)

train$mainland_us <- as.numeric(train$native_country == "United-States")
test$mainland_us <- as.numeric(test$native_country == "United-States")

train$high_income <- ifelse(train$income_bin == ">50K", 1, 0)
table(train$high_income, train$income_bin, useNA="ifany")

test$high_income <- ifelse(test$income_bin == ">50K", 1, 0)
table(test$high_income, test$income_bin, useNA="ifany")

#the CPS weights. these are real weird
hist(train$fnlwgt)
summary(train$fnlwgt)

# =========================================================================
### The model bake-off!

summary(train$high_income)
summary(test$high_income)

##########
## Logistic regression
##########
mod <- glm(high_income ~ age + workclass + education_num +
                              marital_status + occupation + relationship + race + sex +
                              capital_gain + capital_loss + hours_per_week + mainland_us,
                 data=train, family=binomial("logit"))
summary(mod)

yPred <- predict(mod, newdata=test, type="response")
pred <- prediction(yPred, test$income_bin)

#Confusion matrix at 0.5 threshold
prop.table(table(test$high_income, ifelse(yPred >= 0.5, 1, 0)))

#ROC curve
perf <- performance(pred, "tpr", x.measure="fpr")
plot(perf)

#AUC: area under the ROC curve
as.numeric(performance(pred, "auc")@y.values)

##########
## Support vector machine
##########
library(e1071)

##########
## Random forest
##########
library(randomForest)

##########
## Neural network with a single hidden layer
##########
library(nnet)

##########
## k-NN
##########
library(class)

##########
## Lasso / ridge / elastic-net
##########
library(glmnet)

##########
## Gaussian process model
##########
library(gptk)
