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

train$high_income <- as.factor(ifelse(train$income_bin == ">50K", 1, 0))
table(train$high_income, train$income_bin, useNA="ifany")

test$high_income <- as.factor(ifelse(test$income_bin == ">50K", 1, 0))
table(test$high_income, test$income_bin, useNA="ifany")

#the CPS weights. these are real weird
hist(train$fnlwgt)
summary(train$fnlwgt)

# =========================================================================
### The model bake-off!

summary(train$high_income)
summary(test$high_income)

form <- high_income ~ age + workclass + education_num +
  marital_status + occupation + relationship + race + sex +
  capital_gain + capital_loss + hours_per_week + mainland_us

dv <- all.vars(form)[1]
ivs <- all.vars(form)[-1]

##########
## Logistic regression
##########
mod <- glm(form, data=train, family=binomial("logit"))
summary(mod)

yPred <- predict(mod, newdata=test, type="response")
pred <- prediction(yPred, test$high_income)

#Confusion matrix at 0.5 threshold
prop.table(table(true = test$high_income, pred = ifelse(yPred >= 0.5, 1, 0)))

#ROC curve
perf <- performance(pred, "tpr", x.measure="fpr")
plot(perf)

#AUC: area under the ROC curve
as.numeric(performance(pred, "auc")@y.values)

##########
## Support vector machine
##########
library(e1071)

mod <- svm(form, data=train)
summary(mod)

yPred <- as.numeric(as.character(predict(mod, newdata=test)))
pred <- prediction(yPred, test$high_income)

#Confusion matrix at 0.5 threshold
prop.table(table(true = test$high_income, pred = yPred))

#ROC curve
perf <- performance(pred, "tpr", x.measure="fpr")
plot(perf)

#AUC: area under the ROC curve
as.numeric(performance(pred, "auc")@y.values)

##########
## Random forest
##########
library(randomForest)

mod <- randomForest(form, data=train)
summary(mod)

yPred <- as.numeric(as.character(predict(mod, newdata=test)))
pred <- prediction(yPred, test$high_income)

#Confusion matrix at 0.5 threshold
prop.table(table(true = test$high_income, pred = yPred))

#ROC curve
perf <- performance(pred, "tpr", x.measure="fpr")
plot(perf)

#AUC: area under the ROC curve
as.numeric(performance(pred, "auc")@y.values)

##########
## Neural network with a single hidden layer
##########
library(nnet)

mod <- nnet(form, data=train, size=10)
summary(mod)

yPred <- as.numeric(as.character(predict(mod, newdata=test)))
pred <- prediction(yPred, test$high_income)

#Confusion matrix at 0.5 threshold
prop.table(table(true = test$high_income, pred = ifelse(yPred >= 0.5, 1, 0)))

#ROC curve
perf <- performance(pred, "tpr", x.measure="fpr")
plot(perf)

#AUC: area under the ROC curve
as.numeric(performance(pred, "auc")@y.values)
