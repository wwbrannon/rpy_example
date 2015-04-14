### R vs pandas + scikit-learn testing script
### We have a dataset from the 1990 Census of individuals, with demographic and
### economic traits, and a binary indicator for whether the individual made
### more than $50,000 a year. Let's take this dataset, explore and clean it, and
### have a model bake-off to see how well we can predict the high-income indicator.
### We'll conduct the same analysis a) in R, b) with pandas / scikit-learn in Python.

### We'll use these metrics of performance:
###     o) inspecting the confusion matrix
###     o) the ROC curve, plotted
###     o) the area under the ROC curve
###     o) the accuracy and error rate
# ==================================================================================

library(pscl)
library(ROCR)
library(ggplot2)

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

## Logistic regression
logit.mod <- glm(high_income ~ age + workclass + education_num +
                              marital_status + occupation + relationship + race + sex +
                              capital_gain + capital_loss + hours_per_week + mainland_us,
                 data=train, family=binomial("logit"))
summary(logit.mod); pR2(logit.mod)

yPred <- ifelse(predict(logit.mod, newdata=test, type="response") >= 0.5, 1, 0)

#Confusion matrix
prop.table(table(test$income_bin, yPred))

#ROC curve, AUC, accuracy/error rates
pred <- prediction(yPred, test$income_bin)
perf <- performance(pred, "tpr", x.measure="fpr")
plot(perf)

## Logistic regression with PCA

# Support vector machine

# Random forest

# Neural network with a single hidden layer

# k-NN

# Lasso / ridge / elastic-net

# Bayesian logit

# Gaussian process model

# 