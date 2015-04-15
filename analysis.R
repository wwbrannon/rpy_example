### R vs pandas + scikit-learn testing script

#Confusion matrix at 0.5 threshold
#ROC curve
#AUC: area under the ROC curve

# ==================================================================================
library(caret)
library(pROC)
setwd("~/Downloads/rpy_example")
set.seed(447647079)

## Read in the data and preprocess it
df <- read.csv("data/census_data.txt")

df$mainland_us <- as.numeric(df$native_country == "United-States")
df$high_income <- as.factor(ifelse(df$income_bin == ">50K", 'y', 'n'))

## Explore it and investigate variable coding
summary(df)
with(df, table(education, education_num))

hist(df$age)
table(df$sex)

hist(df$capital_loss)
hist(df$capital_gain)
hist(df$hours_per_week)

table(df$high_income, df$income_bin, useNA="ifany")
summary(df$high_income)

#the CPS weights. these are real weird
summary(df$fnlwgt)
hist(df$fnlwgt)

## The predictors and the DV
form <- high_income ~ age + workclass + education_num +
  marital_status + occupation + relationship + race + sex +
  capital_gain + capital_loss + hours_per_week + mainland_us
dv <- all.vars(form)[1]
ivs <- all.vars(form)[-1]

## Create a training/test split
df$in_train <- sample(0:1, nrow(df), replace=TRUE)
balance_form <- as.formula(paste("in_train ~ ", paste(ivs, collapse="+")))
summary(glm(balance_form, data=df, family=binomial("logit")))

training <- df[df$in_train == 1,]
test     <- df[df$in_train == 0,]

# =========================================================================
### The model bake-off!

##########
## Logistic regression
##########
mod <- train(form,
             data=training,
             method="glm",
             trControl=trainControl(method="none"))

yPred <- predict(mod, newdata=test, type="prob")
plot(roc(test$high_income, yPred$y))

yPred <- predict(mod, newdata=test, type="raw")
confusionMatrix(yPred, test$high_income, positive='y')

#overfit? nope!
yPred <- predict(mod, newdata=training, type="raw")
confusionMatrix(yPred, training$high_income, positive='y')

##########
## Support vector machine
##########
mod <- train(form, data=training, method="svm", trControl=tc)


##########
## Random forest
##########

##########
## Neural network with a single hidden layer
##########
