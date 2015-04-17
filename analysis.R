### R vs pandas + scikit-learn testing script
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
df$train_section <- sample(0:2, nrow(df), replace=TRUE)
balance_form <- as.formula(paste("(train_section == 0) ~ ", paste(ivs, collapse="+")))
summary(glm(balance_form, data=df, family=binomial("logit")))

training <- df[df$train_section != 0,]
test     <- df[df$train_section == 0,]

# =========================================================================
### The model bake-off!

trc <- trainControl(method="repeatedcv", number=10, repeats=3,
                    classProbs=T, savePred=T, )

## Logistic regression
glm_mod <- train(form,
             data=training,
             method="glm",
             trControl=trc)

## Support vector machine
svm_mod <- train(form,
             data=training,
             method="svmLinear",
             tuneGrid=data.frame(C=10^seq(-3, 3)),
             trControl=trc)

## Random forest
rf_mod <- train(form,
             data=training,
             method="rf",
             trControl=trc)

## Neural network with a single hidden layer
nnet_mod <- train(form,
             data=training,
             method="nnet",
             trControl=trc)

## Results
#Cross-validated
results <- resamples(list(glm=glm_mod, svm=svm_mod, rf=rf_mod, nnet=nnet_mod))

summary(results)
bwplot(results)
dotplot(results)

#And on the holdout set
yPred <- predict(glm_mod, newdata=test, type="prob")
plot(roc(test$high_income, yPred$y))

yPred <- predict(glm_mod, newdata=training, type="prob")
plot(roc(training$high_income, yPred$y))

yPred <- predict(svm_mod, newdata=test, type="prob")
plot(roc(test$high_income, yPred$y))

yPred <- predict(svm_mod, newdata=training, type="prob")
plot(roc(training$high_income, yPred$y))

yPred <- predict(rf_mod, newdata=test, type="prob")
plot(roc(test$high_income, yPred$y))

yPred <- predict(rf_mod, newdata=training, type="prob")
plot(roc(training$high_income, yPred$y))

yPred <- predict(nnet_mod, newdata=test, type="prob")
plot(roc(test$high_income, yPred$y))

yPred <- predict(nnet_mod, newdata=training, type="prob")
plot(roc(training$high_income, yPred$y))

