library(tidyverse)
library(skimr)
library(inspectdf)
library(caret)
library(glue)
library(highcharter)
library(h2o)
library(scorecard)#Used for woe bins

setwd("C:/Users/99470/Downloads")
df <- read_csv('Churn_Modelling (1).csv')

df %>% glimpse()

df %>% skim()

df <- df %>% select(-RowNumber,-CustomerId,-Surname)

df$Exited <- df$Exited %>% 
  factor(levels = c(1,0),
         labels = c(1,0))#Target Column should always be in factor data type


#checking inbalance
df$Exited %>% table() %>% prop.table()

# Weight Of Evidence ----

# IV (information values) 
iv <- df %>% 
  iv(y = 'Exited') %>% as_tibble() %>%
  mutate(info_value = round(info_value, 3)) %>%
  arrange(desc(info_value))
#Information values shows the importance of the variables and here we calculate iv according to the 
#target variable

#iv<0.02 not important
#0.02<iv<0.1 Weak predictive power
#0.1<iv<0.3 Medium predictive power
#0.3<iv<0.5 Strong predictive power
#iv>0.5 # Very Strong predictive power
# Exclude not important variables 
ivars <- iv %>% 
  filter(info_value>0.02) %>% 
  select(variable) %>% .[[1]] #Excluding not important variables.

df.iv <- df %>% select(Exited,ivars)#When we calculate iv we remove target column and here
#we merge it again

df.iv %>% dim()
# woe binning-------------------------------------------- 
bins <- df.iv %>% woebin("Exited")# Creating woe bins. 

# converting into woe values

df.woe <- df.iv %>% woebin_ply(bins)

names(df.woe) <- df.woe %>% names() %>% gsub("_woe","",.)

# Multicollinearity --------------------------

# coef_na
target <- 'Exited'
features <- df.woe %>% select(-Exited) %>% names()

f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = df.woe, family = "binomial")#Binomial means for binary classification
glm %>% summary()

coef_na <- attributes(alias(glm)$Complete)$dimnames[[1]]#Choosing linearly dependent columns
features <- features[!features %in% coef_na] #Removing them from the dataset

f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = df.woe, family = "binomial")

# VIF (Variance Inflation Factor) 

while(glm %>% vif() %>% arrange(desc(gvif)) %>% .[1,2] >= 1.5){
  afterVIF <- glm %>% vif() %>% arrange(desc(gvif)) %>% .[-1,variable] #Removing first variable with the highest
  f <- as.formula(paste(target, paste(afterVIF, collapse = " + "), sep = " ~ "))
  glm <- glm(f, data = df.woe, family = "binomial")
}

glm %>% vif() %>% arrange(desc(gvif)) %>% pull(variable) -> features 

df.woe <- df.woe %>% select(target,features)

dt_list <- df.woe %>% 
  split_df("Exited", ratio = 0.8, seed = 123)#This time we used split_df instead of h2o.split
#because we are going to work with woe bins

train_woe <- dt_list$train
test_woe <- dt_list$test 


# Modeling with H2O ----
h2o.init()

train_h2o <- train_woe %>%  as.h2o()
test_h2o <- test_woe %>%  as.h2o()

model <- h2o.glm(
  x = features, y = target, family = "binomial", 
  training_frame = train_h2o, validation_frame = test_h2o,
  nfolds = 10, seed = 123, remove_collinear_columns = T,
  balance_classes = T, lambda = 0, compute_p_values = T)

while(model@model$coefficients_table %>%
      as.data.frame() %>%
      select(names,p_value) %>%
      mutate(p_value = round(p_value,3)) %>%
      .[-1,] %>%
      arrange(desc(p_value)) %>%
      .[1,2] >= 0.05){
  model@model$coefficients_table %>%
    as.data.frame() %>%
    select(names,p_value) %>%
    mutate(p_value = round(p_value,3)) %>%
    filter(!is.nan(p_value)) %>%
    .[-1,] %>%
    arrange(desc(p_value)) %>%
    .[1,1] -> v
  features <- features[features!=v]
  
  train_h2o <- train_woe %>% select(target,features) %>% as.h2o()
  test_h2o <- test_woe %>% select(target,features) %>% as.h2o()
  
  model <- h2o.glm(
    x = features, y = target, family = "binomial", 
    training_frame = train_h2o, validation_frame = test_h2o,
    nfolds = 10, seed = 123, remove_collinear_columns = T,
    balance_classes = T, lambda = 0, compute_p_values = T)
}

model@model$coefficients_table %>%
  as.data.frame() %>%
  select(names,p_value) %>%
  mutate(p_value = round(p_value,3)) %>% arrange(desc(p_value))

model@model$coefficients %>%
  as.data.frame() %>%
  mutate(names = rownames(model@model$coefficients %>% as.data.frame())) %>%
  `colnames<-`(c('coefficients','names')) %>%
  select(names,coefficients)#Pulling the coefficients for each variables 

h2o.varimp(model) %>% as.data.frame() %>% .[.$percentage != 0,] %>%
  select(variable, percentage) %>%
  hchart("pie", hcaes(x = variable, y = percentage)) %>%
  hc_colors(colors = 'orange') %>%
  hc_xAxis(visible=T) %>%
  hc_yAxis(visible=T)#Visualizing the Influence of each variable for prediction


# Evaluation Metrices ----------------------------

# Predictions
pred_test <- model %>% h2o.predict(newdata = test_h2o) %>% 
  as.data.frame() %>% select(p1,predict)#Predicting values
pred_train <- model %>% h2o.predict(newdata = train_h2o) %>% 
  as.data.frame() %>% select(p1,predict)#Predicting values
#high performance
model %>% h2o.performance(newdata = test_h2o)
model %>% h2o.performance(newdata = train_h2o)

#confusionMatrix&accuracy
model %>% h2o.confusionMatrix(test_h2o) %>% as.tibble() %>% select("0","1") %>% .[1:2,] %>% t() %>% fourfoldplot(conf.level=0,color=c("Red","Green"),main=paste("Accuracy=",round(sum(diag(.))/sum(.)*100,1),"%"))
model %>% h2o.confusionMatrix(train_h2o) %>% as.tibble() %>% select("0","1") %>% .[1:2,] %>% t() %>% fourfoldplot(conf.level=0,color=c("Red","Green"),main=paste("Accuracy=",round(sum(diag(.))/sum(.)*100,1),"%"))
#precision
model %>% h2o.performance(newdata = test_h2o) %>% h2o.precision()
model %>% h2o.performance(newdata = train_h2o) %>% h2o.precision()
#recall
model %>% h2o.performance(newdata = test_h2o) %>% h2o.recall()
model %>% h2o.performance(newdata = train_h2o) %>% h2o.recall()
#f1
model %>% h2o.performance(newdata = test_h2o) %>% h2o.F1()
model %>% h2o.performance(newdata = train_h2o) %>% h2o.F1()
#auc/roc/gini
perf_eva(
  pred = pred_test %>% pull(p1),
  label = dt_list$test$Exited %>% as.character() %>% as.numeric(),
  binomial_metric = c("auc","gini"),
  show_plot = "roc")
perf_eva(
  pred = pred_train %>% pull(p1),
  label = dt_list$train$Exited %>% as.character() %>% as.numeric(),
  binomial_metric = c("auc","gini"),
  show_plot = "roc")
model %>%
  h2o.auc(train = T,
          valid = T,
          xval = T) %>%
  as_tibble() %>%
  round(2) %>%
  mutate(data = c('train','test','cross_val')) %>%
  mutate(gini = 2*value-1) %>%
  select(data,auc=value,gini)#Checking Performance by gini score
#ROC curve shows the performance of the model in distinguishing between positive and negative cases by plotting the TPR against the FPR at different threshold settings. In our test case, ROC performs much better, as the TPR increases but FPR does not increase that much. It is much better than the case where all the values are given randomly.
#The AUC is a numerical measure of the performance of the ROC curve, representing the area under the curve. An AUC score of 0.5 indicates that all the values are given randomly, while a higher score indicates a better ability of the model to distinguish between positive and negative cases. As the AUC is 0.81 in our case, it is considered a better model.
