library(tidyverse);library(GGally);library(lubridate);library(xgboostExplainer)
library(xgboost);library(pROC);library(plotROC);library(DALEX)
library(modelplotr)

df1 = read.csv("model.csv")

df1 = df1 %>% filter(year(CREATETIME)>2015)


df1$CREATETIME = as.numeric(month(df1$CREATETIME))


####  X:take CASE=1 and remove ACTIONID,CASE,DOHB_Y,DOHBY_Y(high correlation)
X = filter(df1,CASE==1) %>% select(everything(),-c(ACTIONID,CASE,DOHB_Y,DOHBY_Y) ) %>% as.matrix()
y = filter(df1,CASE==1) %>% select(DOHBY_Y) %>% as.matrix() %>% as.numeric()
set.seed(123)
index = sample(1:nrow(X),floor(0.8*nrow(X)),replace = F)

xgb.train.data = xgb.DMatrix(data = X[index,], label = y[index])
xgb.test.data  = xgb.DMatrix(data = X[-index,])


#### Grid search
#for(eta_ in c(0.01,0.02,0.03,0.04)){
#  for(max_depth_ in c(1,2,3,4,5,6)){
#    for(subsample_ in c(0.6,0.8,1)){
#      for (colsample_bytree_ in c(0.6,0.8,1)) {
#        xgb_params = list(subsample=subsample_,max_depth=max_depth_,eta=eta_,
#                          colsample_bytree =colsample_bytree_,
#                          eval_metric = "error",objective = "binary:logistic")
#        cv_model1 = xgb.cv(params=xgb_params,  data= as.matrix(X),
#                           label =  y ,nrounds = 100,nfold = 8,verbose = F)
#        print(paste('eta:',eta_,',max_depth:',max_depth_,',subsample:',subsample_,
#                    ',colsample_bytree:',colsample_bytree_,
#                    ',train_error:',round(cv_model1$evaluation_log$train_error_mean[100],4),
#                    ',test_error:',round(cv_model1$evaluation_log$test_error_mean[100],4)))
#}}}}




#[1] "eta: 0.03 ,max_depth: 4 ,subsample: 0.8 ,
#colsample_bytree: 0.6 ,train_error: 0.0207 ,test_error: 0.0222"



#### CV model
param = list(subsample=0.8,max_depth=4,eta=0.03,
             colsample_bytree =0.6,
             eval_metric = "error",objective = "binary:logistic")
xgboost.cv = xgb.cv(param=param, data = xgb.train.data, nfold = 8, print_every_n = 100,
                    nrounds = 200, early_stopping_rounds = 100, metrics='error')
#### CV performance
plot(xgboost.cv$evaluation_log$train_error_mean,col=1,
     ylim = c(min(xgboost.cv$evaluation_log$train_error_mean)-0.0001,
              max(xgboost.cv$evaluation_log$test_error_mean)+0.0001),
     main="overfitting check",type="l",ylab='error')
points(xgboost.cv$evaluation_log$test_error_mean,col=2,type="l")
legend("topright", c("train","test"),col=c(1,2),lty=2,cex=0.7)

####  XGBoost model
best_iteration = xgboost.cv$best_iteration
xgb.model <- xgboost(param =param,  data = xgb.train.data, 
                     nrounds=best_iteration,verbose = F)
xgb.preds = predict(xgb.model, xgb.test.data)
xgb.roc_obj <- roc(y[-index],xgb.preds)

#### Confused matrix(test)
table('true'=y[-index],'pred'=as.numeric(xgb.preds>0.5) )

#### ROC curve
K = data.frame('true'=y[-index],'prob'=xgb.preds)
basicplot = ggplot(K,aes(d = true, m = prob)) + geom_roc() 
basicplot + style_roc(theme = theme_grey) +
  theme(axis.text = element_text(colour = "blue")) +
  ggtitle("ROC curve(xgboost_test)") +
  geom_abline(slope=1,col=4) +
  annotate("text", x = .75, y = .29, 
           label = paste("AUC =",round(calc_auc(basicplot)$AUC, 4) )) +
  annotate("text", x = .775, y = .25, 
           label = paste("Predict error =",round(mean(as.numeric(K$prob>0.5)==K$true), 4)  )) +
  scale_x_continuous("Specificity", breaks = seq(0, 1, by = .1)) +
  scale_y_continuous("Sensitivity", breaks = seq(0, 1, by = .1)) 

#### Xgb importance
col_names = attr(xgb.train.data, ".Dimnames")[[2]]
imp = xgb.importance(col_names, xgb.model)
xgb.plot.importance(imp[1:10,],main='xgboost feature important')



QW = as.tibble(data.frame( 'index'=setdiff(1:length(y),index),
                           'new_index' = 1:length(y[-index]),
                           'ACTIONID' = (filter(df1,CASE==1))[-index,1],
                           'SEX' = (filter(df1,CASE==1))[-index,69],
                           'true'=y[-index],
                           'pred_prob'=xgb.preds,
                           'pred'=as.numeric(xgb.preds>0.5) ) )


#### THE XGBoost Explainer
explainer = buildExplainer(xgb.model,xgb.train.data, type="binary", 
                           base_score = 0.5, trees_idx = NULL)
pred.breakdown = explainPredictions(xgb.model, explainer, xgb.test.data)
cat('Breakdown Complete','\n')
weights = rowSums(pred.breakdown)
pred.xgb = 1/(1+exp(-weights))



showWaterfall_ = function (xgb.model, explainer, DMatrix, data.matrix, idx, type = "binary", 
          threshold = 1e-02, limits = c(NA, NA)) {
  breakdown = explainPredictions(xgb.model, explainer, slice(DMatrix,  as.integer(idx)))
  logit = function(x) {
    return(log(x/(1 - x)))
  }
  
  weight = rowSums(breakdown)
  if (type == "regression") {
    pred = weight
  }
  else {
    pred = 1/(1 + exp(-weight))
  }
  breakdown_summary = as.matrix(breakdown)[1, ]
  data_for_label = data.matrix[idx, ]
  i = order(abs(breakdown_summary), decreasing = TRUE)
  breakdown_summary = breakdown_summary[i]
  data_for_label = data_for_label[i]
  intercept = breakdown_summary[names(breakdown_summary) == 
                                  "intercept"]
  data_for_label = data_for_label[names(breakdown_summary) != 
                                    "intercept"]
  breakdown_summary = breakdown_summary[names(breakdown_summary) != 
                                          "intercept"]
  i_other = which(abs(breakdown_summary) < threshold)
  other_impact = 0
  if (length(i_other > 0)) {
    other_impact = sum(breakdown_summary[i_other])
    names(other_impact) = "other"
    breakdown_summary = breakdown_summary[-i_other]
    data_for_label = data_for_label[-i_other]
  }
  if (abs(other_impact) > 0) {
    breakdown_summary = c(intercept, breakdown_summary, other_impact)
    data_for_label = c("", data_for_label, "")
    labels = paste0(names(breakdown_summary), " = ", data_for_label)
    labels[1] = "intercept"
    labels[length(labels)] = "other"
  }
  else {
    breakdown_summary = c(intercept, breakdown_summary)
    data_for_label = c("", data_for_label)
    labels = paste0(names(breakdown_summary), " = ", data_for_label)
    labels[1] = "intercept"
  }
  if (!is.null(getinfo(DMatrix, "label"))) {
    cat("\nActual: ", getinfo(slice(DMatrix, as.integer(idx)), 
                              "label"))
  }
  cat("\nPrediction: ", pred)
  cat("\nWeight: ", weight)
  cat("\nBreakdown")
  cat("\n")
  print(breakdown_summary)
  if (type == "regression") {
    waterfalls::waterfall(values = breakdown_summary, 
                          rect_text_labels = round(breakdown_summary,  2), 
                          labels = labels, total_rect_text = round(weight,2), 
                          calc_total = TRUE, total_axis_text = "Prediction") + 
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  }
  else {
    inverse_logit_trans <- scales::trans_new("inverse logit", 
                                             transform = plogis, inverse = qlogis)
    inverse_logit_labels = function(x) {
      return(1/(1 + exp(-x)))
    }
    logit = function(x) {
      return(log(x/(1 - x)))
    }
    ybreaks <- logit(seq(2, 98, 2)/100)
    waterfalls::waterfall(values = breakdown_summary,
                          rect_text_labels = round(breakdown_summary,  2), 
                          labels = labels, total_rect_text = round(weight,  2), 
                          calc_total = TRUE, total_axis_text = "Prediction") + 
      scale_y_continuous(labels = inverse_logit_labels, breaks = ybreaks, limits = limits) + 
      theme(axis.text.x = element_text(angle = 45,  hjust = 1)) + 
      labs(title= paste("XGBoost break down for case:",as.matrix(QW[which(QW$new_index==idx),3])))
  }
}

# pred_prob = 0.00714 (true=0,pred=0)
(QW %>% arrange(pred_prob))[1:20,]
idx_to_get = 985
showWaterfall_(xgb.model, explainer, xgb.test.data, X[-index,] ,idx_to_get, 
               type = "binary")
idx_to_get = 126
showWaterfall_(xgb.model, explainer, xgb.test.data, X[-index,] ,idx_to_get, 
               type = "binary")

# pred_prob = 0.955 (true=1,pred=1)
(QW %>% arrange(desc(pred_prob)))[1:20,]
idx_to_get = c(2658)
showWaterfall_(xgb.model, explainer, xgb.test.data, X[-index,] ,idx_to_get, 
               type = "binary")

idx_to_get = c(2719)
showWaterfall_(xgb.model, explainer, xgb.test.data, X[-index,] ,idx_to_get, 
               type = "binary")

#### DALEX 
model_martix_train <- model.matrix(y ~ . -1 , data.frame(y=y,X) )


predict_logit <- function(model, x) {
  raw_x <- predict(model, x)
  exp(raw_x)/(1 + exp(raw_x))
}
logit <- function(x) exp(x)/(1+exp(x))

explainer_xgb <- explain(xgb.model, 
                         data = model_martix_train, 
                         y = y, 
                         predict_function = predict_logit,
                         link = logit,
                         label = "xgboost")
explainer_xgb

#### Single variable(AGE)
sv_xgb_satisfaction_level  <- variable_response(explainer_xgb, 
                                                variable = "AGE",
                                                type = "ale")
p1 = sv_xgb_satisfaction_level %>% ggplot(aes(x=x,y=y,col=label)) + 
  geom_point() + labs(x="AGE",y="yhat",title='Variable response',col="model")+geom_line()
p1

ggplotly(p1)


#### Single variable(CREATETIME)
sv_xgb_satisfaction_level  <- variable_response(explainer_xgb, 
                                                variable = "CREATETIME",
                                                type = "pdp")
p2 = sv_xgb_satisfaction_level %>% ggplot(aes(x=x,y=y,col=label)) + 
      geom_point() + labs(y="yhat",title='Variable response',col="model")+
        scale_x_continuous("CREATETIME",breaks = seq(1, 12))+geom_line()
plot(sv_xgb_satisfaction_level)

i=1
variable_response(explainer_xgb, 
                  variable = "MERRIAGEFLG_Y",
                  type = "pdp") %>% plot


i=i+1

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

filter(df1,CASE==1)%>% as.tibble() %>% select(CREATETIME,AGE) %>% 
  ggplot(aes(x=AGE,fill=as.factor(CREATETIME) ,alpha=0.01))+geom_density()+
  labs(fill="CREATETIME",title="CASE=1,CREATETIME vs. DOHBY")


filter(df1,CASE==1)%>% as.tibble() %>% select(CREATETIME,DOHBY_Y) %>% 
  ggplot(aes(x=CREATETIME,fill=as.factor(DOHBY_Y) ,alpha=0.01))+geom_bar()+ 
  labs(fill="DOHBY_Y",title="CASE=1,CREATETIME vs. DOHBY")+
  scale_x_continuous("CREATETIME",breaks = seq(1,12))


plot(variable_response(explainer_xgb, variable = "MAIMED_X",type = "pdp"))

library(randomForest)

#tuning parameter(RF)
data = data.frame(y,X)
fit4.1 = randomForest(y~.,data=data[index,],ntree=1000)
plot(fit4.1)
tuneRF(data[train,-1], data[train,1])
#--------------------------
rc1 = data.frame(B = y[index] , A=X[index,]  )

fit4 = randomForest(B~.,data=rc1,ntree=400, mytry=12)

num4 = predict(fit4,data.frame(A=X[-index,]) )

table("true"=y[-index],"pred"=as.numeric(num4>0.5))
varImpPlot(fit4,main = "RF Importance Vaiables")



################################################################################



dtrain = xgb.DMatrix(data = X, label = y)

#### CV model
xgboost.cv2 = xgb.cv(param=param, data = dtrain, nfold = 8, print_every_n = 100,
                     nrounds = 200, early_stopping_rounds = 100, metrics='error')
#### CV performance
plot(xgboost.cv2$evaluation_log$train_error_mean,col=1,
     ylim = c(min(xgboost.cv2$evaluation_log$train_error_mean)-0.0001,
              max(xgboost.cv2$evaluation_log$test_error_mean)+0.0001),
     main="overfitting check",type="l",ylab='error')
points(xgboost.cv2$evaluation_log$test_error_mean,col=2,type="l")
legend("topright", c("train","test"),col=c(1,2),lty=2,cex=0.7)

####  XGBoost model
best_iteration2 = xgboost.cv2$best_iteration
xgb.model_ <- xgboost(param =param,  data = dtrain, 
                      nrounds=best_iteration2,verbose = F)
xgb.preds_ = predict(xgb.model_, dtrain)

#### Confused matrix(train)
table('true'=y,'pred'=as.numeric(xgb.preds_>0.5) )

AA = data.frame("ID"=filter(df1,CASE==1)%>%select(ACTIONID) ,
                "y"=y,"pred_prob"=xgb.preds_,
                "pred"=as.numeric(xgb.preds_>0.5),
                "index"=1:length(y))


qqq = which((AA$y==0) &(AA$pred==1))



high_risk = left_join(df1[qqq,c("ACTIONID","SEXID_M",imp[1:10,1]$Feature)],AA[qqq,])
high_risk %>% arrange(desc(pred_prob)) 


explainer_ = buildExplainer(xgb.model_,dtrain, type="binary", 
                           base_score = 0.5, trees_idx = NULL)


idx_to_get = high_risk$index[1]
showWaterfall(xgb.model_, explainer_, dtrain, X,idx_to_get, type = "binary")



