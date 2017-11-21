# Helpful R_packages


+ library(class) | knn algorithm
+ library(gmodels) | CrossTable
+ require(maps) | maps
+ require(mapdata) | maps_dataset
+ library(ggplot2) | plot
+ library(ggrepel) | 
+ library("TTR")   | decompose (observed, trend, seasonal and random portions)  
+ library(vcd) | ```assocstats(mytable), kappa(mytable)```
+ library(DAAG) | ```cv.lm(df=mydata, fit, m=3) # 3 fold cross-validation```
+ library(bootstrap)| ```results<-crossval(X,y,theta.fit,theta.predict,ngroup=10)```

+ library(MASS)
  * ```stepAIC(fit, direction="both") # forward , backward``` for Variable Selection
  * ```rlm()``` for Robust Regression
  * ```lda(G ~ x1 + x2 + x3, data=mydata, na.action="na.omit", CV=TRUE)``` for Linear Discriminant Analysis
  * ```qda(G ~ x1 + x2 + x3 + x4, data=na.omit(mydata),  prior=c(1,1,1)/3))``` for Quadratic DA
+ library(leaps)
  * ```leaps<-regsubsets(y~x1+x2+x3+x4,data=mydata,nbest=10)``` for all-subsets regression
+ library(relaimpo)
  * ```calc.relimp(fit,type=c("lmg","last","first","pratt"),rela=TRUE)``` for Relative importance of variables
+ library(gvlma)
  * ```gvmodel <- gvlma(fit)      summary(gvmodel)```  global validation of linear model assumptions

+ library(klaR)
  * ```partimat(G~x1+x2+x3,data=mydata,method="lda")``` display the lda,qda for 2 variables results
+ library(forecast)
  * ```seasonplot(myts)```
  * ```accuracy(fit)```
  * ```forecast(fit,#no) ``` predict future values
  * ```fit<- ets(ts)```  # Automated forecasting using an exponential model
  * ```fit<- auto.arima(ts)``` # Automated forecasting using an ARIMA model
+ library(sem) #Confirmatory Factor Analysis
+ library(ca) #Correspondence Analysis
