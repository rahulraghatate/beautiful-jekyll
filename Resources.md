### Data Normalization for nueral network

Adjusting the data to a common scale so that the predicted values of the neural network can be accurately compared to that of the actual data. 
Failure to normalize the data will typically result in the prediction value remaining the same across all observations regardless of the input.

Two ways in R:

+ Using the scale function in R
+ Transform the data using user build function for min-max normalization technique

```{r}
#Scaled Normalization
scaleddata<-scale(mydata)

#Max-Min Normalization
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

maxmindf <- as.data.frame(lapply(mydata, normalize))
```


### Box-Cox Transformation for Non-Normal Data

Obtain a lambda value – that is the value at which the data is transformed into a normal shape.

```{r}
#Shapiro- Wilkey Test for normality check
shapiro.test(Income)
library(MASS)
boxcoxreg1<-boxcox(reg1)
summary(boxcoxreg1)

bc<-boxcox(reg1,plotit=T)
title("Lambda and Log-Likelihood")
which.max(bc$y)

(lambda<-bc$x[which.max(bc$y)])

######## Example
bc_transformed<-lm(I(Income^lambda)~Population,data=mydata)
plot(bc_transformed,which=1)
## Perform BP test for heteroskedasticity
bptest(bc_transformed)
```



#Time Series
http://a-little-book-of-r-for-time-series.readthedocs.io/en/latest/src/timeseries.html
