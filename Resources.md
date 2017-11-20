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
