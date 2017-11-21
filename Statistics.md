Tests of Independence
1. Chi-Square Test | ```chisq.test(table)```
2. Fisher Exact Test | ```fisher.test(x)```
3. Mantel-Haenszel test -perform a Cochran-Mantel-Haenszel chi-squared test | ```mantelhaen.test(x)```
4. Loglinear Models 
  + Mutual Independence | ```mytable <- xtabs(~A+B+C, data=mydata) \ loglm(~A+B+C, mytable)```
  + Partial Independence | ```loglin(~A+B+C+B*C, mytable)```
  + Conditional Independence | ```loglm(~A+B+C+A*C+B*C, mytable)```
  + No Three-Way Interaction | ```loglm(~A+B+C+A*B+A*C+B*C, mytable)```

Correlations
```
cor(x, use=, method= )#pearson, spearman or kendall.
cor(x, y)
library(Hmisc)
rcorr(x, type="pearson")

# polychoric correlation
# x is a contingency table of counts
library(polycor)
polychor(x) 


# heterogeneous correlations in one matrix 

# pearson (numeric-numeric), 
# polyserial (numeric-ordinal), 
# and polychoric (ordinal-ordinal)
# x is a data frame with ordered factors 
# and numeric variables
library(polycor)
hetcor(x) 
```

t-tests
```
##### var.equal = TRUE if equal variances


# independent 2-group t-test
t.test(y~x) # where y is numeric and x is a binary factor
t.test(y1,y2) # where y1 and y2 are numeric

# paired t-test
t.test(y1,y2,paired=TRUE) # where y1 & y2 are numeric

# one sample t-test
t.test(y,mu=3) # Ho: mu=3 #alternative="less" or alternative="greater"
```

Non-parametric Tests of Group Differences
```
# independent 2-group Mann-Whitney U Test 
wilcox.test(y~A) # where y is numeric and A is A binary factor
# independent 2-group Mann-Whitney U Test
wilcox.test(y,x) # where y and x are numeric
# dependent 2-group Wilcoxon Signed Rank Test 
wilcox.test(y1,y2,paired=TRUE) # where y1 and y2 are numeric
# Kruskal Wallis Test One Way Anova by Ranks 
kruskal.test(y~A) # where y1 is numeric and A is a factor
```

Diagnostic Plots
```
#provide checks for heteroscedasticity, normality, and influential observerations.
layout(matrix(c(1,2,3,4),2,2)) # optional 4 graphs/page 
plot(fit)
```

*Variable Selection*

```
# Stepwise Regression
library(MASS)
fit <- lm(y~x1+x2+x3,data=mydata)
step <- stepAIC(fit, direction="both")
step$anova # display results
```


**Relative Importance**
  + link : https://www.statmethods.net/stats/rdiagnostics.html

**ANOVA/MANOVA (more than one dependent variable)**
  + link : https://www.statmethods.net/stats/anova.html


** Time Series**
```
ts<-ts(vector,start=,end=,frequency=)

#subset the time-series
window(ts, start=,end=)

#plot
plot(ts)

#Seasonal
fit<- stl(ts, s.window="period")
plot(fit)
monthplots(ts)
seasonplot(myts)
```

**Exponential Models**
```
# simple exponential - models level
fit <- HoltWinters(myts, beta=FALSE, gamma=FALSE)
# double exponential - models level and trend
fit <- HoltWinters(myts, gamma=FALSE)
# triple exponential - models level, trend, and seasonal components
fit <- HoltWinters(myts)
```

**ARIMA**


**Auto Forescasting**


**Factor Analysis**
  + PCA
    ```
    # Pricipal Components Analysis
    # entering raw data and extracting PCs 
    # from the correlation matrix 
    fit <- princomp(mydata, cor=TRUE) #provide unrotated PCA
    summary(fit) # print variance accounted for 
    loadings(fit) # pc loadings 
    plot(fit,type="lines") # scree plot 
    fit$scores # the principal components
    biplot(fit)
    ```
    
  + Exploratory Factor Analysis
    ```
    # Maximum Likelihood Factor Analysis
    # entering raw data and extracting 3 factors, 
    # with varimax rotation 
    fit <- factanal(mydata, 3, rotation="varimax")
    print(fit, digits=2, cutoff=.3, sort=TRUE)
    # plot factor 1 by factor 2 
    load <- fit$loadings[,1:2] 
    plot(load,type="n") # set up plot 
    text(load,labels=names(mydata),cex=.7) # add variable names
    ```
    
    
