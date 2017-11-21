# Frequency Tables

```
mytable <- table(A,B) # A will be rows, B will be columns 
margin.table(mytable, 1) # A frequencies (summed over B) 
prop.table(mytable) # cell percentages
prop.table(mytable, 1) # row percentages 
```
# 3-Way Frequency Table 
```
mytable <- table(A, B, C) 
ftable(mytable)
summary(mytable) # chi-square test of indepedence
```

#Crosstab
```
library(gmodels)
CrossTable(mydata$myrowvar, mydata$mycolvar)
```
