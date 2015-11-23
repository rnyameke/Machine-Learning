# Course Project
Rose Nyameke  
November 21, 2015  
I built my model with considerations for speed and accuracy. My first thought was
to try different models (specifically, random forests and multiple boosting methods),
and to pick the best model based on accuracy. However, due to computation limitations,
I built one model with Generalized Boosted Regression Modeling with 10-fold cross-
validation, and because I was satisfied with the accuracy, I did not build any other
models. When I used my model to predict the values of "classe" on the test data
set, I was able to obtain the correct values for all the cases, confirming the
accuracy of the model.

##Data Processing

```r
data <- read.csv("pml-training.csv")
library(plyr)
library(dplyr)
library(caret)
library(gbm)
```

I took a simple approach to cleaning the data, and did not use any of the pre-
processing functions in the caret package. Instead, I first removed any columns
that were not numeric (except the response variable), because I noticed that these
were missing many values. I then removed columns that had NAs, after which I 
removed columns that were highly correlated with other columns (threshold of 0.75).
Lastly, I removed the timestamp variables (as they were not, strictly speaking, device
measurements), and this left me with 31 predictor variables, with which I built my model.


```r
#keeping only the numeric variables
data_wo_factors <- data[sapply(data, is.numeric)]

#re-introducing the classe variable
data_wo_factors <- cbind(data$classe, data_wo_factors)

#removing the row numbering variable
data_wo_factors <- data_wo_factors[, -2]

#removing columns with nas
data_wo_nas <- data_wo_factors[, colSums(is.na(data_wo_factors)) == 0]

#finding highly correlated predictor variables
correlationMatrix <- cor(data_wo_nas[,-1])
highcorr <- findCorrelation(correlationMatrix, cutoff = 0.75)

#removing the highly correlated columns
data_wo_corr <- data_wo_nas[,-highcorr]

#removing the timestamp columns
data_wo_time <- data_wo_corr[, -(2:3)]

#renaming the classe column
names(data_wo_time)[1] <- "classe"

clean_data <- data_wo_time
```

I built one model, using 10-fold cross-validation and generalized boosted regression
modeling.

```r
#building model using gbm and cross-fold validation (for speed reasons)
model <- train(classe ~ ., method = "gbm", trControl = trainControl(method = "cv",
               number = 10), data = clean_data, verbose = TRUE)
```

I then selected the same variables in the test data set that were used to train my
model:

```r
#predict with model
test_data <- read.csv("pml-testing.csv")

#keeping only the variables from the train data model
keepvariables <- names(clean_data)

#removing the classe variable
keepvariables <- keepvariables[-1]

#subsetting the test data
clean_test_data <- subset(test_data, select = keepvariables)
```

After cleaning the test data set, I used the model to predict the values of the
20 test cases, which were then written to 20 text files.

```r
answers <- predict(model, clean_test_data)

pml_write_files <- function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(answers)
```

## Results
Below is the model that was used in prediction, which yielded correct predictions
for all 20 test cases:

```r
print(model)
```

```
## Stochastic Gradient Boosting 
## 
## 19622 samples
##    31 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 17658, 17660, 17660, 17662, 17659, 17659, ... 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa      Accuracy SD
##   1                   50      0.7014537  0.6198966  0.012214439
##   1                  100      0.7785641  0.7193546  0.008181493
##   1                  150      0.8166858  0.7678726  0.007547475
##   2                   50      0.8330967  0.7887322  0.007821469
##   2                  100      0.8858930  0.8556406  0.004875729
##   2                  150      0.9108143  0.8871566  0.005697480
##   3                   50      0.8816643  0.8502867  0.003775216
##   3                  100      0.9216201  0.9008239  0.004741262
##   3                  150      0.9447563  0.9301056  0.003217309
##   Kappa SD   
##   0.015762498
##   0.010527711
##   0.009614176
##   0.009930577
##   0.006166995
##   0.007188689
##   0.004801633
##   0.005991556
##   0.004064709
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.
```
As the model incorporates 10-fold cross-validation into the train function that
was used to build it, the out of sample (OOS) error is:  
**OOS = 1 - accuracy = 1-0.95 = 0.05**
