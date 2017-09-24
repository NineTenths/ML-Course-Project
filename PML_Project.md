# Practical Machine Learning Project
David P  
9/24/2017  

##Introduction
###Executive Summary
Data collected from human-wearable fitness devices has been collected and analyzed so that the manner in which the participants wearing the devices could be predicted. Three different machine learning algorithms (decision trees, random forest, and general boosted models) were applied to a subset of the training data set. The random forest model has the best accuracy of the three methods, so it was used to predict the classe of test data set. 


###Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

###Data
The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment. 
##Project Setup
The following packages will be needed for the subsequent analysis and set the seed for random number generation used in the project.

```r
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(rattle)
library(e1071)

set.seed(56789)
```

##Getting and Cleaning the Data
###Download & Load the Data
The following commands will download the data from the URLs noted in the _Data_ section and then load the data into the environment

```r
trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

training <- read.csv(url(trainUrl))
testing <- read.csv(url(testUrl))
```

###Cleaning the Data
The first thing that we should do with the data is to clean the data sets. We will perform the cleaning on the _training_ data set, then extract and use the "cleaned" variables from _test_ data set. This will ensure the same number of variables apper in both the training and test data sets. The following command will identify the near zero variance predictors and remove those variables from the data sets.


```r
nzvTraining <- nearZeroVar(training,saveMetrics = T)
training <- training[,nzvTraining$nzv==F]
```

We will now remove any variables that contain missing data as well as the _X_ variable, since this variable is simply a counting varaible.


```r
missingCount <- sapply(training,function(x) sum(is.na(x)))
training <- training[missingCount==0]
training <- training[,-1]
```

Now that we have identified the variables that can be used for modeling in the training data set, we will subset the testing data set using these same variables. 


```r
testing <- testing[,names(testing) %in% names(training)]
```

To develop the machine learning algorithms, we have to divide the training data set into training and testing sub-sets. The following code will do this.


```r
subInd <- createDataPartition(training$classe, p=0.6, list = F)
trainingSub <- training[subInd,]
testingSub <- training[-subInd,]
```

##Prediction
###Cross Validation
Cross validation is done for each model with K = 3. This is set  using the fitControl object as defined in the following code.

```r
fitControl <- trainControl(method='cv', number = 3)
```

###Decision Trees
The following code will fit a prediction model using decision trees and then display a plot of the resulting decision tree.

```r
model_dTree <- train(
  classe ~ ., 
  data=trainingSub,
  trControl=fitControl,
  method='rpart'
)
fancyRpartPlot(model_dTree$finalModel)
```

![](PML_Project_files/figure-html/FitDecisionTree-1.png)<!-- -->

Now that we have trained the decision tree model, we can apply the model to the testing sub-set and obtaing the confusion matrix to determine how well the model can predict the data. The following code will do this. We will also display the overall statistics for the resulting confusion matrix. 


```r
pred_dTree <- predict(model_dTree, newdata=testingSub)
conf_dTree <- confusionMatrix(pred_dTree, testingSub$classe)
conf_dTree$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.5672954      0.4539799      0.5562455      0.5782953      0.2844762 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN
```
We can see from the resulting statistics that the accuracy of the model is 0.5672954. 

###Random Forests
The following code will fit a prediction model using random forests, use the resulting model to predict the results of the testing sub-set, and display the overall statistics of the prediction. 


```r
model_rf <- train(
  classe ~ ., 
  data=trainingSub,
  trControl=fitControl,
  method='rf',
  ntree=100
)
pred_rf <- predict(model_rf, newdata=testingSub)
conf_rf <- confusionMatrix(pred_rf, testingSub$classe)
conf_rf$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.9988529      0.9985491      0.9978236      0.9994754      0.2844762 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN
```
We can see from the resulting statistcs that the accuracy of the model is 0.9988529, which is better than the accuracy of the decision tree model. 

###General Boosted Model
The following code will fit a prediction model using general boosted models, use the resulting model to predict the results of the testing sub-set, and display the overall statistics of the prediction. 

```r
model_gbm <- train(
  classe ~ ., 
  data=trainingSub,
  trControl=fitControl,
  method='gbm',
  verbose=F
)
pred_gbm <- predict(model_gbm, newdata=testingSub)
conf_gbm <- confusionMatrix(pred_gbm, testingSub$classe)
conf_gbm$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.9970686      0.9962923      0.9956046      0.9981408      0.2844762 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN
```
We can see from the resulting statistcs that the accuracy of the model is 0.9970686.

###Fit Summary
The following code will produce a table to compare the accuracy of the three methods. 

```r
AccuracyResults <- data.frame(
  Model = c('D TREE', 'GBM', 'RF'),
  Accuracy = rbind(conf_dTree$overall[1], conf_gbm$overall[1], conf_rf$overall[1])
)
print(AccuracyResults)
```

```
##    Model  Accuracy
## 1 D TREE 0.5672954
## 2    GBM 0.9970686
## 3     RF 0.9988529
```
We can see from this table that the random forest method results in the highest accuracy. Therefore, we will use this model to classify the test data set. 

##Predicting the Test Set
We can now use the random forest model previously described to predict the _classe_ of the testing data. The following code will make the prediction and show the results.

```r
predTestData <- predict(model_rf, newdata=testing)
predTestData
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

