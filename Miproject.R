---
title: "ProyectoMachineLearning"
author: "Cristian Alejandro Rojas"
date: "05/24/2015"
output: html_document
---

#Abstract 

Human Activity Recognition - HAR - has emerged as a key research area in the last years and is gaining increasing attention by the pervasive computing research community (see picture below, that illustrates the increasing number of publications in HAR with wearable accelerometers), especially for the development of context-aware systems. 
In this work we took a dataset containing measures of devices located in the main parts of the body associated with a position (sitting-down, standing-up, standing, walking, and sitting) with the purpose of generate a model to  determinate the activity of an individual given a set of measures.




#Loading the data

The data has been previously downloaded from coursera project notes and is in our workspaces. First we will load the raw data. In this step we also set the seed for reproducibility purposes.

```{r}
set.seed(999)
testingRaw <- read.csv("pml-testing.csv")
trainingRaw <- read.csv("pml-training.csv")
```

#Cleaning the data
Once the data has been loaded we need to remove the columns containing "NANs" because these values will not be useful for our final model.

```{r}

trainingRawClean <- trainingRaw[, colSums(is.na(trainingRaw)) == 0] 
testingRawClean <- testingRaw[, colSums(is.na(testingRaw)) == 0] 

```

We will look the current columns of our dataset
```{r}
names(trainingRawClean)
```

Columns with names such as window, timestamp aren't so useful for our model so we'll remove them.

```{r}
classe <- trainingRawClean$classe
trainRemove <- grepl("^X|timestamp|window", names(trainingRawClean))
trainingRawClean <- trainingRawClean[, !trainRemove]
trainCleaned <- trainingRawClean[, sapply(trainingRawClean, is.numeric)]
trainCleaned$classe <- classe
testRemove <- grepl("^X|timestamp|window", names(testingRawClean))
testingRawClean <- testingRawClean[, !testRemove]
testCleaned <- testingRawClean[, sapply(testingRawClean, is.numeric)]
```
Our cleaned dataset has 53 variables.

#Slicing the data
Select the data to train our model
```{r}
inTrain <- createDataPartition(trainCleaned$classe, p=0.70, list=FALSE)
trainData <- trainCleaned[inTrain, ]
testData <- trainCleaned[-inTrain, ]
```

#Getting the model 
For close validation purposes we will use a 5-fold. 
```{r}
controlK <- trainControl(method="cv", 5)
```

Since the variable to predict is a factor variable we won't use a linear model.  We will use Random Forest since with this approach the most important variables are selected to use in the model and our outcome is a factor variable.

```{r}
modelRf <- train(classe ~ ., data=trainData, method="rf", trControl=controlK, ntree=250)
```

#Evaluating our model
Then we will use our training set to check the confusion matrix and the accuracy of our model.
```{r}
predictRf <- predict(modelRf, testData)
confusionMatrix(testData$classe, predictRf)
accuracy <- postResample(predictRf, testData$classe)
accuracy
oose <- 1 - as.numeric(confusionMatrix(testData$classe, predictRf)$overall[1])
oose

```
The accuracy for the proposed model is 99.2% and the out-of-sample error is 0.007.


#Predicting with the model
Now with the given test set we'll generate the predictions for the 20 test cases.

```{r}
result <- predict(modelRf, testCleaned[, -length(names(testCleaned))])
result
```

# References
-Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

