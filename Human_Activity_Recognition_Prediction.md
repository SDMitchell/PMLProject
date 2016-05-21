# Human Activity Recognition Prediction
SDMitchell  
May 20, 2016  
  



  


## Synopsis
We are performing an analysis of Human Activity data collected while participants were exercising both correctly and incorrectly. Using the data collected, we are attempting to predict if the subjects were indeed performing the activities correctly.

## Exploration
The data comes pre-divided into training and test sets, but since the training set is large enough, we are going to further divide the training data into a training and calibration set and leave the test set untouched until the final evaluation.
  
Simply looking at the training data, we can see that there are rows of summary statistics that look like samples but are in actuality completely dependent upon other samples. They seem to have been denoted by setting the *new_window* feature to the value "yes" so we can safely leave out these rows plus the *new_window* feature itself.
  
The summary statistics are also captured in their own set of features which we can safely ignore; they seem to be any feature starting with any of the following prefixes:
  
*"max_"*  
*"min_"*  
*"amplitude_"*  
*"var_"*  
*"stddev_"*  
*"kurtosis_"*  
*"skewness_"*  
  
The *classe* column denotes the value we are actually trying to predict, so we'll be using it in our models as the response. It seems to consist of a single capital letter in the range of A through E.
  
Several other statistics are of limited value to prediction algorithms (user name, trial number and timestamps for example) and they are also ignored in the analysis. It would appear that all other features are potentially usable after this cursory exploration.
  
With so much data and several features, we are going to investigate two methods for analysing the data: Support Vector Machines (which were used in the fourth quiz) and Random Forest (which were in a quiz and the lectures). The reason for selecting two methods to use is both for extra validation and because of pure curiousity. There is a quiz-based test set upon which both methods will not perform perfectly; there is a personal interest in which predictions were wrong for each set (i.e. would they misclassify the same samples or different ones).
***
  
## Data Preparation
To save a bit of time, we are going to construct our test set and save it into an R session with only the relevant samples and features. We also want to divide this set up into training and calibration sets; note that we set the random number generation seed at the beginning of the report in a hidden section. The code for *acquireData* is also in a hidden section and can be seen in the committed code rather than in this report.

```r
trainingdata <- acquireData("pml-training")
trainingdata$calibration <- runif(1:nrow(trainingdata)) > 0.70
realTrainingData <- subset(trainingdata[!trainingdata$calibration,], select=-c(calibration))
realCalibrationData <- subset(trainingdata[trainingdata$calibration,], select=-c(calibration))
realTestingData <- acquireData("pml-testing")
```

### Generating fold information for cross validation
Since we are going to do k-fold cross validation, we need a way to partition the data. This is simply an easy way; assign each row a random number between 1 and k, then we can later use the for loop index to select the test data for that fold.

```r
# Generate a vector to be used for k-fold cross-validation. It simply consists of nrow(realTrainingData) integers from 1-10
# that can be used in a simple for loop
foldNumber <- as.integer(runif(1:nrow(realTrainingData)) * numberOfFoldsForKFold) + 1
inSampAccuracy <- replicate(numberOfFoldsForKFold, 0)
outSampAccuracy <- replicate(numberOfFoldsForKFold, 0)
```

## Initial Investigation
First let us check the counts of the various levels of the outcome to make sure that there isn't any sort of obvious bias (e.g. rare occurences which may not be properly represented when we sub-sample):


```r
count(realTrainingData$classe)[2]/dim(realTrainingData)[1]*100
```

```
##       freq
## 1 28.55017
## 2 19.22190
## 3 17.28037
## 4 16.62575
## 5 18.32180
```

Where these counts are percent of the overall total, they all look relatively well balanced.
  
One interesting observation about the data is that the outcome variable (*classe*) seems to be ordering the entire data set (i.e. all of the "A" values come first, then "B" and so on). This is not a real problem, but it does mean that we cannot be lazy when selecting our fold partitions (they will have to be random and uniformly distributed, not simple slices).
  
### Clustering
Perhaps the data will fall into convenient clusters using kmeans; it doesn't hurt to check given that it is a relatively cheap process:

```r
trainingScaled <- scale(subset(realTrainingData, select=-classe))
km <- kmeans(trainingScaled, 5)
#summary(km)
clusplot(trainingScaled, km$cluster, color=TRUE, shade=FALSE,labels=4, lines=0, main="PCA Cluster Plot")
```

<img src="Human_Activity_Recognition_Prediction_files/figure-html/A Quick Clustering-1.png" style="display: block; margin: auto;" />

Given the PCA components produced, this doesn't look like a winning algorithm on its own. The two main components only explain 30-ish percent of the variability in the data and there is some massive overlap in three of the clusters. This calls for more extensive analysis.
  
## Analysis

### Support vector machine
The support vector machine algorithm performs a one-vs-the-rest analysis when it comes to classifying non-binary classifiers. We were able to use all of the training data (instead of a subset) because the algorithm finished in a reasonable amount of time and it seemed to perform better with more samples.
  
The *cross* parameter in the SVM algorithm is just for model parameter optimization/selection, not actual cross validation of models, so we have to perform our own cross validation. A comparison of the prediction against the validation data set versus the actual correct response in the validation set was used to calculate our out-of-sample error rate for comparison to the in-sample rate using the training data.

```r
dataWithFoldingInfo <- cbind(realTrainingData, foldNumber)

for(currentFold in 1:numberOfFoldsForKFold)
{
	#print(paste("Starting fold", currentFold))
	currentFoldTest <- dataWithFoldingInfo[dataWithFoldingInfo$foldNumber==currentFold, ]
	currentFoldTrain <- dataWithFoldingInfo[dataWithFoldingInfo$foldNumber!=currentFold, ]
	svmFit <- svm(classe~., data=subset(currentFoldTrain, select=-c(foldNumber)))
	svmPredTrain <- predict(svmFit, newdata=subset(currentFoldTrain, select=-c(foldNumber)))
	svmPredTest <- predict(svmFit, newdata=subset(currentFoldTest, select=-c(foldNumber)))
	correct <- count(currentFoldTrain$classe == svmPredTrain)
	inSampAccuracy[currentFold] <- as.numeric(correct[correct$x,][2] / length(svmPredTrain))
	correct <- count(currentFoldTest$classe == svmPredTest)
	outSampAccuracy[currentFold] <- as.numeric(correct[correct$x,][2] / length(svmPredTest))
}
accInSampleSVM <- mean(unlist(inSampAccuracy))
accOutSampleSVM <- mean(unlist(outSampAccuracy))
```
The SVM seems to perform rather well, with a mean out-of-sample success rate of 93.64% as compared to the in-sample success rate of 94.43%.
  
For fun, let's take a look at its success rate versus the actual test data; we'll see what it would get if it were used to blindly submit the answers to the quiz:

```r
svmFitAll <- svm(classe~., data=realTrainingData)
svmPredQuiz <- predict(svmFitAll, newdata=realTestingData)
correct <- count(quizAnswers == svmPredQuiz)
quizSuccessSVM <- as.numeric(correct[correct$x,][2] / length(svmPredQuiz))
```

Well, 90% isn't too shabby; certainly much better than guessing!

###Random Forest
Random forest should perform very well given the type of data we have, but it has the disadvantage of wanting to melt your computer if you don't constrain it in some way. There was an attempt made to use all of the training data, but it resulted in an hour-long wait and complete usage of 10 of 12 CPU cores and nearly all of 32GB of RAM (and it likely still ended up deep in swap). To make matters worse, the model it produced was not very good. Running with one fifth of the training set data (randomly selected) generated a very good model and took a lot less time; running k-fold cross validation to get some error rates was much more reasonable with the resouces at hand.


```r
computationCluster <- makeCluster(8)
registerDoParallel(computationCluster)
dataWithFoldingInfo <- cbind(realTrainingData, foldNumber)[runif(1:nrow(realTrainingData)) > 0.80,]

for(currentFold in 1:numberOfFoldsForKFold)
{
	#print(paste("Starting fold", currentFold))
	currentFoldTest <- dataWithFoldingInfo[dataWithFoldingInfo$foldNumber==currentFold, ]
	currentFoldTrain <- dataWithFoldingInfo[dataWithFoldingInfo$foldNumber!=currentFold, ]
	rfFit <- train(classe ~ ., data=subset(currentFoldTrain, select=-c(foldNumber)), method="rf", importance=TRUE)
	rfPredTrain <- predict(rfFit, newdata=subset(currentFoldTrain, select=-c(foldNumber)))
	rfPredTest <- predict(rfFit, newdata=subset(currentFoldTest, select=-c(foldNumber)))
	correct <- count(currentFoldTrain$classe == rfPredTrain)
	inSampAccuracy[currentFold] <- as.numeric(correct[correct$x,][2] / length(rfPredTrain))
	correct <- count(currentFoldTest$classe == rfPredTest)
	outSampAccuracy[currentFold] <- as.numeric(correct[correct$x,][2] / length(rfPredTest))
}

stopCluster(computationCluster)

accInSampleRF <- mean(unlist(inSampAccuracy))
accOutSampleRF <- mean(unlist(outSampAccuracy))
```
The random forest seems to perform better than the SVM, with a mean out-of-sample success rate of 95.99% as compared to the in-sample success rate of 100%.
  
For an equal amount of fun as last time, let's take a look at its success rate versus the actual test data; we'll see what it would get if it were used to blindly submit the answers to the quiz:

```r
computationCluster <- makeCluster(8)
registerDoParallel(computationCluster)

rfData <- realTrainingData[runif(1:nrow(realTrainingData)) > 0.80,]
rfFitAll <- train(classe~., data=rfData, method="rf", importance=TRUE)
rfPredQuiz <- predict(rfFitAll, newdata=realTestingData)
correct <- count(quizAnswers == rfPredQuiz)
quizSuccessRF <- as.numeric(correct[correct$x,][2] / length(rfPredQuiz))

stopCluster(computationCluster)
```

Well, 90% is pretty decent here as well.
   
If this report was not constrained by size and time, the investigation would likely continue exploring using the random forest algorithm. We computed the variable importance during the random forest calculation; it appears that we could likely focus our (future) analysis on a handful of features instead of the entire set:
  

```r
varImpPlot(rfFitAll$finalModel, main="Random Forest Variable Importance")
```

<img src="Human_Activity_Recognition_Prediction_files/figure-html/Variable Importance-1.png" style="display: block; margin: auto;" />

Choosing the top ten features in terms of importance would make for some interesting analysis versus using all of them.

## Conclusions
It appears that random forest wins a battle between the two methods for this type of data, although support vector machines certainly gave an admirable result. Random forest required much more resources to do its computations (so much so that *doParallel* was installed to make it somewhat reasonable) and SVM completed much faster. As it turns out, a combination of the two methods helped answer the quiz questions, as an SVM model fit resulted in a 19/20 correct and a random forest fit supplied us with the answer to the question that SVM got incorrect.
  
***
  
## Appendix A - About the Analysis
  
### Metadata
File Characteristic | Value
----------------- | -------
File Name | pml-training.csv
URL | [https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)
File Size | 12,202,745 bytes
Compression | None
MD5 | 56926c78af383dcdc2060407942e52e9
 | 
File Name | pml-testing.csv
URL | [https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)
File Size | 15,113 bytes
Compression | None
MD5 | bc4174f3ec5dfcc5c570a1d2709272d9

### Environment
**System Information**

```r
sysinf <- Sys.info()
```
Parameter | Value
-------- | --------
Operating System | Windows7 x64
version | build 7601, Service Pack 1
machine arch | x86-64

**Session Information**

```r
sessionInfo()
```

```
## R version 3.2.2 (2015-08-14)
## Platform: x86_64-w64-mingw32/x64 (64-bit)
## Running under: Windows 7 x64 (build 7601) Service Pack 1
## 
## locale:
## [1] LC_COLLATE=English_Canada.1252  LC_CTYPE=English_Canada.1252   
## [3] LC_MONETARY=English_Canada.1252 LC_NUMERIC=C                   
## [5] LC_TIME=English_Canada.1252    
## 
## attached base packages:
## [1] parallel  tools     stats     graphics  grDevices utils     datasets 
## [8] methods   base     
## 
## other attached packages:
##  [1] randomForest_4.6-12             cluster_2.0.3                  
##  [3] ROCR_1.0-7                      gplots_3.0.1                   
##  [5] plyr_1.8.3                      doParallel_1.0.10              
##  [7] iterators_1.0.7                 foreach_1.4.2                  
##  [9] e1071_1.6-7                     AppliedPredictiveModeling_1.1-6
## [11] caret_6.0-57                    lattice_0.20-33                
## [13] ggplot2_2.1.0                   knitr_1.13                     
## 
## loaded via a namespace (and not attached):
##  [1] Rcpp_0.12.2        compiler_3.2.2     formatR_1.2.1     
##  [4] nloptr_1.0.4       bitops_1.0-6       class_7.3-13      
##  [7] rpart_4.1-10       digest_0.6.8       lme4_1.1-9        
## [10] evaluate_0.8       nlme_3.1-121       gtable_0.1.2      
## [13] mgcv_1.8-7         Matrix_1.2-2       yaml_2.1.13       
## [16] SparseM_1.7        stringr_1.0.0      caTools_1.17.1    
## [19] gtools_3.5.0       MatrixModels_0.4-1 stats4_3.2.2      
## [22] grid_3.2.2         nnet_7.3-10        rmarkdown_0.9.6   
## [25] gdata_2.17.0       minqa_1.2.4        reshape2_1.4.1    
## [28] car_2.1-0          magrittr_1.5       scales_0.3.0      
## [31] codetools_0.2-14   htmltools_0.2.6    MASS_7.3-43       
## [34] splines_3.2.2      pbkrtest_0.4-2     colorspace_1.2-6  
## [37] quantreg_5.19      KernSmooth_2.23-15 stringi_1.0-1     
## [40] munsell_0.4.2      CORElearn_1.47.1
```
  
## References
* [Human Activity Recognition](http://groupware.les.inf.puc-rio.br/har) - The original study where the data was collected
* [R Graphics Cookbook](http://shop.oreilly.com/product/0636920023135.do) - Winston Chang
* [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/) - Gareth James, et al
* [The Elements of Statistical Learning](http://statweb.stanford.edu/~tibs/ElemStatLearn/) - Trevor Hastie, Robert Tibshirani, Jerome Friedman
* [Applied Predictive Modeling](http://appliedpredictivemodeling.com/) - Max Kuhn, Kjell Johnson
* [Practical Data Science With R](https://www.manning.com/books/practical-data-science-with-r) - Nina Zumel, John Mount

  
