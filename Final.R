# Data Mining Research
wine_data = read.csv("winequality-white.csv", header = FALSE, sep = ";")
# Randomly shuffle
wine_data <- wine_data[sample(nrow(wine_data)),]
wine_cls = wine_data[, c(12)]
wine_data = wine_data[,c(1:11)]
summary(wine_data)


set.seed(777)

#-------------------------------------------------------------------------------------
# Without Normalizing
# Perform 5 fold cross validation
# Creating 5 equal size fold
folds <- cut(seq(1, nrow(wine_data)), breaks=10, labels=FALSE)

# Segment data by fold using the which() function
result = vector("numeric", 10)
for (i in 1:10) {
  testIndex <- which(folds==i, arr.ind=TRUE)
  
  test_data = wine_data[testIndex, ]
  training_data = wine_data[-testIndex, ]
  test_target = wine_cls[testIndex]
  training_target = wine_cls[-testIndex]
  
  require(class)
  pred = knn(train=training_data, test=test_data, cl=training_target, k=5)
  result[i] = mean(test_target != pred)
}
mean(result)

#-----------------------------------------------------------------------------------

# Normalizing function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# Normalize the data
norm_wine = as.data.frame(lapply(wine_data[,c(1:11)], normalize))

# Perform 10 fold cross validation
# Creating 10 equal size fold
folds <- cut(seq(1, nrow(norm_wine)), breaks=10, labels=FALSE)

# Segment data by fold using the which() function
result_n = vector("numeric", 10)
cumulative_misclass = vector("numeric", 6)

# KNN with k=5
for (i in 1:10) {
  testIndex <- which(folds==i, arr.ind=TRUE)
  
  test_data = norm_wine[testIndex, ]
  training_data = norm_wine[-testIndex, ]
  test_target = wine_cls[testIndex]
  training_target = wine_cls[-testIndex]
  
  require(class)
  pred = knn(train=training_data, test=test_data, cl=training_target, k=5)
  result_n[i] = mean(test_target != pred)
}
cumulative_misclass[1] = mean(result_n)


# KNN with k=9
for (i in 1:10) {
  testIndex <- which(folds==i, arr.ind=TRUE)
  
  test_data = norm_wine[testIndex, ]
  training_data = norm_wine[-testIndex, ]
  test_target = wine_cls[testIndex]
  training_target = wine_cls[-testIndex]
  
  require(class)
  pred = knn(train=training_data, test=test_data, cl=training_target, k=9)
  result_n[i] = mean(test_target != pred)
}
cumulative_misclass[2] = mean(result_n)


# KNN with k=13
for (i in 1:10) {
  testIndex <- which(folds==i, arr.ind=TRUE)
  
  test_data = norm_wine[testIndex, ]
  training_data = norm_wine[-testIndex, ]
  test_target = wine_cls[testIndex]
  training_target = wine_cls[-testIndex]
  
  require(class)
  pred = knn(train=training_data, test=test_data, cl=training_target, k=13)
  result_n[i] = mean(test_target != pred)
}
cumulative_misclass[3] = mean(result_n)


# KNN with k=17
for (i in 1:10) {
  testIndex <- which(folds==i, arr.ind=TRUE)
  
  test_data = norm_wine[testIndex, ]
  training_data = norm_wine[-testIndex, ]
  test_target = wine_cls[testIndex]
  training_target = wine_cls[-testIndex]
  
  require(class)
  pred = knn(train=training_data, test=test_data, cl=training_target, k=17)
  result_n[i] = mean(test_target != pred)
}
cumulative_misclass[4] = mean(result_n)


# KNN with k=21
for (i in 1:10) {
  testIndex <- which(folds==i, arr.ind=TRUE)
  
  test_data = norm_wine[testIndex, ]
  training_data = norm_wine[-testIndex, ]
  test_target = wine_cls[testIndex]
  training_target = wine_cls[-testIndex]
  
  require(class)
  pred = knn(train=training_data, test=test_data, cl=training_target, k=21)
  result_n[i] = mean(test_target != pred)
}
cumulative_misclass[5] = mean(result_n)


# KNN with k=25
for (i in 1:10) {
  testIndex <- which(folds==i, arr.ind=TRUE)
  
  test_data = norm_wine[testIndex, ]
  training_data = norm_wine[-testIndex, ]
  test_target = wine_cls[testIndex]
  training_target = wine_cls[-testIndex]
  
  require(class)
  pred = knn(train=training_data, test=test_data, cl=training_target, k=25)
  result_n[i] = mean(test_target != pred)
}
cumulative_misclass[6] = mean(result_n)
cumulative_accuracy = 1 - cumulative_misclass

cumulative_accuracy = cumulative_accuracy * 100
cumulative_misclass = cumulative_misclass * 100

num_k = vector("numeric", 6)
num_k[1] = 5
num_k[2] = 9
num_k[3] = 13
num_k[4] = 17
num_k[5] = 21
num_k[6] = 25


plot(cumulative_accuracy, xaxt="n",type = 'o', col = "blue", ylim = c(40,70), xlab = "Number of K in KNN", ylab = "Percentage")
axis(1, at=1:6, labels=num_k)
lines(cumulative_misclass, type = 'o', col = "red", pch = 22)
title(main = "Accuracy and errors of K-nearest neighbor with different K")
legend("topright", c("Accuracy", "Errors"), pt.cex=1.5, cex=0.75, ncol=2, col=c("blue","red"), pch=21:22, lty=1)


#--------------------------------------------------------------------------------------
#Decision Tree
wine_data = read.csv("winequality-white.csv", header = FALSE, sep = ";")
wine_data$V12 = ifelse(wine_data$V12 < 5, 0, 1)

require(rpart)
require(rpart.plot)

#tree = rpart(V12 ~ wine_data$V1+wine_data$V2+wine_data$V3+wine_data$V4+wine_data$V5+wine_data$V6+wine_data$V7+wine_data$V8+wine_data$V9+wine_data$V10+wine_data$V11, data=wine_data, method = "class")
#plot(tree)
#text(tree, pretty=0)

#rpart.plot(tree)
#table(wine_data$V12) #3% = 0, 97% = 1
#printcp(tree)
#plotcp(tree)

# Cross Validation to find accuracy
# Perform 10 fold cross validation
# Creating 10 equal size fold
folds <- cut(seq(1, nrow(wine_data)), breaks=10, labels=FALSE)

# Segment data by fold using the which() function
result_tree = vector("numeric", 10)
cumulative_misclass_tree = vector("numeric", 6)

# before pruning
for (i in 1:10) {
  testIndex <- which(folds==i, arr.ind=TRUE)
  test_data = wine_data[testIndex, -c(12)]
  training_data = wine_data[-testIndex, ]
  test_target = wine_data[testIndex, c(12)]
  training_target = wine_data[-testIndex, c(12)]
  
  dtree_model = rpart(training_data$V12~., data=training_data, method = "class")
  #rpart.plot(dtree_model)
  #plotcp(dtree_model)
  dtree_pred = predict(dtree_model, test_data, type = "class")
  result_tree[i]= mean(dtree_pred != test_target)
}

# after pruning
result_ptree = vector("numeric", 10)
for (i in 1:10) {
  testIndex <- which(folds==i, arr.ind=TRUE)
  test_data = wine_data[testIndex, -c(12)]
  training_data = wine_data[-testIndex, ]
  test_target = wine_data[testIndex, c(12)]
  training_target = wine_data[-testIndex, c(12)]
  
  dtree_model = rpart(training_data$V12~., data=training_data, method = "class")
  #rpart.plot(dtree_model)
  #plotcp(dtree_model)
  ptree_model = prune(dtree_model, cp= dtree_model$cptable[which.min(dtree_model$cptable[,"xerror"]),"CP"])
  rpart.plot(ptree_model)
  #plotcp(ptree_model)
  ptree_pred = predict(ptree_model, test_data, type = "class")
  
  result_ptree[i]= mean(ptree_pred != test_target)
}

result_tree = mean(result_tree)
result_ptree = mean(result_ptree)



plot((1-result_tree)*100, xaxt="n",type = 'o', col = "blue", ylim = c(90,100), xlab = "Test_Set", ylab = "Percentage")
lines((1-result_ptree)*100, type = 'o', col = "red", pch = 22)
axis(1, at=1:10)
title(main = "Accuracy of Decision Tree")
legend("bottomright", c("Before Pruning", "After Pruning"), pt.cex=1.5, cex=0.75, ncol=2, col=c("blue","red"), pch=21:22, lty=1)

plot((result_tree)*100, xaxt="n",type = 'o', col = "blue", ylim = c(0,10), xlab = "Test_Set", ylab = "Percentage")
lines((result_ptree)*100, type = 'o', col = "red", pch = 22)
axis(1, at=1:10)
title(main = "Errors of Decision Tree")
legend("bottomright", c("Before Pruning", "After Pruning"), pt.cex=1.5, cex=0.75, ncol=2, col=c("blue","red"), pch=21:22, lty=1)



#--------------------------------------------------------------------------------------
#EM Clustering
wine_data = read.csv("winequality-white.csv", header = FALSE, sep = ";")

#Randomly shuffle
wine_data <- wine_data[sample(nrow(wine_data)),]
wine_cls = wine_data[, c(12)]
wine_data = wine_data[,c(1:11)]

# Normalizing function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}


wine_data = as.data.frame(lapply(wine_data[,c(1:11)], normalize))
str(wine_data)
summary(wine_data)

require(mclust)
#Clustering
table(wine_cls)
clPairs(wine_data, wine_cls)
fit = Mclust(wine_data, G=7)
summary(fit)


plot(fit, what = "BIC")
plot(fit, what = "classification")
plot(fit, what = "uncertainty")
plot(fit, what = "density")

# Creating 10 equal size fold
folds <- cut(seq(1, nrow(wine_data)), breaks=10, labels=FALSE)

# Segment data by fold using the which() function
result_em = vector("numeric", 10)
cumulative_misclass_em = vector("numeric", 6)

for (i in 1:10) {
  testIndex <- which(folds==1, arr.ind=TRUE)
  test_data = wine_data[testIndex, -c(12)]
  training_data = wine_data[-testIndex, -c(12)]
  
  mod = Mclust(training_data, G=7)
  em_pred = predict.Mclust(mod, test_data)
  
  table(em_pred$classification)
  table(mod$classification)
  
  plot(training_data, col = em_pred$classification, pch = em_pred$classification)
  
  result_em[i]= mean(_pred != test_target)
}






#MclustDA
dataMclustDA <- MclustDA(wine_data, wine_cls, modelType = "EDDA", modelNames = "EEE")
cv <- cvMclustDA(dataMclustDA, nfold = 10) 
cv[c("classification","error", "se")]
table(wine_cls)
table(cv$classification)

summary.MclustDA(dataMclustDA)
plot.MclustDA(dataMclustDA)

#--------------------------------------------------------------------------------------
#Kmeans
wine_data = read.csv("winequality-white.csv", header = FALSE, sep = ";")

wine_features = wine_data[,c(1:11)]

# Normalizing function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

set.seed(777)
require(cluster)
require(fpc)

wine_features = as.data.frame(lapply(wine_data[,c(1:11)], normalize))
str(wine_features)
summary(wine_features)

result = kmeans(wine_features, 7, iter.max=10000, nstart = 20, trace = TRUE)

plotcluster(wine_features, result$cluster)

result$size
result$centers
table(result$cluster)
table(wine_data$V12)
table(wine_data$V12, result$cluster)

mean(wine_data$V12 != result$cluster)

plot(wine_features, col=result$cluster, main="Kmeans clustering", pch=20, cex=1)


require(broom)
tidy(result)
