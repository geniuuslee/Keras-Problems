## Logistic Regrssion
## Dataset: Graduate Admissions (file name: Admission_Predict_Ver1.1.csv)
## (https://www.kaggle.com/mohansacharya/graduate-admissions)


#library
install.packages("lubridate")
install.packages("pastecs")
install.packages("corrplot")
install.packages("moments")
install.packages("e1071")
install.packages
library(moments)
library(lubridate)
library(pastecs)
library(corrplot)
library(e1071)


#Load dataset
admission <- read.csv("Admission_Predict_Ver1.1.csv")

#id variable 제거
id_idx <- 1
admission_data <- cbind(admission[,-id_idx])

#statistics of variables
stat.desc(admission_data, basic=FALSE)
kurtosis(admission_data)
skewness(admission_data)

#boxplot
boxplot(admission_data$GRE.Score, main = "GRE SCORE")
boxplot(admission_data$TOEFL.Score, main = "TOEFL Score")
boxplot(admission_data$University.Rating, main = "University.Rating")
boxplot(admission_data$SOP, main = "sop")
boxplot(admission_data$LOR, main = "LOR")
boxplot(admission_data$CGPA, main = "CGPA")
boxplot(admission_data$Research, main = "Research")

#remove outliers
admission_sub <- admission_data
admission_sub$LOR <- ifelse(admission_data$LOR < boxplot(admission_data$LOR)$stats[1,1]|admission_data$LOR > boxplot(admission_data$LOR)$stats[5,1],NA, admission_data$LOR)

admission_removed <- na.omit(admission_sub)
boxplot(admission_removed$LOR, main = "Removed outlier")

#scatter plot
nVars <- ncol(admission_removed)
par(mfrow=c(3,3))
for(i in 1:nVars){
  plot(admission_removed[,i],admission_removed$GRE.Score, ylab ="GRE score")
}
par(mfrow=c(3,3))
for(i in 1:nVars){
  plot(admission_removed[,i],admission_removed$TOEFL.Score, ylab ="TOEFL.Score")
}
par(mfrow=c(3,3))
for(i in 1:nVars){
  plot(admission_removed[,i],admission_removed$University.Rating, ylab ="University.Rating")
}
par(mfrow=c(3,3))
for(i in 1:nVars){
  plot(admission_removed[,i],admission_removed$SOP, ylab ="SOP")
}
par(mfrow=c(3,3))
for(i in 1:nVars){
  plot(admission_removed[,i],admission_removed$LOR, ylab ="LOR")
}
par(mfrow=c(3,3))
for(i in 1:nVars){
  plot(admission_removed[,i],admission_removed$CGPA, ylab ="CGPA")
}
par(mfrow=c(3,3))
for(i in 1:nVars){
  plot(admission_removed[,i],admission_removed$Research, ylab ="Research")
}

#Correlation plot
admission_corr <- cor(admission_removed)
admission_corr
corrplot(admission_corr, method ="number")

#target variable
target_idx <- 8
admit_idx <- which(admission_removed[,target_idx]>0.8)
admission_factor <- admission_removed
admission_factor[admit_idx,target_idx] <- 1
admission_factor[-admit_idx,target_idx] <- 0

#Normalization
input_idx <- c(1,2,3,4,5,6,7)
admission_input <- admission_factor[,input_idx]
admission_input <- scale(admission_input, center = TRUE, scale = TRUE)
admission_target <- as.factor(admission_factor[,target_idx])
admission_reg <- data.frame(admission_input,admission_target)

#Split the data into the training/test sets
set.seed(2015170829)
nObjects <- nrow(admission_reg)
trn_idx <- sample(1:nObjects, round(0.7*nObjects))
admission_trn <- admission_reg[trn_idx,]
admission_tst <- admission_reg[-trn_idx,]

#Train the Logistic Regression Model
admission_lr <- glm(admission_target~., family=binomial, admission_trn)
summary(admission_lr)

lr_response <- predict(admission_lr, type = "response", newdata = admission_tst)
lr_target <- admission_tst$admission_target
lr_predicted <- rep(0, length(lr_target))
lr_predicted[which(lr_response>0.8)] <- 1
cm_full <- table(lr_target, lr_predicted)
cm_full

#Performance matrix
perf_eval <- function(cm){
  
  # True positive rate: TPR (Recall)
  TPR <- cm[2,2]/sum(cm[2,])
  # Precision
  PRE <- cm[2,2]/sum(cm[,2])
  # True negative rate: TNR
  TNR <- cm[1,1]/sum(cm[1,])
  # False Positive Rate
  FPR <- cm[1,2]/sum(cm[1,])
  # False Negative Rate
  FNR <- cm[2,1]/sum(cm[2,])
  # Simple Accuracy
  ACC <- (cm[1,1]+cm[2,2])/sum(cm)
  # Balanced Correction Rate
  BCR <- sqrt(TPR*TNR)
  # F1-Measure
  F1 <- 2*TPR*PRE/(TPR+PRE)
  
  return(c(TPR, PRE, TNR, FPR, FNR, ACC, BCR, F1))
}

perf_mat <- matrix(0, 1, 8)
colnames(perf_mat) <- c("TPR ", "Precision", "TNR", "FPR", "FNR", "ACC", "BCR", "F1")
rownames(perf_mat) <- "Logstic Regression"

perf_mat[1,] <- perf_eval(cm_full)
perf_mat

#AUROC, seed number: (1,2,3,4,5)
set.seed(5)
roc_trn_idx <- sample(1:nObjects, round(0.7*nObjects))
roc_trn <- admission_reg[roc_trn_idx,]
roc_tst <- admission_reg[-roc_trn_idx,]

roc_lr <- glm(admission_target~., family = binomial, roc_trn)
summary(roc_lr)
roc_probs <- predict(roc_lr, type ="response", roc_tst)
roc_target <- roc_tst$admission_target
roc_probs

#plotting, calculate AUROC
auroc <- function(probs, target){
  probs_dec <- sort(probs, decreasing = TRUE, index.return = TRUE)
  true_y <- target[probs_dec$ix]
  
  fpr <- cumsum(true_y == 0)/sum(true_y == 0)
  tpr <- cumsum(true_y == 1)/sum(true_y == 1)
  plot(fpr, tpr, xlim = c(0,1), ylim = c(0,1), main ="ROC5")
  auc <- sum((fpr[2:length(fpr)]-fpr[1:length(fpr)-1])*tpr[2:length(tpr)])
  auc
  return(auc)
}

auroc(roc_probs, roc_target)
