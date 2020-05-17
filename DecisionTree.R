## Decision Tree
## Dataset: Heart Disease UCI (https://www.kaggle.com/ronitf/heart-disease-uci)



#Load dataset and data preprocessing
heart <- read.csv("heart.csv")
summary(heart)

input_idx <- c(1:13)
target_idx <- 14

heart_input <- heart[,input_idx]

#target 변수를 integer -> factor로 변경
heart_target <- as.factor(heart[,target_idx])

#Split dataset into Training/Validation set
set.seed(2015170829)
nObjects <- nrow(heart)
trn_idx <- sample(1:nObjects, 200)

heart_trn_data <- data.frame(heart_input[trn_idx,], heartYN = heart_target[trn_idx])
heart_tst_data <- data.frame(heart_input[-trn_idx,], heartYN = heart_target[-trn_idx])

#Performance evaluation function
perf_eval <- function(cm){
  
  # True positive rate: TPR (Recall)
  TPR <- cm[2,2]/sum(cm[2,])
  # Precision
  PRE <- cm[2,2]/sum(cm[,2])
  # True negative rate: TNR
  TNR <- cm[1,1]/sum(cm[1,])
  # False Negative Rate
  FNR <- cm[2,1]/sum(cm[2,])
  # Simple Accuracy
  ACC <- (cm[1,1]+cm[2,2])/sum(cm)
  # Balanced Correction Rate
  BCR <- sqrt(TPR*TNR)
  # F1-Measure
  F1 <- 2*TPR*PRE/(TPR+PRE)
  
  return(c(TPR, PRE, TNR, FNR, ACC, BCR, F1))
}

#Performance Matrix
perf_mat <- matrix(0, 2, 7)
colnames(perf_mat) <- c("TPR ", "Precision", "TNR", "FNR", "ACC", "BCR", "F1")
rownames(perf_mat) <- c("Post-pruning", "Pruned")
perf_mat


#install packages
#install.packages("tree")
library(tree)

#Post-pruning tree 학습
heart_post <- tree(heartYN~., heart_trn_data)
summary(heart_post)

#Plot the tree
plot(heart_post)
text(heart_post, pretty = 1)

#Prediction and Performance
heart_post_prey <- predict(heart_post, heart_tst_data, type = "class")
heart_post_cm <- table(heart_tst_data$heartYN, heart_post_prey)
heart_post_cm

perf_mat[1,] <- perf_eval(heart_post_cm)
perf_mat


#Pruning - Cross validation
heart_post_cv <- cv.tree(heart_post, FUN = prune.misclass)
heart_post_cv

#Plot the pruning result
plot(heart_post_cv$size, heart_post_cv$dev, type = 'b')

#불순도가 가장 낮은 leaf node 수 = 8
#Select the final model
heart_post_pruned <- prune.misclass(heart_post, best = 8)
summary(heart_post_pruned)

plot(heart_post_pruned)
text(heart_post_pruned, pretty = 1)

#Prediction and perforamnce
heart_pruned_prey <- predict(heart_post_pruned, heart_tst_data, type = "class")
heart_pruned_cm <- table(heart_tst_data$heartYN, heart_pruned_prey)
heart_pruned_cm

perf_mat[2,] <- perf_eval(heart_pruned_cm)
perf_mat





##Classification Tree package 1 : rpart

#Install 'rpart' package
#install.packages("rpart")
library(rpart)

#Train the tree
heart_rpart <- rpart(heartYN ~ ., heart_trn_data, method = "class")
summary(heart_rpart)

#Plot the tree
plot(heart_rpart)
text(heart_rpart, pretty = 1)

#Other Plotting package install
#install.packages("rattle")
library(rattle)
fancyRpartPlot(heart_rpart)

#Prediction and performance
heart_rpart_prey <- predict(heart_rpart, heart_tst_data, type = "class")
heart_rpart_cm <- table(heart_tst_data$heartYN, heart_rpart_prey)
heart_rpart_cm

#Performance matrix
rpart_perf <- matrix(0,4,7)
rownames(rpart_perf) <- c("Post-pruning", "Pruned", "Dev_Post-pruning", "Dev_Pruned")
colnames(rpart_perf) <- c("TPR ", "Precision", "TNR", "FNR", "ACC", "BCR", "F1")

rpart_perf[1,] <- perf_eval(heart_rpart_cm)
rpart_perf

#Find the best model
printcp(heart_rpart)
plotcp(heart_rpart)

#Pruning
heart_rpart_pruned <- prune(heart_rpart, cp= heart_rpart$cptable[which.min(
  heart_rpart$cptable[,"xerror"]), "CP"])
summary(heart_rpart_pruned)
printcp(heart_rpart_pruned)

#Plot the Pruned tree
fancyRpartPlot(heart_rpart_pruned)

#Predict with Pruned tree
rpart_pruned_prey <- predict(heart_rpart_pruned, heart_tst_data,type = "class" )
rpart_pruend_cm <- table(heart_tst_data$heartYN, rpart_pruned_prey)
rpart_pruend_cm

#Performance of the pruned tree
rpart_perf[2,] <- perf_eval(rpart_pruend_cm)
rpart_perf



##Changing parameter
heart_rpart2 <- rpart(heartYN ~ ., heart_trn_data, method = "class", parms = list(split = 'deviation'))
summary(heart_rpart2)
printcp(heart_rpart2)

#Plot the tree
fancyRpartPlot(heart_rpart2)

#Prediction and performance
heart_rpart2_prey <- predict(heart_rpart2, heart_tst_data, type = "class")
heart_rpart2_cm <- table(heart_tst_data$heartYN, heart_rpart2_prey)
heart_rpart2_cm

rpart_perf[3,] <- perf_eval(heart_rpart2_cm)
rpart_perf

#Pruning
plotcp(heart_rpart2)
printcp(heart_rpart2)

#현재 모델이 best 모델이므로 pruning이 필요 없음.
rpart_perf[4,] <- rpart_perf[3,]
rpart_perf


##Classification Tree package 2 : party

#Install 'party' package
install.packages("party")
library(party)

#Train the tree
heart_party <- ctree(heartYN ~ ., heart_trn_data)
heart_party

#plot the tree
plot(heart_party)

#Prediction and performance
heart_party_prey <- predict(heart_party, heart_tst_data)
heart_party_cm <- table(heart_tst_data$heartYN, heart_party_prey)
heart_party_cm

#Performance Matrix for 'party' package
party_perf <- matrix(0,2,7)
rownames(party_perf) <- c("Default", "Changed option")
colnames(party_perf) <- c("TPR ", "Precision", "TNR", "FNR", "ACC", "BCR", "F1")
party_perf

party_perf[1,] <- perf_eval(heart_party_cm)
party_perf

#Train the tree by changing option (significance level 0.05 -> 0.01)
heart_party2 <- ctree(heartYN ~ ., heart_trn_data, controls = ctree_control(mincriterion = 0.99))
heart_party2

#Plot the tree
plot(heart_party2)

#Prediction and performance
heart_party2_prey <- predict(heart_party2, heart_tst_data)
heart_party2_cm <- table(heart_tst_data$heartYN, heart_party2_prey)
heart_party2_cm

party_perf[2,] <- perf_eval(heart_party2_cm)
party_perf

##Classification Tree package 3 : evtree
#Install package
install.packages("evtree")
library(evtree)

#Train the tree
heart_evtree <- evtree(heartYN ~ ., heart_trn_data, seed = 29)
heart_evtree

#Plot the tree
plot(heart_evtree)

#Prediction and performance
heart_evtree_prey <- predict(heart_evtree, heart_tst_data)
heart_evtree_cm <- table(heart_tst_data$heartYN, heart_evtree_prey)
heart_evtree_cm

#Performance matrix for 'party' package
evtree_perf <- matrix(0, 2, 7)
rownames(evtree_perf) <- c("Default", "Changed option")
colnames(evtree_perf) <- c("TPR ", "Precision", "TNR", "FNR", "ACC", "BCR", "F1")

evtree_perf[1,] <- perf_eval(heart_evtree_cm)
evtree_perf

#Train the tree by changing option
heart_evtree2 <- evtree(heartYN ~ ., heart_trn_data, operatorprob = list(pprune = 0.5), seed = 29)
heart_evtree2

#Plot the tree
plot(heart_evtree2)

#Prediction and Performance
heart_evtree2_prey <- predict(heart_evtree2, heart_tst_data)
heart_evtree2_cm <- table(heart_tst_data$heartYN, heart_evtree2_prey)
heart_evtree2_cm

evtree_perf[2,] <- perf_eval(heart_evtree2_cm)
evtree_perf



#Performance of Best model of each packages
best_perf <- matrix(0, 4, 7)
rownames(best_perf) <- c("tree", "rpart", "party", "evtree")
colnames(best_perf) <- c("TPR ", "Precision", "TNR", "FNR", "ACC", "BCR", "F1")
best_perf

best_perf[1,] <- perf_mat[2,]
best_perf[2,] <- rpart_perf[2,]
best_perf[3,] <- party_perf[1,]
best_perf[4,] <- evtree_perf[1,]

leaf_nodes <- c(8,4,6,5)
colnames(leaf_nodes) <- c("# of leaf nodes")
best_perf <- cbind(best_perf, leaf_nodes)
best_perf