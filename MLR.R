## Multivariate Linear Regression
## Dataset: House Sales in King Country, USA, kc_house_data.csv (https://www.kaggle.com/harlfoxem/housesalesprediction)

#packages
install.packages("lubridate")
install.packages("pastecs")
install.packages("corrplot")
install.packages("moments")
install.packages("e1071")
library(moments)
library(lubridate)
library(pastecs)
library(corrplot)
library(e1071)

#read dataset
House <- read.csv("kc_house_data.csv")

#transform the date variable to integer
House$date <- (substr(House$date, 1,8))
House$date <- ymd(House$date)
House$date <- as.numeric(as.Date(House$date, origin = "1900-01-01"))

id_idx <- 1
price_idx <- 2

#id variable 제거
house_mlr_data <- cbind(House[,-id_idx])

#statistics
stat.desc(house_mlr_data, basic=FALSE)
kurtosis(house_mlr_data)
skewness(house_mlr_data)

#boxplot
boxplot(house_mlr_data$date, main = "date")
boxplot(house_mlr_data$bedrooms, main = "bedrooms")
boxplot(house_mlr_data$bathrooms, main="bathrooms")
boxplot(house_mlr_data$sqft_living, main="sqft_living")
boxplot(house_mlr_data$sqft_lot, main="sqft_lot")
boxplot(house_mlr_data$floors, main="floors")
boxplot(house_mlr_data$waterfront, main ="waterfront")
boxplot(house_mlr_data$view, main="view")
boxplot(house_mlr_data$condition, main ="condition")
boxplot(house_mlr_data$grade, main="grade")
boxplot(house_mlr_data$sqft_above, main = "sqft_above")
boxplot(house_mlr_data$sqft_basement, main = "sqft_basement")
boxplot(house_mlr_data$yr_built, main="yr_built")
boxplot(house_mlr_data$yr_renovated, main="yr_renovated")
boxplot(house_mlr_data$zipcode, main="zipcode")
boxplot(house_mlr_data$lat, main="lat")
boxplot(house_mlr_data$long, main="long")
boxplot(house_mlr_data$sqft_living15, main="sqft_living15")
boxplot(house_mlr_data$sqft_lot15, main="sqft_lot15")

#remove outliers

#outlier로 판단하기 어려운 varaible 빼기
no_out_idx <- c(8,15)
house_removed <- cbind(house_mlr_data[,-no_out_idx])
nVar_removed <- ncol(house_removed)

#outlier NA로 대체
house_sub <- house_removed
for (i in 1:nVar_removed){
  Q1 <- quantile(house_removed[,i], probs=c(0.25), na.rm=TRUE)
  Q3 <- quantile(house_removed[,i], probs=c(0.75), na.rm=TRUE)
  
  LC <- Q1-1.5*(Q3-Q1)
  UC <- Q3+1.5*(Q3-Q1)
  
  house_sub[,i] <- ifelse(house_removed[,i] < boxplot(house_removed[,i])$stats[1,1]|house_removed[,i] > boxplot(house_removed[,i])$stats[5,1],NA, house_removed[,i])
}

#제거했던 variables 추가
house_rejoin <- cbind(house_sub, house_mlr_data[,no_out_idx])

#결측치 포함 객체 제거
house_clean <- na.omit(house_rejoin)

#scatter plot
house_scatter <- cbind(house_clean[,-price_idx])
nVar_scatter <- ncol(house_scatter)

# 컴퓨터 속도 문제로 house_scatter$date~ house_scatter$yr_renovated 직접 변경
par(mfrow=c(5,5))
for(i in 1:nVar_scatter){
  plot(house_scatter[,i],house_scatter$yr_renovated)
}

#correlation
house_corr <- cor(house_clean)
house_corr
corrplot(house_corr, method ="number")

##MLR

#phase 1
#split the data into training/validation sets
nVar <- ncol(house_clean)
nHouse <-nrow(house_clean)

set.seed(2015170829)
house_trn_idx <- sample(1:nHouse, round(0.7*nHouse))
house_trn_data <- house_clean[house_trn_idx,]
house_val_data <- house_clean[-house_trn_idx,]

#train the MLR
mlr_house <- lm(price~., data = house_trn_data)
mlr_house
summary(mlr_house)
plot(mlr_house)

#Performance evaluation function for regression
perf_eval_reg <- function(tgt_y, pre_y){
  
  #RMSE
  rmse <- sqrt(mean((tgt_y - pre_y)^2))
  #MAE
  mae <- mean(abs(tgt_y - pre_y))
  #MAPE
  mape <- 100*mean(abs((tgt_y - pre_y)/tgt_y))
  
  return(c(rmse, mae, mape))
}

perf_mat <- matrix(0, nrow = 2, ncol = 3)
rownames(perf_mat) <- c("house price #1", "house price #2")
colnames(perf_mat) <- c("RMSE", "MAE", "MAPE")
perf_mat

mlr_house_haty <- predict(mlr_house, newdata = house_val_data)

perf_mat[1,] <- perf_eval_reg(house_val_data$price, mlr_house_haty)
perf_mat


#phase 2

remove_var_idx <- c(3,5,6,8,9,11,12,15,16,17,18,19)
house_new <- cbind(house_clean[,-remove_var_idx])

nVar_new <- ncol(house_new)
nHouse_new <-nrow(house_new)

house_trn_idx_new <- sample(1:nHouse_new, round(0.7*nHouse_new))
house_trn_data_new <- house_new[house_trn_idx_new,]
house_val_data_new <- house_new[-house_trn_idx_new,]

#train the MLR
mlr_house_new <- lm(price~., data = house_trn_data_new)
mlr_house_new
summary(mlr_house_new)
plot(mlr_house_new)

#performance
mlr_house_haty_new <- predict(mlr_house_new, newdata = house_val_data_new)

perf_mat[2,] <- perf_eval_reg(house_val_data_new$price, mlr_house_haty_new)
perf_mat

#extra
anova(mlr_house, mlr_house_new)
AIC(mlr_house, mlr_house_new)
