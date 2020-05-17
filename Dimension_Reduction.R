## Dimensionality Reductino Practice
## Dataset: Weahter Ankara (https://sci2s.ugr.es/keel/dataset.php?cod=41) 

#packages
#install.packages("GA")
library(GA)

#read dataset
weather <- read.csv("Weather_Ankara.csv")
temp_idx <- 10

#MLR
#Split the data into trainig/test sets
set.seed(2015170829)
weather_trn_idx <- sample(1:250)
weather_trn_data <- weather[weather_trn_idx,]
weather_tst_data <- weather[-weather_trn_idx,]

#Train the MLR
mlr_weather <- lm(Mean_temperature~., data = weather_trn_data)
mlr_weather
summary(mlr_weather)
plot(mlr_weather)

#Adjusted R-square Matrix
adj_mat <- matrix(0, nrow = 1, ncol = 6)
colnames(adj_mat) <- c("All Vars", "Exhaustive", "Forward","Backward","Stepwise","GA")
rownames(adj_mat) <- c("Adj R_square")

adj_mat[,1] <- summary(mlr_weather)$adj
adj_mat

#Performance evalutaion function for regression
perf_eval_reg <- function(tgt_y, pre_y){
  
  #RMSE
  rmse <- sqrt(mean((tgt_y - pre_y)^2))
  #MAE
  mae <- mean(abs(tgt_y - pre_y))
  #MAPE
  mape <- 100*mean(abs((tgt_y -pre_y)/tgt_y))

  
  return(c(rmse, mae, mape))
}

perf_mat <- matrix(0, nrow = 6, ncol = 3)
rownames(perf_mat) <- c("All Vars", "Exhaustive", "Forward","Backward","Stepwise","GA")
colnames(perf_mat) <- c("RMSE", "MAE", "MAPE")
perf_mat

#Prediction
mlr_weather_haty <- predict(mlr_weather, newdata = weather_tst_data)
perf_mat[1,] <- perf_eval_reg(weather_tst_data$Mean_temperature, mlr_weather_haty)
perf_mat

##Exhaustive Search -> Depth-First Search 방식을 통한 탐색.
#변수 조합 설정을 위한 binary matrix 생성 by DFS
varMat <<- 1:length(9)
dfs <- function(idx,arr){
  if(idx==length(arr)+1){
    varMat <<- cbind(varMat,arr)
    return()
  }
  arr[idx] <- 0
  dfs(idx+1, arr)
  arr[idx] <- 1
  dfs(idx+1, arr)
}
dfs(1, 1:9)

#모든 변수가 0 인경우 제외
varMat <- data.frame(varMat[,-c(1,2)])
varMat

#소요 시간 matrix
time_mat <- matrix(0, nrow = 1, ncol = 5)
rownames(time_mat) <- c("time")
colnames(time_mat) <- c("Exhaustive", "Forward", "Backward", "Stepwise", "GA")
time_mat

#exhaustive search 함수
varNames <- colnames(weather[,-10])
exhaustive_search <- function(){
  #탐색 시작 시각 측정
  start_time<-Sys.time()
  ex_lm_result <- rep(0:9)
  names(ex_lm_result) <- c("Max_temperature","Min_temperature","Dewpoint",
                        "Precipitation","Sea_level_pressure","Standard_pressure",
                        "Visibility","Wind_speed","Max_wind_speed","Adj R_square")
  #모든 변수 조합에 대해 학습
  for(i in 1:511){
    tmp_x <- paste(varNames[which(varMat[i]==1)],collapse = " + ")
    string <- paste("Mean_temperature ~ ", tmp_x, collapse = "")
    model <- lm(as.formula(string), data = weather_trn_data)
    
    ex_lm_result <- rbind(ex_lm_result, c(varMat[,i], summary(model)$adj))
  }
  #탐색 종료 시각 측정
  end_time <- Sys.time()
  time_mat[1,1] <<- end_time - start_time
  return(ex_lm_result)
}

exhaustive_result <- exhaustive_search()
exhaustive_result <- exhaustive_result[-1,]

#Adj R_sqaure 내림차순 정렬 후 가장 높은 변수 조합을 찾음
ex_rank <- order(exhaustive_result[,10], decreasing = TRUE)
ex_rank[1]
ex_best_vars <- which(varMat[,ex_rank[1]]>0)
ex_best_vars
names(weather[,ex_best_vars])

#Best Adjusted R-square
adj_mat[,2] <- exhaustive_result[ex_rank[1],10]
adj_mat

#time of Exhaustive search
time_mat

#Split data into the training/validation sets
ex_best_trn <- weather_trn_data[,ex_best_vars]
ex_best_trn <- cbind(ex_best_trn, weather_trn_data[,10])
names(ex_best_trn)[9] <- c("Mean_temperature")

#Training
mlr_ex_best <- lm(Mean_temperature ~ Max_termperature + Min_temperature
                  + Dewpoint + Sea_level_pressure + Standard_pressure
                  + Visibility + Wind_speed + Max_wind_speed, data = weather_trn_data)
mlr_ex_best
summary(mlr_ex_best)
plot(mlr_ex_best)

#Prediction
mlr_ex_haty <- predict(mlr_ex_best, newdata = weather_tst_data)
perf_mat[2,] <- perf_eval_reg(weather_tst_data$Mean_temperature, mlr_ex_haty)
perf_mat

##Forwad Selection
#Training
start_forward <- Sys.time()
forward_model <- step(lm(Mean_temperature ~ 1, data = weather_trn_data),
                      scope = list(upper = mlr_weather,
                                   lower = Mean_temperature ~ 1),
                      dircetion = "forward", trace = 1)
end_forward <- Sys.time()
time_mat[,2] <- end_forward - start_forward
time_mat

forward_model
summary(forward_model)
plot(forward_model)
adj_mat[,3] <- summary(forward_model)$adj
adj_mat

#Prediction
mlr_forward_haty <- predict(forward_model, newdata = weather_tst_data)
perf_mat[3,] <- perf_eval_reg(weather_tst_data$Mean_temperature, mlr_forward_haty)
perf_mat

##Backward Selection
#Training
start_backward <- Sys.time()
backward_model <- step(mlr_weather,
                       scope = list(upeer = mlr_weather,
                                    lower = Mean_temperature ~ 1),
                       direction = "backward", trace =1)
end_backward <- Sys.time()
time_mat[,3] <- end_backward - start_backward
time_mat

backward_model
summary(backward_model)
plot(backward_model)
adj_mat[,4] <- summary(backward_model)$adj
adj_mat

#Prediction
mlr_backward_haty <- predict(backward_model, newdata = weather_tst_data)
perf_mat[4,] <- perf_eval_reg(weather_tst_data$Mean_temperature, mlr_backward_haty)
perf_mat

##Stepwise Selection
#Training
start_stepwise <- Sys.time()
stepwise_model <- step(lm(Mean_temperature ~ 1, data = weather_trn_data),
                       scope = list(upper = mlr_weather, lower = Mean_temperature ~ 1),
                       direction = "both", trace = 1)
end_stepwise <- Sys.time()
time_mat[,4] <- end_stepwise - start_stepwise
time_mat

summary(stepwise_model)
plot(stepwise_model)
adj_mat[,5] <- summary(stepwise_model)$adj
adj_mat

#Prediction
mlr_stepwise_haty <- predict(stepwise_model, newdata = weather_tst_data)
perf_mat[5,] <- perf_eval_reg(weather_tst_data$Mean_temperature, mlr_stepwise_haty)
perf_mat

##Genetic Algorithm
#Fintess function : Adjusted R-square
fit_adj <- function(string){
  sel_var_idx <- which(string == 1)
  #Use variables whose gene value is 1
  sel_x <- x[, sel_var_idx]
  xy <- data.frame(sel_x, y)
  
  #Training model
  GA_lr <- lm(y~., data = xy)
  GA_lr_prey <- predict(GA_lr, newdata = xy)
  
  return(summary(GA_lr)$adj)
}

x <- as.matrix(weather_trn_data[,-temp_idx])
y <- weather_trn_data[,temp_idx]

#Variable selection via genetic algorithm
start_GA <- Sys.time()
GA_adj <- ga(type = "binary", fitness = fit_adj, nBits=ncol(x),
             names = colnames(x), popSize = 50, pcrossover = 0.5, 
             pmutation = 0.01, maxiter = 100, elitism = 2, seed = 29)
end_GA <- Sys.time()
time_mat[,5] <- end_GA - start_GA
time_mat

best_vars_GA <- which(GA_adj@solution == 1)

# Model training based on the best vas by GA
GA_trn_data <- weather_trn_data[,c(best_vars_GA,temp_idx)]
GA_tst_data <- weather_tst_data[,c(best_vars_GA,temp_idx)]

GA_model <- lm(Mean_temperature ~ . , data = GA_trn_data)
GA_model
summary(GA_model)
plot(GA_model)
adj_mat[,6] <- summary(GA_model)$adj
adj_mat

mlr_ga_haty <- predict(GA_model, newdata = GA_tst_data)

perf_mat[6,] <- perf_eval_reg(weather_tst_data$Mean_temperature, mlr_ga_haty)
perf_mat

##Compare GA models by different hyperparameters
#Times of each GA models
time_GA <- matrix(0, nrow = 1, ncol = 13)
rownames(time_GA) <- c("time")
colnames(time_GA) <- c("#0","#1","#2","#3","#4","#5","#6","#7","#8","#9","#10","#11","#12")
time_GA

# 0) Population size: 50 / Cross-over: 0.5 / Mutation rate: 0.01
var_idx0 <- best_vars_GA
time_GA[,1] <- time_mat[,5]

# 1) Population size: 50 / Cross-over: 0.7 / Mutation rate: 0.01
start_1 <- Sys.time()
GA_1 <- ga(type = "binary", fitness = fit_adj, nBits=ncol(x),
             names = colnames(x), popSize = 50, pcrossover = 0.7, 
             pmutation = 0.01, maxiter = 100, elitism = 2, seed = 29)
end_1 <- Sys.time()
time_GA[,2] <- end_1 - start_1
var_idx1 <- which(GA_1@solution == 1)
var_idx1

# 2) Population size: 100 / Cross-over: 0.5 / Mutation rate: 0.01
start_2 <- Sys.time()
GA_2 <- ga(type = "binary", fitness = fit_adj, nBits=ncol(x),
             names = colnames(x), popSize = 100, pcrossover = 0.5, 
             pmutation = 0.1, maxiter = 100, elitism = 2, seed = 29)
end_2 <- Sys.time()
time_GA[,3] <- end_2 - start_2
var_idx2 <- which(GA_2@solution == 1)
var_idx2

# 3) Population size: 50 / Cross-over: 0.5 / Mutation rate: 0.05
start_3 <- Sys.time()
GA_3 <- ga(type = "binary", fitness = fit_adj, nBits=ncol(x),
             names = colnames(x), popSize = 50, pcrossover = 0.5, 
             pmutation = 0.05, maxiter = 100, elitism = 2, seed = 29)
end_3 <- Sys.time()
time_GA[,4] <- end_3 - start_3
var_idx3 <- which(GA_3@solution == 1)
var_idx3

# 4) Population size: 100 / Cross-over: 0.7 / Mutation rate: 0.01
start_4 <- Sys.time()
GA_4 <- ga(type = "binary", fitness = fit_adj, nBits=ncol(x),
           names = colnames(x), popSize = 100, pcrossover = 0.7, 
           pmutation = 0.01, maxiter = 100, elitism = 2, seed = 29)
end_4 <- Sys.time()
time_GA[,5] <- end_4 - start_4
var_idx4 <- which(GA_4@solution == 1)
var_idx4

# 5) Population size: 100 / Cross-over: 0.5 / Mutation rate: 0.05
start_5 <- Sys.time()
GA_5 <- ga(type = "binary", fitness = fit_adj, nBits=ncol(x),
           names = colnames(x), popSize = 100, pcrossover = 0.5, 
           pmutation = 0.05, maxiter = 100, elitism = 2, seed = 29)
end_5 <- Sys.time()
time_GA[,6] <- end_5 - start_5
var_idx5 <- which(GA_5@solution == 1)
var_idx5

# 6) Population size: 50 / Cross-over: 0.7 / Mutation rate: 0.05
start_6 <- Sys.time()
GA_6 <- ga(type = "binary", fitness = fit_adj, nBits=ncol(x),
           names = colnames(x), popSize = 50, pcrossover = 0.7, 
           pmutation = 0.05, maxiter = 100, elitism = 2, seed = 29)
end_6 <- Sys.time()
time_GA[,7] <- end_6 - start_6
var_idx6 <- which(GA_6@solution == 1)
var_idx6

# 7) Population size: 100 / Cross-over: 0.7 / Mutation rate: 0.05
start_7 <- Sys.time()
GA_7 <- ga(type = "binary", fitness = fit_adj, nBits=ncol(x),
           names = colnames(x), popSize = 100, pcrossover = 0.7, 
           pmutation = 0.05, maxiter = 100, elitism = 2, seed = 29)
end_7 <- Sys.time()
time_GA[,8] <- end_7 - start_7
var_idx7 <- which(GA_7@solution == 1)
var_idx7

# 8) Max iteration: 10 / elitism: 2 / Population size: 50 / Cross-over: 0.5 / Mutation rate: 0.01
start_8 <- Sys.time()
GA_8 <- ga(type = "binary", fitness = fit_adj, nBits=ncol(x),
           names = colnames(x), popSize = 50, pcrossover = 0.5, 
           pmutation = 0.01, maxiter = 10, elitism = 2, seed = 29)
end_8 <- Sys.time()
time_GA[,9] <- end_8 - start_8
var_idx8 <- which(GA_8@solution == 1)
var_idx8

# 9) Max iteration: 3 / elitism: 2 / Population size: 50 / Cross-over: 0.5 / Mutation rate: 0.01
start_9 <- Sys.time()
GA_9 <- ga(type = "binary", fitness = fit_adj, nBits=ncol(x),
           names = colnames(x), popSize = 50, pcrossover = 0.5, 
           pmutation = 0.01, maxiter = 3, elitism = 2, seed = 29)
end_9 <- Sys.time()
time_GA[,10] <- end_9 - start_9
var_idx9 <- which(GA_9@solution == 1)
var_idx9

# 10) Max iteration: 100 / elitism: 5 / Population size: 50 / Cross-over: 0.5 / Mutation rate: 0.01
start_10 <- Sys.time()
GA_10 <- ga(type = "binary", fitness = fit_adj, nBits=ncol(x),
           names = colnames(x), popSize = 50, pcrossover = 0.5, 
           pmutation = 0.01, maxiter = 100, elitism = 5, seed = 29)
end_10 <- Sys.time()
time_GA[,11] <- end_10 - start_10
var_idx10 <- which(GA_10@solution == 1)
var_idx10

# 11) Max iteration: 100 / elitism: 10 / Population size: 50 / Cross-over: 0.5 / Mutation rate: 0.01
start_11 <- Sys.time()
GA_11 <- ga(type = "binary", fitness = fit_adj, nBits=ncol(x),
            names = colnames(x), popSize = 50, pcrossover = 0.5, 
            pmutation = 0.01, maxiter = 100, elitism = 10, seed = 29)
end_11 <- Sys.time()
time_GA[,12] <- end_11 - start_11
var_idx11 <- which(GA_11@solution == 1)
var_idx11

# 12) Max iteration: 10 / elitism: 10 / Population size: 50 / Cross-over: 0.5 / Mutation rate: 0.01
start_12 <- Sys.time()
GA_12 <- ga(type = "binary", fitness = fit_adj, nBits=ncol(x),
            names = colnames(x), popSize = 50, pcrossover = 0.5, 
            pmutation = 0.01, maxiter = 50, elitism = 10, seed = 29)
end_12 <- Sys.time()
time_GA[,13] <- end_12 - start_12
var_idx12 <- which(GA_12@solution == 1)
var_idx12

time_GA
