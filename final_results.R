#get final model results 
library(tidymodels)
library(tidyverse)
library(corrplot)

tidymodels_prefer()

test_data <- read_rds("data/processed/test_data.rds")
test <- read_csv("data/raw/test.csv")

result_files <- list.files("data/data/results/attempt_1/", "*.rda", full.names = TRUE)

for(i in result_files){
  load(i)
}

load("data/results/attempt_2/info_rf.rda")


tuned_rf <- load("data/results/attempt_1/info_rf.rda")
tuned_en <- load("data/results/attempt_1/info_en.rda")
tuned_bt <- load("data/results/attempt_1/info_boost.rda")
tuned_knn <- load("data/results/attempt_1/info_knn.rda")
tuned_mars <- load("data/results/attempt_1/info_mars.rda")
tuned_nn <- load("data/results/attempt_1/info_nn.rda")
tuned_svm_poly <- load("data/results/info_1/info_svm_poly.rda")
tuned_svm_rad <- load("data/results/info_1/info_svm_rad.rda")

tuned_rf2 <- load("data/results/attempt_2/info_rf.rda")
tuned_bt2 <- load("data/results/attempt_2/info_boost.rda")
tuned_knn2 <- load("data/results/attempt_2/info_knn.rda")
tuned_mars2 <- load("data/results/attempt_2/info_mars.rda")
tuned_nn2 <- load("data/results/attempt_2/info_nn.rda")
tuned_svm_poly2 <- load("data/results/attempt_2/info_svm_poly.rda")
tuned_svm_rad2 <- load("data/results/attempt_2/info_svm_rad.rda")

result_files <- list.files("data/results/attempt_2/", "*.rda", full.names = TRUE)
for(i in result_files){
  load(i)
}



#################################################################
#organize results to find best overall

#Individual model results
#Put in the appendix
autoplot(tuned_knn, metric = "rmse")

tuned_knn %>% 
  show_best(metric = "rmse")

####################################
#Results
best_rf <- show_best(tuned_rf, metric = "rmse")[1,]
best_rf
best_bt <- show_best(tuned_bt, metric = "rmse")[1,]
best_bt
best_knn <- show_best(tuned_knn, metric = "rmse")[1,]
best_knn
best_mars <- show_best(tuned_knn, metric = "rmse")[1,]
best_mars
best_nn <- show_best(tuned_knn, metric = "rmse")[1,]
best_nn
best_svm_poly <- show_best(tuned_knn, metric = "rmse")[1,]
best_svm_poly
best_svm_rad <- show_best(tuned_svm_rad, metric = "rmse")[1,]
best_svm_rad
best_en <- show_best(tuned_en, metric = "rmse")[1,]
best_en

##attempt 1 #####################################################
bt_workflow_tuned <- boost_workflow %>%
  finalize_workflow(select_best(tuned_bt, metric = "rmse"))

# Fit entire training data set to workflow
bt_results <- fit(bt_workflow_tuned, train_data)

bt_pred <- test %>%
  bind_cols(predict(bt_results, test)) %>%
  select(id, .pred) %>%
  rename(y = .pred)

write_csv(bt_pred, file = "data/results/attempt_1/a1_submissions/bt_pred.csv")


#attempt 2 #################################################

#results
#Results
best_rf <- show_best(tuned_rf2, metric = "rmse")[1,]
best_rf #8.59
best_bt <- show_best(tuned_bt2, metric = "rmse")[1,]
best_bt #10.6
best_knn <- show_best(tuned_knn2, metric = "rmse")[1,]
best_knn #10.8
best_mars <- show_best(tuned_mars2, metric = "rmse")[1,]
best_mars #10.7
best_nn <- show_best(tuned_nn2, metric = "rmse")[1,]
best_nn #10.6
best_svm_poly <- show_best(tuned_knn2, metric = "rmse")[1,]
best_svm_poly
best_svm_rad <- show_best(tuned_svm_rad2, metric = "rmse")[1,]
best_svm_rad #10.2

#making the csv for submission
rf_workflow_tuned2 <- rf_workflow2 %>%
  finalize_workflow(select_best(tuned_rf2, metric = "rmse"))

# Fit entire training data set to workflow
rf_results2 <- fit(rf_workflow_tuned2, train2)

rf_pred2 <- test %>%
  bind_cols(predict(rf_results2, test)) %>%
  select(id, .pred) %>%
  rename(y = .pred)

write_csv(rf_pred2, file = "data/results/attempt_2/a2_submissions/rf_pred.csv")

#attempt 3 #################################################

load("data/results/attempt_3/info_rf.rda")
tuned_rf3 <- load("data/results/attempt_3/tuned_rf.rds")


#making the csv for submission
rf_workflow_tuned3 <- rf_workflow3 %>%
  finalize_workflow(select_best(tuned_rf3, metric = "rmse"))

# Fit entire training data set to workflow
rf_results3 <- fit(rf_workflow_tuned3, train2)

rf_pred3 <- test %>%
  bind_cols(predict(rf_results3, test)) %>%
  select(id, .pred) %>%
  rename(y = .pred)

write_csv(rf_pred3, file = "data/results/attempt_3/a3_submissions/rf_pred.csv")

#attempt 4 #################################################

load("data/results/attempt_4/info_rf.rda")
tuned_rf4 <- load("data/results/attempt_4/tuned_rf.rds")

best_rf4 <- show_best(tuned_rf4, metric = "rmse")[1,]
best_rf4 #8.59

#making the csv for submission
rf_workflow_tuned4 <- rf_workflow4 %>%
  finalize_workflow(select_best(tuned_rf4, metric = "rmse"))

# Fit entire training data set to workflow
rf_results4 <- fit(rf_workflow_tuned4, train2)

rf_pred4 <- test %>%
  bind_cols(predict(rf_results4, test)) %>%
  select(id, .pred) %>%
  rename(y = .pred)

write_csv(rf_pred4, file = "data/results/attempt_4/a4_submissions/rf_pred.csv")
write_csv(rf_pred4, file = "data/results/attempt_4/a4_submissions/rf_pred2.csv")

#attempt 5 #################################################

load("data/results/attempt_5/info_rf.rda")
tuned_rf5 <- load("data/results/attempt_5/tuned_rf.rds")

best_rf5 <- show_best(tuned_rf5, metric = "rmse")[1,]
best_rf5 #8.59

#making the csv for submission
rf_workflow_tuned5 <- rf_workflow5 %>%
  finalize_workflow(select_best(tuned_rf5, metric = "rmse"))

# Fit entire training data set to workflow
rf_results5 <- fit(rf_workflow_tuned5, train2)

rf_pred5 <- test %>%
  bind_cols(predict(rf_results5, test)) %>%
  select(id, .pred) %>%
  rename(y = .pred)

write_csv(rf_pred5, file = "data/results/attempt_5/a5_submissions/rf_pred.csv")

#rf2 
load("data/results/attempt_5/info_rf.rda")
tuned_rf5b <- load("data/results/attempt_5/tuned_rf2.rds")

best_rf5b <- show_best(tuned_rf5b, metric = "rmse")[1,]
best_rf5b #8.59

#making the csv for submission
rf_workflow_tuned5b <- rf_workflow5 %>%
  finalize_workflow(select_best(tuned_rf5b, metric = "rmse"))

# Fit entire training data set to workflow
rf_results5b <- fit(rf_workflow_tuned5b, train2)

rf_pred5b <- test %>%
  bind_cols(predict(rf_results5b, test)) %>%
  select(id, .pred) %>%
  rename(y = .pred)

write_csv(rf_pred5b, file = "data/results/attempt_5/a5_submissions/rf_predb.csv")

#svm_rad
load("data/results/attempt_5/info_svm_rad.rda")
tuned_svm_rad <- load("data/results/attempt_5/tuned_svm_rad.rds")

best_svm_rad <- show_best(tuned_svm_rad, metric = "rmse")[1,]
best_svm_rad #8.59

#making the csv for submission
svm_rad_workflow_tuned <- svm_rad_workflow6 %>%
  finalize_workflow(select_best(tuned_svm_rad, metric = "rmse"))

# Fit entire training data set to workflow
svm_rad_results <- fit(svm_rad_workflow_tuned, train2)

svm_rad_pred <- test %>%
  bind_cols(predict(svm_rad_results, test)) %>%
  select(id, .pred) %>%
  rename(y = .pred)

write_csv(svm_rad_pred, file = "data/results/attempt_5/a5_submissions/svm_rad_pred.csv")

#rfc
load("data/results/attempt_5/info_rfc.rda")
tuned_rf5c <- load("data/results/attempt_5/tuned_rfc.rds")

best_rf5c <- show_best(tuned_rf5c, metric = "rmse")[1,]
best_rf5c #8.59

#making the csv for submission
rf_workflow_tuned5c <- rf_workflow6 %>%
  finalize_workflow(select_best(tuned_rf5c, metric = "rmse"))

# Fit entire training data set to workflow
rf_results5c <- fit(rf_workflow_tuned5c, train2)

rf_pred5c <- test %>%
  bind_cols(predict(rf_results5c, test)) %>%
  select(id, .pred) %>%
  rename(y = .pred)

write_csv(rf_pred5c, file = "data/results/attempt_5/a5_submissions/rf_predc.csv")

load("data/results/attempt_5/info_rfc.rda")
tuned_rf5c <- load("data/results/attempt_5/tuned_rfc.rds")

best_rf5c <- show_best(tuned_rf5c, metric = "rmse")[1,]
best_rf5c #8.59

#making the csv for submission
rf_workflow_tuned5c <- rf_workflow6 %>%
  finalize_workflow(select_best(tuned_rf5c, metric = "rmse"))

# Fit entire training data set to workflow
rf_results5c <- fit(rf_workflow_tuned5c, train2)

rf_pred5c <- test %>%
  bind_cols(predict(rf_results5c, test)) %>%
  select(id, .pred) %>%
  rename(y = .pred)

write_csv(rf_pred5c, file = "data/results/attempt_5/a5_submissions/rf_predc.csv")
#RESULTS FOR RECIPE 6 (part of a5)--------------------------------------

#RESULTS FOR RECIPE 6 (part of a5)--------------------------------------

load("data/results/attempt_5/info_boost.rda")
load("data/results/attempt_5/info_knn.rda")
load("data/results/attempt_5/info_mars.rda")
load("data/results/attempt_5/info_nn.rda")
load("data/results/attempt_5/info_svm_rad.rda")
load("data/results/attempt_5/info_rf.rda")

tuned_rf5 <- read_rds("data/results/attempt_5/tuned_rf.rds")
tuned_bt5 <- read_rds("data/results/attempt_5/tuned_bt.rds")
tuned_knn5 <- read_rds("data/results/attempt_5/tuned_knn.rds")
tuned_mars5 <- read_rds("data/results/attempt_5/tuned_mars.rds")
tuned_nn5 <- read_rds("data/results/attempt_5/tuned_nn.rds")
tuned_svm_rad5 <- read_rds("data/results/attempt_5/tuned_svm_rad.rds")

best_rf5 <- show_best(tuned_rf5, metric = "rmse")[1,]
best_rf5 #9.76

best_bt5 <- show_best(tuned_bt5, metric = "rmse")[1,]
best_bt5 #10.3

best_knn5 <- show_best(tuned_knn5, metric = "rmse")[1,]
best_knn5 #10.2

best_mars5 <- show_best(tuned_mars5, metric = "rmse")[1,]
best_mars5 #10.6

best_nn5 <- show_best(tuned_nn5, metric = "rmse")[1,]
best_nn5 #9.85

best_svm_rad5 <- show_best(tuned_svm_rad5, metric = "rmse")[1,]
best_svm_rad5 #9.99








