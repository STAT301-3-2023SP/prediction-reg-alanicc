# load packages 
library(tidyverse)
library(tidymodels)
library(corrplot)
library(parsnip)
library(kernlab)
library(recipes)
library(rsample)
library(naniar)
library(earth)
library(xgboost)
library(skimr)
library(knitr)
library(randomForest)
library(ggplot2)

# load data
train_data <- read_rds("data/processed/train_data.rds")
train_2 <- na.omit(train_data)
write_rds(train_2, "data/processed/train_2.rds")

#Making the folds 
reg_fold <- vfold_cv(train_data, v = 5, repeats = 3, 
                     strata = y)

reg_rec <- recipe(y ~ x146 + x102 + x014 + x619 + x687 + x651 + x696 + x755 + x569 + x543 + 
                       x749 + x591 + x427 + x561 + x670 + x669 + x430 + x638 + x118 + x724 + 
                       x265 + x661 + x105 + x731 + x609, data = train_data) %>%
  step_impute_knn(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors()) %>% 
  step_corr(all_numeric_predictors()) %>% 
  step_YeoJohnson(all_numeric_predictors()) #%>%
#step_center(all_numeric_predictors()) %>%
#step_scale(all_numeric_predictors())

reg_rec %>% 
  prep() %>% 
  bake(new_data = NULL)

reg_fold2 <- vfold_cv(train_2, v = 5, repeats = 3, 
                      strata = y)

reg_rec2 <- recipe(y ~ ., data = train_2) %>% 
  step_impute_knn(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors()) %>% 
  step_corr(all_numeric_predictors()) %>% 
  step_YeoJohnson(all_numeric_predictors())

reg_rec2 %>% 
  prep() %>% 
  bake(new_data = NULL)

reg_rec3 <- recipe(y ~ x017 + x035 + x036 + x051 + x092 + x108 + x111
                      + x114 + x118 + x142 + x161 + x215 + x223 + x233 
                      + x244 + x252 + x253 + x281 + x289 + x356 + x365
                      + x445 + x456 + x477 + x483 + x516 + x532 + x538
                      + x548 + x553 + x567 + x581 + x598 + x623 + x626
                      + x660 + x675 + x702 + x721 + x741 + x721 + x741
                      + x752 + x753, data = train_2) %>% 
  step_impute_knn(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors()) %>% 
  step_corr(all_numeric_predictors()) %>% 
  step_YeoJohnson(all_numeric_predictors())

reg_rec3 %>% 
  prep() %>% 
  bake(new_data = NULL)


reg_rec4 <- recipe(y ~ x017 + x035 + x036 + x051 + x092 + x108 + x111
                      + x114 + x118 + x142 + x161 + x215 + x223 + x233 
                      + x244 + x252 + x253 + x281 + x289 + x356 + x365
                      + x445 + x456 + x477 + x483 + x516 + x532 + x538
                      + x548 + x553 + x567 + x581 + x598 + x623 + x626
                      + x660 + x675 + x702 + x721 + x741 + x721 + x741
                      + x752 + x753, x146 + x102 + x014 + x619 + x687
                      + x651 + x696 + x755 + x569 + x543 + 
                        x749 + x591 + x427 + x561 + x670 + 
                        x669 + x430 + x638 + x118 + x724 + 
                        x265 + x661 + x105 + x731 + x609, data = train_2) %>% 
  step_impute_knn(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors()) %>% 
  step_corr(all_numeric_predictors()) 

reg_rec4 %>% 
  prep() %>% 
  bake(new_data = NULL)

#making a recipe with the top 10 variables from train_2

reg_rec5 <- recipe(y ~ x102 + x146 + x619 + x687 + x696 + x651
                      + x118 + x724 + x014, data = train_2) %>%
  step_impute_knn(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors()) %>% 
  step_YeoJohnson(all_numeric_predictors())

## BEST RECIPE ###############################################################
#making a recipe with the top 25 from the RF Model

reg_rec6 <- recipe(y ~ x146 + x102 + x105 + x702 + x073 + x670
                      + x753 + x118 + x724 + x147 + x561 + x014 +
                        x687 + x619 + x755 + x420 + x548 + x744 + x355
                      + x589 + x366 + x253 + x725 + x748 + x488 , data = train_2) %>%
  step_impute_knn(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors()) %>% 
  step_YeoJohnson(all_numeric_predictors())

#saving for stacked model

# objects required for tuning
# data objects
save(reg_fold, file = "data/reg_fold.rda")
# model info object
save(reg_rec6, train_2, file = "data/reg_rec6.rda")

#############################################################################

#making a recipe with the top 25 from the RF Model

reg_rec7 <- recipe(y ~ x146 + x102 + x105 + x702 + x073 + x670
                      + x753 + x118 + x724 + x147 + x561 + x014 +
                        x687 + x619 + x755 + x420 + x548 + x744 + x355
                      + x589 + x366 + x253 + x725 + x748 + x488 + x223
                      + x037 + x257 + x051 + x361 + x477 + x203 + x508
                      + x466 + x114, data = train_2) %>%
  step_impute_knn(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors()) %>% 
  step_YeoJohnson(all_numeric_predictors())

######################################################################

#making a recipe with the top 20 from the RF Model

reg_rec8 <- recipe(y ~ x146 + x102 + x105 + x702 + x073 + x670
                      + x753 + x118 + x724 + x147 + x561 + x014 +
                        x687 + x619 + x755 + x420 + x548 + x744 + x355
                      + x589, data = train_2) %>%
  step_impute_knn(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors()) %>% 
  step_YeoJohnson(all_numeric_predictors())


#saving for stacked model

# model info object
save(reg_rec8, train_2, file = "data/reg_rec8.rda")


## random forest #########################################################
rf_model <- rand_forest(mode = "regression",
                        min_n = tune(),
                        mtry = tune()) %>% 
  set_engine("ranger")


rf_params <- parameters(rf_model) %>% 
  update(mtry = mtry(range = c(1, 10)))

## create our grid 
rf_grid <- grid_regular(rf_params, levels = 5)

#attempt 1
## define workflows
rf_workflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(reg_rec)

# attempt 2

rf_model2 <- rand_forest(mode = "regression", 
                         min_n = tune(), 
                         mtry = tune()) %>% 
  set_engine("ranger")

rf_params2 <- parameters(rf_model2) %>% 
  update(mtry = mtry(range = c(1,10)))

rf_grid2 <- grid_regular(rf_params2, levels = 5)

rf_workflow2 <- workflow() %>% 
  add_model(rf_model2) %>% 
  add_recipe(reg_rec3)

#attempt 3
rf_workflow3 <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(reg_rec4)

#attempt 4 

rf_workflow4 <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(reg_rec5)

#attempt 5
rf_workflow5 <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(reg_rec6)

#attempt 6
rf_workflow6 <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(reg_rec7)


## save necessary parts to run in separate script 
save(rf_workflow, rf_grid, reg_fold, file = "data/results/attempt_1/info_rf.rda")
save(rf_workflow2, train_2, rf_grid2, reg_fold2, file = "data/results/attempt_2/info_rf.rda")
save(rf_workflow3, rf_grid, reg_fold, file = "data/results/attempt_3/info_rf.rda")
save(rf_workflow4, rf_grid, reg_fold, file = "data/results/attempt_4/info_rf.rda")
save(rf_workflow5, rf_grid, reg_fold, file = "data/results/attempt_5/info_rf.rda")
save(rf_workflow6, rf_grid, reg_fold, file = "data/results/attempt_5/info_rfc.rda")
## boosted ###############################################################

#boost model
boost_model <- boost_tree(mode = "regression", 
                          mtry = tune(), 
                          min_n = tune(), 
                          learn_rate = tune()) %>% 
  set_engine("xgboost") #%>% 
# set_args(nrounds = tune(), 
#          colsample_bytree = tune(),
#          max_depth = tune(),
#          lambda = tune(), 
#          gamma = tune(), 
#          subsample = tune())

boost_params <- parameters(boost_model) %>% 
  update(mtry = mtry(range = c(1, 10)),
         learn_rate = learn_rate(range = c(-5, -0.2)))


## create our grid 
boost_grid <- grid_regular(boost_params, levels = 5)

#attempt 1-----
## define workflow 
boost_workflow <- workflow() %>% 
  add_model(boost_model) %>% 
  add_recipe(reg_rec)

#attempt 2-------
## define workflow 
boost_workflow2 <- workflow() %>% 
  add_model(boost_model) %>% 
  add_recipe(reg_rec3)

#attempt w/ r6-------

boost_workflow6 <- workflow() %>% 
  add_model(boost_model) %>% 
  add_recipe(reg_rec6)

## save necessary parts to run in separate script 
save(boost_workflow, boost_grid, reg_fold, file = "data/results/attempt_1/info_boost.rda")
save(boost_workflow2, boost_grid, reg_fold, file = "data/results/attempt_2/info_boost.rda")
save(boost_workflow6, boost_grid, reg_fold, file = "data/results/attempt_3/info_boost.rda")

## knn ###############################################################

#k model
k_model <- nearest_neighbor(mode = "regression", 
                            neighbors = tune()) %>% 
  set_engine("kknn")

knn_params <- parameters(k_model)

# create our grid 
knn_grid <- grid_regular(knn_params, levels = 5)

#attempt 1-----
# define workflow 
knn_workflow <- workflow() %>% 
  add_model(k_model) %>% 
  add_recipe(reg_rec)

#attempt 2-----
# define workflow 
knn_workflow2 <- workflow() %>% 
  add_model(k_model) %>% 
  add_recipe(reg_rec3)

#attempt w r6-----
# define workflow 
knn_workflow6 <- workflow() %>% 
  add_model(k_model) %>% 
  add_recipe(reg_rec6)

# save necessary parts to run in separate script 
save(knn_workflow, knn_grid, reg_fold, file = "data/results/attempt_1/info_knn.rda")
save(knn_workflow2, knn_grid, reg_fold, file = "data/results/attempt_2/info_knn.rda")
save(knn_workflow6, knn_grid, reg_fold, file = "data/results/attempt_5/info_knn.rda")
## elastic ###############################################################

#elastic model
en_model <- linear_reg(mode = "regression", penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet")

en_params <- extract_parameter_set_dials(en_model)

en_grid <- grid_regular(en_params, levels = 5)

en_workflow <- workflow() %>% 
  add_model(en_model) %>% 
  add_recipe(reg_rec)


# with r6-----------

en_workflow6 <- workflow() %>% 
  add_model(en_model) %>% 
  add_recipe(reg_rec6)


#save necessary parts to run in a separate script 
save(en_workflow, en_grid, reg_fold, file = "data/results/attempt_1/info_en.rda")
save(en_workflow6, en_grid, reg_fold, file = "data/results/attempt_5/info_en.rda")

## mars #################################################################

mars_model <- mars(
  mode = "regression", 
  num_terms = tune(),
  prod_degree = tune()
) %>%
  set_engine("earth")

mars_params <- extract_parameter_set_dials(mars_model)

mars_grid <- grid_regular(mars_params, levels = 5)

#attempt 1-----

mars_workflow <- workflow() %>% 
  add_model(mars_model) %>% 
  add_recipe(reg_rec)

#attempt 2-----
mars_workflow2 <- workflow() %>% 
  add_model(mars_model) %>% 
  add_recipe(reg_rec3)

#attempt r6-----
mars_workflow6 <- workflow() %>% 
  add_model(mars_model) %>% 
  add_recipe(reg_rec6)

#save necessary parts to run in a separate script 
save(mars_workflow, mars_grid, reg_fold, file = "data/results/attempt_1/info_mars.rda")
save(mars_workflow2, mars_grid, reg_fold, file = "data/results/attempt_2/info_mars.rda")
save(mars_workflow6, mars_grid, reg_fold, file = "data/results/attempt_5/info_mars.rda")
## nn #################################################################

nn_model <- mlp(
  mode = "regression", 
  hidden_units = tune(),
  penalty = tune()
) %>%
  set_engine("nnet")

nn_params <- extract_parameter_set_dials(nn_model)

nn_grid <- grid_regular(nn_params, levels = 5)

#attempt 1-----
nn_workflow <- workflow() %>% 
  add_model(nn_model) %>% 
  add_recipe(reg_rec)

#attempt 2-----
nn_workflow2 <- workflow() %>% 
  add_model(nn_model) %>% 
  add_recipe(reg_rec3)

#attempt r6----

nn_workflow6 <- workflow() %>% 
  add_model(nn_model) %>% 
  add_recipe(reg_rec6)

#save necessary parts to run in a separate script 
save(nn_workflow, nn_grid, reg_fold, file = "data/results/attempt_1/info_nn.rda")
save(nn_workflow6, nn_grid, reg_fold, file = "data/results/attempt_5/info_nn.rda")
## svm poly #################################################################

svm_poly_model <- svm_poly(
  mode = "regression",
  cost = tune(),
  degree = tune(),
  scale_factor = tune()
) %>%
  set_engine("kernlab")

svm_poly_params <- extract_parameter_set_dials(svm_poly_model)

svm_poly_grid <- grid_regular(svm_poly_params, levels = 5)

#attempt 1-----

svm_poly_workflow <- workflow() %>% 
  add_model(svm_poly_model) %>% 
  add_recipe(reg_rec)

#attempt 2-----

svm_poly_workflow2 <- workflow() %>% 
  add_model(svm_poly_model) %>% 
  add_recipe(reg_rec3)

#attempt 6-----

svm_poly_workflow6 <- workflow() %>% 
  add_model(svm_poly_model) %>% 
  add_recipe(reg_rec6)

#save necessary parts to run in a separate script 
save(svm_poly_workflow, svm_poly_grid, reg_fold, file = "data/results/attempt_1/info_svm_poly.rda")
save(svm_poly_workflow2, svm_poly_grid, reg_fold, file = "data/results/attempt_2/info_svm_poly.rda")
save(svm_poly_workflow6, svm_poly_grid, reg_fold, file = "data/results/attempt_5/info_svm_poly.rda")
## svm rad #################################################################

svm_rad_model <- svm_rbf(
  mode = "regression", 
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab")

svm_rad_params <- parameters(svm_rad_model)

svm_rad_grid <- grid_regular(svm_rad_params, levels = 5)

#attempt 1-----

svm_rad_workflow <- workflow() %>% 
  add_model(svm_rad_model) %>% 
  add_recipe(reg_rec)

#attempt 2-----

svm_rad_workflow2 <- workflow() %>% 
  add_model(svm_rad_model) %>% 
  add_recipe(reg_rec3)

#attempt r6-----

svm_rad_workflow6 <- workflow() %>% 
  add_model(svm_rad_model) %>% 
  add_recipe(reg_rec6)


#save necessary parts to run in a separate script 
save(svm_rad_workflow, svm_rad_grid, reg_fold, file = "data/results/attempt_1/info_svm_rad.rda")
save(svm_rad_workflow2, svm_rad_grid, reg_fold, file = "data/results/attempt_2/info_svm_rad.rda")
save(svm_rad_workflow6, svm_rad_grid, reg_fold, file = "data/results/attempt_5/info_svm_rad.rda")

## MARS to find most important #############################################

reg_model <- earth(y ~ .,  data = train_2)
print(reg_model)
summary(reg_model)


#other mars fit 

mars_vars <- mars_fit %>%
  extract_fit_parsnip() %>%
  vip::vi()

## Random Forest to find most important #####################################
library(randomForest)
library(caret)

# Build a random forest model
rf_import <- randomForest(y ~ ., data = train_2, ntree = 1000)

# Get variable importance
var_importance <- varImp(rf_import)

# Extract variable names
var_names <- rownames(var_importance)

# Sort variable names based on importance
sorted_var_names <- var_names[order(var_importance$Overall, decreasing = TRUE)]

# Print sorted variable names
print(sorted_var_names)



# other 

set.seed(4543)
reg_rf <- randomForest(y ~ ., data=train_2, ntree=1000,
                       keep.forest=FALSE, importance=TRUE)
importance(reg_rf)
importance(reg_rf, type=1)

## Corr graph with train_2 ######################################################
cor_matrix <- cor(train_2)

# Extract the correlations between "y" and all other variables
cor_with_y <- cor_matrix["y", -1]

# Sort the correlations in descending order
sorted_cor <- sort(cor_with_y, decreasing = TRUE)

# Print the top 10 and 40 variables with the highest correlation to "y"

top_10_vars <- names(sorted_cor)[1:10]
print(top_10_vars)

#Correlation plot with top 10:

train_top_10_vars <- train_2[, top_10_vars]

corr_top_10_vars <- cor(train_top_10_vars)

corrplot(corr_top_10_vars, method = "circle")



