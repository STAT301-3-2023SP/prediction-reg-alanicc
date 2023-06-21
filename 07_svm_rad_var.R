# svm rad tuning ----

# load packages ----
library(tidyverse)
library(tidymodels)
library(stacks)

tidymodels_prefer()

# set seed
set.seed(1234)

# load data ----
load("data/reg_rec8.rda")
load("data/reg_fold.rda")

# create model ----
svm_rad_model_stack <- svm_rbf(
  mode = "regression", 
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab")

# check tuning parameters
hardhat::extract_parameter_set_dials(svm_rad_model_stack)

# set-up tuning grid ----
svm_rad_params_stack <- parameters(svm_rad_model_stack)

# define grid
svm_rad_grid_stack <- grid_regular(svm_rad_params_stack, levels = 5)

# workflow ----
svm_rad_workflow_stack <- workflow() %>% 
  add_model(svm_rad_model_stack) %>% 
  add_recipe(reg_rec8)

# tune model ----
svm_rad_res <- svm_rad_workflow_stack %>%
  tune_grid(
    resamples = reg_fold,
    grid = svm_rad_grid_stack,
    control = control_stack_grid()
  )

# Write out results & workflow
save(svm_rad_res, file = "data/stack_model_attempt_r6/results/svm_rad_res.rda")

