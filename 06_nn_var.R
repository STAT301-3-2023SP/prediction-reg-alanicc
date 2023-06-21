# nn var tuning ----

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
nn_model_stack <- mlp(
  mode = "regression", 
  hidden_units = tune(),
  penalty = tune()
) %>%
  set_engine("nnet")

# check tuning parameters
hardhat::extract_parameter_set_dials(nn_model_stack)

# create params ----
nn_params_stack <- extract_parameter_set_dials(nn_model_stack)

#create grid
nn_grid_stack <- grid_regular(nn_params_stack, levels = 5)

#create workflow ----
nn_workflow_stack <- workflow() %>% 
  add_model(nn_model_stack) %>% 
  add_recipe(reg_rec8)

# tune model ----
nn_res <- nn_workflow_stack %>%
  tune_grid(
    resamples = reg_fold,
    grid = nn_grid_stack,
    control = control_stack_grid()
  )

# save results
save(nn_res, file = "data/stack_model_attempt_r6/results/nn_res.rda")


