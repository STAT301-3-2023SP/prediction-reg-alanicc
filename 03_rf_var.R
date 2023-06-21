# rf stack tuning ----

# load packages ----
library(tidyverse)
library(tidymodels)
library(stacks)


tidymodels_prefer()

# set.seed
set.seed(1234)

# load data
load("data/reg_rec8.rda")
load("data/reg_fold.rda")

# create model ----
rf_model_stack <- rand_forest(mode = "regression",
                              min_n = tune(),
                              mtry = tune()) %>% 
  set_engine("ranger")

# check tuning parameters
hardhat::extract_parameter_set_dials(rf_model_stack)

# create params ----
rf_params_stack <- parameters(rf_model_stack) %>% 
  update(mtry = mtry(range = c(1, 10)))

# create grid ----
rf_grid_stack <- grid_regular(rf_params_stack, levels = 5)

# create workflow ----
rf_workflow_stack <- workflow() %>% 
  add_model(rf_model_stack) %>% 
  add_recipe(reg_rec8)

# tune model ----
rf_res <- rf_workflow_stack %>%
  tune_grid(
    resamples = reg_fold,
    grid = rf_grid_stack,
    control = control_stack_grid()
  )

# save results
save(rf_res, file = "data/stack_model_attempt_r6/results/rf_res.rda")


