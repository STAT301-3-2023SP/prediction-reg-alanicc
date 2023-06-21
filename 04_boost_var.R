# bt stack tuning ----

# load packages ----
library(tidyverse)
library(tidymodels)
library(stacks)


tidymodels_prefer()

# set seed
set.seed(13579)

# load data ----
load("data/reg_rec6.rda")
load("data/reg_fold.rda")

# create model ----
boost_model_stack <- boost_tree(mode = "regression", 
                                mtry = tune(), 
                                min_n = tune(), 
                                learn_rate = tune()) %>% 
  set_engine("xgboost")

# check tuning parameters
hardhat::extract_parameter_set_dials(boost_model_stack)

# create params ----
boost_params_stack <- hardhat::extract_parameter_set_dials(boost_model_stack) %>% 
  update(mtry = mtry(range = c(1, 10)),
         learn_rate = learn_rate(range = c(-5, -0.2)))

# create grid
boost_grid_stack <- grid_regular(boost_params_stack, levels = 5)

# create workflow ----
boost_workflow_stack <- workflow() %>% 
  add_model(boost_model_stack) %>% 
  add_recipe(reg_rec6)

# tune model ----
boost_res <- boost_workflow_stack %>%
  tune_grid(
    resamples = reg_fold,
    grid = boost_grid_stack,
    control = control_stack_grid()
  )

# save results
save(boost_res, file = "data/stack_model_attempt_r6/results/boost_res.rda")