# knn var tuning ----

# load packages ----
library(tidyverse)
library(tidymodels)
library(stacks)


tidymodels_prefer()

# set seed
set.seed(13579)

# load required objects ----
load("data/reg_rec8.rda")
load("data/reg_fold.rda")

# Define model ----
#k model
knn_model_stack <- nearest_neighbor(mode = "regression", 
                                    neighbors = tune()) %>% 
  set_engine("kknn")

# check tuning parameters
hardhat::extract_parameter_set_dials(knn_model_stack)

# set-up tuning grid ----
knn_params_stack <- parameters(knn_model_stack)

# define grid
# create our grid 
knn_grid_stack <- grid_regular(knn_params_stack, levels = 5)

# workflow ----
knn_workflow_stack <- workflow() %>% 
  add_model(knn_model_stack) %>% 
  add_recipe(reg_rec8)

# Tuning/fitting ----
knn_res <- knn_workflow_stack %>%
  tune_grid(
    resamples = reg_fold,
    grid = knn_grid_stack,
    control = control_stack_grid()
  )

# Write out results & workflow
save(knn_res, file = "data/stack_model_attempt_r6/results/knn_res.rda")


