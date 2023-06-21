# Load package(s) ----
library(tidymodels)
library(tidyverse)
library(stacks)

tidymodels_prefer()

# load stacked models ----
load("data/results/attempt_5/nn_res.rda")
load("data/results/attempt_5/svm_rad_res.rda")
load("data/results/attempt_5/rf_res.rda")
load("data/results/attempt_5/knn_res.rda")

test <- read_csv("data/raw/test.csv")

# create stack
reg_stack <-
  stacks() %>% 
  add_candidates(nn_res) %>% 
  add_candidates(svm_rad_res) %>% 
  add_candidates(rf_res) %>% 
  add_candidates(knn_res)

# blend predictions using penalty defined above (tuning step, set seed)

# penalty values for blending (set penalty argument when blending)
blend_penalty <- c(10^(-6:-1), 0.5, 1, 1.5, 2)

set.seed(9876)

reg_stack_blend <- reg_stack %>% 
  blend_predictions(penalty = blend_penalty)

# save blended model stack for reproducibility & easy reference (Rmd report)
save(reg_stack_blend, file = "data/results/attempt_5/reg_stack_blend.rda")

# fit to ensemble to entire training set ----
reg_stacked_model_fit <- reg_stack_blend %>% 
  fit_members()

# save results (Rmd report)

save(reg_stacked_model_fit, file = "data/results/attempt_5/reg_stacked_model_fit.rda")

# explore model

stacked_pred <- test %>% 
  bind_cols(predict(reg_stacked_model_fit, test)) %>% 
  select(id, .pred) %>% 
  rename(y = .pred)

write_csv(stacked_pred, file = "data/results/attempt_5/submissions/stacked_predb.csv")



