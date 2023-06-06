# lasso var select

# load packages
library(tidyverse)
library(doMC)
library(tidymodels)

tidymodels_prefer()

# load data
load("data/processed/data_setup.rda")
# load("results/kitchen_sink.rda") ##COMPLETE


# set seed
set.seed(1234)

# create folds
fold_var_select <- vfold_cv(train_data, folds = 5, repeats = 1, strata = y)


## tune model
lasso_mod <- linear_reg(
  mode = "regression",
  penalty = tune(),
  mixture = 1) %>% 
  set_engine("glmnet")

# set params
lasso_params <- extract_parameter_set_dials(lasso_mod) %>% 
  update(penalty = penalty(range = c(0.01,0.1), trans = NULL))

# create grid
lasso_grid <- grid_regular(lasso_params, levels = 5)

# create workflow
lasso_wkflow <- workflow() %>% 
  add_model(lasso_mod) %>% 
  add_recipe(kitchen_sink)


registerDoMC(cores = 8)

# tune model
lasso_tune <- tune_grid(
  lasso_wkflow,
  resamples = fold_var_select,
  grid = lasso_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = TRUE,
                         parallel_over = "everything"),
  metric = metric_set(rmse)
)

# save results
save(lasso_tune,
     file = "results/lasso_var_tune.rda")



# load data
load("results/lasso_var_tune.rda")
load("results/lasso_var_specs.rda")


# final workflow
lasso_workflow_final_1 <- lasso_wkflow %>%
  finalize_workflow(select_best(lasso_tune, metric = "rmse"))


# fit final workflow to training set
lasso_fit_1 <- fit(lasso_workflow_final_1, data = train_data)


lasso_coef <- lasso_fit_1 %>%
  tidy() %>%
  view()


lasso_vars_1 <- lasso_fit_1 %>%
  tidy() %>%
  filter(estimate != 0, term != "(Intercept)" ) %>%
  pull(term)

print(lasso_vars_1)

train_clean_1 <- train_data %>%
  select(any_of(lasso_vars_1))

train_clean_2 <- train_data %>%
  select("x006", "x014", "x017", "x022", "x045", "y", "id")

train_cleanest <- merge(x = train_clean-1, y = train_clean_2)

write_rds(train_cleanest, file = "data/processed/train_cleanest.rds")

save(lasso_fit_1, lasso_vars_1, file = "results/lasso_var_specs.rda")

save(train_clean_1, file = "results/train_clean-1.rda")
