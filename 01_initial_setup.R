# initial setup ----

## load packages ----

library(tidyverse)
library(tidymodels)
library(naniar)
library(doMC)

tidymodels_prefer()

## load data ---

train <- read_csv("data/train.csv")
test <- read_csv("data/test.csv")

# split training data

my_split <- initial_split(train, prop = 0.75, strata = y)
train_data <- training(my_split)
test_data <- testing(my_split)


# exploration

boxplot_fun <- function(var = NULL) {
  ggplot(train_data, aes(factor(x = !!sym(var)), y = y)) +
    geom_boxplot()
}

boxplot_log_fun <- function(var = NULL) {
  ggplot(train_data, aes(factor(x = !!sym(var)), y = log(y))) +
    geom_boxplot()
}


# y distribution

ggplot(train_data, aes(x = y)) +
  geom_histogram()

MASS::boxcox(lm(y~1, train_data))

ggplot(train_data, aes(x = log(y))) +
  geom_histogram()


missing_list <- list() # making an empty list
var <- "x001"

for (var in colnames(train_data)) {
  missing_list[var] <- train_data %>% 
    select(any_of(var)) %>% 
    filter(is.na(!!sym(var))) %>%  # !!SYM IS HOW YOU TURN A CHARACTER INTO A USABLE VARIABLE
    summarize(num_missing = n())
}

missing_tibble <- enframe(unlist(missing_list))

missing_tibble %>% 
  mutate(pct = value/4034) %>% 
  arrange(desc(pct)) 


var_list <- list() # making an empty list

for (var in colnames(train_data)) {
  var_list[var] <- train_data %>% 
    select(any_of(var)) %>% 
    summarize(sd = sd(!!sym(var), na.rm = TRUE))
}

var_table <- enframe(unlist(var_list))


zero_var <- var_table %>% 
  filter(value == 0) %>% 
  pull(name) #data$variable

# update dataset ----

train_data <- train_data %>% 
  select(!all_of(zero_var))


cat_list <- list()

for (var in colnames(train_data)) {
  cat_list[var] <- train_data %>% 
    select(any_of(var)) %>% 
    summarize(unique = length(unique(!!sym(var))))
}

cat_table <- enframe(unlist(cat_list))

cat_var <- cat_table %>% 
  filter(value <= 10) %>% 
  pull(name)

boxplot_fun(var = "x025")
boxplot_log_fun(var = "x025")

map(cat_var, boxplot_fun)
map(cat_var, boxplot_log_fun)

# save results 

save(train_data, test_data, train, test, my_split, file = "data/processed/data_setup.rda")

# find . -size +100M | sed 's|^\./||g' | cat >> .gitignore; awk '!NF || !seen[$0]++' .gitignore
