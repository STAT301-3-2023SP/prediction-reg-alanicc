library(tidyverse)
library(tidymodels)
library(skimr)
library(knitr)
library(naniar)
library(ggplot2)
library(corrplot)

test <- read_csv("data/raw/test.csv")
train <- read_csv("data/raw/train.csv")

# address missingness
train_missing_table <- train %>% 
  miss_var_summary() %>% 
  kable()
train_missing_table

# reorder
y <- train$y

# remove column
mydata_minus_col <- train[, -which(names(train) == "y")]

train2 <- cbind(y, mydata_minus_col)


# find variables w/ highest correlation -- correlation matrix
cor_matrix <- cor(train2)

# extract correlations w/ y
cor_with_y <- cor_matrix["y", -1]

# sort correlations
sorted_cor <- sort(cor_with_y, decreasing = TRUE)

# print results
top_40_vars <- names(sorted_cor)[1:40]
print(top_40_vars)

top_10_vars <- names(sorted_cor)[1:10]
print(top_10_vars)

# plot results -- top 10
train2_top_10_vars <- train2[, top_10_vars]

corr_top_10_vars <- cor(train2_top_10_vars)

corrplot(corr_top_10_vars, method = "circle")

# plot results -- top 40
train2_top_40_vars <- train2[, top_40_vars]

corr_top_40_vars <- cor(train2_top_40_vars)

corrplot(corr_top_40_vars, method = "circle")


# from class ----
set.seed(1234)

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

# check for missingness
missing_list <- list()
var <- "x001"
for(var in colnames(train_data)){
  missing_list[var] <- train_data %>% 
    select(any_of(var)) %>% 
    filter(is.na(!!sym(var))) %>% 
    summarize(num_missing = n())
}

missing_tbl <- enframe(unlist(missing_list))

missing_tbl %>% 
  mutate(pct = value/4034) %>% 
  arrange(desc(pct))

# remove zero variance
var_list <- list()
for(var in colnames(train_data)){
  var_list[var] <- train_data %>% 
    select(any_of(var)) %>% 
    summarize(sd = sd(!!sym(var), na.rm = TRUE))
}

var_tbl <- enframe(unlist(var_list)) 

zero_var <- var_tbl %>% 
  filter(value == 0) %>% 
  pull(name)

# update data
train_data <- train_data %>% 
  select(!all_of(zero_var))


# misscoded categorical variables
cat_list <- list()

for(var in colnames(train_data)){
  cat_list[var] <- train_data %>% 
    select(any_of(var)) %>% 
    summarize(unique = length(unique(!!sym(var))))
}

cat_tbl <- enframe(unlist(cat_list))

cat_var <- cat_tbl %>% 
  filter(value <= 10) %>% 
  pull(name)

boxplot_fun(var = "x025")

boxplot_log_fun(var = "x025")

map(cat_var, boxplot_fun)
map(cat_var, boxplot_log_fun)

# create correlation matrix
cor_matrix <- cor(train_data)

# extract correlations
cor_with_y <- cor_matrix["y", -1]

# sort correlations
sorted_cor <- sort(cor_with_y, decreasing = TRUE)

# print top 10
top_40_vars <- names(sorted_cor)[1:40]
print(top_40_vars)

top_10_vars <- names(sorted_cor)[1:10]
print(top_10_vars)

# correlation plot w/ top 10
train_top_10_vars <- train_data[, top_10_vars]

corr_top_10_vars <- cor(train_top_10_vars)

corrplot(corr_top_10_vars, method = "circle")

# correlation plot w/ top 40
train_top_40_vars <- train_data[, top_40_vars]

corr_top_40_vars <- cor(train_top_40_vars)

corrplot(corr_top_40_vars, method = "circle")

# save results
write_rds(train_data, "data/processed/train_data.rds")
write_rds(test_data, "data/processed/test_data.rds")
