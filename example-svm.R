# Example of tidymodels flow for supervised ML
# 26-09-2021

# Packages ----
library(tidyverse)
library(tidymodels)
library(themis)

# Data
data(income, package = "kernlab")

# Explore ----
glimpse(income)
fct_count(income$INCOME)

## Change income level
income <- 
  income %>% 
  mutate(INCOME2 = fct_collapse(INCOME, 
                                lowest = c("-10.000)", "[10.000-15.000)"),
                                lower_middle = c("[15.000-20.000)", "[20.000-25.000)"), 
                                middle = c("[25.000-30.000)", "[30.000-40.000)"), 
                                upper_middle = c("[40.000-50.000)", "[50.000-75.000)"), 
                                highest = "[75.000-"), 
         INCOME2 = factor(INCOME2, ordered = F)) %>% 
  select(-INCOME) %>% 
  janitor::clean_names()

## Check for missing data 
DataExplorer::profile_missing(income) 

# Split data ----
set.seed(2021)

dat_index <- initial_split(income, strata = income2)
dat_train <- training(dat_index)
dat_test <- testing(dat_index)

# Explore training data
DataExplorer::plot_missing(dat_train) # > 20%, not good to impute
skimr::skim(dat_train)
table(dat_train$income2) # imbalance

# Recipe + preprocessing ----
dat_rec <- 
  recipe(income2 ~., data = dat_train) %>% 
  step_impute_knn(all_predictors(), -c(sex, age, dual_incomes, under18)) %>% 
  step_nzv(all_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_upsample(income2, seed = 2021)

dat_processed <- 
  dat_rec %>% 
  prep() %>% 
  bake(new_data = NULL) # or juice()

## See preprocessed data
table(dat_processed$income2)
DataExplorer::profile_missing(dat_processed)

# Resample ----
set.seed(2021)
dat_cv <- vfold_cv(dat_processed, v = 10, repeats = 1, strata = income2)
dat_boot <- bootstraps(dat_processed, strata = income2)

# Specify model ----
svm_model <- 
  svm_poly(cost = tune(),
           degree = tune(), 
           scale_factor = tune()) %>% 
  set_engine("kernlab") %>% 
  set_mode("classification")

show_engines("rand_forest")
show_model_info("rand_forest")

# Specify workflow
svm_wf <- 
  workflow() %>% 
  add_recipe(dat_rec) %>% 
  add_model(svm_model)

# Tuning parameters ----
# library(doParallel) # Do not run ! (~1 hour run time with 2 cores)
# cl <- makePSOCKcluster(detectCores() - 1)
# registerDoParallel(cl)
# set.seed(2021)
# 
# ctrl <- control_resamples(save_pred = TRUE, verbose = T)
# svm_tune <-  
#   svm_wf %>% 
#   tune_grid(resamples = dat_cv,
#             metrics = metric_set(accuracy, pr_auc, roc_auc),
#             control = ctrl,
#             grid = 10)
# 
# stopCluster(cl)

# HERE!!! TRY save as rds <------ HERE !!! ----
load("svm_tune.rda")

## Explore result of tuning
svm_tune %>% 
  collect_metrics()

svm_tune %>% 
  show_best("accuracy")

svm_tune %>% autoplot()

svm_tune %>% 
  collect_predictions() %>% 
  conf_mat(income2, .pred_class)

svm_tune %>% 
  collect_predictions() %>% 
  group_by(id) %>% 
  roc_curve(truth = income2, estimate = .pred_lowest:.pred_highest) %>% 
  autoplot()

svm_tune %>% 
  collect_metrics() %>% 
  autoplot() # not working with rda

# Finalize workflow ----
svm_best <- svm_tune %>% select_best("roc_auc")

svm_wf_final <- 
  svm_wf %>% 
  finalize_workflow(svm_best)

# Fit and predict on testing data ----

## 1) 1st Method ----
### Fit on entire training data 
svm_train <- # this should be save for model deployment (readr::write_rds())
  svm_wf_final %>% 
  fit(data = dat_processed)

### Fit on processed testing data
dat_test_processed <- 
  dat_rec %>% 
  prep() %>% 
  bake(new_data = dat_test)

dat_pred <- 
  dat_test_processed %>% 
  bind_cols(predict(svm_train, new_data = dat_test_processed)) %>% 
  bind_cols(predict(svm_train, new_data = dat_test_processed, type = "prob"))

# w/o workflow for model deployment
predict(svm_train, 
        new_data = dat_rec %>% 
          prep() %>% 
          bake(new_data = dat_train[3:4, ])) # lower_middle lower_middle

## 2) 2nd method - workflow ----
#### Fit on entire training data then testing data
dat_fit <- 
  svm_wf_final %>% 
  last_fit(dat_index)

# workflow below should be saved for model deployment (readr::write_rds())
wf_saved <- dat_fit$.workflow[[1]] 
predict(wf_saved, dat_train[3:4, ]) # lower_middle lower_middle

# Performance metrics ----

## W/o workflow
dat_pred %>% 
  accuracy(income2, .pred_class)

dat_pred %>% 
  roc_auc(income2, .pred_lowest:.pred_highest)

dat_fit %>% 
  collect_metrics()

## With workflow
dat_fit %>% 
  collect_predictions() %>% 
  roc_curve(income2, .pred_lowest:.pred_highest) %>% 
  autoplot()

dat_fit %>% 
  collect_predictions() %>% 
  roc_curve(income2, .pred_lowest:.pred_highest) %>% 
  ggplot(aes(1-specificity, sensitivity, color = .level))  +
  geom_line() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  theme_bw()

# Model explainer ----
library(DALEX)
library(DALEXtra)

explainer_svm <- # ~1 min run time
  explain_tidymodels(
    wf_saved, 
    data = dat_processed %>% select(-income2), 
    y = as.numeric(dat_processed$income2),
    label = "svm_poly",
    verbose = T
  )

## Global explanation - variable importance
vip_svm <- model_parts(explainer_svm) # ~6 min run time

source("var_imp.R")
ggplot_imp(vip_svm)
