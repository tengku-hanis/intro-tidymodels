# Example of tidymodels with workflows for supervised ML
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
  step_dummy(all_nominal_predictors()) %>% 
  step_upsample(income2)  # has built in set seed 

dat_processed <- 
  dat_rec %>% 
  prep() %>% 
  bake(new_data = NULL) # or juice()

## See preprocessed data
table(dat_processed$income2)
DataExplorer::profile_missing(dat_processed)

# Resample ----
set.seed(2021)
dat_cv <- vfold_cv(dat_train, v = 10, repeats = 1, strata = income2)
dat_boot <- bootstraps(dat_train, strata = income2)

# Specify model ----
lasso_model <- 
  multinom_reg(penalty = tune(), 
               mixture = 1) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification")

show_engines("multinom_reg")
show_model_info("multinom_reg")

# Specify workflow
lasso_wf <- 
  workflow() %>% 
  add_recipe(dat_rec) %>% 
  add_model(lasso_model)

# Tuning parameters ----
# library(doParallel) 
# cl <- makePSOCKcluster(detectCores() - 1)
# registerDoParallel(cl)
set.seed(2021)

ctrl <- control_resamples(save_pred = TRUE, verbose = T)
lasso_tune <-
  lasso_wf %>%
  tune_grid(resamples = dat_cv, # fit_resample() if no tuning
            metrics = metric_set(accuracy, pr_auc, roc_auc),
            control = ctrl,
            grid = 100)

# stopCluster(cl)

## Explore result of tuning
lasso_tune %>% 
  collect_metrics()

lasso_tune %>% 
  show_best("roc_auc")

lasso_tune %>% autoplot() + theme_bw()

lasso_tune %>% 
  collect_predictions() %>% 
  conf_mat(income2, .pred_class)

lasso_tune  %>% 
  conf_mat_resampled(parameters = lasso_tune %>% select_best("roc_auc"), tidy = F) # average cell count in resample

lasso_tune %>% 
  collect_predictions() %>% 
  group_by(id) %>% 
  roc_curve(truth = income2, estimate = .pred_lowest:.pred_highest) %>% 
  autoplot()

# Finalize workflow ----

lasso_best <- lasso_tune %>% select_best("roc_auc") # select_by_one_std_err() 

lasso_wf_final <- 
  lasso_wf %>% 
  finalize_workflow(lasso_best)

# Fit and predict on testing data ----

## 1) 1st Method ----
### Fit on entire training data 
lasso_train <- # this should be save for model deployment (readr::write_rds())
  lasso_wf_final %>% 
  fit(data = dat_train)

### Fit on processed testing data
dat_pred <- 
  dat_test %>% 
  bind_cols(predict(lasso_train, new_data = dat_test)) %>% 
  bind_cols(predict(lasso_train, new_data = dat_test, type = "prob"))

# for model deployment
predict(lasso_train, 
        new_data = dat_train[3:4, ]) # lower_middle lower_middle

## 2) 2nd method - easier ----
#### Fit on entire training data then testing data
dat_fit <- 
  lasso_wf_final %>% 
  last_fit(dat_index)

# workflow below should be saved for model deployment (readr::write_rds())
wf_saved <- dat_fit$.workflow[[1]] 
wf_saved <- extract_workflow(dat_fit)

predict(wf_saved, dat_train[3:4, ]) # lower_middle lower_middle
predict(wf_saved, dat_train[3:4, ], type = "prob")

# Performance metrics ----

## For 1st method
dat_pred %>% 
  accuracy(income2, .pred_class)

dat_pred %>% 
  roc_auc(income2, .pred_lowest:.pred_highest)

## For 2nd method
dat_fit %>% 
  collect_metrics()

dat_fit %>% 
  collect_predictions() %>% 
  conf_mat(income2, .pred_class)

dat_fit %>% 
  collect_predictions() %>% 
  conf_mat(income2, .pred_class) %>% 
  autoplot("heatmap") + # or "mosaic"
  scale_fill_gradient(low="white", high="purple")

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

# specific for regression stat model
dat_fit %>% 
  extract_fit_parsnip() %>% 
  tidy() %>% 
  group_by(term) %>% 
  filter(term != "(Intercept)") %>% 
  ggplot(aes(estimate, term, fill = class, color = class)) +
  geom_col(alpha = 0.7) +
  labs(x = "Lasso coefficients", y = "") +
  theme_minimal() 
  # facet_wrap(vars(class))

# Model explainer ----
library(DALEX)
library(DALEXtra)

explainer_lasso <- 
  explain_tidymodels(
    wf_saved, 
    data = dat_train %>% select(-income2), 
    y = as.numeric(dat_processed$income2),
    label = "lasso_regression",
    verbose = T
  )

## Global explanation - variable importance
vip_lasso <- model_parts(explainer_lasso) 

source("var_imp.R")
ggplot_imp(vip_lasso)

## vip package (not suitable to all models)
library(vip)

dat_fit %>%
  extract_fit_parsnip() %>%
  vi_model(lambda = lasso_best$penalty) %>%
  mutate(
    Importance = abs(Importance),
    Variable = fct_reorder(Variable, Importance)
  ) %>%
  ggplot(aes(x = Importance, y = Variable, fill = Sign)) +
  geom_col() +
  scale_x_continuous(expand = c(0, 0)) +
  labs(y = NULL)

  # Or another approach

dat_fit %>% 
  extract_fit_parsnip() %>% 
  vi_model(lambda = lasso_best$penalty) %>% 
  vip(num_features = 70)


# Explainer:
# 1) Global - permutation approach; partial dependence plot
# 2) Local (for single observation) - Shapley additive explanation (SHAP)
# Related packages: DALEX, DALEXtra, vip, lime