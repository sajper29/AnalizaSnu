# PROJEKT: Analiza snu i stylu życia
# CEL: Przewidywanie czasu snu (Sleep Duration) na kilku modelach z różnymi interakcjami oraz najlepszymi hiperparametrami

# Setup
library(tidyverse)
library(tidymodels)
library(skimr)
library(naniar)
library(GGally)
library(vip)
library(finetune)
library(doParallel)
library(rules)
library(Cubist)
library(finetune)
library(dplyr)
library(purrr)

tidymodels_prefer()
set.seed(123)

# Wczytanie danych
df <- read_csv("Sleep_health_and_lifestyle_dataset.csv")

# Eksploracja danych

# Przegląd struktury
glimpse(df)
skim(df)

# Wizualizacja braków danych
vis_missing_plot <- vis_miss(df) + ggtitle("Mapa brakujących danych")
print(vis_missing_plot)


# Rozdzielenie ciśnienia krwi na dwie kolumny
df <- df %>%
  separate(`Blood Pressure`, into = c("Systolic_BP", "Diastolic_BP"), sep = "/", convert = TRUE) %>%
  mutate(across(where(is.character), as.factor))

# Rozkład zmiennej docelowej

ggplot(df, aes(x = `Sleep Duration`)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white") +
  theme_minimal() +
  labs(title = "Rozkład czasu snu", x = "Sleep Duration", y = "Liczba obserwacji")

# Korelacje między zmiennymi numerycznymi
df %>%
  select(where(is.numeric)) %>%
  ggpairs()

# Boxploty do wykrycia wartości odstających
df %>%
  select(where(is.numeric)) %>%
  pivot_longer(everything()) %>%
  ggplot(aes(x = name, y = value)) +
  geom_boxplot(fill = "lightblue") +
  theme_minimal() +
  labs(title = "Wartości odstające w danych")

# Interakcje: aktywność fizyczna vs stres a sen
ggplot(df, aes(x = `Physical Activity Level`, y = `Stress Level`, color = `Sleep Duration`)) +
  geom_jitter(width = 1, height = 1, size = 3, alpha = 0.6) +
  theme_minimal() +
  labs(title = "Aktywność fizyczna vs stres a sen")

# Interakcje: jakość snu vs zaburzenia
ggplot(df, aes(x = `Sleep Disorder`, y = `Quality of Sleep`, fill = `Sleep Disorder`)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Jakość snu a zaburzenia snu")

# Wpływ zawodu na stres i aktywność
ggplot(df, aes(x = `Occupation`, y = `Stress Level`, fill = `Occupation`)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Poziom stresu w różnych zawodach") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggplot(df, aes(x = `Occupation`, y = `Physical Activity Level`, fill = `Occupation`)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Aktywność fizyczna w różnych zawodach") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#Daily Steps są nieistotne(związane z aktywnością fizyczną)
#Nie ma braków danych
#Jest korelacja między snem, chorobami sennymi, jakością snu,stresem oraz aktywnością fizyczną

# Przygotowanie danych 

# Usunięcie zmiennej Daily Steps
df <- df %>% select(-`Daily Steps`)

# Usunięcie braków danych
df <- df %>% drop_na()

# Podział danych 
split <- initial_split(df, prop = 0.8, strata = `Sleep Duration`)
train <- training(split)
test <- testing(split)
folds <- vfold_cv(train, v = 5,repeats = 5)

# Przepisy (interakcje te które zkorelowane)

# Przepis bazowy: dummy + normalizacja
rec_base <- recipe(`Sleep Duration` ~ ., data = train) %>%
  update_role(`Person ID`, new_role = "ID") %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# Przepis z interakcjami 1
rec_interact <- recipe(`Sleep Duration` ~ ., data = train) %>%
  update_role(`Person ID`, new_role = "ID") %>%
  step_dummy(all_nominal_predictors()) %>%
  step_interact(terms = ~ `Quality of Sleep`:`Stress Level` ) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# Przepis z interakcjami 2
rec_interact2 <- recipe(`Sleep Duration` ~ ., data = train) %>%
  update_role(`Person ID`, new_role = "ID") %>%
  step_dummy(all_nominal_predictors()) %>%
  step_interact(terms = ~ `Stress Level`:`Physical Activity Level` + `Quality of Sleep`:`Sleep Disorder_None` + `Quality of Sleep`:`Sleep Disorder_Sleep.Apnea`) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

rec_dummy <- recipe(`Sleep Duration` ~ ., data = train) %>%
  step_dummy(all_nominal_predictors())

prep(rec_dummy) %>% juice() %>% names()

# Specyfikacje modeli z tunowaniem hiperparametrów

# Model liniowy zwykły (lm) – brak hiperparametrów do tuningu
lm_spec <- 
  linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

# Ridge 
ridge_spec <- 
  linear_reg(
    penalty = tune(), 
    mixture = 0        
  ) %>%
  set_engine("glmnet") %>%
  set_mode("regression")

# Las losowy (Random Forest)
rf_spec <- 
  rand_forest(
    mtry = tune(), 
    min_n = tune(), 
    trees = tune()
  ) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")

# Cubist
cubist_spec <- 
  cubist_rules(
    committees = tune(), 
    neighbors = tune()
  ) %>%
  set_engine("Cubist") %>% 
  set_mode("regression")

# XGBoost
xgb_spec <- 
  boost_tree(
    tree_depth = tune(),
    learn_rate = tune(),
    loss_reduction = tune(),
    min_n = tune(),
    sample_size = tune(),
    trees = tune()
  ) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

# Lista modeli
models <- list(
  lm = lm_spec,
  ridge = ridge_spec,
  rf = rf_spec,
  xgb = xgb_spec,
  cubist = cubist_spec
)


# Testowanie przepisu + modelu
recipes <- list(
  base = rec_base,
  interact = rec_interact,
  interact2 = rec_interact2
)

wf_set <- workflow_set(preproc = recipes, models = models)
wf_set

# Random Forest
rf_param <- 
  rf_spec |>
  extract_parameter_set_dials() |> 
  update(
    min_n = min_n(c(8, 14)),
    mtry = mtry(c(3, 8)),
    trees = trees(c(50, 500))
  )

# XGBoost
xgb_param <- 
  xgb_spec |> 
  extract_parameter_set_dials() |> 
  update(
    min_n = min_n(c(8, 14)), 
    trees = trees(c(50, 500)), 
    tree_depth = tree_depth(c(2, 10)),
    learn_rate = learn_rate(c(0.01, 0.3)),
    loss_reduction = loss_reduction(c(0, 5)),
    sample_size = sample_prop(c(0.5, 1))
  )

# Cubist
cubist_param <- 
  cubist_spec |>
  extract_parameter_set_dials() |> 
  update(
    committees = committees(c(1, 10)),
    neighbors = neighbors(c(0, 5))
  )

# Ridge (regresja grzbietowa)
ridge_param <- 
  ridge_spec |>
  extract_parameter_set_dials() |>
  update(
    penalty = penalty(c(0.001, 1))
  )

# LM (brak parametrów do tuningu)
lm_param <- lm_spec |> extract_parameter_set_dials()

# Dodanie zakresów parametrów do workflow options
wf_set
wf_set <- wf_set |> option_add(param_info = rf_param, id = "base_rf")
wf_set <- wf_set |> option_add(param_info = xgb_param, id = "base_xgb")
wf_set <- wf_set |> option_add(param_info = cubist_param, id = "base_cubist")
wf_set <- wf_set |> option_add(param_info = ridge_param, id = "base_ridge")
wf_set <- wf_set |> option_add(param_info = lm_param, id = "base_lm")

wf_set <- wf_set |> option_add(param_info = rf_param, id = "interact_rf")
wf_set <- wf_set |> option_add(param_info = xgb_param, id = "interact_xgb")
wf_set <- wf_set |> option_add(param_info = cubist_param, id = "interact_cubist")
wf_set <- wf_set |> option_add(param_info = ridge_param, id = "interact_ridge")
wf_set <- wf_set |> option_add(param_info = lm_param, id = "interact_lm")

wf_set <- wf_set |> option_add(param_info = rf_param, id = "interact2_rf")
wf_set <- wf_set |> option_add(param_info = xgb_param, id = "interact2_xgb")
wf_set <- wf_set |> option_add(param_info = cubist_param, id = "interact2_cubist")
wf_set <- wf_set |> option_add(param_info = ridge_param, id = "interact2_ridge")
wf_set <- wf_set |> option_add(param_info = lm_param, id = "interact2_lm")

wf_set

wf_set |>
  split(~wflow_id) |>
  map(
    \(x) extract_parameter_set_dials(
      x = x,
      id = x$wflow_id
    ) |>
      _$object
  )

race_ctrl <-
  control_race(
    save_pred = TRUE,
    parallel_over = "everything",
    save_workflow = FALSE
  )

tune_result <-
  wf_set  |> 
  workflow_map(
    "tune_race_anova",
    seed = 1503,
    resamples = folds,
    grid = 50,                 
    control = race_ctrl,
    verbose = TRUE,
    metrics = metric_set(rmse, mae, rsq)
  )

best_results <-
  tune_result |>
  split(~wflow_id) |>
  map(
    \(x)
    extract_workflow_set_result(x = x, id = x$wflow_id) |>
      select_best(metric = "rmse",)
  )


best_models <-
  tune_result |>
  split(~wflow_id) |>
  map(
    \(x) 
    extract_workflow(x = x, id = x$wflow_id) |>
      finalize_workflow(best_results[[x$wflow_id]]) |>
      last_fit(
        split = split, 
        metrics = metric_set(rmse, rsq, mae)
      )
  )

tune_result
best_models
best_params_tbl <- map_dfr(names(best_results), function(id) {
  tibble(wflow_id = id, config = list(best_results[[id]]))
}) %>%
  mutate(config = map_chr(config, ~ paste(names(.x), "=", as.character(.x), collapse = "; ")))

#Najlepsze hiperparametry np. base_rf mtry=8 trees=270 min_n=8
best_params_tbl

res <- tune_result
# Porównanie wyników 
collect_metrics(res) %>%
  ggplot(aes(x = wflow_id, y = mean, fill = .metric)) +
  geom_col(position = "dodge") +
  theme_minimal() +
  labs(title = "Porównanie modeli i przepisów")

tune_result |>
  split(~wflow_id) |>
  map(
    \(x) 
    extract_workflow_set_result(x = x, id = x$wflow_id) |> 
      show_best(metric = "rmse", n = 1) |> 
      select(-n, -.metric, -.config)
  ) |>
  knitr::kable()

tune_result |> 
  rank_results(select_best = T) |> 
  unite("rate", c("mean", "std_err"), sep = "/") |> 
  pivot_wider(names_from = .metric, values_from = rate) |>
  separate_wider_delim(
    cols = mae:rsq, 
    delim = "/", 
    names = c("", "_std_err"), 
    names_sep = "") |> 
  select(-preprocessor, -n, -model) |>
  mutate(.config = str_sub(.config, 20, 30)) |> 
  mutate(across(
    .cols = mae:rsq_std_err, 
    .fns = \(x) signif(x = as.numeric(x), digits = 3)
  )) |> arrange(mae) |> 
  gt::gt() |> 
  gt::tab_header(title = "Wyniki oceny dla zestawu walidacyjnego")

ranked_results <- tune_result |>  
  rank_results(select_best = TRUE) |>
  mutate(wflow_id = fct_reorder(wflow_id, rank))

# --- RMSE ---
plot_rmse <- ranked_results |> 
  filter(.metric == "rmse") |> 
  ggplot(aes(x = wflow_id, y = mean, colour = wflow_id)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = mean - 1.96 * std_err,
                    ymax = mean + 1.96 * std_err), width = 0.4) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(
    title = "Porównanie modeli – RMSE",
    x = "Model",
    y = "RMSE",
    colour = "Model"
  )

# --- MAE ---
plot_mae <- ranked_results |> 
  filter(.metric == "mae") |> 
  ggplot(aes(x = wflow_id, y = mean, colour = wflow_id)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = mean - 1.96 * std_err,
                    ymax = mean + 1.96 * std_err), width = 0.4) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(
    title = "Porównanie modeli – MAE",
    x = "Model",
    y = "MAE",
    colour = "Model"
  )

# --- R² ---
plot_rsq <- ranked_results |> 
  filter(.metric == "rsq") |> 
  ggplot(aes(x = wflow_id, y = mean, colour = wflow_id)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = mean - 1.96 * std_err,
                    ymax = mean + 1.96 * std_err), width = 0.4) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(
    title = "Porównanie modeli – R²",
    x = "Model",
    y = "R²",
    colour = "Model"
  )

# Wyświetlenie wykresów
plot_rmse
plot_mae
plot_rsq

all_fit_metrics <-
  tune_result |>
  split(~wflow_id) |>
  map(
    \(x)
    extract_workflow_set_result(x = x, id = x$wflow_id) |>
      _$.metrics |>
      _[[1]] |> 
      mutate(.config = str_sub(.config, 20, 24))
  )    
all_fit_metrics

#baseRanger
c("mae", "rmse", "rsq") |> 
  map_dfr(
    \(x) 
    tune_result |> 
      extract_workflow_set_result(id = "base_rf") |> 
      select_best(metric = x), .id = ".metric") |> 
  gt::gt()

all_fit_metrics[["base_rf"]] |>                              # Zmień model ...
  filter(.metric == "rsq") |>                               # Zmień statystykę rsq, mae 
  ggplot(aes(trees, .estimate, color = factor(min_n))) +
  geom_point() +
  facet_wrap(~mtry) +
  scale_x_continuous(limits = c(0, 300), breaks = seq(0, 300, 50)) +
  ggtitle(label = "ranger")

# Najlepszy model na zbiorze testowym

test_metrics_list <- list()
test_preds_list   <- list()

# Pętla po wszystkich najlepszych modelach
for (id in names(best_results)) {
  cat("Dopasowuję i oceniam model:", id, "\n")
  
  # Finalizacja i dopasowanie workflow
  final_wf <- 
    extract_workflow(wf_set, id = id) |>
    finalize_workflow(best_results[[id]]) |>
    fit(data = train)
  
  # Predykcje na zbiorze testowym
  preds <- 
    predict(final_wf, new_data = test) |>
    bind_cols(test %>% select(`Sleep Duration`)) |>
    mutate(wflow_id = id)
  
  # Predykcje
  test_preds_list[[id]] <- preds
  
  # Metryki
  mets <- 
    metrics(preds, truth = `Sleep Duration`, estimate = .pred) |>
    mutate(wflow_id = id)
  
  test_metrics_list[[id]] <- mets
}

test_metrics_df <- bind_rows(test_metrics_list)

test_metrics_df %>%
  select(wflow_id, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  arrange(rmse) %>%
  gt::gt() %>%
  gt::tab_header(title = "Wyniki najlepszych modeli na zbiorze testowym")

# Wszystkie predykcje razem w jednej ramce
all_test_preds <- bind_rows(test_preds_list)

all_test_preds %>% 
  filter(wflow_id == "base_rf") %>% 
  head(20)

#Najlepsze wyniki base_cubist
#Interakcje między zmiennymi nie wpłyneły znacząco na poprawe modelu