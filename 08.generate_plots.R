#--------------------------------------------------------------------------#
# Project: plankton_classif
# Script purpose: Generate plots fo the paper
# Date: 30/08/2023
# Author: Thelma Panaïotis
#--------------------------------------------------------------------------#

library(tidyverse)
library(ggpattern)



# Path to save figures
dir.create("figures", showWarnings = FALSE)


## Read and format results ----
#--------------------------------------------------------------------------#
df <- read_csv("perf/prediction_metrics.csv", col_types = cols())

# Prepare nice names for models
models <- tibble(
  model = c("mobilenet600nw", "mobilenet600", "mobilenet_rf", "effnetv2s", "effnetv2xl", "mobilenet50", "mobilenet_pca50_rf", "native_rf_noweights", "native_rf", "random"),
  name = c("Mob + MLP600 (NW)", "Mob + MLP600", "Mob + RF", "Eff S + MLP600", "Eff XL + MLP600", "Mob + MLP50", "Mob + PCA + RF", "Nat + RF (NW)", "Nat + RF", "Random")
) %>% 
  mutate(name = fct_inorder(name))

# Rename models
df <- df %>% 
  left_join(models, by = join_by(model)) %>% 
  select(-model) %>% 
  rename(model = name) %>% 
  drop_na(model)

# Generate nice names for datasets
datasets <- tibble(
  dataset = c("flowcam", "ifcb", "isiis", "uvp6", "zoocam", "zooscan"),
  name = c("FlowCam", "IFCB", "ISIIS", "UVP6", "ZooCam", "ZooScan")
)

# Rename datasets
df <- df %>% 
  left_join(datasets, by = join_by(dataset)) %>% 
  select(-dataset) %>% 
  rename(dataset = name)

# Reformat results in a long dataframe
# For this we need to separate detailed and grouped results
df_d <- df %>% 
  select(model, dataset, accuracy:plankton_recall) %>% 
  pivot_longer(accuracy:plankton_recall, names_to = "metric", values_to = "score")

df_g <- df %>% 
  select(model, dataset, accuracy_g:plankton_recall_g) %>% 
  pivot_longer(accuracy_g:plankton_recall_g, names_to = "metric_g", values_to = "score_g")

df <- df_d %>% 
  mutate(metric_g = str_c(metric, "_g")) %>% 
  left_join(df_g)


# Create labels for faceting
labels <- c(
  accuracy = "Accuracy", 
  balanced_accuracy = "Balanced accuracy",
  plankton_precision = "Plankton averaged precision",
  plankton_recall = "Plankton averaged recall"
)

# Generate nice colours
my_cols <- c(
  "Random" = "#bebebeff",
  "Mob + MLP600" = "#3a62bfff",
  "Mob + MLP600 (NW)" = "#839fe0ff",
  "Nat + RF" = "#ffbc63ff",
  "Nat + RF (NW)" = "#ffd194ff",
  "Eff S + MLP600" = "#77d1daff",
  "Eff XL + MLP600" = "#29a8b4ff",
  "Mob + MLP50" = "#63c6ffff",
  "Mob + RF" = "#7f7bc5ff",
  "Mob + PCA + RF" = "#b0aaf8ff"
)


## Random VS. MobileNet (W & NW) VS. RF (W & NW) ----
#--------------------------------------------------------------------------#
p2 <- df %>% 
  filter(model %in% c("Random", "Mob + MLP600", "Mob + MLP600 (NW)", "Nat + RF", "Nat + RF (NW)")) %>% 
  ggplot() +
  geom_col_pattern(
    aes(x = dataset, y = score_g, pattern_colour = model, colour = model), position = position_dodge2(), show.legend = FALSE, fill = "white", linewidth = 0.3,
    pattern = "stripe", pattern_density = 0.01, pattern_spacing = 0.025, pattern_angle = 45, pattern_key_scale_factor = 0.6
  ) +
  geom_col(aes(x = dataset, y = score, fill = model, colour = model), position = position_dodge2()) +
  scale_fill_manual(
    values = my_cols,
    labels = c(
      `Mob + MLP600` = expression(Mob + MLP[600]),
      `Mob + MLP600 (NW)` = expression(Mob + MLP[600] ~(NW))
  )) +
  scale_colour_manual(values = my_cols) +
  scale_pattern_color_manual(values = my_cols) +
  guides(colour = "none", pattern = "none") +
  scale_y_continuous(limits = c(0,1), expand = c(0,0)) +
  facet_wrap(~metric, labeller = labeller(metric = labels)) +
  theme_minimal() +
  labs(x = "Dataset", y = "Score", fill = "Model") +
  theme(
    legend.position = "bottom", 
    panel.grid.major.x = element_blank(),
    strip.background = element_rect(colour="white", fill="white"),
    text = element_text(size = 10, family = "Helvetica")
  )
ggsave(p2, file = "figures/figure_2.png", width = 180, height = 100, unit = "mm", dpi = 300, bg = "white")


# Performance increase summary for accuracy
df %>% 
  filter(model %in% c("Mob + MLP600", "Mob + MLP600 (NW)", "Nat + RF", "Nat + RF (NW)")) %>% 
  # Keep accuracy only
  filter(metric == "accuracy") %>% 
  # Flag in weights were used
  mutate(weights = !str_detect(model, "NW")) %>% 
  select(model, dataset, weights, score) %>% 
  # Distinguish between CNN and RF
  mutate(model = ifelse(str_detect(model, "RF"), "RF", "CNN")) %>% 
  pivot_wider(names_from = model, values_from = score) %>% 
  mutate(diff = CNN - RF) %>% 
  group_by(weights) %>% 
  summarise(
    diff_min = min(diff),
    diff_mean = mean(diff),
    diff_max = max(diff)
    )

df %>% 
  filter(model %in% c("Mob + MLP600", "Mob + MLP600 (NW)", "Nat + RF", "Nat + RF (NW)")) %>% 
  # Keep accuracy only
  filter(metric == "accuracy") %>% 
  # Flag in weights were used
  mutate(weights = !str_detect(model, "NW")) %>% 
  select(model, dataset, weights, score) %>% 
  # Distinguish between CNN and RF
  mutate(model = ifelse(str_detect(model, "RF"), "RF", "CNN")) %>% 
  pivot_wider(names_from = model, values_from = score) %>% 
  mutate(diff = CNN - RF) %>% 
  summarise(
    diff_min = min(diff),
    diff_mean = mean(diff),
    diff_max = max(diff)
  )



## Bigger CNN do not improve classification performance and a smaller CNN performs just as well ----
#--------------------------------------------------------------------------#
p3 <- df %>% 
  filter(model %in% c("Mob + MLP600", "Eff S + MLP600", "Eff XL + MLP600", "Mob + MLP50")) %>% 
  ggplot() +
  geom_col_pattern(
    aes(x = dataset, y = score_g, pattern_colour = model, colour = model), position = position_dodge2(), show.legend = FALSE, fill = "white", linewidth = 0.3,
    pattern = "stripe", pattern_density = 0.01, pattern_spacing = 0.025, pattern_angle = 45, pattern_key_scale_factor = 0.6
  ) +
  geom_col(aes(x = dataset, y = score, fill = model, colour = model), position = position_dodge2()) +
  scale_fill_manual(
    values = my_cols,
    labels = c(
      `Mob + MLP600` = expression(Mob + MLP[600]),
      `Eff S + MLP600` = expression(Eff~S + MLP[600]),
      `Eff XL + MLP600` = expression(Eff~XL + MLP[600]),
      `Mob + MLP50` = expression(Mob + MLP[50])
    )) +
  scale_colour_manual(values = my_cols) +
  scale_pattern_color_manual(values = my_cols) +
  guides(colour = "none", pattern = "none") +
  scale_y_continuous(limits = c(0,1), expand = c(0,0)) +
  facet_wrap(~metric, labeller = labeller(metric = labels)) +
  theme_minimal() +
  labs(x = "Dataset", y = "Score", fill = "Model") +
  theme(
    legend.position = "bottom", 
    panel.grid.major.x = element_blank(),
    strip.background = element_rect(colour="white", fill="white"),
    text = element_text(size = 10, family = "Helvetica")
  )
ggsave(p3, file = "figures/figure_3.png", width = 180, height = 100, unit = "mm", dpi = 300, bg = "white")


## Is it the features or the classifier?  ----
#--------------------------------------------------------------------------#
p4 <- df %>% 
  filter(model %in% c("Mob + MLP600", "Mob + RF", "Mob + PCA + RF", "Nat + RF")) %>% 
  ggplot() +
  geom_col_pattern(
    aes(x = dataset, y = score_g, pattern_colour = model, colour = model), position = position_dodge2(), show.legend = FALSE, fill = "white", linewidth = 0.3,
    pattern = "stripe", pattern_density = 0.01, pattern_spacing = 0.025, pattern_angle = 45, pattern_key_scale_factor = 0.6
  ) +
  geom_col(aes(x = dataset, y = score, fill = model, colour = model), position = position_dodge2()) +
  scale_fill_manual(
    values = my_cols,
    labels = c(
      `Mob + MLP600` = expression(Mob + MLP[600])
    )) +
  scale_colour_manual(values = my_cols) +
  scale_pattern_color_manual(values = my_cols) +
  guides(colour = "none", pattern = "none") +
  scale_y_continuous(limits = c(0,1), expand = c(0,0)) +
  facet_wrap(~metric, labeller = labeller(metric = labels)) +
  theme_minimal() +
  labs(x = "Dataset", y = "Score", fill = "Model") +
  theme(
    legend.position = "bottom", 
    panel.grid.major.x = element_blank(),
    strip.background = element_rect(colour="white", fill="white"),
    text = element_text(size = 10, family = "Helvetica")
  )
ggsave(p4, file = "figures/figure_4.png", width = 180, height = 100, unit = "mm", dpi = 300, bg = "white")


