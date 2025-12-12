#--------------------------------------------------------------------------#
# Project: plankton_classif
# Script purpose: Generate plots fo the paper
# Date: 30/08/2023
# Author: Thelma Panaïotis
#--------------------------------------------------------------------------#

library(tidyverse)
library(ggpattern)
library(patchwork)
library(chroma)


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


## Figure 2: Random VS. MobileNet (W & NW) VS. RF (W & NW) ----
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



## Figure 3: Bigger CNN do not improve classification performance and a smaller CNN performs just as well ----
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


## Figure 4: Is it the features or the classifier?  ----
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


## Figure 5: Performance increase, from native RF to deep ----
#--------------------------------------------------------------------------#

times_100 <- function(x){x * 100}

# Dataset to work with
dataset <- "zooscan"

# Sheet with taxonomy
taxo <- read_csv("taxonomy_match/Taxonomy match - zooscan.csv")%>% 
  select(taxon, grouped = level2, plankton)


# Read classification reports for given dataset
files <- list.files("perf", full.names = TRUE, pattern = "report*") %>% str_subset(dataset)

## Detailed metrics
df <- read_csv(files %>% str_subset("detailed"), show_col_types = FALSE) %>% 
  select(taxon, everything()) %>% 
  # reformat to longer
  pivot_longer(`precision-native_rf`:`f1-effnetv2s`) %>% 
  # get metric and model from name
  separate(name, into = c("metric", "model"), sep = "-") %>% 
  mutate(level = "detailed")

# Separate RF values which are the reference
df_rf <- df %>% 
  filter(model == "native_rf") %>% 
  rename(ref = value) %>% 
  select(-model)

# Compute the difference between metrics and reference
df <- df %>% 
  filter(model != "native_rf") %>% 
  left_join(df_rf, by = join_by(taxon, grouped, plankton, metric, level)) %>% 
  mutate(value = value - ref) %>% 
  select(-ref)

## Grouped metrics
df_g <- read_csv(files %>% str_subset("grouped"), show_col_types = FALSE) %>% 
  select(taxon, everything()) %>% 
  # reformat to longer
  pivot_longer(`precision-native_rf`:`f1-effnetv2s`) %>% 
  # get metric and model from name
  separate(name, into = c("metric", "model"), sep = "-") %>% 
  mutate(level = "grouped")

# Separate RF values which are the reference
df_g_rf <- df_g %>% 
  filter(model == "native_rf") %>% 
  rename(ref = value) %>% 
  select(-model)

# Compute the difference between metrics and reference
df_g <- df_g %>% 
  filter(model != "native_rf") %>% 
  left_join(df_g_rf, by = join_by(taxon, plankton, metric, level)) %>% 
  mutate(value = value - ref) %>% 
  select(-ref)

# Generate nice colours
my_cols <- c(
  "Mob + MLP600" = "#3a62bfff",
  #"Eff S + MLP600" = "#77d1daff",
  "Eff S + MLP600" = darken("#77d1daff"),
  #"Mob + PCA + RF" = "#b0aaf8ff"
  "Mob + PCA + RF" = darken("#b0aaf8ff")
)



# Prepare nice names for models
models <- tibble(
  model = c("mobilenet600", "effnetv2s", "mobilenet_pca50_rf"),
  name = c("Mob + MLP600", "Eff S + MLP600", "Mob + PCA + RF")
) %>% 
  mutate(name = fct_inorder(name))


# Join detailed and grouped metrics together
df_all <- df %>% 
  bind_rows(df_g) %>% 
  mutate(value = value * 100) %>% 
  # metric as factor for plotting
  mutate(
    metric = str_to_sentence(metric),
    level = str_to_sentence(level)
  ) %>% 
  mutate(metric = factor(metric, levels = c("Precision", "Recall", "F1"))) %>% 
  left_join(models, by = join_by(model)) %>% 
  select(-model) %>% 
  rename(model = name)

override.linetype <- c(1, 3, 2) 
# Plot it
p5 <- df_all %>% 
  ggplot(aes(x = value, color = model, linetype = model)) +
  geom_vline(xintercept = 0, colour = "gray80", linewidth = 1) +
  geom_density(adjust=0.75) +
  scale_color_manual(
    guide = none,
    values = my_cols,
    labels = c(
      `Mob + MLP600` = expression(Mob + MLP[600]),
      `Eff S + MLP600` = expression(Eff~S + MLP[600])
    )) +
  scale_y_continuous(breaks = c(0, 0.01, 0.02)) +
  theme_classic() +
  theme(strip.background = element_blank(), legend.text.align = 0, legend.position = "bottom", text = element_text(family = "Helvetica")) +
  facet_grid(rows = vars(level), cols = vars(metric)) +
  labs(x = "Percent increase in metric, from RF on native features", y = "Estimated probability density", color = "Model") +
  guides(colour = guide_legend(override.aes = list(linetype = override.linetype))) +
  scale_linetype(guide = "none")
ggsave(p5, file = "figures/figure_5.png", width = 180, height = 100, unit = "mm", dpi = 300, bg = "white")


## Figure S1: CNN learning curve for UVP6 dataset ----
#--------------------------------------------------------------------------#

## Plot two models for illustration: Mob + MLP600 and Eff S + MLP600

# Path to saved models
models_dir <- "/home/tpanaiotis/complex/plankton_classif/out/models"

# Read training log for Eff S + MLP600
df_eff <- read_tsv(file.path(models_dir, "uvp6_effnetv2s", "checkpoints", "training_log.tsv"), show_col_types = FALSE) |> 
  mutate(mod_name = "Eff S + MLP600") |> 
  select(mod_name, epoch, val_loss, val_accuracy)

# Read training log for Mob + MLP600
df_mob <- read_tsv(file.path(models_dir, "uvp6_mobilenet600", "checkpoints", "training_log.tsv"), show_col_types = FALSE) |> 
  mutate(mod_name = "Mob + MLP600") |> 
  select(mod_name, epoch, val_loss, val_accuracy)

# Assemble together
df <- bind_rows(df_eff, df_mob) |> 
  pivot_longer(val_loss:val_accuracy, names_to = "metric") |> 
  mutate(metric = str_remove_all(metric, "val_"))

metric_labels <- c(
  accuracy = "Validation accuracy",
  loss = "Validation loss"
)

# And plot
ps1 <- ggplot(df) +
  geom_path(aes(x = epoch, y = value, colour = mod_name)) +
  scale_x_continuous(breaks = c(1:10)) +
  scale_colour_manual(
    values = my_cols,
    labels = c(
      `Mob + MLP600` = expression(Mob + MLP[600]),
      `Eff S + MLP600` = expression(Eff~S + MLP[600])
    )) +
  labs(x = "Epoch", y = "Value", colour = "Model") +
  facet_wrap(~metric, scales = "free_y", labeller = labeller(metric = metric_labels)) +
  theme_classic() +
  theme(
    strip.background = element_blank()
  )

# Save
ggsave(ps1, file = "figures/figure_s1.png", width = 180, height = 50, unit = "mm", dpi = 300, bg = "white")


## Figure S2: F1 vs accuracy ----
#--------------------------------------------------------------------------#
# F1 micro VS accuracy
ps2a <- df %>% 
  filter(metric %in% c("accuracy", "f1_micro")) |> 
  select(model:score) |> 
  pivot_wider(names_from = metric, values_from = score) |> 
  ggplot() +
  geom_abline(slope = 1, intercept = 0, colour = "grey20", linetype = "dotted") +
  geom_point(aes(x = accuracy, y = f1_micro, colour = model, shape = dataset), size = 2) +
  scale_colour_manual(values = my_cols) +
  xlim(c(0,1)) + ylim(c(0,1)) +
  labs(x = "Accuracy", y = "F1 micro", colour = "Model", shape = "Dataset") +
  coord_fixed() +
  theme_classic() +
  theme(text = element_text(size = 10, family = "Helvetica"))

# F1 macro VS balanced accuracy
ps2b <- df %>% 
  filter(metric %in% c("balanced_accuracy", "f1_macro")) |> 
  select(model:score) |> 
  pivot_wider(names_from = metric, values_from = score) |> 
  ggplot() +
  geom_abline(slope = 1, intercept = 0, colour = "grey20", linetype = "dotted") +
  geom_point(aes(x = balanced_accuracy, y = f1_macro, colour = model, shape = dataset), size = 2) +
  scale_colour_manual(values = my_cols) +
  xlim(c(0,1)) + ylim(c(0,1)) +
  labs(x = "Balanced accuracy", y = "F1 macro", colour = "Model", shape = "Dataset") +
  coord_fixed() +
  theme_classic() +
  theme(text = element_text(size = 10, family = "Helvetica"))

ps2 <- ps2a + ps2b + plot_layout(guides = 'collect') & theme(
  legend.position = 'bottom',
  legend.direction = "vertical"
)

# Save
ggsave(ps2, file = "figures/figure_s2.png", width = 180, height = 160, unit = "mm", dpi = 300, bg = "white")
