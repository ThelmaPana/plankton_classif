#--------------------------------------------------------------------------#
# Project: plankton_classif
# Script purpose: Generate a nice formatting for classification reports.
# Date: 09/04/2025
# Author: Thelma Panaïotis
#--------------------------------------------------------------------------#

library(tidyverse)
library(gt)
library(officer)
library(openxlsx)

# Directory where to save formatted reports
reports_dir <- "reports/"
dir.create(reports_dir, showWarnings = FALSE)


## List detailed classification reports ----
#--------------------------------------------------------------------------#
files <- list.files("perf", pattern = "detailed", full.names = TRUE)

# List CR to generate as supplementary
tables <- tribble(
  ~dataset, ~dataset_name, ~table_number,
  "flowcam", "FlowCAM", "S2",
  "ifcb",    "IFCB",    "S3",
  "isiis",   "ISIIS",   "S4",
  "uvp6",    "UVP6",    "S5",
  "zoocam",  "ZooCAM",  "S6",
  "zooscan", "Zooscan", "S7"
)

for (file in files) {
  # Get dataset name
  dataset <- str_split_fixed(file, pattern = "_", n = 3)[1, 2]
  
  # Read file
  df <- read_csv(file, show_col_types = FALSE) %>%
    # Select only F1 columns
    select(taxon, grouped, `n-native_rf`, plankton, contains("f1")) %>%
    # Rename columns with nice formatting
    rename(
      Taxon = taxon,
      Grouped = grouped,
      `N test` = `n-native_rf`,
      `Nat + RF` = `f1-native_rf`,
      `Mob + MLP600` = `f1-mobilenet_pca50_rf`,
      `Eff S + MLP600` = `f1-mobilenet600`,
      `Mob + PCA + RF` = `f1-effnetv2s`
    ) %>%
    mutate(plankton = ifelse(plankton, "Plankton", "Non plankton"))
  
  # For zooscan, generate great table, write in a single file as docx for the manuscript
  if (dataset == "zooscan"){
    
    cr <- df %>%
      mutate(plankton = str_c(plankton, " classes")) |> 
      gt(rowname_col = "Taxon", groupname_col = "plankton") %>%
      tab_stubhead(label = "Class") %>%
      tab_header(title = paste0("Classification report for detailed classes in the ", dataset, " dataset.")) %>%
      fmt_number(columns = contains("+"),
                 decimals = 1,
                 scale_by = 100) %>%
      summary_rows(
        groups = everything(),
        columns = contains("+"),
        fns = list(average = ~ mean(., na.rm = TRUE)),
        fmt = list( ~ fmt_number(
          ., decimals = 1, scale_by = 100
        )),
        missing_text = ""
      ) %>%
      # Italicize group cells
      tab_style(style = list(cell_text(style = "italic", align = "center")),
                locations = cells_row_groups()) %>%
      # Italicize, bold and center summary cells
      tab_style(style = list(cell_text(style = "italic", weight = "bold", align = "center")),
                locations = list(cells_summary(), cells_stub_summary())) %>%
      # Center column names
      tab_style(
        style = list(cell_text(align = "center", v_align = "middle")),
        locations = list(cells_column_labels(), cells_stubhead())
      ) %>%
      # Remove vertical line after the "Taxon" column
      tab_style(
        style = cell_borders(sides = c("r"), weight = px(0)),
        locations = list(cells_stub(), cells_stub_summary())
      ) %>%
      # No horizontal line in the table
      tab_options(table_body.hlines.width = 0) %>%
      # Cell colours
      #data_color(
      #  columns = contains("+"),
      #  method = "numeric",
      #  palette = palette <- colorRampPalette(colors=c("#FFFFFF", "#A6D8DB"))(500),
      #  domain = c(0, 1),
      #  reverse = FALSE,
      #  autocolor_text = FALSE,
      #) %>%
      # Equal width for number columns
      cols_width(contains("+") ~ px(80)) %>% 
      # Change font
      opt_table_font(
        font = list(
          google_font(name = "Palatino"),
          "Cochin", "serif"
        )
      )
    
    gtsave(cr, paste0(reports_dir, dataset, ".tex"))
    gtsave(cr, paste0(reports_dir, dataset, ".docx"))
    
  } 
  
  ## And save all to Gsheet
  # Separate plankton from non plankton lines and convert values to percent
  df_plankton <- df %>% filter(plankton == "Plankton") %>% select(-plankton) %>% mutate(across(`Nat + RF`:`Mob + PCA + RF`, ~(.x*100)))
  df_nonplankton <- df %>% filter(plankton != "Plankton") %>% select(-plankton) %>% mutate(across(`Nat + RF`:`Mob + PCA + RF`, ~(.x*100)))
  
  # Prepare average summary rows
  summ_plankton <- bind_cols(
    tribble(
      ~Taxon, ~Grouped, ~`N test`,
      "average", "", "",
    ) |> 
      mutate(`N test` = as.numeric(`N test`)),
    df_plankton |> summarise(across(`Nat + RF`:`Mob + PCA + RF`, mean))
  )
  summ_nonplankton <- bind_cols(
    tribble(
      ~Taxon, ~Grouped, ~`N test`,
      "average", "", "",
    ) |> 
      mutate(`N test` = as.numeric(`N test`)),
    df_nonplankton |> summarise(across(`Nat + RF`:`Mob + PCA + RF`, mean))
  )
  
  # Add these rows to dataframes
  df_plankton <- bind_rows(df_plankton, summ_plankton)
  df_nonplankton <- bind_rows(df_nonplankton, summ_nonplankton)
  
  # Round numbers to 1 decimal
  df_plankton <- df_plankton %>% mutate_if(is.numeric, ~round(.x, digits = 1))
  df_nonplankton <- df_nonplankton %>% mutate_if(is.numeric, ~round(.x, digits = 1))
  
  # Prepare table caption
  dataset_name <- tables %>% filter(dataset == !!dataset) %>% pull(dataset_name)
  table_number <- tables %>% filter(dataset == !!dataset) %>% pull(table_number)
  table_caption <- paste0("Table ", table_number, ": Classification report for detailed classes in the ", dataset_name, " dataset. Reported values are F1-scores. The models are described in Figure 1. N test indicates the number of objects in the test set for each class.")
  
  ## Write to excel
  # Create workbook if it does not exist 
  if (!exists("wb")) {
    wb <- createWorkbook()  
  }
  
  # Create tab for dataset
  addWorksheet(wb, sheetName = dataset, gridLines = FALSE)
  
  # Write caption
  writeData(wb, sheet = dataset, x = table_caption)
  
  # Write column names
  writeData(wb, sheet = dataset, x = df_plankton %>% slice_head(n = 0), startCol = 1, startRow = 2)
  
  # Write "plankton" line
  writeData(wb, sheet = dataset, x = "Plankton classes", startCol = 1, startRow = 3)
  
  # Write plankton report  with conditional formatting
  writeData(wb, sheet = dataset, x = df_plankton, startCol = 1, startRow = 4, colNames = FALSE)
  conditionalFormatting(
    wb, sheet = dataset, type = "colourScale", cols = 4:7, rows = 4:(3 + nrow(df_plankton)),
    style = c("white", "#ccebc5", "#4eb3d3"), rule = c(0, 50, 100)
  )
  
  # Write "non plankton" line
  writeData(wb, sheet = dataset, x = "Non plankton classses", startCol = 1, startRow = nrow(df_plankton) + 4)
  
  # Write non plankton report with conditional formatting
  writeData(wb, sheet = dataset, x = df_nonplankton, startCol = 1, startRow = nrow(df_plankton) + 5, colNames = FALSE)
  conditionalFormatting(
    wb, sheet = dataset, type = "colourScale", cols = 4:7, rows = (nrow(df_plankton) + 5):(nrow(df_plankton) + nrow(df_nonplankton) + 4),
    style = c("white", "#ccebc5", "#4eb3d3"), rule = c(0, 50, 100)
  )
}

# Save the workbook
saveWorkbook(wb, "perf/classif_reports_supp_2.xlsx", overwrite = TRUE)
