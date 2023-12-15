#--------------------------------------------------------------------------#
# Project: plankton_classif
# Script purpose: Generate a nice formatting for classification reports.
# Date: 14/12/2023
# Author: Thelma Panaïotis
#--------------------------------------------------------------------------#

library(tidyverse)
library(gt)
library(officer)
library(gto)
library(googlesheets4)
gs4_auth(email = "thelma.panaiotis@imev-mer.fr")


# ss <- "https://docs.google.com/spreadsheets/d/1HBBOfNsSG1kwQILiwSFw8hGAeeqQhXJdqXeQ6_BDsrQ/edit?usp=sharing" # Not used any more

# Docx file to save supplementary tables
file_name <- "reports/supplementary.docx"
if (file.exists(file_name)){file.remove(file_name)}


## List detailed classification reports ----
#--------------------------------------------------------------------------#
files <- list.files("perf", pattern = "detailed", full.names = TRUE)

for (file in files) {
  # Get dataset name
  dataset <- str_split_fixed(file, pattern = "_", n = 3)[1, 2]
  
  # Read file
  df <- read_csv(file, show_col_types = FALSE) %>%
    # Select only F1 columns
    select(taxon, grouped, plankton, contains("f1")) %>%
    # Rename columns with nice formatting
    rename(
      Taxon = taxon,
      Grouped = grouped,
      `Nat + RF` = `f1-native_rf`,
      `Mob + MLP600` = `f1-mobilenet_pca50_rf`,
      `Eff S + MLP600` = `f1-mobilenet600`,
      `Mob + PCA + RF` = `f1-effnetv2s`
    ) %>%
    mutate(plankton = ifelse(plankton, "Plankton", "Non plankton"))

  cr <- df %>%
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
    data_color(
      columns = contains("+"),
      method = "numeric",
      palette = palette <- colorRampPalette(colors=c("#FFFFFF", "#A6D8DB"))(500),
      domain = c(0, 1),
      reverse = FALSE,
      autocolor_text = FALSE,
    ) %>%
    # Equal width for number columns
    cols_width(contains("+") ~ px(80)) %>% 
    # Change font
    opt_table_font(
      font = list(
        google_font(name = "Palatino"),
        "Cochin", "serif"
      )
    )
  
  # For zooscan, write in a single file
  if (dataset == "zooscan"){
    
    gtsave(cr, paste0("reports/", dataset, ".tex"))
    gtsave(cr, paste0("reports/", dataset, ".docx"))
    
  } else {# For others, all in the same docx file
    
    gtsave(cr, paste0("reports/", dataset, ".tex"))
    
    # If file does not exist, create it
    if (!file.exists(file_name)){
      doc <- read_docx()
      doc <- body_add_gt(doc, value = cr)
      print(doc, target = file_name)
    } else { # Otherwise just read it
      doc <- read_docx(file_name)
      doc <- doc %>% 
        body_add_break() %>% 
        body_add_gt(value = cr)
      print(doc, target = file_name)
    }
  }
}
