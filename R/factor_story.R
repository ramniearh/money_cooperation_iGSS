# Compact figure of how the final outcome changes with one parameter (or a few), around one environment.
#
#   Rscript R/factor_story.R --out <directory> --name <file name> --title <title> --scoring aggregate \
#       --panel "<sweep directory>|<parameter>|<filter>" [--panel ...]
#
# <parameter> is a column of runs.csv (for example b, rounds, tournament_size), two columns joined by
# "+" (n_c+n_d), or "generation@50,100,200" to read the most common strategy at those generations from
# generations.csv. <filter> is empty or name=value,... and fixes other parameters of that sweep.
#
# One figure: one row of panels per model and one column per mode; in each panel one stacked bar per
# value of the parameter, showing the share of runs in which each strategy name is alone the most
# common (ties are their own category). Names and colours come from labels.csv, with the colour keys
# of R/figures.R. Nothing is computed here that is not in the CSV files (SPEC §10).

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
})

# --- Arguments ----------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
usage <- paste("usage: Rscript R/factor_story.R --out <dir> --name <name> --title <title>",
               "--scoring <option> --panel \"<sweep dir>|<parameter>|<filter>\" [--panel ...]")
option <- function(name) {
  i <- match(name, args)
  if (is.na(i) || i == length(args)) NULL else args[i + 1]
}
out_dir <- option("--out")
file_name <- option("--name")
title <- option("--title")
scoring_option <- option("--scoring")
panel_specs <- args[which(args == "--panel") + 1]
if (is.null(out_dir) || is.null(file_name) || is.null(title) || is.null(scoring_option) ||
    length(panel_specs) == 0) stop(usage)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# --- Colours and theme (as in R/figures.R) ------------------------------------------

key_colours <- c(named_1 = "#2a78d6", cooperate = "#eb6834", named_2 = "#1baf7a",
                 defect = "#eda100", named_3 = "#e87ba4", named_4 = "#008300",
                 unnamed = "#898781", tie = "#c3c2b7")
key_order <- names(key_colours)
surface <- "#fcfcfb"
ink <- "#0b0b0b"
ink_secondary <- "#52514e"

theme_story <- function() {
  theme_minimal(base_size = 10) +
    theme(
      plot.background = element_rect(fill = surface, colour = NA),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_line(colour = "#e1e0d9", linewidth = 0.3),
      panel.spacing = unit(1, "lines"),
      text = element_text(colour = ink),
      axis.text = element_text(colour = ink_secondary),
      strip.text = element_text(colour = ink, face = "bold", hjust = 0),
      plot.title = element_text(face = "bold"),
      plot.subtitle = element_text(colour = ink_secondary),
      plot.caption = element_text(colour = ink_secondary),
      plot.margin = margin(8, 14, 8, 8),
      legend.position = "bottom",
      legend.title = element_blank()
    )
}

short_names <- c(b = "b", e_action = "error", rounds = "R", tournament_size = "tournament",
                 depth_limit = "depth limit", parsimony_tax = "penalty", generation = "generation",
                 `n_c+n_d` = "C/D", episodes_per_score = "E")
mode_titles <- c(`1` = "mode 1: action rule evolves",
                 `2` = "mode 2: assessment rule evolves",
                 `3` = "mode 3: both evolve")

# --- Read every panel ----------------------------------------------------------------

version_column <- cols(code_version = "c")
parse_filter <- function(text) {
  if (is.na(text) || text == "") return(character(0))
  pairs <- strsplit(strsplit(text, ",")[[1]], "=")
  setNames(vapply(pairs, `[`, "", 2), vapply(pairs, `[`, "", 1))
}
apply_filter <- function(data, filter) {
  for (column in names(filter)) {
    data <- data[as.numeric(data[[column]]) == as.numeric(filter[[column]]), ]
  }
  data
}

labels_table <- NULL
captions <- character(0)
rows <- list()
for (k in seq_along(panel_specs)) {
  parts <- strsplit(panel_specs[k], "|", fixed = TRUE)[[1]]
  sweep_dir <- parts[1]
  parameter <- parts[2]
  filter <- parse_filter(if (length(parts) >= 3) parts[3] else "")
  if (is.null(labels_table)) {
    labels_table <- read_csv(file.path(sweep_dir, "labels.csv"), show_col_types = FALSE)
  }
  runs <- read_csv(file.path(sweep_dir, "runs.csv"), show_col_types = FALSE,
                   col_types = version_column) |>
    filter(scoring == scoring_option) |>
    apply_filter(filter)
  if (nrow(runs) == 0) stop("no runs in ", sweep_dir, " match the panel ", panel_specs[k])

  if (startsWith(parameter, "generation@")) {
    at <- as.integer(strsplit(sub("generation@", "", parameter), ",")[[1]])
    data <- read_csv(file.path(sweep_dir, "generations.csv"), show_col_types = FALSE,
                     col_types = version_column) |>
      semi_join(runs, by = "run_id") |>
      filter(generation %in% at) |>
      mutate(tied = as.integer(grepl("+", most_common, fixed = TRUE)), value = generation)
    parameter_name <- "generation"
  } else {
    columns <- strsplit(parameter, "+", fixed = TRUE)[[1]]
    data <- runs
    data$value <- if (length(columns) == 1) runs[[columns]] else
      do.call(paste, c(lapply(columns, function(cl) runs[[cl]]), sep = "/"))
    parameter_name <- parameter
  }
  # Every other parameter must be fixed within a value: at most one run per seed.
  repeats <- data |> count(mode, model, value, run_seed) |> filter(n > 1)
  if (nrow(repeats) > 0) stop("the panel ", panel_specs[k], " mixes several values of another parameter")

  values <- sort(unique(data$value))
  label <- short_names[[parameter_name]]
  data <- data |>
    mutate(level = paste0(label, " ", value),
           level_order = k * 1000 + match(value, values),
           group = label) |>
    select(mode, model, level, level_order, group, most_common_name, tied)
  rows[[k]] <- data
  captions <- c(captions, paste0("'", runs$sweep[1], "' (code ", runs$code_version[1], ")"))
}
data <- bind_rows(rows)

key_of <- function(model, name) {
  labels_table$colour_key[match(paste(model, name), paste(labels_table$model, labels_table$strategy_name))]
}
key_labels <- labels_table |>
  group_by(colour_key) |>
  summarise(label = if (n_distinct(strategy_name) == 1) strategy_name[1]
                    else paste(paste0(strategy_name, " (", model, ")"), collapse = " / "),
            .groups = "drop")
key_labels <- c(setNames(key_labels$label, key_labels$colour_key), tie = "tie for most common")

# --- Plot ----------------------------------------------------------------------------

shares <- data |>
  mutate(key = ifelse(tied == 1, "tie", key_of(model, most_common_name))) |>
  count(mode, model, level, level_order, group, key) |>
  group_by(mode, model, level) |>
  mutate(share = n / sum(n), runs = sum(n)) |>
  ungroup() |>
  mutate(key = factor(key, levels = key_order),
         level = factor(level, levels = unique(level[order(level_order)])),
         mode_label = factor(mode_titles[as.character(mode)], levels = mode_titles),
         model = factor(model, levels = c("DR", "IR")))
n_runs <- shares |> distinct(mode, model, level, runs) |> pull(runs) |> range()
run_text <- if (n_runs[1] == n_runs[2]) paste(n_runs[1]) else paste(n_runs, collapse = " to ")

# Thin separators between groups when several parameters share a figure.
levels_table <- shares |> distinct(level, group) |> arrange(level)
breaks_at <- which(levels_table$group[-1] != levels_table$group[-nrow(levels_table)]) + 0.5

plot <- ggplot(shares, aes(level, share, fill = key)) +
  geom_col(width = 0.75, colour = NA) +
  { if (length(breaks_at) > 0) geom_vline(xintercept = breaks_at, colour = ink_secondary, linewidth = 0.3) } +
  facet_grid(model ~ mode_label) +
  scale_fill_manual(values = key_colours, labels = key_labels, breaks = key_order, drop = TRUE) +
  scale_y_continuous(labels = scales::percent, expand = c(0, 0)) +
  guides(fill = guide_legend(nrow = 2)) +
  labs(title = title,
       subtitle = paste0("Share of runs in which each strategy name was alone the most common, ",
                         "by value of the parameter (", scoring_option, " scoring, ", run_text,
                         " runs per bar)"),
       x = NULL, y = "share of runs", caption = paste("sweeps", paste(unique(captions), collapse = ", "))) +
  theme_story() +
  theme(axis.text.x = element_text(angle = if (nlevels(shares$level) > 5) 45 else 0,
                                   hjust = if (nlevels(shares$level) > 5) 1 else 0.5))
width <- max(9, 3 + 0.45 * nlevels(shares$level) * 3)
path <- file.path(out_dir, paste0(file_name, ".png"))
ggsave(path, plot, width = min(width, 14), height = 5.6, dpi = 200)
cat("wrote", path, "\n")
