# Figures and tables from the output files of a sweep (SPEC §10).
#
#   Rscript R/figures.R <sweep directory> [--baseline <baseline directory>] [--runs all | id1,id2,...]
#
# Reads runs.csv, generations.csv and rules.csv from the sweep directory and,
# if given, baseline.csv from the baseline directory. Writes PNG and CSV files
# into <sweep directory>/figures/. Nothing is computed here that is not in the
# CSV files.
#
#   F1  composition over time: share of rules with each bit string, by generation
#   F2  final composition: share of runs in which each bit string ends most common
#   F3  baseline, rules alone: payoff per round of each rule
#   F4  baseline, rules in pairs: payoff of X's carriers minus payoff of Y's
#   F5  one run (only with --runs): bit-string counts and payoffs by generation
#   T1  search against baseline (needs --baseline)
#   T2  expressions behind each bit string in final generations

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
})

# --- Arguments ----------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: Rscript R/figures.R <sweep directory> [--baseline <directory>] [--runs all|ids]")
sweep_dir <- args[1]
option <- function(name) {
  i <- match(name, args)
  if (is.na(i) || i == length(args)) NULL else args[i + 1]
}
baseline_dir <- option("--baseline")
runs_wanted <- option("--runs")

out_dir <- file.path(sweep_dir, "figures")
dir.create(out_dir, showWarnings = FALSE)

# --- Rules, labels and colours ---------------------------------------------------

# Order and wording follow SPEC §6. Each bit string keeps its colour in every figure.
rule_levels <- c("10", "11", "01", "00")
# Legend text comes from labels.csv, written by the Python code from SPEC §16:
# what each bit string does and, where the model has one assessment rule in
# this sweep, what the literature calls the resulting strategy.
labels_table <- read_csv(file.path(sweep_dir, "labels.csv"), show_col_types = FALSE)
labels_for <- function(the_model) {
  rows <- labels_table |> filter(model == the_model)
  with_names <- n_distinct(rows$assessment_rule) == 1
  rows <- rows |> distinct(bit_string, .keep_all = TRUE)
  text <- paste0(sub("^TT:", "", rows$bit_string), "  ", rows$description)
  if (with_names) {
    named <- rows$literature_name != "unnamed"
    text[named] <- paste0(text[named], ": ", rows$literature_name[named])
  }
  text <- paste0(text, ifelse(rows$canonical == 1, " (canonical)", ""))
  c(setNames(text, sub("^TT:", "", rows$bit_string)), "tie" = "tie for most common")
}
rule_colours <- c("10" = "#2a78d6", "11" = "#eb6834", "01" = "#1baf7a", "00" = "#eda100",
                  "tie" = "#898781")
surface <- "#fcfcfb"
ink <- "#0b0b0b"
ink_secondary <- "#52514e"
ink_muted <- "#898781"

strip_tt <- function(x) sub("^TT:", "", x)

theme_igss <- function() {
  theme_minimal(base_size = 11) +
    theme(
      plot.background = element_rect(fill = surface, colour = NA),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(colour = "#e1e0d9", linewidth = 0.3),
      panel.spacing = unit(1.4, "lines"),
      text = element_text(colour = ink),
      axis.text = element_text(colour = ink_secondary),
      strip.text = element_text(colour = ink, hjust = 0),
      plot.title = element_text(face = "bold"),
      plot.subtitle = element_text(colour = ink_secondary),
      plot.margin = margin(10, 20, 10, 10),
      legend.position = "bottom",
      legend.title = element_blank()
    )
}

save_figure <- function(plot, name, width = 9, height = 6) {
  path <- file.path(out_dir, paste0(name, ".png"))
  ggsave(path, plot, width = width, height = min(height, 40), dpi = 150, limitsize = FALSE)
  cat("wrote", path, "\n")
}

# --- Which parameters vary in this sweep -------------------------------------------

parameter_columns <- c("n_igss", "n_c", "n_d", "b", "c", "e_action", "rounds", "pop_size",
                       "generations", "episodes_per_score", "tournament_size", "cxpb", "mutpb",
                       "init_depth_min", "init_depth_max", "mut_depth_min", "mut_depth_max",
                       "depth_limit", "parsimony_tax")

varying <- function(data) {
  columns <- intersect(parameter_columns, names(data))
  columns[vapply(columns, function(column) n_distinct(data[[column]]) > 1, logical(1))]
}

# One label per combination of the varying parameters, for example "n_c=0, n_d=10".
add_cell <- function(data, columns) {
  if (length(columns) == 0) {
    data$cell <- "base configuration"
  } else {
    parts <- lapply(columns, function(column) paste0(column, "=", data[[column]]))
    data$cell <- do.call(paste, c(parts, sep = ", "))
  }
  data$cell <- factor(data$cell, levels = unique(data$cell))
  data
}

# --- Read ------------------------------------------------------------------------

runs <- read_csv(file.path(sweep_dir, "runs.csv"), show_col_types = FALSE)
generations <- read_csv(file.path(sweep_dir, "generations.csv"), show_col_types = FALSE)
rule_rows <- read_csv(file.path(sweep_dir, "rules.csv"), show_col_types = FALSE)

cell_columns <- varying(runs)
runs <- add_cell(runs, cell_columns)
generations <- add_cell(generations, cell_columns)
n_cells <- n_distinct(runs$cell)
n_seeds <- runs |> count(model, scoring, cell) |> pull(n) |> max()
sweep_name <- runs$sweep[1]
caption <- paste0("sweep '", sweep_name, "', code ", runs$code_version[1],
                  ", up to ", n_seeds, " runs per panel or bar")

counts_long <- generations |>
  pivot_longer(starts_with("count_"), names_to = "rule", names_prefix = "count_",
               values_to = "count") |>
  mutate(rule = factor(rule, levels = rule_levels), share = count / pop_size)

# --- F1: composition over time ------------------------------------------------------

for (the_model in unique(generations$model)) {
  rule_labels <- labels_for(the_model)
  for (the_scoring in unique(generations$scoring)) {
    data <- counts_long |>
      filter(model == the_model, scoring == the_scoring) |>
      group_by(cell, generation, rule) |>
      summarise(share = mean(share), .groups = "drop")
    if (nrow(data) == 0) next
    plot <- ggplot(data, aes(generation, share, fill = rule)) +
      geom_area(colour = surface, linewidth = 0.3, show.legend = c(fill = TRUE)) +
      facet_wrap(~cell, ncol = min(4, n_cells)) +
      scale_fill_manual(values = rule_colours, labels = rule_labels, drop = FALSE) +
      scale_y_continuous(labels = scales::percent, expand = c(0, 0)) +
      scale_x_continuous(expand = c(0, 0)) +
      labs(title = paste0("F1. Composition over time: ", the_model, ", ", the_scoring, " scoring"),
           subtitle = "Share of the rules with each bit string, averaged over runs",
           x = "generation", y = "share of rules", caption = caption) +
      theme_igss() +
      guides(fill = guide_legend(nrow = 2))
    save_figure(plot, paste0("F1_composition_", the_model, "_", the_scoring),
                width = 3 + 2.6 * min(4, n_cells), height = 2.5 + 2.2 * ceiling(n_cells / 4))
  }
}

# --- F2: final composition ---------------------------------------------------------------

final <- runs |>
  mutate(rule = ifelse(tied == 1, "tie", strip_tt(most_common)),
         rule = factor(rule, levels = c(rule_levels, "tie")))

for (the_model in unique(final$model)) {
  rule_labels <- labels_for(the_model)
  data <- final |>
    filter(model == the_model) |>
    count(scoring, cell, rule) |>
    group_by(scoring, cell) |>
    mutate(share = n / sum(n)) |>
    ungroup()
  plot <- ggplot(data, aes(share, cell, fill = rule)) +
    geom_col(colour = surface, linewidth = 0.6, width = 0.7, show.legend = c(fill = TRUE)) +
    facet_wrap(~scoring, labeller = as_labeller(function(x) paste(x, "scoring"))) +
    scale_fill_manual(values = rule_colours, labels = rule_labels, drop = FALSE) +
    scale_x_continuous(labels = scales::percent, expand = c(0, 0)) +
    scale_y_discrete(limits = rev) +
    labs(title = paste0("F2. Final composition: ", the_model),
         subtitle = "Share of runs in which each bit string is the most common in the final generation",
         x = "share of runs", y = NULL, caption = caption) +
    theme_igss() +
    theme(panel.grid.major.y = element_blank()) +
    guides(fill = guide_legend(nrow = 2))
  save_figure(plot, paste0("F2_final_", the_model), width = 10, height = 2.8 + 0.35 * n_cells)
}

# --- T2: expressions behind each bit string ------------------------------------------------

t2 <- rule_rows |>
  inner_join(runs |> select(run_id, model, scoring, generations), by = "run_id") |>
  filter(generation == generations) |>
  count(model, scoring, bit_string, simplified, name = "rules") |>
  arrange(model, scoring, bit_string, desc(rules))
write_csv(t2, file.path(out_dir, "T2_expressions.csv"), na = "")
cat("wrote", file.path(out_dir, "T2_expressions.csv"), "\n")

# --- F5: single runs, on request --------------------------------------------------------

if (!is.null(runs_wanted)) {
  ids <- if (runs_wanted == "all") runs$run_id else strsplit(runs_wanted, ",")[[1]]
  type_labels <- c(payoff_igss = "iGSS agents", payoff_c = "unconditional cooperators",
                   payoff_d = "unconditional defectors")
  for (id in ids) {
    one <- generations |> filter(run_id == id)
    if (nrow(one) > 0) rule_labels <- labels_for(one$model[1])
    if (nrow(one) == 0) { cat("no such run:", id, "\n"); next }
    top <- counts_long |> filter(run_id == id) |>
      mutate(panel = "number of rules with each bit string")
    bottom <- one |>
      pivot_longer(c(payoff_igss, payoff_c, payoff_d), names_to = "type", values_to = "payoff") |>
      filter(!is.na(payoff)) |>
      mutate(type = factor(type_labels[type], levels = type_labels),
             panel = "payoff per round")
    plot <- ggplot() +
      geom_area(data = top, aes(generation, count, fill = rule), colour = surface, linewidth = 0.3,
                show.legend = c(fill = TRUE)) +
      geom_line(data = bottom, aes(generation, payoff, linetype = type), colour = ink,
                linewidth = 0.7) +
      facet_wrap(~panel, ncol = 1, scales = "free_y") +
      scale_fill_manual(values = rule_colours, labels = rule_labels, drop = FALSE) +
      scale_linetype_manual(values = c("iGSS agents" = "solid", "unconditional cooperators" = "dashed",
                                       "unconditional defectors" = "dotted")) +
      labs(title = paste0("F5. One run: ", id),
           subtitle = paste0(one$model[1], ", ", one$scoring[1], " scoring, ", one$cell[1],
                             ", seed ", one$run_seed[1]),
           x = "generation", y = NULL) +
      theme_igss() +
      theme(legend.box = "vertical") +
      guides(fill = guide_legend(nrow = 2),
             linetype = guide_legend(override.aes = list(fill = NA)))
    save_figure(plot, paste0("F5_run_", id), width = 8, height = 7)
  }
}

# --- Baseline: F3, F4, T1 -------------------------------------------------------------------

if (!is.null(baseline_dir)) {
  baseline <- read_csv(file.path(baseline_dir, "baseline.csv"), show_col_types = FALSE) |>
    mutate(rule_x = strip_tt(rule_x), rule_y = strip_tt(rule_y))
  baseline_columns <- varying(baseline)
  baseline <- add_cell(baseline, baseline_columns)
  n_baseline_cells <- n_distinct(baseline$cell)
  baseline_caption <- paste0("baseline for sweep '", baseline$sweep[1], "', code ",
                             baseline$code_version[1])

  alone <- baseline |>
    filter(kind == "alone") |>
    mutate(rule_x = factor(rule_x, levels = rule_levels))
  pairs <- baseline |> filter(kind == "pair")

  for (the_model in unique(baseline$model)) {
    rule_labels <- labels_for(the_model)
    # F3: one box per rule; the background types' mean payoffs as marks beside them.
    data <- alone |> filter(model == the_model)
    background <- data |>
      group_by(cell, rule_x) |>
      summarise(`unconditional cooperators` = mean(payoff_c),
                `unconditional defectors` = mean(payoff_d), .groups = "drop") |>
      pivot_longer(-c(cell, rule_x), names_to = "type", values_to = "payoff") |>
      filter(!is.na(payoff))
    plot <- ggplot(data, aes(rule_x, payoff_x)) +
      geom_boxplot(aes(fill = rule_x), colour = ink_secondary, linewidth = 0.3, width = 0.55,
                   outlier.size = 0.8, show.legend = c(fill = TRUE)) +
      geom_point(data = background, aes(rule_x, payoff, shape = type), colour = ink, size = 2.5) +
      facet_wrap(~cell, ncol = min(4, n_baseline_cells)) +
      scale_fill_manual(values = rule_colours, labels = rule_labels, drop = FALSE) +
      scale_shape_manual(values = c(3, 4)) +
      labs(title = paste0("F3. Baseline, rules alone: ", the_model),
           subtitle = "Boxes: payoff per round of the iGSS agents when all carry one rule, over seeds.
Marks: mean payoff per round of each background type.",
           x = "rule carried by every iGSS agent", y = "payoff per round",
           caption = baseline_caption) +
      theme_igss() +
      theme(legend.box = "vertical", panel.grid.major.x = element_blank()) +
      guides(fill = guide_legend(nrow = 2),
             shape = guide_legend(override.aes = list(fill = NA, linetype = 0)))
    save_figure(plot, paste0("F3_baseline_alone_", the_model),
                width = 3 + 2.8 * min(4, n_baseline_cells),
                height = 3.2 + 2.4 * ceiling(n_baseline_cells / 4))

    # F4: payoff of X's carriers minus payoff of Y's carriers, one cell per pair.
    data <- pairs |>
      filter(model == the_model) |>
      group_by(cell, rule_x, rule_y) |>
      summarise(difference = mean(payoff_x - payoff_y), .groups = "drop") |>
      mutate(rule_x = factor(rule_x, levels = rule_levels),
             rule_y = factor(rule_y, levels = rule_levels))
    limit <- max(abs(data$difference))
    plot <- ggplot(data, aes(rule_y, rule_x, fill = difference)) +
      geom_tile(colour = surface, linewidth = 1) +
      geom_text(aes(label = formatC(difference, format = "f", digits = 2)), colour = ink, size = 3) +
      facet_wrap(~cell, ncol = min(4, n_baseline_cells)) +
      scale_fill_gradient2(low = "#eb6834", mid = "#f0efec", high = "#2a78d6", midpoint = 0,
                           limits = c(-limit, limit), name = "X minus Y") +
      labs(title = paste0("F4. Baseline, rules in pairs: ", the_model),
           subtitle = "Mean payoff per round of the carriers of X minus that of the carriers of Y.
Blue: X earns more. Orange: Y earns more.",
           x = "rule Y", y = "rule X", caption = baseline_caption) +
      theme_igss() +
      theme(panel.grid.major = element_blank(), legend.title = element_text(),
            legend.key.width = unit(1.2, "cm"))
    save_figure(plot, paste0("F4_baseline_pairs_", the_model),
                width = 3 + 2.6 * min(4, n_baseline_cells),
                height = 3 + 2.2 * ceiling(n_baseline_cells / 4))
  }

  # T1: for each environment and scoring option, what the baseline shows and what the search did.
  environment_columns <- c("model", "n_igss", "n_c", "n_d", "b", "c", "e_action", "rounds")
  ranked <- alone |>
    group_by(across(all_of(environment_columns)), rule_x) |>
    summarise(payoff = mean(payoff_x), zero_read = sum(reads_zero_x) > 0, .groups = "drop")
  baseline_view <- ranked |>
    group_by(across(all_of(environment_columns))) |>
    summarise(
      best_rule_alone = paste0("TT:", paste(rule_x[payoff == max(payoff)], collapse = "+")),
      input_zero_read_under_10 = zero_read[rule_x == "10"],
      .groups = "drop")
  t1 <- runs |>
    group_by(across(all_of(environment_columns)), scoring, cell) |>
    summarise(runs = n(), recovery_rate = mean(recovered), tie_rate = mean(tied), .groups = "drop") |>
    left_join(baseline_view, by = environment_columns)
  write_csv(t1, file.path(out_dir, "T1_search_against_baseline.csv"), na = "")
  cat("wrote", file.path(out_dir, "T1_search_against_baseline.csv"), "\n")
}
