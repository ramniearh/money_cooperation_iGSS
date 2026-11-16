# Figures and tables from the output files of a sweep (SPEC §10).
#
#   Rscript R/figures.R <sweep directory> [--baseline <baseline directory>] [--runs all | id1,id2,...]
#
# Reads runs.csv, generations.csv, composition.csv, rules.csv and labels.csv
# from the sweep directory and, if given, baseline.csv from the baseline
# directory. Writes PNG and CSV files into <sweep directory>/figures/. Nothing
# is computed here that is not in the CSV files.
#
#   F1  composition over time: share of the population with each strategy name
#   F2  final composition: share of runs in which each strategy name ends most common
#   F3  baseline: payoff per round of each strategy when every iGSS agent uses it
#   F5  one run (only with --runs): composition and payoffs by generation
#   F6  final composition against one varying parameter, one line per strategy name
#   F7  final composition over two varying parameters, as shaded cells
#   F8  most common strategy by generation, as a share of runs
#   F9  cooperation rates of the final generation against one varying parameter
#   T4  final composition by name: the numbers behind F6 and F7
#   T5  cooperation by combination of parameters: the numbers behind F9
#   T1  search against baseline (needs --baseline)
#   T2  expressions behind each strategy in final generations
#   T3  most common strategies in final generations, by bit strings
#
# Strategies are grouped by the name SPEC §16 gives them. The names, and the
# key that fixes each name's colour, come from labels.csv; none is typed here.

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

# --- Colours and theme -----------------------------------------------------------

# One colour per kind of name, the same in every figure and every model. The
# order below is the order in which names are stacked and listed; neighbouring
# colours were checked to stay distinguishable under colour-vision deficiency.
key_colours <- c(named_1 = "#2a78d6", cooperate = "#eb6834", named_2 = "#1baf7a",
                 defect = "#eda100", named_3 = "#e87ba4", named_4 = "#008300",
                 unnamed = "#898781", tie = "#c3c2b7")
tie_label <- "tie for most common"
surface <- "#fcfcfb"
ink <- "#0b0b0b"
ink_secondary <- "#52514e"

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

# The narrowest a figure may be: room for the title and a two-row legend of names.
min_width <- 9

save_figure <- function(plot, name, width = 9, height = 6) {
  path <- file.path(out_dir, paste0(name, ".png"))
  ggsave(path, plot, width = width, height = min(height, 40), dpi = 150, limitsize = FALSE)
  cat("wrote", path, "\n")
}

save_table <- function(table, name) {
  path <- file.path(out_dir, paste0(name, ".csv"))
  write_csv(table, path, na = "")
  cat("wrote", path, "\n")
}

# --- Names and their colours, per model ---------------------------------------------

labels_table <- read_csv(file.path(sweep_dir, "labels.csv"), show_col_types = FALSE)
names_of <- function(the_model) {
  rows <- labels_table |>
    filter(model == the_model) |>
    arrange(match(colour_key, names(key_colours)))
  list(strategies = rows$strategy_name,
       with_tie = c(rows$strategy_name, tie_label),
       colours = c(setNames(key_colours[rows$colour_key], rows$strategy_name),
                   setNames(key_colours[["tie"]], tie_label)))
}
fill_scale <- function(n, with_tie = FALSE) {
  scale_fill_manual(values = n$colours, limits = if (with_tie) n$with_tie else n$strategies,
                    drop = FALSE)
}

# --- Which parameters vary in this sweep -------------------------------------------

# `assessment_rule` is the fixed (base) assessment rule; it varies in sweeps that compare such rules.
parameter_columns <- c("n_igss", "n_c", "n_d", "b", "c", "e_action", "rounds", "assessment_rule",
                       "pop_size",
                       "generations", "episodes_per_score", "tournament_size", "cxpb", "mutpb",
                       "init_depth_min", "init_depth_max", "mut_depth_min", "mut_depth_max",
                       "depth_limit", "parsimony_tax")

varying <- function(data) {
  columns <- intersect(parameter_columns, names(data))
  varies <- function(column) {
    if (column == "assessment_rule" && "model" %in% names(data)) {
      # DR and IR have different base rules, so this varies only if it does within a model.
      any(tapply(data[[column]], data$model, n_distinct) > 1)
    } else {
      n_distinct(data[[column]]) > 1
    }
  }
  columns[vapply(columns, varies, logical(1))]
}

# One label per combination of the varying parameters, for example "n_c=0, n_d=32".
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

text_columns <- cols(action_expression = "c", action_simplified = "c",
                     assessment_expression = "c", assessment_simplified = "c")
# A commit id such as "61e0808" looks like a number in scientific notation, so it is read as text.
version_column <- cols(code_version = "c")
runs <- read_csv(file.path(sweep_dir, "runs.csv"), show_col_types = FALSE, col_types = version_column)
generations <- read_csv(file.path(sweep_dir, "generations.csv"), show_col_types = FALSE,
                        col_types = version_column)
composition <- read_csv(file.path(sweep_dir, "composition.csv"), show_col_types = FALSE)
rule_rows <- read_csv(file.path(sweep_dir, "rules.csv"), show_col_types = FALSE,
                      col_types = text_columns)

cell_columns <- varying(runs)
runs <- add_cell(runs, cell_columns)
generations <- add_cell(generations, cell_columns)
n_cells <- n_distinct(runs$cell)
n_seeds <- runs |> count(mode, model, scoring, cell) |> pull(n) |> max()
caption <- paste0("sweep '", runs$sweep[1], "', code ", runs$code_version[1],
                  ", up to ", n_seeds, " runs per panel or bar")
run_info <- runs |> select(run_id, mode, model, scoring, cell, pop_size, generations)

# Number of rules with each strategy name, per run and generation.
by_name <- composition |>
  group_by(run_id, generation, strategy_name) |>
  summarise(count = sum(count), .groups = "drop")

groups <- runs |> distinct(mode, model) |> arrange(mode, model)

# --- F1 and F2, per mode and model ------------------------------------------------------

for (g in seq_len(nrow(groups))) {
  the_mode <- groups$mode[g]
  the_model <- groups$model[g]
  n <- names_of(the_model)
  tag <- paste0("mode", the_mode, "_", the_model)
  heading <- paste0("mode ", the_mode, ", ", the_model)

  # F1: one figure per scoring option. A name absent from a generation counts as 0.
  for (the_scoring in unique(runs$scoring)) {
    selected <- run_info |> filter(mode == the_mode, model == the_model, scoring == the_scoring)
    if (nrow(selected) == 0) next
    data <- by_name |>
      semi_join(selected, by = "run_id") |>
      mutate(strategy_name = factor(strategy_name, levels = n$strategies)) |>
      complete(run_id, generation, strategy_name, fill = list(count = 0)) |>
      inner_join(selected, by = "run_id") |>
      group_by(cell, generation, strategy_name) |>
      summarise(share = mean(count / pop_size), .groups = "drop")
    plot <- ggplot(data, aes(generation, share, fill = strategy_name)) +
      geom_area(colour = surface, linewidth = 0.3, show.legend = c(fill = TRUE)) +
      facet_wrap(~cell, ncol = min(4, n_cells)) +
      fill_scale(n) +
      scale_y_continuous(labels = scales::percent, expand = c(0, 0)) +
      scale_x_continuous(expand = c(0, 0)) +
      labs(title = paste0("F1. Composition over time: ", heading, ", ", the_scoring, " scoring"),
           subtitle = "Share of the population with each strategy, averaged over runs",
           x = "generation", y = "share of the population", caption = caption) +
      theme_igss() +
      guides(fill = guide_legend(nrow = 2))
    save_figure(plot, paste0("F1_composition_", tag, "_", the_scoring),
                width = max(min_width, 3 + 2.6 * min(4, n_cells)), height = 2.5 + 2.2 * ceiling(n_cells / 4))
  }

  # F2: which name ends most common, one bar per cell.
  data <- runs |>
    filter(mode == the_mode, model == the_model) |>
    mutate(name = factor(ifelse(tied == 1, tie_label, most_common_name), levels = n$with_tie)) |>
    count(scoring, cell, name) |>
    group_by(scoring, cell) |>
    mutate(share = n / sum(n)) |>
    ungroup()
  plot <- ggplot(data, aes(share, cell, fill = name)) +
    geom_col(colour = surface, linewidth = 0.6, width = 0.7, show.legend = c(fill = TRUE)) +
    facet_wrap(~scoring, labeller = as_labeller(function(x) paste(x, "scoring"))) +
    fill_scale(n, with_tie = TRUE) +
    scale_x_continuous(labels = scales::percent, expand = c(0, 0)) +
    scale_y_discrete(limits = rev) +
    labs(title = paste0("F2. Final composition: ", heading),
         subtitle = "Share of runs in which each strategy is the most common in the final generation",
         x = "share of runs", y = NULL, caption = caption) +
    theme_igss() +
    theme(panel.grid.major.y = element_blank()) +
    guides(fill = guide_legend(nrow = 2))
  save_figure(plot, paste0("F2_final_", tag), width = 10, height = 2.8 + 0.35 * n_cells)
}

# --- T2 and T3: what is behind the names ------------------------------------------------

# rules.csv carries the scoring option since v1.8; run_info supplies it for older files too.
t2 <- rule_rows |>
  select(-any_of("scoring")) |>
  inner_join(run_info, by = "run_id") |>
  filter(generation == generations) |>
  count(mode, model, scoring, strategy, strategy_name, action_simplified, assessment_simplified,
        name = "rules") |>
  arrange(mode, model, scoring, desc(rules))
save_table(t2, "T2_expressions")

t3 <- runs |>
  count(mode, model, scoring, cell, most_common, most_common_name, name = "runs") |>
  arrange(mode, model, scoring, cell, desc(runs))
save_table(t3, "T3_most_common_strategies")

# --- Final composition by name: T4, F6, F7; and F8 --------------------------------------

# Two measures of how often a strategy name is found at the end of the runs (SPEC §10):
#   ends most common: the share of runs in which the name alone is the most common
#                     in the final generation; ties are their own category;
#   final share:      the share of the P rules with the name in the final generation,
#                     averaged over runs.
measure_labels <- c(ends_most_common = "ends most common\n(share of runs)",
                    final_share = "final share\n(mean share of the population)")
sequential <- c("#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")  # one hue, light to dark

# "n_d=0, b=2" for each row of `data`, as a factor ordered by the columns' values.
label_of <- function(data, columns, empty) {
  if (length(columns) == 0) return(factor(rep(empty, nrow(data))))
  parts <- lapply(columns, function(column) paste0(column, "=", data[[column]]))
  labels <- do.call(paste, c(parts, sep = ", "))
  factor(labels, levels = unique(labels[do.call(order, unname(as.list(data[columns])))]))
}

final_measures <- function(the_mode, the_model, the_scoring) {
  n <- names_of(the_model)
  selected <- runs |> filter(mode == the_mode, model == the_model, scoring == the_scoring)
  cells <- selected |> distinct(cell, across(all_of(cell_columns)))
  ends <- selected |>
    mutate(strategy_name = factor(ifelse(tied == 1, tie_label, most_common_name),
                                  levels = n$with_tie)) |>
    count(cell, strategy_name, name = "runs_most_common") |>
    complete(cell = cells$cell, strategy_name, fill = list(runs_most_common = 0))
  shares <- by_name |>
    inner_join(selected |> select(run_id, cell, pop_size, generations), by = "run_id") |>
    filter(generation == generations) |>
    mutate(strategy_name = factor(strategy_name, levels = n$with_tie)) |>
    group_by(cell, strategy_name) |>
    summarise(total_share = sum(count / pop_size), .groups = "drop")
  ends |>
    left_join(shares, by = c("cell", "strategy_name")) |>
    left_join(selected |> count(cell, name = "runs"), by = "cell") |>
    left_join(cells, by = "cell") |>
    mutate(mode = the_mode, model = the_model, scoring = the_scoring,
           ends_most_common = runs_most_common / runs,
           # A name absent from a run's final generation has share 0 in that run.
           final_share = ifelse(strategy_name == tie_label, NA,
                                coalesce(total_share, 0) / runs))
}

# Share of interactions in which the donor cooperated, by who the donor and the partner are.
rate_labels <- c(coop_rate = "every donor",
                 coop_igss = "iGSS agents",
                 coop_igss_to_igss = "iGSS agents,
partner an iGSS agent",
                 coop_igss_to_c = "iGSS agents,
partner an unconditional
cooperator",
                 coop_igss_to_d = "iGSS agents,
partner an unconditional
defector")

scoring_groups <- runs |> distinct(mode, model, scoring) |> arrange(mode, model, scoring)
t4_parts <- list()
t5_parts <- list()

for (g in seq_len(nrow(scoring_groups))) {
  the_mode <- scoring_groups$mode[g]
  the_model <- scoring_groups$model[g]
  the_scoring <- scoring_groups$scoring[g]
  n <- names_of(the_model)
  tag <- paste0("mode", the_mode, "_", the_model, "_", the_scoring)
  heading <- paste0("mode ", the_mode, ", ", the_model, ", ", the_scoring, " scoring")

  measures <- final_measures(the_mode, the_model, the_scoring)
  t4_parts[[g]] <- measures |>
    select(mode, model, scoring, all_of(cell_columns), strategy_name, runs,
           ends_most_common, final_share)
  long <- measures |>
    pivot_longer(c(ends_most_common, final_share), names_to = "measure", values_to = "value") |>
    filter(!is.na(value)) |>
    mutate(measure_label = factor(measure_labels[measure], levels = measure_labels))
  # Names that are 0 in every panel of a figure are left out of it.
  present <- function(data) {
    found <- data |> group_by(strategy_name) |> summarise(any = any(value > 0)) |> filter(any)
    n$with_tie[n$with_tie %in% found$strategy_name]
  }

  # F6: both measures against each varying parameter, one line per name.
  for (parameter in cell_columns) {
    others <- setdiff(cell_columns, parameter)
    data <- long
    data$panel <- label_of(data, others, "all other parameters at their base values")
    data$x <- factor(data[[parameter]], levels = sort(unique(data[[parameter]])))
    shown <- present(data)
    data <- data |> filter(strategy_name %in% shown)
    plot <- ggplot(data, aes(x, value, colour = strategy_name, group = strategy_name)) +
      geom_line(linewidth = 0.8) +
      geom_point(size = 2.6) +
      facet_grid(measure_label ~ panel) +
      scale_colour_manual(values = n$colours, limits = shown) +
      scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
      labs(title = paste0("F6. Final composition against ", parameter, ": ", heading),
           subtitle = "How often each strategy is found in the final generation, by two measures",
           x = parameter, y = NULL, caption = caption) +
      theme_igss() +
      theme(strip.text.y = element_text(angle = 0, hjust = 0)) +
      guides(colour = guide_legend(nrow = if (length(shown) > 4) 2 else 1))
    save_figure(plot, paste0("F6_final_by_", parameter, "_", tag),
                width = max(min_width, 4.5 + 2.2 * n_distinct(data$panel)), height = 6.5)
  }

  # F7: one measure over the first two varying parameters, one row of panels per name.
  if (length(cell_columns) >= 2) {
    others <- cell_columns[-(1:2)]
    for (measure_name in names(measure_labels)) {
      data <- long |> filter(measure == measure_name)
      data$panel <- label_of(data, others, "all other parameters at their base values")
      data$x <- factor(data[[cell_columns[1]]], levels = sort(unique(data[[cell_columns[1]]])))
      data$y <- factor(data[[cell_columns[2]]], levels = sort(unique(data[[cell_columns[2]]])))
      shown <- present(data)
      data <- data |>
        filter(strategy_name %in% shown) |>
        mutate(strategy_name = factor(strategy_name, levels = shown))
      plot <- ggplot(data, aes(x, y, fill = value)) +
        geom_tile(colour = surface, linewidth = 1.2) +
        geom_text(aes(label = scales::percent(value, accuracy = 1), colour = value > 0.6),
                  size = 3.4, show.legend = FALSE) +
        facet_grid(strategy_name ~ panel) +
        scale_fill_gradientn(colours = sequential, limits = c(0, 1), labels = scales::percent) +
        scale_colour_manual(values = c(`TRUE` = "white", `FALSE` = ink)) +
        scale_x_discrete(expand = c(0, 0)) +
        scale_y_discrete(expand = c(0, 0)) +
        labs(title = paste0("F7. Final composition over ", cell_columns[1], " and ",
                            cell_columns[2], ": ", heading),
             subtitle = sub("\n", " ", measure_labels[[measure_name]]),
             x = cell_columns[1], y = cell_columns[2], caption = caption) +
        theme_igss() +
        theme(panel.grid = element_blank(), strip.text.y = element_text(angle = 0, hjust = 0),
              legend.key.width = unit(3, "lines"))
      save_figure(plot, paste0("F7_final_over_", cell_columns[1], "_", cell_columns[2], "_",
                               measure_name, "_", tag),
                  width = max(min_width, 4 + (0.9 * nlevels(data$x) + 0.4) * n_distinct(data$panel)),
                  height = 3 + (0.5 * nlevels(data$y) + 0.3) * length(shown))
    }
  }

  # F8: which name is the most common in each generation, as a share of runs.
  data <- generations |>
    filter(mode == the_mode, model == the_model, scoring == the_scoring) |>
    mutate(name = factor(ifelse(grepl("+", most_common, fixed = TRUE), tie_label, most_common_name),
                         levels = n$with_tie)) |>
    count(cell, generation, name) |>
    complete(nesting(cell, generation), name, fill = list(n = 0)) |>
    group_by(cell, generation) |>
    mutate(share = n / sum(n)) |>
    ungroup()
  plot <- ggplot(data, aes(generation, share, fill = name)) +
    geom_area(colour = surface, linewidth = 0.3, show.legend = c(fill = TRUE)) +
    facet_wrap(~cell, ncol = min(4, n_cells)) +
    fill_scale(n, with_tie = TRUE) +
    scale_y_continuous(labels = scales::percent, expand = c(0, 0)) +
    scale_x_continuous(expand = c(0, 0)) +
    labs(title = paste0("F8. Most common strategy by generation: ", heading),
         subtitle = "Share of runs in which each strategy is the most common in that generation",
         x = "generation", y = "share of runs", caption = caption) +
    theme_igss() +
    guides(fill = guide_legend(nrow = 2))
  save_figure(plot, paste0("F8_most_common_by_generation_", tag),
              width = max(min_width, 3 + 2.6 * min(4, n_cells)),
              height = 2.5 + 2.2 * ceiling(n_cells / 4))

  # F9: cooperation rates of the final generation against each varying parameter.
  # A rate with no agents of that type is an empty cell and drops out here.
  rates <- runs |>
    filter(mode == the_mode, model == the_model, scoring == the_scoring) |>
    select(run_id, all_of(cell_columns), all_of(names(rate_labels))) |>
    pivot_longer(all_of(names(rate_labels)), names_to = "rate", values_to = "value") |>
    filter(!is.na(value)) |>
    mutate(rate_label = factor(rate_labels[rate], levels = rate_labels))
  t5_parts[[g]] <- rates |>
    group_by(across(all_of(cell_columns)), rate) |>
    summarise(runs = n(), mean = mean(value), lowest = min(value), highest = max(value),
              .groups = "drop") |>
    mutate(mode = the_mode, model = the_model, scoring = the_scoring, .before = 1)
  for (parameter in cell_columns) {
    data <- rates
    data$panel <- label_of(data, setdiff(cell_columns, parameter),
                           "all other parameters at their base values")
    data$x <- factor(data[[parameter]], levels = sort(unique(data[[parameter]])))
    means <- data |>
      group_by(panel, rate_label, x) |>
      summarise(value = mean(value), .groups = "drop")
    plot <- ggplot(data, aes(x, value)) +
      geom_point(colour = ink_secondary, alpha = 0.35, size = 1.4,
                 position = position_jitter(width = 0.08, height = 0, seed = 1)) +
      # A rate present at only one value of the parameter gets a point and no line.
      geom_line(data = means |> group_by(panel, rate_label) |> filter(n() > 1),
                aes(group = 1), colour = ink, linewidth = 0.8) +
      geom_point(data = means, colour = ink, size = 2.6) +
      facet_grid(rate_label ~ panel) +
      scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
      labs(title = paste0("F9. Cooperation against ", parameter, ": ", heading),
           subtitle = paste0("Share of interactions in which the donor cooperated, final generation. ",
                             "Line: mean over runs; dots: runs."),
           x = parameter, y = NULL, caption = caption) +
      theme_igss() +
      theme(strip.text.y = element_text(angle = 0, hjust = 0))
    save_figure(plot, paste0("F9_cooperation_by_", parameter, "_", tag),
                width = max(min_width, 5 + 2.2 * n_distinct(data$panel)),
                height = 2.5 + 1.5 * n_distinct(data$rate))
  }
}

save_table(bind_rows(t4_parts), "T4_final_by_name")
save_table(bind_rows(t5_parts), "T5_cooperation")

# --- F5: single runs, on request --------------------------------------------------------

if (!is.null(runs_wanted)) {
  ids <- if (runs_wanted == "all") runs$run_id else strsplit(runs_wanted, ",")[[1]]
  type_labels <- c(payoff_igss = "iGSS agents", payoff_c = "unconditional cooperators",
                   payoff_d = "unconditional defectors")
  for (id in ids) {
    one <- generations |> filter(run_id == id)
    if (nrow(one) == 0) {
      cat("no such run:", id, "\n")
      next
    }
    n <- names_of(one$model[1])
    top <- by_name |>
      filter(run_id == id) |>
      mutate(strategy_name = factor(strategy_name, levels = n$strategies)) |>
      complete(generation, strategy_name, fill = list(count = 0)) |>
      mutate(panel = "number of rules with each strategy")
    bottom <- one |>
      pivot_longer(c(payoff_igss, payoff_c, payoff_d), names_to = "type", values_to = "payoff") |>
      filter(!is.na(payoff)) |>
      mutate(type = factor(type_labels[type], levels = type_labels),
             panel = "payoff per round")
    plot <- ggplot() +
      geom_area(data = top, aes(generation, count, fill = strategy_name), colour = surface,
                linewidth = 0.3, show.legend = c(fill = TRUE)) +
      geom_line(data = bottom, aes(generation, payoff, linetype = type), colour = ink,
                linewidth = 0.7) +
      facet_wrap(~panel, ncol = 1, scales = "free_y") +
      fill_scale(n) +
      scale_linetype_manual(values = c("iGSS agents" = "solid", "unconditional cooperators" = "dashed",
                                       "unconditional defectors" = "dotted")) +
      labs(title = paste0("F5. One run: ", id),
           subtitle = paste0("mode ", one$mode[1], ", ", one$model[1], ", ", one$scoring[1],
                             " scoring, ", one$cell[1], ", seed ", one$run_seed[1]),
           x = "generation", y = NULL) +
      theme_igss() +
      theme(legend.box = "vertical") +
      guides(fill = guide_legend(nrow = 2),
             linetype = guide_legend(override.aes = list(fill = NA)))
    save_figure(plot, paste0("F5_run_", id), width = 8, height = 7)
  }
}

# --- Baseline: F3 and T1 ----------------------------------------------------------------

if (!is.null(baseline_dir)) {
  baseline <- read_csv(file.path(baseline_dir, "baseline.csv"), show_col_types = FALSE,
                       col_types = version_column)
  baseline <- add_cell(baseline, varying(baseline))
  n_baseline_cells <- n_distinct(baseline$cell)
  baseline_caption <- paste0("baseline for sweep '", baseline$sweep[1], "', code ",
                             baseline$code_version[1])
  baseline_groups <- baseline |> distinct(mode, model) |> arrange(mode, model)

  # F3: one box per strategy over seeds; background types' mean payoffs as marks.
  for (g in seq_len(nrow(baseline_groups))) {
    the_mode <- baseline_groups$mode[g]
    the_model <- baseline_groups$model[g]
    n <- names_of(the_model)
    data <- baseline |>
      filter(mode == the_mode, model == the_model) |>
      mutate(strategy_name = factor(strategy_name, levels = n$strategies),
             strategy = sub("^TT:", "", strategy))
    n_strategies <- n_distinct(data$strategy)
    background <- data |>
      group_by(cell, strategy) |>
      summarise(`unconditional cooperators` = mean(payoff_c),
                `unconditional defectors` = mean(payoff_d), .groups = "drop") |>
      pivot_longer(-c(cell, strategy), names_to = "type", values_to = "payoff") |>
      filter(!is.na(payoff))
    plot <- ggplot(data, aes(payoff_igss, strategy)) +
      geom_boxplot(aes(fill = strategy_name), colour = ink_secondary, linewidth = 0.3,
                   width = 0.6, outlier.size = 0.6, show.legend = c(fill = TRUE)) +
      geom_point(data = background, aes(payoff, strategy, shape = type), colour = ink, size = 1.8) +
      facet_wrap(~cell, ncol = min(4, n_baseline_cells)) +
      fill_scale(n) +
      scale_shape_manual(values = c(3, 4)) +
      scale_y_discrete(limits = rev) +
      labs(title = paste0("F3. Baseline: mode ", the_mode, ", ", the_model),
           subtitle = paste0("Boxes: payoff per round of the iGSS agents when all use one strategy, over seeds.\n",
                             "Marks: mean payoff per round of each background type. ",
                             "Strategy = action rule / assessment rule."),
           x = "payoff per round", y = NULL, caption = baseline_caption) +
      theme_igss() +
      theme(legend.box = "vertical", panel.grid.major.y = element_blank()) +
      guides(fill = guide_legend(nrow = 2),
             shape = guide_legend(override.aes = list(fill = NA, linetype = 0)))
    save_figure(plot, paste0("F3_baseline_mode", the_mode, "_", the_model),
                width = max(min_width, 3 + 3 * min(4, n_baseline_cells)),
                height = 3.5 + (0.6 + 0.22 * n_strategies) * ceiling(n_baseline_cells / 4))
  }

  # T1: for each environment and scoring option, what the baseline shows and what the search did.
  # Strategies that tie for the best payoff cannot be told apart in that environment.
  environment_columns <- c("mode", "model", "assessment_rule", "n_igss", "n_c", "n_d", "b", "c",
                           "e_action", "rounds")
  baseline_view <- baseline |>
    group_by(across(all_of(environment_columns)), strategy, strategy_name) |>
    summarise(payoff = mean(payoff_igss), .groups = "drop") |>
    group_by(across(all_of(environment_columns))) |>
    summarise(best_alone = paste(strategy[payoff == max(payoff)], collapse = "+"),
              best_alone_name = paste(unique(strategy_name[payoff == max(payoff)]), collapse = "+"),
              .groups = "drop")
  t1 <- runs |>
    group_by(across(all_of(environment_columns)), scoring, cell) |>
    summarise(runs = n(), recovery_rate = mean(recovered), tie_rate = mean(tied), .groups = "drop") |>
    left_join(baseline_view, by = environment_columns)
  save_table(t1, "T1_search_against_baseline")
}
