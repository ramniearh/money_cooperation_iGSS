# Compact figures for one environment of a sweep, every mode and model side by side.
#
#   Rscript R/cell_story.R <sweep directory> --baseline <baseline directory> --where n_c=32,n_d=32,b=3,e_action=0.05
#
# Meant for communication: each figure is a grid of small panels, one row per model
# and one column per mode, all for the one environment named by --where. Reads
# runs.csv, generations.csv, composition.csv and labels.csv from the sweep and
# baseline.csv from the baseline directory. Writes PNG files into
# <sweep directory>/figures/cell_<environment>/. Nothing is computed here that is
# not in the CSV files.
#
#   C1  composition over generations (mean share of the rules with each name)
#   C2  final population of every run (share of the rules with each name)
#   C3  payoff per round by generation, one line per run, with baseline levels
#   C4  baseline payoff of the leading strategies, with the runs that ended on each
#   C5  cooperation of iGSS donors in the final generation, by type of partner
#
# Names and colours come from labels.csv, with the colour keys of R/figures.R.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
})

# --- Arguments ----------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
usage <- "usage: Rscript R/cell_story.R <sweep directory> --baseline <directory> --where name=value,..."
if (length(args) < 1) stop(usage)
sweep_dir <- args[1]
option <- function(name) {
  i <- match(name, args)
  if (is.na(i) || i == length(args)) NULL else args[i + 1]
}
baseline_dir <- option("--baseline")
where_text <- option("--where")
if (is.null(baseline_dir) || is.null(where_text)) stop(usage)

where <- strsplit(strsplit(where_text, ",")[[1]], "=")
where <- setNames(vapply(where, `[`, "", 2), vapply(where, `[`, "", 1))
in_cell <- function(data) {
  for (column in names(where)) {
    data <- data[as.numeric(data[[column]]) == as.numeric(where[[column]]), ]
  }
  data
}
cell_label <- paste(names(where), where, sep = "=", collapse = ", ")
cell_tag <- gsub("[^A-Za-z0-9.]+", "_", paste0(names(where), where, collapse = "_"))

out_dir <- file.path(sweep_dir, "figures", paste0("cell_", cell_tag))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# --- Colours and theme (as in R/figures.R) ------------------------------------------

key_colours <- c(named_1 = "#2a78d6", cooperate = "#eb6834", named_2 = "#1baf7a",
                 defect = "#eda100", named_3 = "#e87ba4", named_4 = "#008300",
                 unnamed = "#898781", tie = "#c3c2b7")
surface <- "#fcfcfb"
ink <- "#0b0b0b"
ink_secondary <- "#52514e"

theme_cell <- function() {
  theme_minimal(base_size = 10) +
    theme(
      plot.background = element_rect(fill = surface, colour = NA),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(colour = "#e1e0d9", linewidth = 0.3),
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

save_figure <- function(plot, name, width = 9, height = 5.6) {
  path <- file.path(out_dir, paste0(name, ".png"))
  ggsave(path, plot, width = width, height = height, dpi = 200)
  cat("wrote", path, "\n")
}

# --- Read ------------------------------------------------------------------------

version_column <- cols(code_version = "c")
runs <- read_csv(file.path(sweep_dir, "runs.csv"), show_col_types = FALSE,
                 col_types = version_column) |> in_cell()
if (nrow(runs) == 0) stop("no runs match ", cell_label)
if (n_distinct(runs$scoring) != 1) stop("expected one scoring option in the selected runs")
generations <- read_csv(file.path(sweep_dir, "generations.csv"), show_col_types = FALSE,
                        col_types = version_column) |> in_cell()
composition <- read_csv(file.path(sweep_dir, "composition.csv"), show_col_types = FALSE) |>
  semi_join(runs, by = "run_id")
baseline <- read_csv(file.path(baseline_dir, "baseline.csv"), show_col_types = FALSE,
                     col_types = version_column) |> in_cell()
labels_table <- read_csv(file.path(sweep_dir, "labels.csv"), show_col_types = FALSE)

# --- Names, colour keys and panel labels ---------------------------------------------

# One legend for both models: each colour key lists the names it stands for, per model.
key_order <- names(key_colours)
key_labels <- labels_table |>
  group_by(colour_key) |>
  summarise(label = if (n_distinct(strategy_name) == 1) strategy_name[1]
                    else paste(paste0(strategy_name, " (", model, ")"), collapse = " / "),
            .groups = "drop")
key_labels <- c(setNames(key_labels$label, key_labels$colour_key), tie = "tie for most common")
fill_keys <- scale_fill_manual(values = key_colours, labels = key_labels, breaks = key_order,
                               drop = TRUE)
colour_keys <- scale_colour_manual(values = key_colours, labels = key_labels, breaks = key_order,
                                   drop = TRUE)
key_of <- function(model, name) {
  labels_table$colour_key[match(paste(model, name), paste(labels_table$model, labels_table$strategy_name))]
}

mode_titles <- c(`1` = "mode 1: action rule evolves",
                 `2` = "mode 2: assessment rule evolves",
                 `3` = "mode 3: both evolve")
panel_label <- function(data) {
  data |> mutate(
    mode_label = factor(mode_titles[as.character(mode)], levels = mode_titles),
    model = factor(model, levels = c("DR", "IR")))
}
facets <- facet_grid(model ~ mode_label)

n_runs <- runs |> count(mode, model) |> pull(n) |> max()
pop <- runs$pop_size[1]
caption <- paste0("sweep '", runs$sweep[1], "', ", runs$scoring[1], " scoring, code ",
                  runs$code_version[1], "; ", cell_label, "; ", n_runs, " runs per panel")
short <- function(strategy) sub("^TT:", "", strategy)

# Final outcome of each run: the colour key of its most common name, or a tie.
final <- runs |>
  mutate(final_key = ifelse(tied == 1, "tie", key_of(model, most_common_name))) |>
  select(run_id, mode, model, final_key, most_common, most_common_count)

# --- C1: composition over generations ------------------------------------------------

by_key <- composition |>
  inner_join(runs |> select(run_id, mode, model), by = "run_id") |>
  mutate(key = key_of(model, strategy_name)) |>
  group_by(run_id, mode, model, generation, key) |>
  summarise(count = sum(count), .groups = "drop")

c1 <- by_key |>
  complete(nesting(run_id, mode, model), generation, key, fill = list(count = 0)) |>
  group_by(mode, model, generation, key) |>
  summarise(share = mean(count) / pop, .groups = "drop") |>
  filter(key %in% unique(by_key$key)) |>
  mutate(key = factor(key, levels = key_order)) |>
  panel_label()

plot <- ggplot(c1, aes(generation, share, fill = key)) +
  geom_area(colour = surface, linewidth = 0.2) +
  facets + fill_keys +
  scale_y_continuous(labels = scales::percent, expand = c(0, 0)) +
  scale_x_continuous(expand = c(0, 0)) +
  guides(fill = guide_legend(nrow = 2)) +
  labs(title = "C1. Composition of the population over generations",
       subtitle = paste0("Share of the ", pop, " rules with each strategy name, averaged over runs"),
       x = "generation", y = "share of rules", caption = caption) +
  theme_cell()
save_figure(plot, "C1_composition_over_generations")

# --- C2: final population of every run -------------------------------------------------

c2 <- by_key |>
  filter(generation == max(generation)) |>
  inner_join(final |> select(run_id, final_key, most_common_count), by = "run_id") |>
  mutate(key = factor(key, levels = key_order),
         final_rank = match(final_key, key_order)) |>
  group_by(mode, model) |>
  mutate(run = as.integer(factor(run_id, levels = unique(run_id[order(final_rank, -most_common_count, run_id)])))) |>
  ungroup() |>
  panel_label()

plot <- ggplot(c2, aes(run, count / pop, fill = key)) +
  geom_col(width = 0.85, colour = NA) +
  facets + fill_keys +
  scale_y_continuous(labels = scales::percent, expand = c(0, 0)) +
  scale_x_continuous(breaks = NULL, expand = c(0.01, 0.01)) +
  guides(fill = guide_legend(nrow = 2)) +
  labs(title = "C2. Final population of every run",
       subtitle = paste0("One bar per run: share of the ", pop,
                         " rules with each strategy name in the final generation; runs sorted by their most common name"),
       x = "runs", y = "share of rules", caption = caption) +
  theme_cell()
save_figure(plot, "C2_final_population_by_run")

# --- Baseline summaries ------------------------------------------------------------------

base_summary <- baseline |>
  group_by(mode, model, strategy, strategy_name) |>
  summarise(payoff = mean(payoff_igss), low = min(payoff_igss), high = max(payoff_igss),
            seeds = n(), .groups = "drop") |>
  mutate(key = key_of(model, strategy_name))

# --- C3: payoff by generation, with baseline levels -----------------------------------------

c3 <- generations |>
  select(run_id, mode, model, generation, payoff_igss) |>
  inner_join(final |> select(run_id, final_key), by = "run_id") |>
  mutate(final_key = factor(final_key, levels = key_order)) |>
  panel_label()
c3_levels <- base_summary |>
  filter(key != "unnamed") |>
  mutate(key = factor(key, levels = key_order)) |>
  panel_label()

plot <- ggplot(c3, aes(generation, payoff_igss)) +
  geom_hline(data = c3_levels, aes(yintercept = payoff, colour = key),
             linetype = "dashed", linewidth = 0.5) +
  geom_line(aes(group = run_id, colour = final_key), linewidth = 0.3, alpha = 0.6) +
  facets + colour_keys +
  scale_x_continuous(expand = c(0, 0)) +
  guides(colour = guide_legend(nrow = 2, override.aes = list(linetype = "solid", linewidth = 1, alpha = 1))) +
  labs(title = "C3. Payoff per round of the iGSS agents, by generation",
       subtitle = paste0("Solid: one line per run, mean over the rules of the generation, coloured by the run's final most common name.\n",
                         "Dashed: baseline payoff of each named strategy, every iGSS agent using it (mean over ",
                         max(base_summary$seeds), " seeds)"),
       x = "generation", y = "payoff per round", caption = caption) +
  theme_cell()
save_figure(plot, "C3_payoff_by_generation", height = 6)

# --- C4: baseline ranking, with the runs that ended on each strategy ---------------------------

ended <- final |>
  filter(final_key != "tie") |>
  count(mode, model, strategy = most_common, name = "runs_ended")
top_n <- 6
c4 <- base_summary |>
  left_join(ended, by = c("mode", "model", "strategy")) |>
  mutate(runs_ended = coalesce(runs_ended, 0L)) |>
  group_by(mode, model) |>
  arrange(desc(payoff), .by_group = TRUE) |>
  mutate(rank = row_number()) |>
  filter(rank <= top_n | runs_ended > 0) |>
  ungroup() |>
  mutate(label = paste0(short(strategy), "  ", strategy_name),
         key = factor(key, levels = key_order),
         panel = factor(paste0(model, ", ", mode_titles[as.character(mode)]),
                        levels = as.vector(t(outer(c("DR", "IR"), mode_titles, paste, sep = ", ")))),
         row = paste(panel, label, sep = "|"))
c4 <- c4 |> arrange(panel, payoff) |> mutate(row = factor(row, levels = unique(row)))
x_right <- max(c4$high) + 0.1

plot <- ggplot(c4, aes(y = row)) +
  geom_segment(aes(x = low, xend = high, yend = row), colour = "#c3c2b7", linewidth = 1.6) +
  geom_point(aes(x = payoff, colour = key), size = 2.4) +
  geom_text(aes(x = x_right, label = ifelse(runs_ended > 0, paste0(runs_ended, "/", n_runs), "–")),
            hjust = 1, size = 3, colour = ink) +
  facet_wrap(~panel, ncol = 3, scales = "free_y") +
  scale_y_discrete(labels = function(x) sub("^.*\\|", "", x)) +
  colour_keys +
  guides(colour = guide_legend(nrow = 2)) +
  labs(title = "C4. Search against baseline",
       subtitle = paste0("Dot: baseline payoff per round when every iGSS agent uses the strategy (mean over ",
                         max(base_summary$seeds), " seeds; bar: lowest to highest).\n",
                         "Right: runs in which the strategy was the single most common in the final generation.\n",
                         "Shown: the ", top_n, " best strategies in the baseline, and any strategy a run ended on."),
       x = "baseline payoff per round", y = NULL, caption = caption) +
  theme_cell() +
  theme(panel.grid.major.y = element_blank())
save_figure(plot, "C4_search_against_baseline", width = 12, height = 6.5)

# --- C5: cooperation in the final generation -------------------------------------------------

partner_types <- c(coop_igss_to_igss = "iGSS agents", coop_igss_to_c = "unconditional\ncooperators",
                   coop_igss_to_d = "unconditional\ndefectors")
c5 <- runs |>
  select(run_id, mode, model, all_of(names(partner_types))) |>
  pivot_longer(all_of(names(partner_types)), names_to = "rate", values_to = "value") |>
  filter(!is.na(value)) |>
  inner_join(final |> select(run_id, final_key), by = "run_id") |>
  mutate(partner = factor(partner_types[rate], levels = partner_types),
         final_key = factor(final_key, levels = key_order)) |>
  panel_label()
c5_means <- c5 |> group_by(mode_label, model, partner) |> summarise(value = mean(value), .groups = "drop")

plot <- ggplot(c5, aes(partner, value)) +
  geom_jitter(aes(colour = final_key), width = 0.18, height = 0, size = 1.4, alpha = 0.8) +
  geom_errorbar(data = c5_means, aes(ymin = value, ymax = value), width = 0.5, linewidth = 0.6, colour = ink) +
  facets + colour_keys +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
  guides(colour = guide_legend(nrow = 2)) +
  labs(title = "C5. How often iGSS donors cooperate, by type of partner (final generation)",
       subtitle = "Dot: one run, coloured by its final most common name. Line: mean over runs.",
       x = NULL, y = "share of interactions in which the donor cooperated", caption = caption) +
  theme_cell()
save_figure(plot, "C5_cooperation_by_partner", height = 6)
