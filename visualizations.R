library(tidyverse)

# Load data
df <- read_csv("results_DR_Mode2.csv")

# ===================================================================================
# CHART 1:  ABM environment parameters (Agent Counts, Duration & Benefit/Cost Ratios)
# ===================================================================================
chart1_overview <- df %>%
  mutate(
    # Create the combined X-axis label. Using \n creates the line break so the text stacks on the axis
    Env_Scenario = paste0(NUM_ROUNDS, " steps\nbc = ", BENEFIT_TO_COST_RATIO),
    Env_Scenario = factor(Env_Scenario, levels = c(
      "10 steps\nbc = 2", 
      "100 steps\nbc = 2", 
      "10 steps\nbc = 5", 
      "100 steps\nbc = 5"
    ))
  ) %>%
  ggplot(aes(x = Env_Scenario, fill = Identified_Strategy)) +
  geom_bar(position = "fill", color = "white", linewidth = 0.2) +
  facet_grid(NUM_D ~ NUM_UC,
             labeller = labeller(
               NUM_D = function(x) paste(x, "Defectors"),
               NUM_UC = function(x) paste(x, "Cooperators"))) +
  scale_y_continuous(labels = function(x) paste0(x * 100, "%")) +
  #scale_fill_brewer(palette = "Dark2", na.value = "gray50") +
  labs(
    title = "Strategy Emergence Map - across ABM parameters",
    subtitle = "(unfiltered by iGSS parameters)",
    x = "Environmental Scenario",
    y = "Proportion of Evolved Strategies",
    fill = "Evolved Strategy"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    strip.background = element_rect(fill = "gray90", color = NA),
    strip.text = element_text(face = "bold", size = 10),
    axis.text.x = element_text(size = 9) 
  )

chart1_overview



# ====================================================================
# CHART 1A: Baseline model (10 cooperators, 10 defectors, 10 iGSS agents)
# ====================================================================
chart1A_baseline <- df %>%
  # Filter for the baseline ecology with 10 agents of each type
  filter(
    NUM_UC == 10, 
    NUM_D == 10, 
    NUM_IGSS == 10
  ) %>%
  # REMOVED: drop_na() so unidentified strategies remain visible as warnings
  mutate(
    # Locked as factors to prevent future alphabetical sorting errors
    Round_Label = factor(paste(NUM_ROUNDS, "Rounds"), 
                         levels = c("10 Rounds", "100 Rounds")),
    BC_Label = factor(paste("Benefit/Cost:", BENEFIT_TO_COST_RATIO), 
                      levels = c("Benefit/Cost: 2", "Benefit/Cost: 5"))
  ) %>%
  # Map strategy to both X and Fill
  ggplot(aes(x = Identified_Strategy, fill = Identified_Strategy)) +
  # Using the default geom_bar() without position="fill" yields raw counts
  geom_bar(color = "black", linewidth = 0.3) +
  # Wrap by the two economic variables
  facet_grid(BC_Label ~ Round_Label) +
  # REVISION: Restored the Dark2 professional palette
  #scale_fill_brewer(palette = "Dark2", na.value = "gray50") +
  labs(
    title = "Evolutionary Dynamics in Baseline Ecology (10 UC / 10 D)",
    subtitle = "(unfiltered by iGSS parameters)",
    x = "Evolved Strategy",
    y = "Count of Emergence (Runs)",
    fill = "Strategy"
  ) +
  theme_minimal() +
  theme(
    legend.position = "none", # Hide legend since x-axis labels the strategies
    strip.background = element_rect(fill = "gray90", color = NA),
    strip.text = element_text(face = "bold", size = 11),
    # Tilt the labels so they don't overlap
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

chart1A_baseline

# ========================================================
# CHART 1 COLLAPSED: aggregating all ABM and iGSS settings
# ========================================================
chart1B_simple <- df %>%
  # Use the dummy variable (x = "") to force a single stacked column
  ggplot(aes(x = "", fill = Identified_Strategy)) +
  geom_bar(position = "fill", color = "black", linewidth = 0.3, width = 0.6) +
  # Strict quadrant matrix using native labeller instead of mutated columns
  facet_grid(NUM_D ~ NUM_UC,
             labeller = labeller(
               NUM_D = function(x) paste(x, "Defectors"),
               NUM_UC = function(x) paste(x, "Cooperators"))) +
  scale_y_continuous(labels = function(x) paste0(x * 100, "%")) +
  scale_fill_brewer(palette = "Dark2", na.value = "gray50") +
  labs(
    title = "Strategy Emergence by Agent Population Environment",
    subtitle = "aggregating all ABM and iGSS settings",
    x = NULL, # Remove the x-axis label
    y = "Overall Frequency of Emergence",
    fill = "Strategy"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom", 
    strip.background = element_rect(fill = "gray90", color = NA),
    strip.text = element_text(face = "bold", size = 11),
    # Hide the x-axis tick marks and text
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    panel.grid.major.x = element_blank()
  )

chart1B_simple


# ====================================================================
# CHART 2: Robustness of TFT Emergence across iGSS Parameters
# ====================================================================


TARGET_STRATEGY <- "Tit-for-tat" ###########

chart2_baseline_one_strategy <- df %>%
  # 1. Isolate the baseline ecology
  filter(
    NUM_UC == 10, 
    NUM_D == 10, 
    NUM_IGSS == 10,
    #BENEFIT_TO_COST_RATIO == 2,
    #NUM_ROUNDS == 100
  ) %>%
  # REMOVED: drop_na() so unidentified strategies don't artificially shrink the denominator
  
  # 2. Group by the three genetic parameters we want to test
  group_by(POP_SIZE, TREE_MAX_DEPTH, PARSIMONY_TAX) %>%
  # 3. Calculate the share of Tit-for-Tat
  summarize(
    Total_Runs = n(),
    # CRITICAL FIX: na.rm = TRUE ensures the sum doesn't crash if an NA is present
    TFT_Count = sum(Identified_Strategy == TARGET_STRATEGY, na.rm = TRUE),
    TFT_Share = TFT_Count / Total_Runs,
    .groups = "drop"
  ) %>%
  # 4 & 5. Plot the calculated shares (mutate removed, using native labeller)
  ggplot(aes(x = as.factor(PARSIMONY_TAX), y = TFT_Share, fill = as.factor(TREE_MAX_DEPTH))) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7, color = "black", linewidth = 0.3) +
  facet_wrap(~ POP_SIZE,
             labeller = labeller(
               POP_SIZE = function(x) paste("Rule Population Size:", x)
             )) +
  # Base R percentage formatting (removes the 'scales' package dependency)
  scale_y_continuous(labels = function(x) paste0(x * 100, "%"), limits = c(0, 1)) +
  # Unified the color language to match the Dark2 palette used in Chart 1
  scale_fill_brewer(palette = "Dark2") +
  labs(
    title = "Robustness of TARGET STRATEGY Emergence to iGSS parameters",
    subtitle = "(unfiltered by ABM parameters)",
    x = "Parsimony Tax Weight",
    y = "Share of Runs resulting in Tit-for-Tat",
    fill = "Max Tree Depth"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    strip.background = element_rect(fill = "gray90", color = NA),
    strip.text = element_text(face = "bold", size = 10)
  )

chart2_baseline_one_strategy



# ====================================================================
# CHART 2 VARIANT: Robustness of TFT Emergence faceted by Economics
# ====================================================================
chart2B_baseline_one_strategy_by_ABM <- df %>%
  # 1. Isolate the baseline ecology
  filter(
    NUM_UC == 10, 
    NUM_D == 10, 
    NUM_IGSS == 10
  ) %>%
  # 2. Group by the ABM economics PLUS the genetic parameters 
  # (Notice POP_SIZE is removed, so it gets averaged out)
  group_by(BENEFIT_TO_COST_RATIO, NUM_ROUNDS, TREE_MAX_DEPTH, PARSIMONY_TAX) %>%
  # 3. Calculate the share of Tit-for-Tat safely
  summarize(
    Total_Runs = n(),
    TFT_Count = sum(Identified_Strategy == TARGET_STRATEGY, na.rm = TRUE),
    TFT_Share = TFT_Count / Total_Runs,
    .groups = "drop"
  ) %>%
  # 4. Lock the facet labels as factors so they sort logically
  mutate(
    Round_Label = factor(paste(NUM_ROUNDS, "Rounds"), 
                         levels = c("10 Rounds", "100 Rounds")),
    BC_Label = factor(paste("Benefit/Cost:", BENEFIT_TO_COST_RATIO), 
                      levels = c("Benefit/Cost: 2", "Benefit/Cost: 5"))
  ) %>%
  # 5. Plot the data
  ggplot(aes(x = as.factor(PARSIMONY_TAX), y = TFT_Share, fill = as.factor(TREE_MAX_DEPTH))) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7, color = "black", linewidth = 0.3) +
  # Use facet_grid to create the clean economic matrix
  facet_grid(BC_Label ~ Round_Label) +
  # Base R percentage formatting
  scale_y_continuous(labels = function(x) paste0(x * 100, "%"), limits = c(0, 1)) +
  scale_fill_brewer(palette = "Dark2") +
  labs(
    title = "Sensitivity of TARGET STRATEGY Emergence to iGSS & ABM Parameters",
    subtitle = "Ecology: 10/10/10 | Aggregated across Rule Population Sizes",
    x = "Parsimony Tax Weight",
    y = "Share of Runs resulting in Tit-for-Tat",
    fill = "Max Tree Depth"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    strip.background = element_rect(fill = "gray90", color = NA),
    strip.text = element_text(face = "bold", size = 10)
  )

chart2B_baseline_one_strategy_by_ABM

