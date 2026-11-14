


library(tidyverse)

# Load data (Updated to Mode 3)
df <- read_csv("results_IR_Mode2.csv")


# ====================================================================
# PIE CHARTS: Faceted by ABM Parameters (Aggregating over iGSS)
# ====================================================================
chart_abm_faceted <- df %>%
  filter(
    NUM_UC == 10, 
    NUM_D == 10, 
    NUM_IGSS == 10
  ) %>%
  # Group only by ABM parameters and Joint Strategy
  group_by(BENEFIT_TO_COST_RATIO, NUM_ROUNDS, Joint_Strategy) %>%
  summarize(Count = n(), .groups = "drop") %>%
  
  # Lock factor levels for clean ordering
  mutate(
    Round_Label = factor(paste(NUM_ROUNDS, "Rounds"), levels = c("10 Rounds", "100 Rounds")),
    BC_Label = factor(paste("B/C Ratio:", BENEFIT_TO_COST_RATIO), levels = c("B/C Ratio: 2", "B/C Ratio: 5"))
  ) %>%
  
  ggplot(aes(x = "", y = Count, fill = Joint_Strategy)) +
  geom_bar(stat = "identity", width = 1, position = "fill", color = "white", linewidth = 0.5) +
  coord_polar(theta = "y", start = 0) +
  
  # Facet grid by ABM economics
  facet_grid(BC_Label ~ Round_Label) +
  
  scale_fill_viridis_d(option = "turbo", na.value = "gray50") + 
  labs(
    title = "Strategy Discovery Space across ABM Environmental Parameters",
    subtitle = "Aggregated across all iGSS search parameters",
    fill = "Discovered Joint Strategy"
  ) +
  theme_void() + 
  theme(
    legend.position = "right",
    plot.title = element_text(face = "bold", hjust = 0.5, margin = margin(b = 5, t = 10)),
    plot.subtitle = element_text(hjust = 0.5, margin = margin(b = 15)),
    strip.text = element_text(face = "bold", size = 10, margin = margin(b = 5, t = 5)),
    strip.background = element_rect(fill = "gray90", color = NA)
  )

chart_abm_faceted


# ====================================================================
# ALTERNATIVE: Fitness Distribution per Discovered Joint Strategy
# ====================================================================
fitness_by_strategy_plot <- df %>%
  filter(
    NUM_UC == 10, 
    NUM_D == 10, 
    NUM_IGSS == 10
  ) %>%
  ggplot(aes(x = reorder(Joint_Strategy, iGSS_Max_Fitness, FUN = median), 
             y = iGSS_Max_Fitness, 
             fill = Joint_Strategy)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.2, size = 1.5, alpha = 0.5) +
  coord_flip() + # Flips to horizontal for easy reading of strategy names
  scale_fill_viridis_d(option = "turbo") +
  labs(
    title = "iGSS Agent Max Fitness by Discovered Joint Strategy",
    subtitle = "Comparing evolutionary success across strategy types",
    x = "Discovered Joint Strategy",
    y = "iGSS Max Fitness (Cumulative Payoff)"
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    plot.title = element_text(face = "bold", size = 12),
    axis.text.y = element_text(face = "bold", size = 10)
  )

fitness_by_strategy_plot

# ====================================================================
# PIE CHARTS: Parsimony Tax vs. Rule Population Size
# ====================================================================
#filtered FOR ONLY BC5 and 100 rounds
chart_parsimony_vs_pop <- df %>%
  filter(
    NUM_UC == 10, 
    NUM_D == 10, 
    NUM_IGSS == 10,
    BENEFIT_TO_COST_RATIO = 5,
    NUM_ROUNDS = 100
  ) %>%
  # Group by Parsimony Tax, Rule Pop Size, and Joint Strategy
  group_by(PARSIMONY_TAX, POP_SIZE, Joint_Strategy) %>%
  summarize(Count = n(), .groups = "drop") %>%
  
  mutate(
    Tax_Label = factor(paste("Parsimony Tax:", PARSIMONY_TAX)),
    Pop_Label = factor(paste("Rule Pop Size:", POP_SIZE))
  ) %>%
  
  ggplot(aes(x = "", y = Count, fill = Joint_Strategy)) +
  geom_bar(stat = "identity", width = 1, position = "fill", color = "white", linewidth = 0.5) +
  coord_polar(theta = "y", start = 0) +
  
  # Facet grid: Parsimony Tax (rows) vs. Rule Population Size (columns)
  facet_grid(Tax_Label ~ Pop_Label) +
  
  scale_fill_viridis_d(option = "turbo", na.value = "gray50") + 
  labs(
    title = "Strategy Discovery Space: Parsimony Tax vs. Rule Population",
    subtitle = "Aggregated across ABM conditions and Tree Depths",
    fill = "Discovered Joint Strategy"
  ) +
  theme_void() + 
  theme(
    legend.position = "right",
    plot.title = element_text(face = "bold", hjust = 0.5, margin = margin(b = 5, t = 10)),
    plot.subtitle = element_text(hjust = 0.5, margin = margin(b = 15)),
    strip.text = element_text(face = "bold", size = 10, margin = margin(b = 5, t = 5)),
    strip.background = element_rect(fill = "gray90", color = NA)
  )

chart_parsimony_vs_pop


# ====================================================================
# PIE CHARTS: Parsimony Tax vs. Max Tree Depth
# ====================================================================
chart_parsimony_vs_depth <- df %>%
  filter(
    NUM_UC == 10, 
    NUM_D == 10, 
    NUM_IGSS == 10
  ) %>%
  # Group by Parsimony Tax, Max Tree Depth, and Joint Strategy
  group_by(PARSIMONY_TAX, TREE_MAX_DEPTH, Joint_Strategy) %>%
  summarize(Count = n(), .groups = "drop") %>%
  
  mutate(
    Tax_Label = factor(paste("Parsimony Tax:", PARSIMONY_TAX)),
    Depth_Label = factor(paste("Max Tree Depth:", TREE_MAX_DEPTH))
  ) %>%
  
  ggplot(aes(x = "", y = Count, fill = Joint_Strategy)) +
  geom_bar(stat = "identity", width = 1, position = "fill", color = "white", linewidth = 0.5) +
  coord_polar(theta = "y", start = 0) +
  
  # Facet grid: Parsimony Tax (rows) vs. Max Tree Depth (columns)
  facet_grid(Tax_Label ~ Depth_Label) +
  
  scale_fill_viridis_d(option = "turbo", na.value = "gray50") + 
  labs(
    title = "Strategy Discovery Space: Parsimony Tax vs. Max Tree Depth",
    subtitle = "Aggregated across ABM conditions and Rule Population Sizes",
    fill = "Discovered Joint Strategy"
  ) +
  theme_void() + 
  theme(
    legend.position = "right",
    plot.title = element_text(face = "bold", hjust = 0.5, margin = margin(b = 5, t = 10)),
    plot.subtitle = element_text(hjust = 0.5, margin = margin(b = 15)),
    strip.text = element_text(face = "bold", size = 10, margin = margin(b = 5, t = 5)),
    strip.background = element_rect(fill = "gray90", color = NA)
  )

chart_parsimony_vs_depth

