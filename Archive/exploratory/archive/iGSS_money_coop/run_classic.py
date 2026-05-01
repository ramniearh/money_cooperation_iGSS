from model_classic import ClassicModel
import matplotlib.pyplot as plt

# 1. Initialize the Simulation Parameters
# These match the default inputs from your NetLogo interface
NUM_AGENTS_PER_STRATEGY = 100
BENEFIT_COST_RATIO = 5.0
LIQUIDITY = 1.0
INITIAL_REPUTATION = 1
STEPS = 3000 # Number of ticks to run the tournament

print(f"Setting up tournament with BCR = {BENEFIT_COST_RATIO} and Liquidity = {LIQUIDITY}...")

model = ClassicModel(
    n_coops=NUM_AGENTS_PER_STRATEGY, 
    n_defs=NUM_AGENTS_PER_STRATEGY, 
    n_dirs=NUM_AGENTS_PER_STRATEGY, 
    n_inds=NUM_AGENTS_PER_STRATEGY, 
    n_mons=NUM_AGENTS_PER_STRATEGY,
    bcr=BENEFIT_COST_RATIO, 
    liq=LIQUIDITY, 
    init_rep=INITIAL_REPUTATION
)

# 2. Run the Simulation
print(f"Running for {STEPS} steps...")
for i in range(STEPS):
    model.step()
    # Print a progress update every 500 steps
    if i % 500 == 0:
        print(f"Tick {i} complete.")

# 3. Extract the Data
# Mesa's DataCollector automatically formats everything into a Pandas DataFrame!
results_df = model.datacollector.get_model_vars_dataframe()

# 4. Plot the Results
print("Simulation complete. Generating chart...")
plt.figure(figsize=(10, 6))
plt.plot(results_df)

plt.title("Evolutionary Tournament: Surviving Strategies", fontsize=14)
plt.xlabel("Ticks", fontsize=12)
plt.ylabel("Number of Agents", fontsize=12)
plt.legend(results_df.columns, loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout() # Ensures the legend doesn't get cut off

# Show the interactive chart
plt.show()