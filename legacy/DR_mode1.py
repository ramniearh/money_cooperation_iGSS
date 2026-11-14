import operator
import random
import numpy as np
import mesa
from deap import base, creator, tools, gp
from datetime import datetime

# Import infrastructure
from config import QUICK_TEST_CONFIG, get_experiment_batches
from utilities import simplify_rule, evaluate_truth_table, log_results_to_csv, generate_dashboard

# =============================================================================
# 1. GENETIC PROGRAMMING SETUP (DEAP)
# =============================================================================
def random_constant():
    return random.choice([-10, -1, 0, 1, 10])

# Primitive Set explicitly tailored for Direct Reciprocity (Mode 1)
pset_action = gp.PrimitiveSet("ActionRule_DR", 1) 
pset_action.renameArguments(ARG0='PartnerInMemory')
pset_action.addPrimitive(operator.add, 2)
pset_action.addPrimitive(operator.sub, 2)
pset_action.addPrimitive(operator.mul, 2)
pset_action.addEphemeralConstant("rand_const_act", random_constant)

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# Note: min/max depths could also be pulled from config, but keeping here for DEAP architecture clarity
toolbox.register("expr_action", gp.genHalfAndHalf, pset=pset_action, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_action)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset_action)

toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_action, pset=pset_action)
toolbox.register("select", tools.selTournament, tournsize=3)

# =============================================================================
# 2. AGENT-BASED MODEL (Asymmetric Topology & Strict Forgiveness)
# =============================================================================
class CoopAgent(mesa.Agent):
    def __init__(self, model, agent_type="iGSS-Agent"):
        super().__init__(model) 
        self.agent_type = agent_type
        self.payoff = 0
        self.memory = set() 
        
    def evaluate_partner(self, partner):
        if self.agent_type == "Unconditional Cooperator": return True
        if self.agent_type == "Defector": return False
        
        # Hardcoded Control Baseline: Tit-for-Tat
        if self.agent_type == "Control-TFT":
            return partner.unique_id not in self.memory
            
        # iGSS Evaluation
        in_memory_signal = 1 if partner.unique_id in self.memory else 0
        decision_score = self.model.igss_rule(in_memory_signal)
        return decision_score > 0

class CooperationModel(mesa.Model):
    def __init__(self, igss_rule, config, control_mode=False):
        super().__init__()
        self.igss_rule = igss_rule
        self.config = config
        self.benefit = config["BENEFIT_TO_COST_RATIO"] * config["COST"]
        self.cost = config["COST"]
        
        target_agent = "Control-TFT" if control_mode else "iGSS-Agent"
        
        for _ in range(config["NUM_IGSS"]): CoopAgent(self, target_agent)
        for _ in range(config["NUM_UC"]): CoopAgent(self, "Unconditional Cooperator")
        for _ in range(config["NUM_D"]): CoopAgent(self, "Defector")

    def step(self):
        agents = list(self.agents)
        random.shuffle(agents)

        # Implementation of the Directed Network topology (Asymmetric matching)
        for donor in agents:
            possible_recipients = [a for a in agents if a.unique_id != donor.unique_id]
            recipient = random.choice(possible_recipients)

            donor_action = donor.evaluate_partner(recipient)
            self.resolve(donor=donor, recipient=recipient, cooperates=donor_action)
            
    def resolve(self, donor, recipient, cooperates):
        if cooperates:
            donor.payoff -= self.cost
            recipient.payoff += self.benefit
        else:
            # Defection occurs: Recipient remembers the defector
            recipient.memory.add(donor.unique_id)
            
            # Forgiveness Execution: If the donor defects against someone they 
            # already remembered, they clear that person from their memory.
            if recipient.unique_id in donor.memory:
                donor.memory.remove(recipient.unique_id)

    def get_fitness_by_type(self):
        fitness_data = {}
        for a_type in ["iGSS-Agent", "Control-TFT", "Unconditional Cooperator", "Defector"]:
            type_agents = [a for a in self.agents if a.agent_type == a_type]
            if type_agents:
                fitness_data[a_type] = sum(a.payoff for a in type_agents) / len(type_agents)
            else:
                fitness_data[a_type] = 0
        return fitness_data

# =============================================================================
# 3. EVOLUTIONARY ENGINE
# =============================================================================
def evaluate_rule(individual, config):
    func_action = toolbox.compile(expr=individual)
    
    tot_igss, tot_uc, tot_d = 0, 0, 0
    runs = 3
    for _ in range(runs): 
        m = CooperationModel(igss_rule=func_action, config=config)
        for _ in range(config["NUM_ROUNDS"]): m.step()
        
        fits = m.get_fitness_by_type()
        tot_igss += fits["iGSS-Agent"]
        tot_uc += fits["Unconditional Cooperator"]
        tot_d += fits["Defector"]
        
    individual.sim_igss = tot_igss / runs
    individual.sim_uc = tot_uc / runs
    individual.sim_d = tot_d / runs
    
    tax = len(individual) * config["PARSIMONY_TAX"]
    final_score = individual.sim_igss - tax
    return final_score, 

def get_control_baseline(config):
    """Runs a strict Tit-For-Tat population to establish the theoretical optimal baseline."""
    tot_tft = 0
    runs = 5
    for _ in range(runs):
        m = CooperationModel(igss_rule=None, config=config, control_mode=True)
        for _ in range(config["NUM_ROUNDS"]): m.step()
        tot_tft += m.get_fitness_by_type()["Control-TFT"]
    return tot_tft / runs

def run_evolution(config):
    # The model uses whatever seed the config handed it
    current_seed = config.get("SEED", 42)
    random.seed(current_seed)
    np.random.seed(current_seed)

    pop = toolbox.population(n=config["POP_SIZE"])
    hof = tools.HallOfFame(config.get("HOF_SIZE", 1)) 
    history = {"max_igss": [], "avg_igss": [], "uc_scores": [], "d_scores": [], "fossil_record": {}}
    
    print("Initializing Direct Reciprocity [MODE 1] Module...")
    control_fitness = get_control_baseline(config)
    print(f"Control Baseline (Tit-for-Tat) established at: {control_fitness:.2f} average payoff.")

    # Evaluate initial population
    fitnesses = [evaluate_rule(ind, config) for ind in pop]
    for ind, fit in zip(pop, fitnesses): ind.fitness.values = fit
    hof.update(pop)
        
    for gen in range(1, config["MAX_GENS"] + 1): 
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        cxpb = config.get("CXPB", 0.5)
        mutpb = config.get("MUTPB", 0.2)

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values; del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values
                
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = [evaluate_rule(ind, config) for ind in invalid_ind]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
                
        pop[:] = offspring
        hof.update(pop) 
        
        if config.get("HOF_SIZE", 1) > 0:
            pop[-1] = toolbox.clone(hof[0])
        
        best_in_gen = tools.selBest(pop, k=1)[0]
        history["max_igss"].append(best_in_gen.sim_igss)
        history["avg_igss"].append(np.mean([ind.sim_igss for ind in pop]))
        history["uc_scores"].append(best_in_gen.sim_uc)
        history["d_scores"].append(best_in_gen.sim_d)
        
        if gen % 10 == 0 or gen == 1:
            history["fossil_record"][gen] = str(best_in_gen)
            print(f" > Gen {gen:02d} | iGSS: {best_in_gen.sim_igss:.1f} | UC: {best_in_gen.sim_uc:.1f} | D: {best_in_gen.sim_d:.1f}")

    return hof[0], history, control_fitness


# =============================================================================
# 4. EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    
    # Truth Table Mapping specific to this Mode.
    # Note on Tuples: (Output when In Memory [1], Output when Not In Memory [0])
    # True = Cooperate, False = Defect
    DR_NORMS_DICT = {
        (False, True): "Tit-for-Tat",         # Defect if in memory, Cooperate if not
        (True, True): "Unconditional Cooperator (ALL-C)",
        (False, False): "Unconditional Defector (ALL-D)",
        (True, False): "Anti-Tit-for-Tat"     # Cooperate if in memory, Defect if not
    }
    
    variables = ["PartnerInMemory"]

    # -------------------------------------------------------------------------
    # OPTION A: SINGLE VISUAL TEST
    # Uncomment the block below to run one test and view the charts on screen.
    # -------------------------------------------------------------------------
    
    print("\n--- RUNNING SINGLE VISUAL TEST ---")
    config = QUICK_TEST_CONFIG
    best_rule, history, control_fit = run_evolution(config)
    
    simp_action = str(simplify_rule(str(best_rule), variables))
    raw_tuple = evaluate_truth_table(simp_action, variables)
    identified_strategy = DR_NORMS_DICT.get(raw_tuple, f"Novel Rule: {raw_tuple}")

    run_log = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M"),
        "Machine_ID": "Local_VSCode", 
        "Run_ID": f"DR1_TEST_{datetime.now().strftime('%H%M%S')}",
        "Mode": "Direct Reciprocity Mode 1 (Action Search)",
        **config, 
        "Best_Rule_Raw": str(best_rule),
        "Best_Rule_SymPy": simp_action,
        "Identified_Strategy": identified_strategy,
        "iGSS_Max_Fitness": round(history["max_igss"][-1], 2),
        "iGSS_Avg_Fitness": round(history["avg_igss"][-1], 2),
        "UC_Fitness": round(history["uc_scores"][-1], 2),
        "D_Fitness": round(history["d_scores"][-1], 2),
        "Control_Baseline": round(control_fit, 2),
        "Manual_Validation": ""
    }
    
    
    log_results_to_csv(run_log, filename="results_DR_Mode1.csv")
    generate_dashboard(best_rule, history, run_log, save_only=False)
    """

    # -------------------------------------------------------------------------
    # OPTION B: BATCH EXPERIMENT
    # -------------------------------------------------------------------------
    
    batches = get_experiment_batches()
    print(f"\n--- STARTING BATCH EXECUTION: {len(batches)} TOTAL RUNS QUEUED ---")
    
    for i, config in enumerate(batches):
        print(f"\n[Run {i+1}/{len(batches)} | {config['CONFIG_GROUP']} Rep {config.get('REPETITION', 1)}] Seed: {config.get('SEED', 42)}")
        
        # 1. Run Evolution
        best_rule, history, control_fit = run_evolution(config)
        
        # 2. Parse the Math & Identify Strategy (The lines that were missing!)
        simp_action = str(simplify_rule(str(best_rule), variables))
        raw_tuple = evaluate_truth_table(simp_action, variables)
        identified_strategy = DR_NORMS_DICT.get(raw_tuple, f"Novel Rule: {raw_tuple}")

        # 3. Package the Data
        run_log = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M"),
            "Machine_ID": "Local_VSCode", 
            "Run_ID": f"DR1_{config.get('CONFIG_GROUP', 'Default')}_R{config.get('REPETITION', 1)}",
            "Mode": "Direct Reciprocity Mode 1",
            **config, # Automatically logs Cost, Pop_Size, SEED, REPETITION, etc.
            "Best_Rule_Raw": str(best_rule),
            "Best_Rule_SymPy": simp_action,
            "Identified_Strategy": identified_strategy,
            "iGSS_Max_Fitness": round(history["max_igss"][-1], 2),
            "iGSS_Avg_Fitness": round(history["avg_igss"][-1], 2),
            "UC_Fitness": round(history["uc_scores"][-1], 2),
            "D_Fitness": round(history["d_scores"][-1], 2),
            "Control_Baseline": round(control_fit, 2),
            "Manual_Validation": ""
        }
        
        # 4. Save and Save Dashboard (save_only=True so it doesn't interrupt the loop)
        log_results_to_csv(run_log, filename="results_DR_Mode1.csv")
        generate_dashboard(best_rule, history, run_log, save_only=True)
        
    print("\n[✓] Batch complete! Check results_DR_Mode1.csv and the /figures directory.")
    """