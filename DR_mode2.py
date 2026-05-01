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
# 1. GENETIC PROGRAMMING SETUP (MEMORY LEDGER SEARCH)
# =============================================================================
def random_constant():
    return random.choice([-10, -1, 0, 1, 10])

# Primitive Set explicitly tailored for Memory Assessment (DR Mode 2)
pset_assessment = gp.PrimitiveSet("AssessmentRule_DR", 2) 
pset_assessment.renameArguments(ARG0='DonorAction', ARG1='DonorInMemory')
pset_assessment.addPrimitive(operator.add, 2)
pset_assessment.addPrimitive(operator.sub, 2)
pset_assessment.addPrimitive(operator.mul, 2)
pset_assessment.addEphemeralConstant("rand_const_ass", random_constant)

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr_assessment", gp.genHalfAndHalf, pset=pset_assessment, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_assessment)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset_assessment)

toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_assessment, pset=pset_assessment)
toolbox.register("select", tools.selTournament, tournsize=3)

# =============================================================================
# 2. AGENT-BASED MODEL (Asymmetric Topology & Memory Mechanics)
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
        
        # MODE 2 ACTION RULE: Hardcoded to Strict Tit-for-Tat for both classes
        return partner.unique_id not in self.memory

class CooperationModel(mesa.Model):
    def __init__(self, igss_rule, config, control_mode=False):
        super().__init__()
        self.igss_rule = igss_rule
        self.config = config
        self.control_mode = control_mode
        self.benefit = config["BENEFIT_TO_COST_RATIO"] * config["COST"]
        self.cost = config["COST"]
        
        target_agent = "Control-Agent" if control_mode else "iGSS-Agent"
        
        for _ in range(config["NUM_IGSS"]): CoopAgent(self, target_agent)
        for _ in range(config["NUM_UC"]): CoopAgent(self, "Unconditional Cooperator")
        for _ in range(config["NUM_D"]): CoopAgent(self, "Defector")

    def step(self):
        agents = list(self.agents)
        random.shuffle(agents)

        for donor in agents:
            possible_recipients = [a for a in agents if a.unique_id != donor.unique_id]
            recipient = random.choice(possible_recipients)

            donor_action = donor.evaluate_partner(recipient)
            self.resolve(donor=donor, recipient=recipient, cooperates=donor_action)
            
    def resolve(self, donor, recipient, cooperates):
        # 1. Economic Execution
        if cooperates:
            donor.payoff -= self.cost
            recipient.payoff += self.benefit
            
        # 2. Institutional Assessment (Updating Memory)
        if self.control_mode:
            # HARDCODED CONTROL: The strict pseudocode implementation
            if not cooperates:
                recipient.memory.add(donor.unique_id)
            else:
                # Execution of Forgiveness
                if donor.unique_id in recipient.memory:
                    recipient.memory.remove(donor.unique_id)
        else:
            # EVOLVED iGSS ASSESSMENT: The Physics of the Grudge
            action_val = 1 if cooperates else 0
            donor_in_mem = 1 if donor.unique_id in recipient.memory else 0
            
            decision_score = self.igss_rule(action_val, donor_in_mem)
            
            # > 0 means "Add/Keep in Memory" (Hold Grudge)
            # <= 0 means "Remove/Keep out of Memory" (Forgive/Trust)
            if decision_score > 0:
                recipient.memory.add(donor.unique_id)
            else:
                recipient.memory.discard(donor.unique_id)

    def get_fitness_by_type(self):
        fitness_data = {}
        for a_type in ["iGSS-Agent", "Control-Agent", "Unconditional Cooperator", "Defector"]:
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
    func_assessment = toolbox.compile(expr=individual)
    
    tot_igss, tot_uc, tot_d = 0, 0, 0
    runs = 3
    for _ in range(runs): 
        m = CooperationModel(igss_rule=func_assessment, config=config)
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
    tot_control = 0
    runs = 5
    for _ in range(runs):
        m = CooperationModel(igss_rule=None, config=config, control_mode=True)
        for _ in range(config["NUM_ROUNDS"]): m.step()
        tot_control += m.get_fitness_by_type()["Control-Agent"]
    return tot_control / runs

def run_evolution(config):
    current_seed = config.get("SEED", 42)
    random.seed(current_seed)
    np.random.seed(current_seed)

    pop = toolbox.population(n=config["POP_SIZE"])
    hof = tools.HallOfFame(config.get("HOF_SIZE", 1)) 
    history = {"max_igss": [], "avg_igss": [], "uc_scores": [], "d_scores": [], "fossil_record": {}}
    
    print("Initializing Direct Reciprocity [MODE 2 - Memory Assessment Search]...")
    control_fitness = get_control_baseline(config)
    print(f"Control Baseline (Strict Memory Ledger) established at: {control_fitness:.2f} average payoff.")

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
    
    # Truth Table Tuple Mapping for the Memory Ledger:
    # Variables: (DonorAction, DonorInMemory)
    # Output: True = Place/Keep in Memory (Hold Grudge), False = Remove/Ignore (Forgive)
    # 4 bits -> (C from Str, C from Foe, D from Str, D from Foe)
    DR_MEMORY_DICT = {
        (False, False, True, True): "Strict Ledger (TFT Base)", # Grudge if D, Forgive if C
        (True, True, True, True): "Paranoid (Remember Everyone)",
        (False, False, False, False): "Amnesia (Remember No One)",
        (False, True, True, True): "Unforgiving (Grim Trigger)", # Grudge if D, and NEVER forgive.
        (False, True, False, True): "Sticky Memory (Ignore actions entirely)"
    }
    
    variables = ["DonorAction", "DonorInMemory"]

    # -------------------------------------------------------------------------
    # OPTION A: SINGLE VISUAL TEST
    # -------------------------------------------------------------------------
    """
    print("\n--- RUNNING SINGLE VISUAL TEST ---")
    config = QUICK_TEST_CONFIG
    best_rule, history, control_fit = run_evolution(config)
    
    simp_assessment = str(simplify_rule(str(best_rule), variables))
    raw_tuple = evaluate_truth_table(simp_assessment, variables)
    identified_strategy = DR_MEMORY_DICT.get(raw_tuple, f"Novel Rule: {raw_tuple}")

    run_log = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M"),
        "Machine_ID": "Local_VSCode", 
        "Run_ID": f"DR2_TEST_{datetime.now().strftime('%H%M%S')}",
        "Mode": "Direct Reciprocity Mode 2 (Memory Ledger Search)",
        **config, 
        "Best_Rule_Raw": str(best_rule),
        "Best_Rule_SymPy": simp_assessment,
        "Identified_Strategy": identified_strategy,
        "iGSS_Max_Fitness": round(history["max_igss"][-1], 2),
        "iGSS_Avg_Fitness": round(history["avg_igss"][-1], 2),
        "UC_Fitness": round(history["uc_scores"][-1], 2),
        "D_Fitness": round(history["d_scores"][-1], 2),
        "Control_Baseline": round(control_fit, 2),
        "Manual_Validation": ""
    }
    
    log_results_to_csv(run_log, filename="results_DR_Mode2.csv")
    generate_dashboard(best_rule, history, run_log, save_only=False)
    """

    # -------------------------------------------------------------------------
    # OPTION B: BATCH EXPERIMENT
    # -------------------------------------------------------------------------
    
    batches = get_experiment_batches()
    print(f"\n--- STARTING BATCH EXECUTION: {len(batches)} TOTAL RUNS QUEUED ---")
    
    for i, config in enumerate(batches):
        print(f"\n[Run {i+1}/{len(batches)} | {config['CONFIG_GROUP']} Rep {config.get('REPETITION', 1)}] Seed: {config.get('SEED', 42)}")
        
        best_rule, history, control_fit = run_evolution(config)
        
        simp_assessment = str(simplify_rule(str(best_rule), variables))
        raw_tuple = evaluate_truth_table(simp_assessment, variables)
        identified_strategy = DR_MEMORY_DICT.get(raw_tuple, f"Novel Rule: {raw_tuple}")

        run_log = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M"),
            "Machine_ID": "Local_VSCode", 
            "Run_ID": f"DR2_{config.get('CONFIG_GROUP', 'Default')}_R{config.get('REPETITION', 1)}",
            "Mode": "Direct Reciprocity Mode 2 (Memory Ledger Search)",
            **config,
            "Best_Rule_Raw": str(best_rule),
            "Best_Rule_SymPy": simp_assessment,
            "Identified_Strategy": identified_strategy,
            "iGSS_Max_Fitness": round(history["max_igss"][-1], 2),
            "iGSS_Avg_Fitness": round(history["avg_igss"][-1], 2),
            "UC_Fitness": round(history["uc_scores"][-1], 2),
            "D_Fitness": round(history["d_scores"][-1], 2),
            "Control_Baseline": round(control_fit, 2),
            "Manual_Validation": ""
        }
        
        log_results_to_csv(run_log, filename="results_DR_Mode2.csv")
        generate_dashboard(best_rule, history, run_log, save_only=True)
        
    print("\n[✓] Batch complete! Check results_DR_Mode2.csv and the /figures directory.")