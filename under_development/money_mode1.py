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
# 1. GENETIC PROGRAMMING SETUP (MONEY ACTION SEARCH)
# =============================================================================
def random_constant():
    return random.choice([-10, -1, 0, 1, 10])

# Primitive Set for Monetary Agents
pset_action = gp.PrimitiveSet("ActionRule_Money", 2) 
pset_action.renameArguments(ARG0='PartnerToken', ARG1='MyToken')
pset_action.addPrimitive(operator.add, 2)
pset_action.addPrimitive(operator.sub, 2)
pset_action.addPrimitive(operator.mul, 2)
pset_action.addEphemeralConstant("rand_const_act", random_constant)

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr_action", gp.genHalfAndHalf, pset=pset_action, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_action)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset_action)

toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_action, pset=pset_action)
toolbox.register("select", tools.selTournament, tournsize=3)

# =============================================================================
# 2. AGENT-BASED MODEL (Closed Economy Physics)
# =============================================================================
class CoopAgent(mesa.Agent):
    def __init__(self, model, agent_type="iGSS-Agent"):
        super().__init__(model) 
        self.agent_type = agent_type
        self.payoff = 0
        self.token = 0 
        
    def evaluate_partner(self, partner):
        # The Unconditional strategies do not care about money
        if self.agent_type == "Unconditional Cooperator": return True
        if self.agent_type == "Defector": return False
        
        # Hardcoded Control Baseline: Accumulating Capitalist
        if self.agent_type == "Control-Agent":
            return partner.token >= 1
            
        # EVOLVED iGSS ACTION: The Monetary Decision
        decision_score = self.model.igss_rule(partner.token, self.token)
        return decision_score > 0

class CooperationModel(mesa.Model):
    def __init__(self, igss_rule, config, control_mode=False):
        super().__init__()
        self.igss_rule = igss_rule
        self.config = config
        self.benefit = config["BENEFIT_TO_COST_RATIO"] * config["COST"]
        self.cost = config["COST"]
        
        target_agent = "Control-Agent" if control_mode else "iGSS-Agent"
        
        agents_list = []
        for _ in range(config["NUM_IGSS"]): agents_list.append(CoopAgent(self, target_agent))
        for _ in range(config["NUM_UC"]): agents_list.append(CoopAgent(self, "Unconditional Cooperator"))
        for _ in range(config["NUM_D"]): agents_list.append(CoopAgent(self, "Defector"))

        # --- THE CENTRAL BANK: PROPRIETARY FIAT GENESIS ---
        # We only inject liquidity into agents who are programmed to use it
        economic_agents = [a for a in agents_list if a.agent_type in ["iGSS-Agent", "Control-Agent"]]
        random.shuffle(economic_agents)
        cutoff = int(len(economic_agents) * config.get("ENDOWMENT_FRACTION", 0.5))
        
        for i, agent in enumerate(economic_agents):
            if i < cutoff:
                agent.token = config.get("INITIAL_ENDOWMENT", 1)

    def step(self):
        agents = list(self.agents)
        random.shuffle(agents)

        for donor in agents:
            possible_recipients = [a for a in agents if a.unique_id != donor.unique_id]
            recipient = random.choice(possible_recipients)

            donor_action = donor.evaluate_partner(recipient)
            self.resolve(donor=donor, recipient=recipient, cooperates=donor_action)
            
    def resolve(self, donor, recipient, cooperates):
        if cooperates:
            # 1. The Real Economy (Non-Zero-Sum)
            donor.payoff -= self.cost
            recipient.payoff += self.benefit
            
            # 2. The Institutional Settlement (Zero-Sum Transfer)
            # A physical transfer of tokens ONLY occurs if the donor is participating in the monetary economy
            donor_cares_about_money = donor.agent_type in ["iGSS-Agent", "Control-Agent"]
            
            # If the recipient has the money to pay, the transaction clears.
            if donor_cares_about_money and recipient.token >= 1:
                recipient.token -= 1
                donor.token += 1

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
    
    print("\nInitializing CLOSED ECONOMY MONEY [MODE 1] Module...")
    control_fitness = get_control_baseline(config)
    print(f"Control Baseline (Accumulating Capitalist) established at: {control_fitness:.2f} avg payoff.")

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

    return hof[0], history, control_fitness

# =============================================================================
# 4. EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    
    # 4-Bit Truth Table Tuple Mapping 
    # Variables: (PartnerToken, MyToken)
    # Assumes binary state for simplicity of identification: (HasMoney/HasMoney, HasMoney/Broke, Broke/HasMoney, Broke/Broke)
    MONEY_ACTION_DICT = {
        (True, True, False, False): "Accumulating Capitalist (Strict Seller)", 
        (True, True, True, True): "Unconditional Cooperator (Charity)",
        (False, False, False, False): "Unconditional Defector (Hoarder)",
        (False, True, False, True): "Desperate Seller (Only works if they are broke)",
        (False, False, True, True): "Altruist to the Poor (Only helps broke people)"
    }
    
    variables = ["PartnerToken", "MyToken"]

    # -------------------------------------------------------------------------
    # OPTION A: SINGLE VISUAL TEST
    # -------------------------------------------------------------------------
    """
    print("\n--- RUNNING SINGLE VISUAL TEST ---")
    config = QUICK_TEST_CONFIG
    # Ensure money configs exist for quick tests
    config.setdefault("INITIAL_ENDOWMENT", 1)
    config.setdefault("ENDOWMENT_FRACTION", 0.5)
    
    best_rule, history, control_fit = run_evolution(config)
    
    simp_action = str(simplify_rule(str(best_rule), variables))
    raw_tuple = evaluate_truth_table(simp_action, variables)
    identified_strategy = MONEY_ACTION_DICT.get(raw_tuple, f"Novel Rule: {raw_tuple}")

    run_log = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M"),
        "Machine_ID": "Local_VSCode", 
        "Run_ID": f"MONEY1_TEST_{datetime.now().strftime('%H%M%S')}",
        "Mode": "Money Mode 1 (Action Search)",
        **config, 
        "Best_Rule_Raw": str(best_rule),
        "Best_Rule_SymPy": simp_action,
        "Identified_Strategy": identified_strategy,
        "iGSS_Max_Fitness": round(history["max_igss"][-1], 2),
        "iGSS_Avg_Fitness": round(history["avg_igss"][-1], 2),
        "Control_Baseline": round(control_fit, 2)
    }
    
    log_results_to_csv(run_log, filename="results_Money_Mode1.csv")
    generate_dashboard(best_rule, history, run_log, save_only=False)
    """

    # -------------------------------------------------------------------------
    # OPTION B: BATCH EXPERIMENT
    # -------------------------------------------------------------------------
    
    batches = get_experiment_batches()
    print(f"\n--- STARTING BATCH EXECUTION: {len(batches)} TOTAL RUNS QUEUED ---")
    
    for i, config in enumerate(batches):
        print(f"\n[Run {i+1}/{len(batches)} | {config['CONFIG_GROUP']} Rep {config.get('REPETITION', 1)}] Seed: {config.get('SEED', 42)}")
        
        # Inject required money params if missing from config.py batches
        config.setdefault("INITIAL_ENDOWMENT", 1)
        config.setdefault("ENDOWMENT_FRACTION", 0.5)
        
        best_rule, history, control_fit = run_evolution(config)
        
        simp_action = str(simplify_rule(str(best_rule), variables))
        raw_tuple = evaluate_truth_table(simp_action, variables)
        identified_strategy = MONEY_ACTION_DICT.get(raw_tuple, f"Novel Rule: {raw_tuple}")

        run_log = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M"),
            "Machine_ID": "Local_VSCode", 
            "Run_ID": f"MONEY1_{config.get('CONFIG_GROUP', 'Default')}_R{config.get('REPETITION', 1)}",
            "Mode": "Money Mode 1 (Action Search)",
            **config,
            "Best_Rule_Raw": str(best_rule),
            "Best_Rule_SymPy": simp_action,
            "Identified_Strategy": identified_strategy,
            "iGSS_Max_Fitness": round(history["max_igss"][-1], 2),
            "iGSS_Avg_Fitness": round(history["avg_igss"][-1], 2),
            "Control_Baseline": round(control_fit, 2)
        }
        
        log_results_to_csv(run_log, filename="results_Money_Mode1.csv")
        generate_dashboard(best_rule, history, run_log, save_only=True)
        
    print("\n[✓] Batch complete! Check results_Money_Mode1.csv and the /figures directory.")