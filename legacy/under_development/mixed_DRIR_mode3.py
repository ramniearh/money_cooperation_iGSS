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
# 1. GENETIC PROGRAMMING SETUP (MIXED CO-EVOLUTION)
# =============================================================================
def random_constant():
    return random.choice([-10, -1, 0, 1, 10])

# --- Tree 1: Action Rule (Mixed - 2 Variables) ---
pset_action = gp.PrimitiveSet("ActionRule_Mixed", 2) 
pset_action.renameArguments(ARG0='PartnerStanding', ARG1='PartnerInMemory')
pset_action.addPrimitive(operator.add, 2)
pset_action.addPrimitive(operator.sub, 2)
pset_action.addPrimitive(operator.mul, 2)
pset_action.addEphemeralConstant("rand_const_act", random_constant)

# --- Tree 2: Assessment Rule (Mixed - 3 Variables) ---
pset_assessment = gp.PrimitiveSet("AssessmentRule_Mixed", 3) 
pset_assessment.renameArguments(ARG0='DonorAction', ARG1='RecipientStanding', ARG2='DonorInMemory')
pset_assessment.addPrimitive(operator.add, 2)
pset_assessment.addPrimitive(operator.sub, 2)
pset_assessment.addPrimitive(operator.mul, 2)
pset_assessment.addEphemeralConstant("rand_const_ass", random_constant)

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr_action", gp.genHalfAndHalf, pset=pset_action, min_=1, max_=3)
toolbox.register("expr_assessment", gp.genHalfAndHalf, pset=pset_assessment, min_=1, max_=3)

def init_individual(container, func1, func2):
    return container([gp.PrimitiveTree(func1()), gp.PrimitiveTree(func2())])

toolbox.register("individual", init_individual, creator.Individual, toolbox.expr_action, toolbox.expr_assessment)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Compilers
toolbox.register("compile_action", gp.compile, pset=pset_action)
toolbox.register("compile_assessment", gp.compile, pset=pset_assessment)

# Custom Genetic Operators for Dual Trees
def cxTwoTrees(ind1, ind2):
    if random.random() < 0.5:
        ind1[0], ind2[0] = gp.cxOnePoint(ind1[0], ind2[0])
    else:
        ind1[1], ind2[1] = gp.cxOnePoint(ind1[1], ind2[1])
    return ind1, ind2

def mutTwoTrees(individual):
    if random.random() < 0.5:
        individual[0], = gp.mutUniform(individual[0], expr=toolbox.expr_action, pset=pset_action)
    else:
        individual[1], = gp.mutUniform(individual[1], expr=toolbox.expr_assessment, pset=pset_assessment)
    return individual,

toolbox.register("mate", cxTwoTrees)
toolbox.register("mutate", mutTwoTrees)
toolbox.register("select", tools.selTournament, tournsize=3)

# =============================================================================
# 2. AGENT-BASED MODEL (Dual Ledger System)
# =============================================================================
class CoopAgent(mesa.Agent):
    def __init__(self, model, agent_type="iGSS-Agent"):
        super().__init__(model) 
        self.agent_type = agent_type
        self.payoff = 0
        
        # The Dual Ledger
        self.memory = set()   # Direct Reciprocity (Private Experience)
        self.standing = 1     # Indirect Reciprocity (Public Gossip)
        
    def evaluate_partner(self, partner):
        if self.agent_type == "Unconditional Cooperator": return True
        if self.agent_type == "Defector": return False
        
        # Hardcoded Baseline 1: Pure Direct Reciprocity
        if self.agent_type == "Control-DR":
            return partner.unique_id not in self.memory
            
        # Hardcoded Baseline 2: Pure Indirect Reciprocity
        if self.agent_type == "Control-IR":
            return partner.standing == 1
            
        # EVOLVED iGSS ACTION: Uses Tree 1
        in_memory_signal = 1 if partner.unique_id in self.memory else 0
        decision_score = self.model.igss_action_rule(partner.standing, in_memory_signal)
        return decision_score > 0

class CooperationModel(mesa.Model):
    def __init__(self, igss_action_rule, igss_assessment_rule, config, control_mode=None):
        super().__init__()
        self.igss_action_rule = igss_action_rule
        self.igss_assessment_rule = igss_assessment_rule
        self.config = config
        self.benefit = config["BENEFIT_TO_COST_RATIO"] * config["COST"]
        self.cost = config["COST"]
        self.control_mode = control_mode
        
        target_agent = "iGSS-Agent"
        if control_mode == "DR": target_agent = "Control-DR"
        if control_mode == "IR": target_agent = "Control-IR"
        
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
            
        # 2. Institutional Assessment
        if self.control_mode == "DR":
            if not cooperates:
                recipient.memory.add(donor.unique_id)
            else:
                recipient.memory.discard(donor.unique_id)
                
        elif self.control_mode == "IR":
            if cooperates:
                donor.standing = 1 
            else:
                if recipient.standing == 1:
                    donor.standing = 0
                    
        else:
            # EVOLVED iGSS ASSESSMENT: Uses Tree 2
            action_val = 1 if cooperates else 0
            donor_in_mem = 1 if donor.unique_id in recipient.memory else 0
            
            decision_score = self.igss_assessment_rule(action_val, recipient.standing, donor_in_mem)
            
            if decision_score > 0:
                donor.standing = 1                      # Public Forgiveness
                recipient.memory.discard(donor.unique_id) # Private Forgiveness
            else:
                donor.standing = 0                      # Public Condemnation
                recipient.memory.add(donor.unique_id)     # Private Grudge

    def get_fitness_by_type(self):
        fitness_data = {}
        for a_type in ["iGSS-Agent", "Control-DR", "Control-IR", "Unconditional Cooperator", "Defector"]:
            type_agents = [a for a in self.agents if a.agent_type == a_type]
            if type_agents:
                fitness_data[a_type] = sum(a.payoff for a in type_agents) / len(type_agents)
            else:
                fitness_data[a_type] = 0
        return fitness_data

# =============================================================================
# 3. EVOLUTIONARY ENGINE
# =============================================================================
def evaluate_rules(individual, config):
    func_action = toolbox.compile_action(expr=individual[0])
    func_assessment = toolbox.compile_assessment(expr=individual[1])
    
    tot_igss, tot_uc, tot_d = 0, 0, 0
    runs = 3
    for _ in range(runs): 
        m = CooperationModel(igss_action_rule=func_action, igss_assessment_rule=func_assessment, config=config)
        for _ in range(config["NUM_ROUNDS"]): m.step()
        
        fits = m.get_fitness_by_type()
        tot_igss += fits["iGSS-Agent"]
        tot_uc += fits["Unconditional Cooperator"]
        tot_d += fits["Defector"]
        
    individual.sim_igss = tot_igss / runs
    individual.sim_uc = tot_uc / runs
    individual.sim_d = tot_d / runs
    
    # Combined Parsimony Tax
    tax = (len(individual[0]) + len(individual[1])) * config["PARSIMONY_TAX"]
    final_score = individual.sim_igss - tax
    return final_score, 

def get_control_baselines(config):
    tot_dr, tot_ir = 0, 0
    runs = 5
    for _ in range(runs):
        m_dr = CooperationModel(igss_action_rule=None, igss_assessment_rule=None, config=config, control_mode="DR")
        for _ in range(config["NUM_ROUNDS"]): m_dr.step()
        tot_dr += m_dr.get_fitness_by_type()["Control-DR"]
        
        m_ir = CooperationModel(igss_action_rule=None, igss_assessment_rule=None, config=config, control_mode="IR")
        for _ in range(config["NUM_ROUNDS"]): m_ir.step()
        tot_ir += m_ir.get_fitness_by_type()["Control-IR"]
        
    return tot_dr / runs, tot_ir / runs

def run_evolution(config):
    current_seed = config.get("SEED", 42)
    random.seed(current_seed)
    np.random.seed(current_seed)

    pop = toolbox.population(n=config["POP_SIZE"])
    hof = tools.HallOfFame(config.get("HOF_SIZE", 1)) 
    history = {"max_igss": [], "avg_igss": [], "uc_scores": [], "d_scores": [], "fossil_record": {}}
    
    print("\nInitializing Mixed Reciprocity [MODE 3 - Co-Evolution]...")
    baseline_dr, baseline_ir = get_control_baselines(config)
    print(f"Baseline DR (Tit-for-Tat)    : {baseline_dr:.2f}")
    print(f"Baseline IR (Strict Discrim) : {baseline_ir:.2f}")
    
    max_baseline = max(baseline_dr, baseline_ir)

    fitnesses = [evaluate_rules(ind, config) for ind in pop]
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
        fitnesses = [evaluate_rules(ind, config) for ind in invalid_ind]
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
            history["fossil_record"][gen] = f"ACT: {best_in_gen[0]} | ASSESS: {best_in_gen[1]}"
            print(f" > Gen {gen:02d} | iGSS: {best_in_gen.sim_igss:.1f} | UC: {best_in_gen.sim_uc:.1f} | D: {best_in_gen.sim_d:.1f}")

    return hof[0], history, max_baseline, baseline_dr, baseline_ir

# =============================================================================
# 4. EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    
    # Truth Table Dictionaries 
    MIXED_ACTION_DICT = {
        (True, True, False, False): "Pure Public (Strict Discriminator)", 
        (False, True, False, True): "Pure Private (Tit-for-Tat)",         
        (True, True, True, True): "Unconditional Cooperator (ALL-C)",
        (False, False, False, False): "Unconditional Defector (ALL-D)",
        (False, True, False, False): "Skeptic (Must be Publicly Good AND Privately Trusted)", 
        (True, True, False, True): "Optimist (Helps if Publicly Good OR Privately Trusted)"
    }

    MIXED_ASSESSMENT_DICT = {
        (True, True, True, True, False, False, True, True): "Pure Public (Simple Standing)",
        (True, True, False, False, False, False, True, True): "Pure Public (Stern Judging)",
        (True, True, False, False, False, False, False, False): "Pure Public (Image Scoring)",
        (True, True, True, True, False, False, False, False): "Pure Private (Strict Ledger / TFT Base)",
        (True, True, True, True, True, True, True, True): "Amnesia (Unconditional Forgiveness)",
        (False, False, False, False, False, False, False, False): "Paranoid (Unconditional Grudge)"
    }
    
    action_vars = ["PartnerStanding", "PartnerInMemory"]
    assess_vars = ["DonorAction", "RecipientStanding", "DonorInMemory"]

    # -------------------------------------------------------------------------
    # OPTION A: SINGLE VISUAL TEST
    # -------------------------------------------------------------------------
    """
    print("\n--- RUNNING SINGLE VISUAL TEST ---")
    config = QUICK_TEST_CONFIG
    best_rule, history, control_fit, b_dr, b_ir = run_evolution(config)
    
    # Parse Tree 1 (Action)
    simp_action = str(simplify_rule(str(best_rule[0]), action_vars))
    act_tuple = evaluate_truth_table(simp_action, action_vars)
    identified_action = MIXED_ACTION_DICT.get(act_tuple, f"Novel Act: {act_tuple}")

    # Parse Tree 2 (Assessment)
    simp_assessment = str(simplify_rule(str(best_rule[1]), assess_vars))
    ass_tuple = evaluate_truth_table(simp_assessment, assess_vars)
    identified_assessment = MIXED_ASSESSMENT_DICT.get(ass_tuple, f"Novel Ass: {ass_tuple}")

    run_log = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M"),
        "Machine_ID": "Local_VSCode", 
        "Run_ID": f"MIXED3_TEST_{datetime.now().strftime('%H%M%S')}",
        "Mode": "Mixed Reciprocity Mode 3 (Co-Evolution)",
        **config, 
        "Best_Act_Raw": str(best_rule[0]),
        "Best_Act_SymPy": simp_action,
        "Identified_Action": identified_action,
        "Best_Ass_Raw": str(best_rule[1]),
        "Best_Ass_SymPy": simp_assessment,
        "Identified_Assessment": identified_assessment,
        "iGSS_Max_Fitness": round(history["max_igss"][-1], 2),
        "iGSS_Avg_Fitness": round(history["avg_igss"][-1], 2),
        "Control_Baseline": round(control_fit, 2),
        "Control_DR_Score": round(b_dr, 2),
        "Control_IR_Score": round(b_ir, 2)
    }
    
    log_results_to_csv(run_log, filename="results_Mixed_Mode3.csv")
    generate_dashboard(best_rule, history, run_log, save_only=False)
    """

    # -------------------------------------------------------------------------
    # OPTION B: BATCH EXPERIMENT
    # -------------------------------------------------------------------------
    
    batches = get_experiment_batches()
    print(f"\n--- STARTING BATCH EXECUTION: {len(batches)} TOTAL RUNS QUEUED ---")
    
    for i, config in enumerate(batches):
        print(f"\n[Run {i+1}/{len(batches)} | {config['CONFIG_GROUP']} Rep {config.get('REPETITION', 1)}] Seed: {config.get('SEED', 42)}")
        
        best_rule, history, control_fit, b_dr, b_ir = run_evolution(config)
        
        # Parse Tree 1 (Action)
        simp_action = str(simplify_rule(str(best_rule[0]), action_vars))
        act_tuple = evaluate_truth_table(simp_action, action_vars)
        identified_action = MIXED_ACTION_DICT.get(act_tuple, f"Novel Act: {act_tuple}")

        # Parse Tree 2 (Assessment)
        simp_assessment = str(simplify_rule(str(best_rule[1]), assess_vars))
        ass_tuple = evaluate_truth_table(simp_assessment, assess_vars)
        identified_assessment = MIXED_ASSESSMENT_DICT.get(ass_tuple, f"Novel Ass: {ass_tuple}")

        run_log = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M"),
            "Machine_ID": "Local_VSCode", 
            "Run_ID": f"MIXED3_{config.get('CONFIG_GROUP', 'Default')}_R{config.get('REPETITION', 1)}",
            "Mode": "Mixed Reciprocity Mode 3 (Co-Evolution)",
            **config,
            "Best_Act_Raw": str(best_rule[0]),
            "Best_Act_SymPy": simp_action,
            "Identified_Action": identified_action,
            "Best_Ass_Raw": str(best_rule[1]),
            "Best_Ass_SymPy": simp_assessment,
            "Identified_Assessment": identified_assessment,
            "iGSS_Max_Fitness": round(history["max_igss"][-1], 2),
            "iGSS_Avg_Fitness": round(history["avg_igss"][-1], 2),
            "Control_Baseline": round(control_fit, 2),
            "Control_DR_Score": round(b_dr, 2),
            "Control_IR_Score": round(b_ir, 2)
        }
        
        log_results_to_csv(run_log, filename="results_Mixed_Mode3.csv")
        generate_dashboard(best_rule, history, run_log, save_only=True)
        
    print("\n[✓] Batch complete! Check results_Mixed_Mode3.csv and the /figures directory.")