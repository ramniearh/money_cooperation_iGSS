# BAD! BABYSAT! 
import operator
import random
import numpy as np
import matplotlib.pyplot as plt
import mesa
import sympy as sp
from deap import base, creator, tools, gp

# =============================================================================
# 1. GLOBAL CONFIGURATIONS & REPRODUCIBILITY
# =============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

MODEL_CONFIG = {
    "BENEFIT_TO_COST_RATIO": 5,
    "COST": 1,
    "NUM_IGSS": 10,
    "NUM_UC": 10,
    "NUM_D": 10,
    "NUM_ROUNDS": 100,
    # --- UNIVERSAL FIAT GENESIS (No Gated Communities) ---
    "INITIAL_ENDOWMENT": 1,        
    "ENDOWMENT_FRACTION": 0.5      # 50% of the ENTIRE population gets 1 token
}

EVO_CONFIG = {
    "POP_SIZE": 60,       # Bumped up slightly to help find the bilateral handshake  
    "MAX_GENS": 100,         
    "PARSIMONY_TAX": 0.1
}

# =============================================================================
# 2. GENETIC PROGRAMMING SETUP (TWO TREES PER AGENT)
# =============================================================================
def random_constant():
    return random.choice([-10, -1, 0, 1, 10])

pset_action = gp.PrimitiveSet("ActionRule_MoneyAcc", 2) 
pset_action.renameArguments(ARG0='PartnerToken', ARG1='MyToken')
pset_action.addPrimitive(operator.add, 2)
pset_action.addPrimitive(operator.sub, 2)
pset_action.addPrimitive(operator.mul, 2)
pset_action.addEphemeralConstant("rand_const_act", random_constant)

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # Individual inherits from list to hold exactly TWO trees
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr_action", gp.genHalfAndHalf, pset=pset_action, min_=1, max_=3)

# --- THE FIX: Explicitly wrap both trees in gp.PrimitiveTree ---
def init_individual(icls):
    tree_donor = gp.PrimitiveTree(toolbox.expr_action())
    tree_recipient = gp.PrimitiveTree(toolbox.expr_action())
    return icls([tree_donor, tree_recipient])

toolbox.register("individual", init_individual, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset_action)

# Custom Crossover & Mutation for Multi-Tree Individuals
def cx_trees(ind1, ind2):
    idx = random.choice([0, 1]) # Randomly cross either Donor tree or Recipient tree
    ind1[idx], ind2[idx] = gp.cxOnePoint(ind1[idx], ind2[idx])
    return ind1, ind2

def mut_trees(individual, expr, pset):
    idx = random.choice([0, 1]) # Randomly mutate either Donor tree or Recipient tree
    individual[idx], = gp.mutUniform(individual[idx], expr=expr, pset=pset)
    return individual,

toolbox.register("mate", cx_trees)
toolbox.register("mutate", mut_trees, expr=toolbox.expr_action, pset=pset_action)
toolbox.register("select", tools.selTournament, tournsize=3)

# =============================================================================
# 3. AGENT-BASED MODEL (F&M DOUBLE-BLIND ESCROW)
# =============================================================================
class CoopAgent(mesa.Agent):
    def __init__(self, model, agent_type="iGSS-Agent"):
        super().__init__(model) 
        self.agent_type = agent_type
        self.payoff = 0
        self.token = 0 
        
    def evaluate_as_donor(self, partner):
        """Do I consent to pay 1 fitness, and gain 1 token?"""
        if self.agent_type == "Unconditional Cooperator": return True
        if self.agent_type == "Defector": return False
        
        # Capitalist Baseline: I will work if you can pay me.
        if self.agent_type == "Control-Agent": return partner.token >= 1
            
        score = self.model.donor_rule(partner.token, self.token)
        return score > 0

    def evaluate_as_recipient(self, partner):
        """Do I consent to gain 5 fitness, and lose 1 token?"""
        if self.agent_type == "Unconditional Cooperator": return True
        if self.agent_type == "Defector": return True # Defectors love +5 fitness, even for a token!
        
        # Capitalist Baseline: Yes, +5 fitness is always worth -1 token.
        if self.agent_type == "Control-Agent": return True
            
        score = self.model.recipient_rule(partner.token, self.token)
        return score > 0

class CooperationModel(mesa.Model):
    def __init__(self, donor_rule, recipient_rule, config=MODEL_CONFIG, control_mode=False):
        super().__init__()
        self.donor_rule = donor_rule
        self.recipient_rule = recipient_rule
        self.config = config
        self.benefit = config["BENEFIT_TO_COST_RATIO"] * config["COST"]
        self.cost = config["COST"]
        
        target_agent = "Control-Agent" if control_mode else "iGSS-Agent"
        
        agents_list = []
        for _ in range(config["NUM_IGSS"]): agents_list.append(CoopAgent(self, target_agent))
        for _ in range(config["NUM_UC"]): agents_list.append(CoopAgent(self, "Unconditional Cooperator"))
        for _ in range(config["NUM_D"]): agents_list.append(CoopAgent(self, "Defector"))

        # --- THE CENTRAL BANK: UNIVERSAL FIAT GENESIS (Matches your Paper) ---
        random.shuffle(agents_list)
        cutoff = int(len(agents_list) * config["ENDOWMENT_FRACTION"])
        
        for i, agent in enumerate(agents_list):
            if i < cutoff:
                agent.token = config["INITIAL_ENDOWMENT"]

    def step(self):
        agents = list(self.agents)
        random.shuffle(agents)

        for donor in agents:
            possible_recipients = [a for a in agents if a.unique_id != donor.unique_id]
            recipient = random.choice(possible_recipients)
            self.resolve(donor=donor, recipient=recipient)
            
    def resolve(self, donor, recipient):
        # 1. The F&M Double-Blind Assessment
        donor_consents = donor.evaluate_as_donor(recipient)
        recipient_consents = recipient.evaluate_as_recipient(donor)
        
        # 2. The Mutual Consent Match
        if donor_consents and recipient_consents:
            # 3. The Physical Liquidity Check
            if recipient.token >= 1:
                # 4. Execute the fully consensual trade
                donor.payoff -= self.cost
                recipient.payoff += self.benefit
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
# 4. EVOLUTIONARY ENGINE
# =============================================================================
def evaluate_rule(individual, model_config, evo_config):
    # Compile both trees
    func_donor = toolbox.compile(expr=individual[0])
    func_recipient = toolbox.compile(expr=individual[1])
    
    tot_igss, tot_uc, tot_d = 0, 0, 0
    runs = 3
    for _ in range(runs): 
        m = CooperationModel(donor_rule=func_donor, recipient_rule=func_recipient, config=model_config)
        for _ in range(model_config["NUM_ROUNDS"]): m.step()
        
        fits = m.get_fitness_by_type()
        tot_igss += fits["iGSS-Agent"]
        tot_uc += fits["Unconditional Cooperator"]
        tot_d += fits["Defector"]
        
    individual.sim_igss = tot_igss / runs
    individual.sim_uc = tot_uc / runs
    individual.sim_d = tot_d / runs
    
    # Parsimony tax applies to the combined length of both trees
    tax = (len(individual[0]) + len(individual[1])) * evo_config["PARSIMONY_TAX"]
    final_score = individual.sim_igss - tax
    return final_score, 

toolbox.register("evaluate", evaluate_rule, model_config=MODEL_CONFIG, evo_config=EVO_CONFIG)

def get_control_baseline(model_config):
    tot_control = 0
    runs = 5
    for _ in range(runs):
        m = CooperationModel(donor_rule=None, recipient_rule=None, config=model_config, control_mode=True)
        for _ in range(model_config["NUM_ROUNDS"]): m.step()
        tot_control += m.get_fitness_by_type()["Control-Agent"]
    return tot_control / runs

def run_evolution():
    pop = toolbox.population(n=EVO_CONFIG["POP_SIZE"])
    hof = tools.HallOfFame(1) 
    history = {"max_igss": [], "avg_igss": [], "uc_scores": [], "d_scores": [], "fossil_record_donor": {}, "fossil_record_recip": {}}
    
    print("Initializing F&M BILATERAL CONSENT [MODE 2.0] Module...")
    control_fitness = get_control_baseline(MODEL_CONFIG)
    print(f"Control Baseline established at: {control_fitness:.2f} average payoff.")

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses): ind.fitness.values = fit
    
    hof.update(pop)
        
    for gen in range(1, EVO_CONFIG["MAX_GENS"] + 1): 
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values; del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values
                
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid_ind, map(toolbox.evaluate, invalid_ind)):
            ind.fitness.values = fit
                
        pop[:] = offspring
        hof.update(pop) 
        
        pop[-1] = toolbox.clone(hof[0])
        
        best_in_gen = tools.selBest(pop, k=1)[0]
        history["max_igss"].append(best_in_gen.sim_igss)
        history["avg_igss"].append(np.mean([ind.sim_igss for ind in pop]))
        history["uc_scores"].append(best_in_gen.sim_uc)
        history["d_scores"].append(best_in_gen.sim_d)
        
        if gen % 10 == 0 or gen == 1:
            history["fossil_record_donor"][gen] = str(best_in_gen[0])
            history["fossil_record_recip"][gen] = str(best_in_gen[1])
            print(f" > Gen {gen:02d} | iGSS: {best_in_gen.sim_igss:.1f} | UC: {best_in_gen.sim_uc:.1f} | D: {best_in_gen.sim_d:.1f}")

    return hof[0], history, control_fitness

# =============================================================================
# 5. SYMPY PARSING & VISUALIZATION
# =============================================================================
def simplify_rule(gp_string):
    mapping = {
        'add': lambda x, y: x + y, 
        'sub': lambda x, y: x - y, 
        'mul': lambda x, y: x * y,
        'PartnerToken': sp.Symbol('PartnerToken'),
        'MyToken': sp.Symbol('MyToken')
    }
    try: 
        return sp.simplify(eval(gp_string, {"__builtins__": {}}, mapping))
    except Exception as e: 
        return f"SymPy Parsing Error: {e}"

def plot_dashboard(best_ind, history, model_config, evo_config, control_fit):
    raw_donor = str(best_ind[0])
    raw_recipient = str(best_ind[1])
    simp_donor = simplify_rule(raw_donor)
    simp_recipient = simplify_rule(raw_recipient)
    
    fossils_str = "\n".join([f"  Gen {g:02d}:\n    Donor: {history['fossil_record_donor'][g]}\n    Recip: {history['fossil_record_recip'][g]}" for g in sorted(history["fossil_record_donor"].keys())])
    
    theoretical_max = model_config["NUM_ROUNDS"] * (model_config["BENEFIT_TO_COST_RATIO"] * model_config["COST"] - model_config["COST"])
    
    report_text = (
        f"\n{'='*60}\n"
        f"      F&M BILATERAL CONSENT [MODE 2.0] REPORT\n"
        f"{'='*60}\n"
        f"--- CONFIGURATIONS ---\n"
        f"Random Seed: {SEED}\n"
        f"Populations: {model_config['NUM_IGSS']} iGSS | {model_config['NUM_UC']} UC | {model_config['NUM_D']} Defectors\n"
        f"Liquidity (Universal Fiat): {model_config['INITIAL_ENDOWMENT']} tokens to {int(model_config['ENDOWMENT_FRACTION'] * 100)}% of ENTIRE population\n\n"
        f"--- BEST EVOLVED TWO-TREE RULE ---\n"
        f"DONOR RULE (Will I pay to help?):\n"
        f"  Raw:   {raw_donor}\n"
        f"  SymPy: {simp_donor}\n"
        f"RECIPIENT RULE (Will I pay to receive?):\n"
        f"  Raw:   {raw_recipient}\n"
        f"  SymPy: {simp_recipient}\n\n"
        f"--- FINAL PERFORMANCE (PER AGENT) ---\n"
        f"Maximum Theoretical Fitness: {theoretical_max}\n"
        f"Control Baseline: {control_fit:.1f}\n"
        f"iGSS Max Payoff: {history['max_igss'][-1]:.1f}\n"
        f"Uncond. Cooperators Avg Payoff: {history['uc_scores'][-1]:.1f}\n"
        f"Defectors Avg Payoff: {history['d_scores'][-1]:.1f}\n\n"
        f"--- FOSSIL RECORD ---\n{fossils_str}\n"
        f"{'='*60}\n"
    )
    print(report_text)

    fig, ax_plot = plt.subplots(figsize=(10, 6))
    ax_plot.plot(history["max_igss"], label='iGSS Agents (Max)', color='blue', linewidth=2.5)
    ax_plot.plot(history["avg_igss"], label='iGSS Agents (Avg)', color='cornflowerblue', linewidth=2, linestyle=':')
    ax_plot.plot(history["uc_scores"], label='Unconditional Cooperators', color='green', linewidth=1.5, linestyle='--')
    ax_plot.plot(history["d_scores"], label='Defectors', color='red', linewidth=1.5, linestyle='-.')
    ax_plot.axhline(y=control_fit, color='black', linestyle=':', label=f'Control Baseline: {control_fit:.1f}')
    
    ax_plot.set_title('F&M Bilateral Consent [Mode 2.0]: Evolutionary Dynamics')
    ax_plot.set_xlabel('Generation')
    ax_plot.set_ylabel('Cumulative Payoff (Per Agent)')
    ax_plot.legend(loc='lower right')
    ax_plot.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show() 

if __name__ == "__main__":
    best_rule, history, control_fit = run_evolution()
    plot_dashboard(best_rule, history, MODEL_CONFIG, EVO_CONFIG, control_fit)