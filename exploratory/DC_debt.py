import operator
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import mesa
import sympy as sp
from deap import base, creator, tools, gp

# =============================================================================
# 1. GLOBAL CONFIGURATIONS (THE SOCIOLOGICAL SHIFT)
# =============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

MODEL_CONFIG = {
    "NUM_IGSS_PER_SPECIES": 15,  # 45 Cooperative AI Agents
    "NUM_DEF_PER_SPECIES": 5,    # 15 Free-Riders (The threat that forces institutions)
    "NUM_ROUNDS": 150,          
    "CONSUMPTION_REWARD": 1.0,  
    "CONSENT_COST": 0.05,        # Friction
    "PRODUCTION_COST": 0.5 # added
}

EVO_CONFIG = {
    "POP_SIZE": 80,        
    "MAX_GENS": 40,         
    "PARSIMONY_TAX": 0.05       
}

# =============================================================================
# 2. GENETIC PROGRAMMING SETUP (The Social Primitives)
# =============================================================================
def random_constant():
    return random.choice([-2, -1, 1, 2])

# PRIMITIVES: The agent evaluates the Social Trust of a transaction.
# Val_Good: 1 if I need it, 0 if I am giving my own away.
# dRep: 1 if I am gaining reputation (selling), -1 if I am spending reputation (buying).
# My_Rep: My net social credit score.
# Partner_Rep: The partner's net social credit score.
pset_debt = gp.PrimitiveSet("DebtRule", 4) 
pset_debt.renameArguments(ARG0='Val', ARG1='dRep', ARG2='My_Rep', ARG3='Partner_Rep')

pset_debt.addPrimitive(operator.add, 2)
pset_debt.addPrimitive(operator.sub, 2)
pset_debt.addPrimitive(operator.mul, 2)
pset_debt.addEphemeralConstant("rand_const", random_constant)

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr_debt", gp.genHalfAndHalf, pset=pset_debt, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_debt)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset_debt)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_debt, pset=pset_debt)
toolbox.register("select", tools.selTournament, tournsize=5)

# =============================================================================
# 3. AGENT-BASED MODEL (The Village of Trust)
# =============================================================================
class VillageAgent(mesa.Agent):
    def __init__(self, model, unique_id, species_id, agent_type):
        super().__init__(model) 
        self.unique_id = unique_id
        
        # 3-Species Ring
        self.my_produce = species_id
        self.my_need = (species_id + 1) % 3 
        
        self.agent_type = agent_type
        self.reputation = 0  # Starts at 0. Can go positive or negative.
        self.fitness_score = 0.0
        
    def consent_to_trade(self, val, drep, partner_rep):
        if self.agent_type == "Defector":
            # Free-Riders take everything and give nothing
            return 1 if val == 1 else -1
            
        if self.agent_type == "Control-CreditLimit":
            # Baseline: Explicit rule that cuts off free-riders at -2 debt
            if val == 1 and drep == -1: return 1
            if val == 0 and drep == 1:  return 1 if partner_rep >= -2 else -1
            return -1
            
        # iGSS evaluates the social physics
        score = self.model.igss_rule(val, drep, self.reputation, partner_rep)
        return 1 if score > 0 else -1

class VillageModel(mesa.Model):
    def __init__(self, igss_rule, config=MODEL_CONFIG, control_mode=False):
        super().__init__()
        self.igss_rule = igss_rule
        self.config = config
        self.traders = []
        
        a_type = "Control-CreditLimit" if control_mode else "iGSS"
        
        # Populate the village with iGSS agents and Defectors
        agent_id = 0
        for s in range(3):
            for _ in range(config["NUM_IGSS_PER_SPECIES"]):
                self.traders.append(VillageAgent(self, agent_id, s, agent_type=a_type))
                agent_id += 1
            for _ in range(config["NUM_DEF_PER_SPECIES"]):
                self.traders.append(VillageAgent(self, agent_id, s, agent_type="Defector"))
                agent_id += 1

    def step(self):
        random.shuffle(self.traders)
        
        for i in range(0, len(self.traders) - 1, 2):
            a0 = self.traders[i]
            a1 = self.traders[i+1]
            
            buyer, seller = None, None
            
            if a1.my_produce == a0.my_need:
                buyer, seller = a0, a1
            elif a0.my_produce == a1.my_need:
                buyer, seller = a1, a0
                
            if buyer and seller:
                # Agents evaluate the trust of the transaction
                act_buyer = buyer.consent_to_trade(val=1, drep=-1, partner_rep=seller.reputation)
                act_seller = seller.consent_to_trade(val=0, drep=1, partner_rep=buyer.reputation)
                
                # Friction
                if act_buyer > 0: buyer.fitness_score -= self.config["CONSENT_COST"]
                if act_seller > 0: seller.fitness_score -= self.config["CONSENT_COST"]
                
                if act_buyer > 0 and act_seller > 0:
                    buyer.reputation -= 1  
                    seller.reputation += 1 
                    seller.fitness_score -= self.config["PRODUCTION_COST"] # THE FIX
                    if buyer.agent_type != "Defector":
                        buyer.fitness_score += self.config["CONSUMPTION_REWARD"]
                

    def get_average_igss_fitness(self):
        # We only care how well the iGSS agents survive
        igss_agents = [a for a in self.traders if a.agent_type != "Defector"]
        return sum(a.fitness_score for a in igss_agents) / len(igss_agents)

# =============================================================================
# 4. EVOLUTIONARY ENGINE & VISUALIZATION
# =============================================================================
def evaluate_rule(individual, config):
    func = toolbox.compile(expr=individual)
    tot_score = 0
    runs = 3 
    for _ in range(runs):
        m = VillageModel(igss_rule=func, config=config)
        for _ in range(config["NUM_ROUNDS"]): m.step()
        tot_score += m.get_average_igss_fitness()
        
    avg_score = tot_score / runs
    tax = len(individual) * EVO_CONFIG["PARSIMONY_TAX"]
    return (avg_score - tax),

toolbox.register("evaluate", evaluate_rule, config=MODEL_CONFIG)

def get_control_baseline(config):
    tot_score = 0
    runs = 5
    for _ in range(runs):
        m = VillageModel(igss_rule=None, config=config, control_mode=True)
        for _ in range(config["NUM_ROUNDS"]): m.step()
        tot_score += m.get_average_igss_fitness()
    return tot_score / runs

def run_evolution():
    pop = toolbox.population(n=EVO_CONFIG["POP_SIZE"])
    hof = tools.HallOfFame(1)
    
    print("\nInitializing Phase 5: The Relational Debt Village...")
    control_fit = get_control_baseline(MODEL_CONFIG)
    print(f"Control Baseline (Credit Limit at -2): {control_fit:.2f}\n")

    for gen in range(EVO_CONFIG["MAX_GENS"]):
        offspring = tools.selBest(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.6: toolbox.mate(c1, c2); del c1.fitness.values; del c2.fitness.values
        for m in offspring:
            if random.random() < 0.3: toolbox.mutate(m); del m.fitness.values
            
        for ind in [i for i in offspring if not i.fitness.valid]:
            ind.fitness.values = toolbox.evaluate(ind)
            
        pop[:] = offspring
        hof.update(pop)
        pop[-1] = toolbox.clone(hof[0]) 
        
        fits = [ind.fitness.values[0] for ind in pop]
        gen_max = max(fits)
        print(f" > Gen {gen:02d} | Max Fit: {gen_max:5.2f}")

    mapping = {
        'add': lambda x, y: x + y, 'sub': lambda x, y: x - y, 'mul': lambda x, y: x * y, 
        'Val': sp.Symbol('Val'), 'dRep': sp.Symbol('dRep'), 
        'My_Rep': sp.Symbol('My_Rep'), 'Partner_Rep': sp.Symbol('Partner_Rep')
    }
    best_str = str(hof[0])
    try: simp_action = sp.simplify(eval(best_str, {'__builtins__': {}}, mapping))
    except Exception: simp_action = "Parse Error"
    
    print("\n" + "="*50)
    print(f"SOCIOLOGICAL EVOLUTION COMPLETE")
    print("="*50)
    print(f"Raw Tree: {best_str}")
    print(f"SymPy:    {simp_action}")
    print("="*50)

if __name__ == "__main__":
    run_evolution()