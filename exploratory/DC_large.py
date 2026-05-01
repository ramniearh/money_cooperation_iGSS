import operator
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import mesa
import sympy as sp
from deap import base, creator, tools, gp

# =============================================================================
# 1. GLOBAL CONFIGURATIONS (THE TWEAKS)
# =============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

MODEL_CONFIG = {
    "NUM_SPECIES": 20,          
    "AGENTS_PER_SPECIES": 10,   
    "NUM_ROUNDS": 300,          
    "CONSUMPTION_REWARD": 1.0,  
    "CONSENT_COST": 0.20,       # TWEAK 1: Punishingly high friction for careless "YES" votes
    "INITIAL_TOKENS": 1         
}

EVO_CONFIG = {
    "POP_SIZE": 100,        
    "MAX_GENS": 50,         
    "PARSIMONY_TAX": 0.01       # TWEAK 2: Lowered tax so they don't fear building complex math
}

# =============================================================================
# 2. GENETIC PROGRAMMING SETUP 
# =============================================================================
def random_constant():
    return random.choice([-1, 1])

pset_money = gp.PrimitiveSet("MoneyRule", 3) 
pset_money.renameArguments(ARG0='Val_Good', ARG1='Delta_Tok', ARG2='My_Tok')

pset_money.addPrimitive(operator.add, 2)
pset_money.addPrimitive(operator.sub, 2)
pset_money.addPrimitive(operator.mul, 2)
pset_money.addEphemeralConstant("rand_const", random_constant)

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr_money", gp.genHalfAndHalf, pset=pset_money, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_money)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset_money)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_money, pset=pset_money)
toolbox.register("select", tools.selTournament, tournsize=5)

# =============================================================================
# 3. AGENT-BASED MODEL 
# =============================================================================
class TokenAgent(mesa.Agent):
    def __init__(self, model, unique_id, species_id, num_species, agent_type="iGSS"):
        super().__init__(model) 
        self.unique_id = unique_id
        
        self.my_produce = species_id
        self.my_need = (species_id + 1) % num_species 
        
        self.tokens = model.config["INITIAL_TOKENS"]
        self.agent_type = agent_type
        self.fitness_score = 0.0
        
    def consent_to_trade(self, val_good, delta_tok):
        if self.agent_type == "Control-Perfect":
            if val_good == 1 and delta_tok == -1: return 1 if self.tokens > 0 else -1
            if val_good == 0 and delta_tok == 1:  return 1 
            return -1
            
        score = self.model.igss_rule(val_good, delta_tok, self.tokens)
        return 1 if score > 0 else -1

class TokenModel(mesa.Model):
    def __init__(self, igss_rule, config=MODEL_CONFIG, control_mode=False):
        super().__init__()
        self.igss_rule = igss_rule
        self.config = config
        self.traders = []
        
        a_type = "Control-Perfect" if control_mode else "iGSS"
        
        agent_id = 0
        for s in range(config["NUM_SPECIES"]):
            for _ in range(config["AGENTS_PER_SPECIES"]):
                self.traders.append(TokenAgent(self, agent_id, species_id=s, num_species=config["NUM_SPECIES"], agent_type=a_type))
                agent_id += 1

    def step(self):
        random.shuffle(self.traders)
        
        for i in range(0, len(self.traders) - 1, 2):
            a0 = self.traders[i]
            a1 = self.traders[i+1]
            
            buyer, seller = None, None
            
            # The universe ensures the physical goods match supply/demand
            if a1.my_produce == a0.my_need:
                buyer, seller = a0, a1
            elif a0.my_produce == a1.my_need:
                buyer, seller = a1, a0
                
            if buyer and seller:
                # Agents are forced to consent and risk the friction penalty
                act_buyer = buyer.consent_to_trade(val_good=1, delta_tok=-1)
                act_seller = seller.consent_to_trade(val_good=0, delta_tok=1)
                
                if act_buyer > 0: buyer.fitness_score -= self.config["CONSENT_COST"]
                if act_seller > 0: seller.fitness_score -= self.config["CONSENT_COST"]
                
                # Liquidity constraint check by the universe
                if act_buyer > 0 and act_seller > 0 and buyer.tokens >= 1:
                    buyer.tokens -= 1
                    seller.tokens += 1
                    buyer.fitness_score += self.config["CONSUMPTION_REWARD"]

    def get_average_fitness(self):
        return sum(a.fitness_score for a in self.traders) / len(self.traders)

# =============================================================================
# 4. EVOLUTIONARY ENGINE & EXECUTION
# =============================================================================
def evaluate_rule(individual, config):
    func = toolbox.compile(expr=individual)
    tot_score = 0
    runs = 2 
    for _ in range(runs):
        m = TokenModel(igss_rule=func, config=config)
        for _ in range(config["NUM_ROUNDS"]): m.step()
        tot_score += m.get_average_fitness()
        
    avg_score = tot_score / runs
    tax = len(individual) * EVO_CONFIG["PARSIMONY_TAX"]
    return (avg_score - tax),

toolbox.register("evaluate", evaluate_rule, config=MODEL_CONFIG)

def get_control_baseline(config):
    tot_score = 0
    runs = 3
    for _ in range(runs):
        m = TokenModel(igss_rule=None, config=config, control_mode=True)
        for _ in range(config["NUM_ROUNDS"]): m.step()
        tot_score += m.get_average_fitness()
    return tot_score / runs

def run_evolution():
    pop = toolbox.population(n=EVO_CONFIG["POP_SIZE"])
    hof = tools.HallOfFame(1)
    
    print(f"\nInitializing TWEAKED N-Species Economy: {MODEL_CONFIG['NUM_SPECIES']} Goods, {MODEL_CONFIG['NUM_SPECIES'] * MODEL_CONFIG['AGENTS_PER_SPECIES']} Agents...")
    control_fit = get_control_baseline(MODEL_CONFIG)
    print(f"Theoretical Fiat Ceiling: {control_fit:.2f}\n")

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
        'Val_Good': sp.Symbol('Val'), 'Delta_Tok': sp.Symbol('dTok'), 'My_Tok': sp.Symbol('Bal')
    }
    best_str = str(hof[0])
    try: simp_action = sp.simplify(eval(best_str, {'__builtins__': {}}, mapping))
    except Exception: simp_action = "Parse Error"
    
    print("\n" + "="*50)
    print(f"EVOLUTION COMPLETE")
    print("="*50)
    print(f"Raw Tree: {best_str}")
    print(f"SymPy:    {simp_action}")
    print("="*50)

if __name__ == "__main__":
    run_evolution()