import operator
import random
import numpy as np
import mesa
import sympy as sp
from deap import base, creator, tools, gp

# =============================================================================
# 1. GLOBAL CONFIGURATIONS
# =============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

MODEL_CONFIG = {
    "NUM_PAIRS": 1,       # 20 pairs (40 agents total)
    "NUM_ROUNDS": 50,      
    "REWARD": 1,
    "PENALTY": -1
}

EVO_CONFIG = {
    "POP_SIZE": 50,        
    "MAX_GENS": 1000,         
    "PARSIMONY_TAX": 0.05  
}

# =============================================================================
# 2. GENETIC PROGRAMMING SETUP (Continuous Math)
# =============================================================================
def random_constant():
    return random.choice([-2, -1, 0, 1, 2])

pset_action = gp.PrimitiveSet("ActionRule_Coord", 2) 
pset_action.renameArguments(ARG0='Env_Clock', ARG1='My_ID')

pset_action.addPrimitive(operator.add, 2)
pset_action.addPrimitive(operator.sub, 2)
pset_action.addPrimitive(operator.mul, 2)
pset_action.addEphemeralConstant("rand_const", random_constant)

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
toolbox.register("select", tools.selTournament, tournsize=5)

# =============================================================================
# 3. AGENT-BASED MODEL (Fixed Initialization for Mesa 2.0+)
# =============================================================================
class CoordAgent(mesa.Agent):
    def __init__(self, model, my_id, agent_type="iGSS-Agent"):
        # Fixed: Aligned with your working DR_mode1.py / IR_mode1.py logic
        super().__init__(model) 
        self.my_id = my_id
        self.agent_type = agent_type
        self.payoff = 0
        
    def get_action(self, clock):
        if self.agent_type == "Control-Perfect":
            return 1 if self.my_id != clock else -1
            
        decision_score = self.model.igss_rule(clock, self.my_id)
        return 1 if decision_score > 0 else -1

class CoordinationModel(mesa.Model):
    def __init__(self, igss_rule, config=MODEL_CONFIG, control_mode=False):
        super().__init__()
        self.igss_rule = igss_rule
        self.config = config
        self.clock = 0 
        
        self.agents_id_0 = []
        self.agents_id_1 = []
        
        a_type = "Control-Perfect" if control_mode else "iGSS-Agent"
        
        for _ in range(config["NUM_PAIRS"]):
            # Fixed: We let Mesa handle unique_id internally
            a0 = CoordAgent(self, my_id=0, agent_type=a_type)
            a1 = CoordAgent(self, my_id=1, agent_type=a_type)
            self.agents_id_0.append(a0)
            self.agents_id_1.append(a1)

    def step(self):
        self.clock = 1 - self.clock
        
        # Shuffle for pairing
        shuffled_0 = list(self.agents_id_0)
        shuffled_1 = list(self.agents_id_1)
        random.shuffle(shuffled_0)
        random.shuffle(shuffled_1)

        for a0, a1 in zip(shuffled_0, shuffled_1):
            act_0 = a0.get_action(self.clock)
            act_1 = a1.get_action(self.clock)
            
            if act_0 != act_1:
                a0.payoff += self.config["REWARD"]
                a1.payoff += self.config["REWARD"]
            else: 
                a0.payoff += self.config["PENALTY"]
                a1.payoff += self.config["PENALTY"]

    def get_average_fitness(self):
        all_agents = self.agents_id_0 + self.agents_id_1
        if not all_agents: return 0
        return sum(a.payoff for a in all_agents) / len(all_agents)

# =============================================================================
# 4. EVOLUTIONARY ENGINE
# =============================================================================
def evaluate_rule(individual, model_config, evo_config):
    func_action = toolbox.compile(expr=individual)
    tot_score = 0
    runs = 3
    for _ in range(runs): 
        m = CoordinationModel(igss_rule=func_action, config=model_config)
        for _ in range(model_config["NUM_ROUNDS"]): m.step()
        tot_score += m.get_average_fitness()
        
    avg_score = tot_score / runs
    tax = len(individual) * evo_config["PARSIMONY_TAX"]
    return (avg_score - tax), 

toolbox.register("evaluate", evaluate_rule, model_config=MODEL_CONFIG, evo_config=EVO_CONFIG)

def get_control_baseline(model_config):
    m = CoordinationModel(igss_rule=None, config=model_config, control_mode=True)
    for _ in range(model_config["NUM_ROUNDS"]): m.step()
    return m.get_average_fitness()

def run_evolution():
    pop = toolbox.population(n=EVO_CONFIG["POP_SIZE"])
    hof = tools.HallOfFame(1) 
    
    print("\nInitializing iGSS Minimal Coordination Sandbox...")
    control_fitness = get_control_baseline(MODEL_CONFIG)
    print(f"Theoretical Max Fitness (Perfect XOR): {control_fitness:.2f}\n")

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses): ind.fitness.values = fit
    hof.update(pop)
        
    for gen in range(1, EVO_CONFIG["MAX_GENS"] + 1): 
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.6:
                toolbox.mate(child1, child2)
                del child1.fitness.values; del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.3:
                toolbox.mutate(mutant)
                del mutant.fitness.values
                
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid_ind, map(toolbox.evaluate, invalid_ind)):
            ind.fitness.values = fit
                
        pop[:] = offspring
        hof.update(pop)
        pop[-1] = toolbox.clone(hof[0]) 
        
        best_in_gen = tools.selBest(pop, k=1)[0]
        print(f"Gen {gen:02d} | Max Fit: {best_in_gen.fitness.values[0]:.2f} | Rule: {best_in_gen}")

    return hof[0]

# =============================================================================
# 5. SYMPY PARSER
# =============================================================================
def simplify_rule(gp_string):
    mapping = {
        'add': lambda x, y: x + y, 'sub': lambda x, y: x - y, 'mul': lambda x, y: x * y,
        'Env_Clock': sp.Symbol('Clock'), 'My_ID': sp.Symbol('ID')
    }
    try: return sp.simplify(eval(gp_string, {"__builtins__": {}}, mapping))
    except Exception: return "Simplify Error"

if __name__ == "__main__":
    best_rule = run_evolution()
    print(f"\nAlgebraic Equivalent: {simplify_rule(str(best_rule))}")