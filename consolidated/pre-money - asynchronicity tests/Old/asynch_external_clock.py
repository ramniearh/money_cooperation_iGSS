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
    "NUM_PAIRS": 20,
    "NUM_ROUNDS": 50,      
    "COLLISION_PENALTY": 0.5 
}

EVO_CONFIG = {
    "POP_SIZE": 60,        
    "MAX_GENS": 30,         
    "PARSIMONY_TAX": 0.1  
}

# =============================================================================
# 2. GP SETUP: One primitive (The Clock) and one internal state (The ID)
# =============================================================================
pset_action = gp.PrimitiveSet("ActionRule_Clock", 2) 
pset_action.renameArguments(ARG0='Env_Clock', ARG1='My_ID')

pset_action.addPrimitive(operator.add, 2)
pset_action.addPrimitive(operator.sub, 2)
pset_action.addPrimitive(operator.mul, 2)

def random_constant():
    return random.choice([-1, 1])

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
# 3. ABM: Leontief Scoring with External Clock
# =============================================================================
class CoordAgent(mesa.Agent):
    def __init__(self, model, my_id):
        super().__init__(model) 
        self.my_id = my_id # 0 or 1
        self.successful_gives = 0
        self.collisions = 0
        
    def get_action(self, clock):
        # We interpret > 0 as Give (1) and <= 0 as Receive (-1)
        score = self.model.igss_rule(clock, self.my_id)
        return 1 if score > 0 else -1

class CoordinationModel(mesa.Model):
    def __init__(self, igss_rule, config=MODEL_CONFIG):
        super().__init__()
        self.igss_rule = igss_rule
        self.config = config
        self.clock = 0 
        
        self.agents_0 = [CoordAgent(self, 0) for _ in range(config["NUM_PAIRS"])]
        self.agents_1 = [CoordAgent(self, 1) for _ in range(config["NUM_PAIRS"])]

    def step(self):
        # The External Alternator
        self.clock = 1 - self.clock
        
        # Match ID=0 agents with ID=1 agents
        random.shuffle(self.agents_0)
        random.shuffle(self.agents_1)

        for a0, a1 in zip(self.agents_0, self.agents_1):
            act_0 = a0.get_action(self.clock)
            act_1 = a1.get_action(self.clock)
            
            if act_0 != act_1:
                if act_0 == 1: a0.successful_gives += 1
                else:          a1.successful_gives += 1
            else: 
                a0.collisions += 1
                a1.collisions += 1

    def get_average_fitness(self):
        total_fit = 0
        for a0, a1 in zip(self.agents_0, self.agents_1):
            # LEONTIEF: Must balance giving to get any score
            pair_score = min(a0.successful_gives, a1.successful_gives)
            pair_score -= (a0.collisions * self.config["COLLISION_PENALTY"])
            total_fit += pair_score
        return total_fit / len(self.agents_0)

# =============================================================================
# 4. EXECUTION
# =============================================================================
def evaluate(individual):
    func = toolbox.compile(expr=individual)
    m = CoordinationModel(igss_rule=func)
    for _ in range(MODEL_CONFIG["NUM_ROUNDS"]): m.step()
    score = m.get_average_fitness()
    return (score - len(individual) * EVO_CONFIG["PARSIMONY_TAX"]),

toolbox.register("evaluate", evaluate)

def run():
    pop = toolbox.population(n=EVO_CONFIG["POP_SIZE"])
    hof = tools.HallOfFame(1)
    
    for gen in range(EVO_CONFIG["MAX_GENS"]):
        offspring = tools.selBest(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5: toolbox.mate(c1, c2); del c1.fitness.values; del c2.fitness.values
        for m in offspring:
            if random.random() < 0.2: toolbox.mutate(m); del m.fitness.values
            
        for ind in [i for i in offspring if not i.fitness.valid]:
            ind.fitness.values = toolbox.evaluate(ind)
            
        pop[:] = offspring
        hof.update(pop)
        print(f"Gen {gen:02d} | Max Fit: {hof[0].fitness.values[0]:.2f} | Rule: {hof[0]}")
    return hof[0]

if __name__ == "__main__":
    best = run()
    mapping = {'add':lambda x,y:x+y, 'sub':lambda x,y:x-y, 'mul':lambda x,y:x*y, 'Env_Clock':sp.Symbol('Clock'), 'My_ID':sp.Symbol('ID')}
    simplified = sp.simplify(eval(str(best), {'__builtins__': {}}, mapping))
    print(f"\nSimplified Best Rule: {simplified}")