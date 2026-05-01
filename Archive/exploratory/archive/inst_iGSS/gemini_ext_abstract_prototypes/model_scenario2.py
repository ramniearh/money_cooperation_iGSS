import mesa
import random

class EconAgent(mesa.Agent):
    "An agent that produces, evaluates partners, and trades conditionally"

    # NOTE: Added unique_id to prevent Mesa version crashes
    def __init__(self, model, production_good):
        super().__init__(model)
        self.production_good = production_good
        
        # Initialize inventory for the 3 goods in our economy
        self.inventory = {0: 0, 1: 0, 2: 0}
        self.inventory[self.production_good] = 2

    def produce(self):
        """Called by the model simultaneously for all agents"""
        self.inventory[self.production_good] += 2

    def consume(self):
        """Agents eat 1 of every good they have"""
        for good in self.inventory:
            if self.inventory[good] > 0:
                self.inventory[good] -= 1

    def evaluation_rule(self, partner):  
        """Uses GP to decide whether to say YES or NO to a trade"""
        my_supply = self.inventory[self.production_good]
        time_now = self.model.current_step 
        partner_specialty = partner.production_good # The Market Signal!
        
        # We pass 3 variables to the AI: my_supply, what the partner has, and the time
        decision_score = self.model.decision_rule(my_supply, partner_specialty, time_now)
        
        return decision_score > 0

    def give(self, partner):
        """Transfers one unit to the partner"""
        if self.inventory[self.production_good] > 0:
            self.inventory[self.production_good] -= 1
            partner.inventory[self.production_good] += 1


class DivisionLaborModel(mesa.Model):
    def __init__(self, N, rule):
        super().__init__()
        self.num_agents = N
        self.decision_rule = rule
        self.current_step = 0

        for i in range(self.num_agents):
            specialty = i % 3
            
            EconAgent(self, specialty)
    
    def step(self):
        self.current_step += 1
        
        # 1. Get all agents and shuffle them
        agent_list = list(self.agents)
        random.shuffle(agent_list)

        # 2. Pair them up for Speed Dating
        for i in range(0, len(agent_list) - 1, 2):
            agent_A = agent_list[i]
            agent_B = agent_list[i+1]

            # 3. Both produce at the exact same time
            agent_A.produce()
            agent_B.produce()

            # 4. THE MUTUAL CONSENT EVALUATION
            a_wants_trade = agent_A.evaluation_rule(agent_B)
            b_wants_trade = agent_B.evaluation_rule(agent_A)

            # 5. THE INSTITUTION OF EXCHANGE
            # Only if BOTH agents agree does the goods transfer happen
            if a_wants_trade and b_wants_trade:
                agent_A.give(agent_B)
                agent_B.give(agent_A)
                
        # 6. Everyone consumes at the end of the day
        for agent in self.agents:
            agent.consume()

    def calculate_fitness(self):
        total_diverse_goods = 0
        for agent in self.agents:
            diverse_goods = sum(1 for amount in agent.inventory.values() if amount > 0)
            total_diverse_goods += diverse_goods
        return total_diverse_goods / self.num_agents