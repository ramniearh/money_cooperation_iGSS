from core import CooperationModel, DEFAULT_CONFIG

def run_test(test_name, action_logic, assessment_logic=None, custom_config=None):
    """Runs a single 20-round MESA model to validate mechanics."""
    config = DEFAULT_CONFIG.copy()
    if custom_config:
        config.update(custom_config)
        
    model = CooperationModel(config=config, action_rule=action_logic, assessment_rule=assessment_logic)
    
    for _ in range(config["NUM_ROUNDS"]):
        model.step()
        
    fitness = model.get_igss_fitness()
    print(f"[{test_name}] Final iGSS Average Payoff: {fitness:.2f}")
    return fitness

if __name__ == "__main__":
    print("Running MESA Core Validations (NetLogo Aligned)...\n")
    # Note: Baseline fitness adds +1 per round, so payoffs will be ~20 points higher than before.

    # TEST 1: Pure Indirect Reciprocity (Hardcoded Assess)
    ir_action = lambda memory, standing, tokens: standing
    run_test("Test 1: Pure Indirect Reciprocity", ir_action)

    # TEST 2: Co-Evolved Strategy Replica (Stern Judging)
    # The new assessment rule takes (helper, recipient, action_signal)
    def stern_judging(helper, recipient, act):
        # Good if you cooperate, OR if you defected against a bad person.
        new_standing = 1 if (act > 0) or (recipient.standing == 0) else 0
        helper.standing = new_standing

    run_test("Test 2: Stern Judging Assessment", ir_action, stern_judging)

    # TEST 3: Pure Money Mechanism
    # Action: Cooperate if they have tokens
    money_action = lambda memory, standing, tokens: tokens
    money_config = {"USE_STANDING": False, "USE_MEMORY": False, "USE_TOKENS": True}
    run_test("Test 3: Pure Money Strategy", money_action, custom_config=money_config)

    # TEST 4: Direct Reciprocity (Tit-for-Tat)
    # Action: Cooperate if they are NOT in memory (memory arg = 1 if they defected previously)
    tft_action = lambda memory, standing, tokens: -1 if memory > 0 else 1
    tft_config = {"USE_STANDING": False, "USE_MEMORY": True, "USE_TOKENS": False}
    run_test("Test 4: Direct Reciprocity (Tit-for-Tat)", tft_action, custom_config=tft_config)
    
    print("\nValidations complete! The core is ready for DEAP.")