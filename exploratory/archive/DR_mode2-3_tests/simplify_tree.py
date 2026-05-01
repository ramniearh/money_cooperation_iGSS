import sympy as sp

def simplify_gp_rule(gp_string):
    """Converts a DEAP GP string into a simplified algebraic equation."""
    
    # 1. Define all possible variables from your models
    # I'm using short variable names (P, A, W) for cleaner final output, 
    # but you can change these back to the full names if you prefer.
    P, A, W = sp.symbols('PartnerInMemory PartnerAction PartnerWasInMemory')
    
    # 2. Map DEAP functional strings to Sympy mathematical objects
    mapping = {
        'add': lambda x, y: x + y,
        'sub': lambda x, y: x - y,
        'mul': lambda x, y: x * y,
        # if_then requires Sympy's Piecewise logic: (Output, Condition)
        'if_then': lambda cond, out: sp.Piecewise((out, cond > 0), (0, True)),
        
        # Map the exact string names outputted by DEAP to our Sympy symbols
        'PartnerInMemory': P,
        'PartnerAction': A,
        'PartnerWasInMemory': W
    }
    
    # 3. Evaluate the string as Python code, using our Sympy mapping
    try:
        # We disable __builtins__ for safety when using eval()
        sympy_expr = eval(gp_string, {"__builtins__": {}}, mapping)
        
        # 4. Use Sympy's powerful simplification engine
        simplified_expr = sp.simplify(sympy_expr)
        
        return simplified_expr
        
    except Exception as e:
        return f"Could not simplify: {e}"

# =============================================================================
# TEST EXECUTIONS
# =============================================================================
if __name__ == "__main__":
    # Let's test it on your massive Tit-for-Tat discovery from the 100-round run!
    
    raw_action = "sub(1, add(mul(10, PartnerAction), add(mul(0, 10), if_then(mul(PartnerAction, PartnerAction), PartnerWasInMemory))))"
    
    raw_memory = "0"
    
    print("========================================")
    print("       iGSS RULE SIMPLIFIER             ")
    print("========================================\n")
    
    print("--- ACTION RULE ---")
    print(f"Raw output : {raw_action}")
    print(f"Simplified : {simplify_gp_rule(raw_action)}")
    
    print("\n--- MEMORY RULE ---")
    print(f"Raw output : {raw_memory}")
    print(f"Simplified : {simplify_gp_rule(raw_memory)}")
    print("\n========================================")