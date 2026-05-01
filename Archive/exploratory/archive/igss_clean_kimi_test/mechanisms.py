"""
Mechanisms Module for iGSS

Simple functions for Direct Reciprocity, Indirect Reciprocity, and Monetary Exchange.
No classes - just pure functions that operate on agent state.
"""

# =============================================================================
# DIRECT RECIPROCITY (DR) - Memory-based
# =============================================================================

def update_memory(helper, recipient, cooperates):
    """
    Update memory sets for Direct Reciprocity.
    
    Logic:
    - If helper defects, recipient remembers them (adds to memory set)
    - If recipient was in helper's memory, remove them (forgiveness on mutual defection)
    
    Args:
        helper: The agent who made the decision
        recipient: The agent who was the target
        cooperates: Boolean - did helper cooperate?
    """
    if not cooperates:
        # Recipient remembers the defection
        recipient.memory.add(helper.unique_id)
        # Forgiveness: if recipient was in helper's memory, remove it
        if recipient.unique_id in helper.memory:
            helper.memory.remove(recipient.unique_id)


def get_memory_signal(focal, partner):
    """
    Get ARG0 signal: 1 if partner is in focal's memory (defected before), else 0.
    
    Args:
        focal: The agent evaluating the partner
        partner: The agent being evaluated
        
    Returns:
        int: 1 if partner defected against focal before, 0 otherwise
    """
    return 1 if partner.unique_id in focal.memory else 0


# =============================================================================
# INDIRECT RECIPROCITY (IR) - Standing/Reputation
# =============================================================================

def update_standing_hardcoded(helper, recipient, cooperates):
    """
    Hardcoded standing update (Mode 1):
    - Cooperating always gives Good standing (1)
    - Defecting against Good recipient gives Bad standing (0)
    
    Args:
        helper: The agent who made the decision
        recipient: The agent who was the target
        cooperates: Boolean - did helper cooperate?
    """
    if cooperates:
        helper.standing = 1
    else:
        # Only lose standing if defecting against a Good recipient
        if recipient.standing == 1:
            helper.standing = 0


def update_standing_evolved(helper, recipient, cooperates, assessment_rule):
    """
    Evolved standing update (Modes 2 & 3):
    Uses an evolved assessment rule to determine new standing.
    
    Args:
        helper: The agent who made the decision
        recipient: The agent who was the target
        cooperates: Boolean - did helper cooperate?
        assessment_rule: Compiled GP function taking (Action, HelperStanding, RecipientStanding)
    """
    action_signal = 1 if cooperates else 0
    assessment_score = assessment_rule(action_signal, helper.standing, recipient.standing)
    helper.standing = 1 if assessment_score > 0 else 0


def get_standing_signal(partner):
    """
    Get ARG1 signal: Partner's current standing (1 = Good, 0 = Bad).
    
    Args:
        partner: The agent being evaluated
        
    Returns:
        int: Partner's standing value
    """
    return partner.standing


# =============================================================================
# MONETARY EXCHANGE - Tokens
# =============================================================================

def update_tokens(helper, recipient, cooperates, config):
    """
    Token transfer logic:
    - If cooperation occurs and recipient has tokens, recipient pays helper
    
    Args:
        helper: The agent who cooperated
        recipient: The agent who received help
        cooperates: Boolean - did helper cooperate?
        config: Model configuration dict with USE_TOKENS flag
        
    Returns:
        bool: True if token transfer occurred, False otherwise
    """
    if cooperates and config.get("USE_TOKENS", False):
        if recipient.tokens > 0:
            recipient.tokens -= 1
            helper.tokens += 1
            return True
    return False


def check_partner_solvent(partner, config):
    """
    Check if partner has tokens to pay (for REQUIRE_TOKENS_TO_COOPERATE mode).
    
    Args:
        partner: The agent being evaluated
        config: Model configuration dict
        
    Returns:
        bool: True if partner can pay (or if solvency check is disabled)
    """
    if not config.get("REQUIRE_TOKENS_TO_COOPERATE", False):
        return True
    return partner.tokens > 0


def get_token_signal(partner):
    """
    Get ARG2 signal: Partner's current token count.
    
    Args:
        partner: The agent being evaluated
        
    Returns:
        int: Partner's token balance
    """
    return partner.tokens


# =============================================================================
# INITIALIZATION HELPERS
# =============================================================================

def initialize_tokens(agent, config):
    """
    Initialize agent's token balance based on INITIAL_LIQUIDITY config.
    
    Args:
        agent: The agent to initialize
        config: Model configuration dict
    """
    liq = config.get("INITIAL_LIQUIDITY", 2.0)
    if liq < 1.0:
        # Probabilistic: e.g., 0.5 means 50% chance of 1 token
        import random
        agent.tokens = 1 if random.random() < liq else 0
    else:
        # Deterministic: integer token amount
        agent.tokens = int(liq)
