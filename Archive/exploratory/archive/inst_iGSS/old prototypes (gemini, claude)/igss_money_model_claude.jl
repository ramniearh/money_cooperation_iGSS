"""
igss_money_model.jl
====================
Julia port of the iGSS Money Emergence Model.
Ferraciolli & Renzini, AAMAS COINE (short paper).

Stack
-----
  Agents.jl          – agent-based simulation (replaces Mesa)
  ExprOptimization.jl – genetic programming on Julia Expr trees (replaces DEAP)
  Statistics          – stdlib, no install needed

Install dependencies (run once in the Julia REPL):
  using Pkg
  Pkg.add(["Agents", "ExprOptimization", "DataFrames"])

Run
---
  julia igss_money_model.jl

Change SCENARIO (1, 2, or 3) and N_GENERATIONS near the top.
"""

using Agents
using ExprOptimization
using ExprOptimization.GeneticProgram   # GP submodule
using Statistics
using Random
using DataFrames

Random.seed!(42)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

const SCENARIO          = 2      # 1 = individualist, 2 = exchange, 3 = debt
const N_GOODS           = 4
const GRID_SIZE         = 10
const N_AGENTS          = 20
const INV_CAPACITY      = 6
const INTERACTION_RANGE = 2
const N_GENERATIONS     = 100    # ← higher than Python prototype; Julia can handle it
const POP_SIZE          = 60
const STEPS_PER_EVAL    = 20
const N_FEATURES        = N_GOODS * 2 + 2 + 1   # 13 floats

const GOOD_TYPES = [Symbol("G$i") for i in 0:N_GOODS-1]  # [:G0, :G1, :G2, :G3]

# ═══════════════════════════════════════════════════════════════════════════════
# ACTION TYPES
# Julia structs replace Python classes — lightweight, zero-overhead dispatch
# ═══════════════════════════════════════════════════════════════════════════════

struct GiveAction
    good      :: Symbol
    target_id :: Int
end

struct SpotExchangeAction
    good_out  :: Symbol
    good_in   :: Symbol
    target_id :: Int
end

struct DebtAction
    good      :: Symbol
    target_id :: Int
end

struct ClearDebtAction
    good        :: Symbol
    creditor_id :: Int
end

const AnyAction = Union{GiveAction, SpotExchangeAction, DebtAction, ClearDebtAction}

# ═══════════════════════════════════════════════════════════════════════════════
# AGENT DEFINITION  (Agents.jl GridAgent)
# ═══════════════════════════════════════════════════════════════════════════════

# Agents.jl requires a mutable struct subtyping AbstractAgent.
# pos is mandatory and managed by the framework.

@agent struct EconomicAgent(GridAgent{2})
    primary_good        :: Symbol
    rule_fn             :: Function          # compiled GP expression

    inventory           :: Dict{Symbol, Int}
    debts_owed          :: Dict{Int, Tuple{Symbol,Int}}   # creditor_id → (good, amount)
    ious_held           :: Dict{Int, Tuple{Symbol,Int}}   # debtor_id  → (good, amount)
    consumed_this_step  :: Set{Symbol}
end

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

# We store the global macro variable (division of labour) as a model property.
# Agents.jl StandardABM accepts a named-tuple of properties.

function make_model(rule_fns; sim_scenario=SCENARIO)
    space = GridSpace((GRID_SIZE, GRID_SIZE); periodic=true, metric=:manhattan)

    properties = Dict(
        :division_of_labour => 0.0,
        :sim_scenario       => sim_scenario,
        :dol_history        => Float64[],
    )

    model = StandardABM(
        EconomicAgent, space;
        properties,
        scheduler = Schedulers.Randomly(),   # random activation order each step
        rng       = MersenneTwister(rand(1:10_000)),
    )

    # Sample N_AGENTS unique positions
    all_positions = [(x, y) for x in 1:GRID_SIZE for y in 1:GRID_SIZE]
    positions     = shuffle(all_positions)[1:N_AGENTS]

    for (idx, pos) in enumerate(positions)
        # Assign primary good by x-region (spatial production sites)
        good_idx     = clamp(ceil(Int, pos[1] * N_GOODS / GRID_SIZE), 1, N_GOODS)
        primary_good = GOOD_TYPES[good_idx]
        rule_fn      = rule_fns[mod1(idx, length(rule_fns))]

        inventory = Dict(g => 0 for g in GOOD_TYPES)
        inventory[primary_good] = INV_CAPACITY

        add_agent!(
            pos, model,
            primary_good,
            rule_fn,
            inventory,
            Dict{Int, Tuple{Symbol,Int}}(),
            Dict{Int, Tuple{Symbol,Int}}(),
            Set{Symbol}(),
        )
    end

    return model
end

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE VECTOR
# Flat array of floats fed into the GP rule function.
# ═══════════════════════════════════════════════════════════════════════════════

function feature_vector(agent::EconomicAgent, neighbour::Union{EconomicAgent,Nothing},
                        macro_dol::Float64)
    own_inv = [Float64(get(agent.inventory, g, 0)) for g in GOOD_TYPES]
    nb_inv  = neighbour === nothing ?
              zeros(Float64, N_GOODS) :
              [Float64(get(neighbour.inventory, g, 0)) for g in GOOD_TYPES]

    own_debts = Float64(length(agent.debts_owed))
    own_ious  = Float64(length(agent.ious_held))

    return [own_inv; nb_inv; own_debts; own_ious; macro_dol]
end

# ═══════════════════════════════════════════════════════════════════════════════
# SAFE ARITHMETIC  (mirrors Python version; prevents NaN/Inf in GP trees)
# ═══════════════════════════════════════════════════════════════════════════════

safe_div(a, b) = abs(b) > 1e-8 ? a / b : 0.0
safe_add(a, b) = a + b
safe_sub(a, b) = a - b
safe_mul(a, b) = a * b
gp_gt(a, b)   = Float64(a > b)
gp_lt(a, b)   = Float64(a < b)
gp_eq(a, b)   = Float64(abs(a - b) < 0.5)
gp_if(c,a,b)  = c > 0.5 ? a : b
gp_neg(a)     = -a
gp_abs(a)     = abs(a)

# ═══════════════════════════════════════════════════════════════════════════════
# GRAMMAR FOR ExprOptimization.jl
#
# ExprOptimization uses a CFG (context-free grammar) defined as a SymbolTable.
# Each non-terminal maps to a vector of production rules (Julia Expr or literals).
# The GP then evolves trees over this grammar.
# ═══════════════════════════════════════════════════════════════════════════════

function build_grammar()
    # Feature terminal names — these become variables in the compiled expression.
    # The GP will produce a Julia Expr; we evaluate it with a local scope
    # that binds these names to the feature vector values.
    feature_terminals = [
        [Symbol("own_G$(i)") for i in 0:N_GOODS-1];
        [Symbol("nb_G$(i)")  for i in 0:N_GOODS-1];
        :own_debts; :own_ious; :macro_dol
    ]

    grammar = @grammar begin
        Real = safe_add(Real, Real)
        Real = safe_sub(Real, Real)
        Real = safe_mul(Real, Real)
        Real = safe_div(Real, Real)
        Real = gp_gt(Real, Real)
        Real = gp_lt(Real, Real)
        Real = gp_eq(Real, Real)
        Real = gp_if(Real, Real, Real)
        Real = gp_neg(Real)
        Real = gp_abs(Real)
        # Feature terminals
        Real = own_G0 | own_G1 | own_G2 | own_G3
        Real = nb_G0  | nb_G1  | nb_G2  | nb_G3
        Real = own_debts | own_ious | macro_dol
        # Numeric constants
        Real = 0.0 | 0.5 | 1.0 | 2.0 | 5.0
    end

    return grammar
end

const GRAMMAR = build_grammar()

# Compile an ExprOptimization individual into a callable Julia function.
# The function signature is f(fv::Vector{Float64}) → Float64.
function compile_rule(ind)
    # ExprOptimization gives us an RuleNode; get_executable turns it into an Expr.
    expr = get_executable(ind, GRAMMAR)

    # Build a function that binds feature names then evaluates the expression.
    fn = @eval function(fv::Vector{Float64})
        # Unpack feature vector into named variables matching the grammar terminals
        own_G0, own_G1, own_G2, own_G3 = fv[1], fv[2], fv[3], fv[4]
        nb_G0,  nb_G1,  nb_G2,  nb_G3  = fv[5], fv[6], fv[7], fv[8]
        own_debts = fv[9]
        own_ious  = fv[10]
        macro_dol = fv[11]
        try
            return Float64($expr)
        catch
            return 0.0
        end
    end

    return fn
end

# ═══════════════════════════════════════════════════════════════════════════════
# AGENT STEP  (called by Agents.jl scheduler each tick)
# ═══════════════════════════════════════════════════════════════════════════════

function agent_step!(agent::EconomicAgent, model)
    produce!(agent)
    decide!(agent, model)
    consume!(agent)
end

function produce!(agent::EconomicAgent)
    deficit = INV_CAPACITY - get(agent.inventory, agent.primary_good, 0)
    if deficit > 0
        agent.inventory[agent.primary_good] =
            get(agent.inventory, agent.primary_good, 0) + max(1, deficit ÷ 2)
    end
end

function consume!(agent::EconomicAgent)
    empty!(agent.consumed_this_step)
    for g in GOOD_TYPES
        if get(agent.inventory, g, 0) > 0
            agent.inventory[g] -= 1
            push!(agent.consumed_this_step, g)
        end
    end
end

# ── Neighbour lookup ──────────────────────────────────────────────────────────

function get_neighbours(agent::EconomicAgent, model)
    # collect_agents_nearby returns agents within Chebyshev distance;
    # we use it as a proxy for Manhattan ≤ INTERACTION_RANGE
    return [a for a in nearby_agents(agent, model, INTERACTION_RANGE)
            if a.id != agent.id]
end

# ── Decision ──────────────────────────────────────────────────────────────────

function decide!(agent::EconomicAgent, model)
    neighbours = get_neighbours(agent, model)
    isempty(neighbours) && return

    best_score  = -Inf
    best_action = nothing

    for nb in neighbours
        fv = feature_vector(agent, nb, model.division_of_labour)

        # ── GIVE actions ──────────────────────────────────────────────────────
        for g in GOOD_TYPES
            if get(agent.inventory, g, 0) > 0
                score = eval_rule(agent.rule_fn, fv)
                bias  = max(0, get(agent.inventory, g, 0) - get(nb.inventory, g, 0))
                score += bias * 0.01
                if score > best_score
                    best_score  = score
                    best_action = GiveAction(g, nb.id)
                end
            end
        end

        # ── SPOT EXCHANGE (scenario ≥ 2) ──────────────────────────────────────
        if model.sim_scenario >= 2
            for g_out in GOOD_TYPES, g_in in GOOD_TYPES
                g_out == g_in && continue
                if get(agent.inventory, g_out, 0) > 0 && get(nb.inventory, g_in, 0) > 0
                    score = eval_rule(agent.rule_fn, fv) + 0.05
                    if score > best_score
                        best_score  = score
                        best_action = SpotExchangeAction(g_out, g_in, nb.id)
                    end
                end
            end
        end

        # ── DEBT (scenario 3) ─────────────────────────────────────────────────
        if model.sim_scenario >= 3
            for g in GOOD_TYPES
                if get(nb.inventory, g, 0) > 0 && !haskey(agent.debts_owed, nb.id)
                    score = eval_rule(agent.rule_fn, fv) - 0.02
                    if score > best_score
                        best_score  = score
                        best_action = DebtAction(g, nb.id)
                    end
                end
            end

            for (creditor_id, (g, _)) in agent.debts_owed
                if get(agent.inventory, g, 0) > 0
                    score = eval_rule(agent.rule_fn, fv) + 0.10
                    if score > best_score
                        best_score  = score
                        best_action = ClearDebtAction(g, creditor_id)
                    end
                end
            end
        end
    end

    best_action !== nothing && execute_action!(agent, best_action, model)
end

function eval_rule(fn::Function, fv::Vector{Float64})
    try
        return Float64(fn(fv))
    catch
        return 0.0
    end
end

# ── Action execution ──────────────────────────────────────────────────────────

function execute_action!(agent::EconomicAgent, action::GiveAction, model)
    target = model[action.target_id]
    if get(agent.inventory, action.good, 0) > 0
        agent.inventory[action.good]  -= 1
        target.inventory[action.good] =
            get(target.inventory, action.good, 0) + 1
    end
end

function execute_action!(agent::EconomicAgent, action::SpotExchangeAction, model)
    target = model[action.target_id]
    if get(agent.inventory, action.good_out, 0) > 0 &&
       get(target.inventory, action.good_in, 0) > 0
        agent.inventory[action.good_out]  -= 1
        target.inventory[action.good_in]  -= 1
        agent.inventory[action.good_in]   =
            get(agent.inventory, action.good_in, 0) + 1
        target.inventory[action.good_out] =
            get(target.inventory, action.good_out, 0) + 1
    end
end

function execute_action!(agent::EconomicAgent, action::DebtAction, model)
    target = model[action.target_id]
    if get(target.inventory, action.good, 0) > 0
        target.inventory[action.good] -= 1
        agent.inventory[action.good]  =
            get(agent.inventory, action.good, 0) + 1
        agent.debts_owed[action.target_id]  = (action.good, 1)
        target.ious_held[agent.id]          = (action.good, 1)
    end
end

function execute_action!(agent::EconomicAgent, action::ClearDebtAction, model)
    creditor = model[action.creditor_id]
    if get(agent.inventory, action.good, 0) > 0 &&
       haskey(agent.debts_owed, action.creditor_id)
        agent.inventory[action.good]    -= 1
        creditor.inventory[action.good] =
            get(creditor.inventory, action.good, 0) + 1
        delete!(agent.debts_owed, action.creditor_id)
        delete!(creditor.ious_held, agent.id)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL STEP  (called once per tick after all agents have acted)
# ═══════════════════════════════════════════════════════════════════════════════

function model_step!(model)
    scores = [length(a.consumed_this_step) / N_GOODS for a in allagents(model)]
    model.division_of_labour = isempty(scores) ? 0.0 : mean(scores)
    push!(model.dol_history, model.division_of_labour)
end

# ═══════════════════════════════════════════════════════════════════════════════
# FITNESS EVALUATION  (iGSS macro-to-micro inversion)
# ═══════════════════════════════════════════════════════════════════════════════

function evaluate(ind, grammar)
    rule_fn  = compile_rule(ind)
    rule_fns = fill(rule_fn, N_AGENTS)
    model    = make_model(rule_fns; sim_scenario=SCENARIO)

    # Run the ABM for STEPS_PER_EVAL ticks
    for _ in 1:STEPS_PER_EVAL
        step!(model, agent_step!, model_step!)
    end

    return model.division_of_labour   # scalar fitness (higher = better)
end

# ═══════════════════════════════════════════════════════════════════════════════
# GP EVOLUTIONARY LOOP  (ExprOptimization.jl)
#
# ExprOptimization.GeneticProgram.optimize() handles:
#   - population initialisation
#   - tournament selection
#   - subtree crossover
#   - subtree mutation
#   - hall of fame tracking
# ═══════════════════════════════════════════════════════════════════════════════

function run_igss()
    println("=" ^ 60)
    println("iGSS Money Emergence Model  [Julia]")
    scenario_name = SCENARIO == 1 ? "individualist" :
                    SCENARIO == 2 ? "exchange" : "debt"
    println("Scenario : $SCENARIO  ($scenario_name)")
    println("Agents   : $N_AGENTS   Grid: $(GRID_SIZE)×$(GRID_SIZE)   Goods: $N_GOODS")
    println("GP       : pop=$POP_SIZE  gen=$N_GENERATIONS  steps/eval=$STEPS_PER_EVAL")
    println("=" ^ 60)

    # ExprOptimization GP configuration
    gp_params = GeneticProgram.GeneticProgramParams(
        pop_size        = POP_SIZE,
        max_depth       = 8,          # max tree depth (bloat control)
        p_crossover     = 0.6,
        p_mutation      = 0.3,
        tournament_size = 3,
    )

    # Minimisation by default in ExprOptimization, so we negate fitness.
    loss(ind, grammar) = -evaluate(ind, grammar)

    result = GeneticProgram.optimize(
        loss,
        GRAMMAR,
        :Real,                # start symbol
        N_GENERATIONS,
        gp_params;
        verbose = true,       # prints gen-by-gen stats
    )

    best_ind     = result.expr
    best_fitness = -result.loss   # un-negate

    println("\n" * "=" ^ 60)
    println("RESULTS")
    println("=" ^ 60)
    println("Best fitness (Division of Labour): $(round(best_fitness, digits=4))")
    println("\nBest rule expression:")
    println("  ", get_executable(best_ind, GRAMMAR))

    interpret_institution(best_ind, best_fitness)

    # ── Final simulation with best rule ───────────────────────────────────────
    best_fn  = compile_rule(best_ind)
    model    = make_model(fill(best_fn, N_AGENTS); sim_scenario=SCENARIO)
    for _ in 1:50
        step!(model, agent_step!, model_step!)
    end

    println("\nDivision-of-Labour time series (last 10 steps):")
    history = model.dol_history
    for (i, v) in enumerate(history[end-9:end])
        println("  step $(length(history)-10+i):  $(round(v, digits=4))")
    end

    return result, model
end

# ═══════════════════════════════════════════════════════════════════════════════
# INSTITUTIONAL INTERPRETATION
# ═══════════════════════════════════════════════════════════════════════════════

function interpret_institution(ind, fitness)
    rule_str = string(get_executable(ind, GRAMMAR))
    println("\n── Institutional Interpretation ──────────────────────────────")

    if SCENARIO == 1
        println("Scenario 1 (individualist): agents use unilateral GIVE only.")
        println("  → Rules likely converge on simple redistribution heuristics.")
        println("  → Limited institutional complexity expected (paper confirms).")

    elseif SCENARIO == 2
        println("Scenario 2 (exchange): SPOT_EXCHANGE primitives active.")
        has_bilateral = any(k -> occursin(k, rule_str), ["nb_G", "gp_gt", "gp_lt"])
        if has_bilateral
            println("  ✓ Rule references neighbour inventories — bilateral awareness.")
            println("  → Possible emergence of COMMODITY MONEY: one good preferred")
            println("    as a medium because it maximises swap opportunities.")
        else
            println("  → Rule does not strongly reference neighbour inventory.")
            println("    Weak bilateral coordination detected.")
        end

    elseif SCENARIO == 3
        println("Scenario 3 (debt): INTERPERSONAL_DEBT & CLEAR_DEBT active.")
        has_debt = any(k -> occursin(k, rule_str), ["own_debts", "own_ious"])
        if has_debt
            println("  ✓ Rule references debt state — intertemporal awareness.")
            println("  → Possible emergence of CREDIT MONEY: IOUs as deferred")
            println("    claims enabling time-space stretching of production chains.")
        else
            println("  → Rule does not reference debt state directly.")
        end
    end

    if fitness > 0.5
        println("\n  DoL = $(round(fitness,digits=3))  (>0.5) → Substantial division of labour.")
        println("  Emergent rule-set qualifies as a MESO-LEVEL INSTITUTION")
        println("  in the paper's sense: shared rule enabling coordinated surplus")
        println("  distribution across spatially separated agents.")
    elseif fitness > 0.25
        println("\n  DoL = $(round(fitness,digits=3))  (>0.25) → Partial coordination achieved.")
        println("  Proto-institutional behaviour; not yet a stable institution.")
    else
        println("\n  DoL = $(round(fitness,digits=3))  → Low coordination.")
        println("  Consistent with paper's finding that asocial agents fail")
        println("  to generate rich institutions.")
    end
    println("─" ^ 60)
end

# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

result, model = run_igss()
