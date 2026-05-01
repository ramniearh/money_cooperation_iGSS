INITIALISE
  └── create 40 random trees (half-and-half, depth 1–3)
  └── evaluate all 40

FOR gen = 1 → 50:

  SELECT
  └── tournament selection (k=3) → 40 offspring candidates

  CLONE
  └── deep-copy all selected individuals

  CROSSOVER  (p = 0.5 per pair)
  └── one-point crossover: swap a random subtree between two parents
  └── invalidate fitness of modified children

  MUTATE  (p = 0.2 per individual)
  └── replace a random subtree with a newly generated one
  └── invalidate fitness of mutated individual

  RE-EVALUATE
  └── only individuals with invalidated fitness are re-evaluated

  REPLACE
  └── pop[:] = offspring  ← full generational replacement, NO elitism

  LOG
  └── max_fitness, avg_fitness recorded every generation
  └── fossil_record: best individual saved every 10 generations (read-only)

RETURN
  └── best individual from generation 50 (not global best)
  └── history dict
  └── final population
