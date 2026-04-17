"""
CHC Metaheuristic for the Talk Allocation Problem
===================================================
CHC: Cross-generational elitist selection, Heterogeneous recombination (HUX),
     Cataclysmic mutation restart.

Reference: Eshelman (1991) "The CHC Adaptive Search Algorithm"
"""

import random
import math
import copy
from typing import Dict, List, Optional, Tuple

from models import School, Talk, Researcher
from fitness import compute_fitness, repair_gene, DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Chromosome = List[int]   # length T; gene i in {-1, 0 .. R-1}


# ---------------------------------------------------------------------------
# Population initialisation
# ---------------------------------------------------------------------------

def _random_chromosome(
    talks: List[Talk],
    valid_map: Dict[int, List[str]],
    r_id_list: List[str],
) -> Chromosome:
    """Generate one random feasible chromosome, respecting valid_map."""
    chrom: Chromosome = []
    usage: Dict[str, int] = {}
    for talk in talks:
        candidates = valid_map.get(talk.talk_id, [])
        if not candidates:
            chrom.append(-1)
            continue
        random.shuffle(candidates)
        chosen = -1
        for r_str in candidates:
            if usage.get(r_str, 0) < 2:  # respect max_talks <= 2
                chosen = r_id_list.index(r_str)
                usage[r_str] = usage.get(r_str, 0) + 1
                break
        if chosen == -1:
            # Fallback: pick any valid even if over-limit (fitness will penalise)
            r_str = random.choice(candidates)
            chosen = r_id_list.index(r_str)
        chrom.append(chosen)
    return chrom


def hamming_distance(a: Chromosome, b: Chromosome) -> int:
    """Count positions where two chromosomes differ."""
    return sum(x != y for x, y in zip(a, b))


def initialise_population(
    pop_size: int,
    talks: List[Talk],
    valid_map: Dict[int, List[str]],
    r_id_list: List[str],
) -> List[Chromosome]:
    """
    Create an initial population of pop_size diverse chromosomes.
    Attempts basic diversity enforcement but does not block completeness.
    """
    population: List[Chromosome] = []
    attempts = 0
    max_attempts = pop_size * 20
    min_hamming = max(1, len(talks) // 10)  # at least 10% different genes

    while len(population) < pop_size and attempts < max_attempts:
        candidate = _random_chromosome(talks, valid_map, r_id_list)
        too_similar = any(hamming_distance(candidate, existing) < min_hamming
                          for existing in population)
        if not too_similar:
            population.append(candidate)
        attempts += 1

    # Fill remainder without diversity constraint if needed
    while len(population) < pop_size:
        population.append(_random_chromosome(talks, valid_map, r_id_list))

    return population


# ---------------------------------------------------------------------------
# HUX Crossover
# ---------------------------------------------------------------------------

def hux_crossover(parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
    """
    Half-Uniform Crossover (HUX):
    Identify all differing positions; swap exactly half of them (random selection).
    Returns two offspring.
    """
    diff_positions = [i for i, (a, b) in enumerate(zip(parent1, parent2)) if a != b]

    if len(diff_positions) < 2:
        return copy.copy(parent1), copy.copy(parent2)

    half = len(diff_positions) // 2
    swap_positions = set(random.sample(diff_positions, half))

    child1 = copy.copy(parent1)
    child2 = copy.copy(parent2)
    for pos in swap_positions:
        child1[pos], child2[pos] = child2[pos], child1[pos]

    return child1, child2


# ---------------------------------------------------------------------------
# Cataclysmic Mutation (Restart)
# ---------------------------------------------------------------------------

def cataclysmic_mutation(
    best: Chromosome,
    pop_size: int,
    mutation_rate: float,
    talks: List[Talk],
    valid_map: Dict[int, List[str]],
    r_id_list: List[str],
) -> List[Chromosome]:
    """
    Restart: keep `best` and fill the rest by mutating `mutation_rate` fraction of genes.
    Each mutated gene is replaced by a new valid researcher for that talk.
    """
    new_population: List[Chromosome] = [copy.copy(best)]
    T = len(talks)
    n_mutate = max(1, int(mutation_rate * T))

    for _ in range(pop_size - 1):
        mutant = copy.copy(best)
        positions = random.sample(range(T), n_mutate)
        for pos in positions:
            mutant[pos] = repair_gene(pos, valid_map, r_id_list)
        new_population.append(mutant)

    return new_population


# ---------------------------------------------------------------------------
# CHC Main Loop
# ---------------------------------------------------------------------------

def chc(
    talks: List[Talk],
    schools: Dict[str, School],
    researchers: Dict[str, Researcher],
    valid_map: Dict[int, List[str]],
    pop_size: int = 50,
    max_generations: int = 200,
    mutation_rate: float = 0.35,
    config: Optional[Dict] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[Chromosome, float, List[float]]:
    """
    Run the CHC algorithm for talk allocation.

    Args:
        talks:          ordered list of Talk objects (length T).
        schools:        dict school_id -> School.
        researchers:    dict researcher_id -> Researcher.
        valid_map:      talk_id -> [valid_researcher_id_strings].
        pop_size:       population size (even number recommended).
        max_generations: stopping criterion.
        mutation_rate:  fraction of genes mutated during cataclysmic restart (default 35%).
        config:         fitness weight config dict.
        seed:           random seed for reproducibility.
        verbose:        print progress.

    Returns:
        (best_chromosome, best_fitness, convergence_curve)
    """
    if seed is not None:
        random.seed(seed)

    if config is None:
        config = DEFAULT_CONFIG

    r_id_list = list(researchers.keys())
    T = len(talks)

    # Initial Hamming threshold:  d0 = T / 4  (standard CHC initialisation)
    threshold = T // 4

    # Initialise population
    population = initialise_population(pop_size, talks, valid_map, r_id_list)

    # Evaluate
    def evaluate(chrom: Chromosome) -> float:
        return compute_fitness(chrom, talks, schools, researchers, valid_map, config)

    fitness_values = [evaluate(c) for c in population]

    # Track best
    best_idx = min(range(pop_size), key=lambda i: fitness_values[i])
    best_chrom = copy.copy(population[best_idx])
    best_fitness = fitness_values[best_idx]

    convergence: List[float] = [best_fitness]
    restarts = 0

    for gen in range(max_generations):
        # Shuffle population for pairing
        indices = list(range(pop_size))
        random.shuffle(indices)
        pairs = [(indices[i], indices[i + 1]) for i in range(0, pop_size - 1, 2)]

        new_individuals: List[Tuple[Chromosome, float]] = []

        for i1, i2 in pairs:
            p1, p2 = population[i1], population[i2]
            d = hamming_distance(p1, p2)

            # Incest prevention: only mate if Hamming distance > threshold
            if d > threshold:
                c1, c2 = hux_crossover(p1, p2)
                f1, f2 = evaluate(c1), evaluate(c2)

                # Child survives only if strictly better than both parents
                if f1 < fitness_values[i1] or f1 < fitness_values[i2]:
                    new_individuals.append((c1, f1))
                if f2 < fitness_values[i1] or f2 < fitness_values[i2]:
                    new_individuals.append((c2, f2))

        if new_individuals:
            # Elitist replacement: combine old + new, keep top pop_size
            combined = list(zip(population, fitness_values)) + new_individuals
            combined.sort(key=lambda x: x[1])
            population = [c for c, _ in combined[:pop_size]]
            fitness_values = [f for _, f in combined[:pop_size]]
            threshold = T // 4  # reset threshold on improvement (optional; keeps search active)
        else:
            # No children improved parents: decrease threshold
            threshold -= 1
            if verbose:
                print(f"  Gen {gen:4d} | threshold={threshold} | best_fit={best_fitness:.2f}")

        # Update global best
        gen_best_idx = min(range(len(fitness_values)), key=lambda i: fitness_values[i])
        if fitness_values[gen_best_idx] < best_fitness:
            best_fitness = fitness_values[gen_best_idx]
            best_chrom = copy.copy(population[gen_best_idx])

        convergence.append(best_fitness)

        # Cataclysmic restart when threshold <= 0
        if threshold <= 0:
            restarts += 1
            if verbose:
                print(f"  [RESTART #{restarts}] Gen {gen} | best_fit={best_fitness:.2f}")
            population = cataclysmic_mutation(
                best_chrom, pop_size, mutation_rate, talks, valid_map, r_id_list
            )
            fitness_values = [evaluate(c) for c in population]
            # Ensure best is in population after restart
            fitness_values[0] = best_fitness
            population[0] = copy.copy(best_chrom)
            threshold = T // 4  # reset threshold

        if verbose and gen % 20 == 0:
            print(f"  Gen {gen:4d} | threshold={threshold} | best_fit={best_fitness:.2f} | restarts={restarts}")

    return best_chrom, best_fitness, convergence
