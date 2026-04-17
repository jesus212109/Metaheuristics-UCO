"""
Main entry point for Practice 3: Talk Allocation via CHC Metaheuristic.

Usage:
    python main.py --schools  <schools.csv>
                  --talks    <requested_talks.csv>
                  --researchers <proposed_talks.csv>
                  [--pop-size 50] [--generations 200] [--seed 42]
                  [--verbose]

If no CSVs are provided, a synthetic instance is generated and saved to ./data/.
"""

import argparse
import random
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np

# Ensure src/ is on the path when running from the project root
sys.path.insert(0, str(Path(__file__).parent))

from models import School, Talk, Researcher
from data_loader import load_instance, build_valid_researchers_per_talk
from fitness import compute_fitness, DEFAULT_CONFIG
from chc import chc


# ---------------------------------------------------------------------------
# Synthetic data generator (for quick testing without CSV files)
# ---------------------------------------------------------------------------

def _generate_synthetic_instance(
    num_schools: int = 10,
    num_talks: int = 20,
    num_researchers: int = 15,
    seed: int = 0,
) -> tuple:
    """Generate a small synthetic instance and return (schools, talks, researchers)."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    topics = ["biology", "chemistry", "physics", "computer science", "maths",
              "education", "agronomy", "electronics", "psychology", "engineering"]
    levels = ["preschool", "primary", "secondary", "high school", "vocational training"]
    locations = ["city", "province"]
    school_types = ["public", "concerted", "private"]

    schools: Dict[str, School] = {}
    for i in range(num_schools):
        sid = f"school{i+1}"
        loc = rng.choice(locations)
        stype = rng.choice(school_types)
        dis = rng.random() < 0.3 if stype == "public" else False
        schools[sid] = School(
            school_id=sid,
            location=loc,
            disadvantaged_area=dis,
            school_type=stype,
            first_year=rng.random() < 0.4,
        )

    school_ids = list(schools.keys())
    talks: List[Talk] = []
    for i in range(num_talks):
        p_any = 0.3  # 30% chance of "any" topic
        top = "any" if rng.random() < p_any else rng.choice(topics)
        talks.append(Talk(
            talk_id=i,
            topic=top,
            level=rng.choice(levels),
            school_id=rng.choice(school_ids),
        ))

    researchers: Dict[str, Researcher] = {}
    for i in range(num_researchers):
        rid = f"researcher{i+1}"
        prev_prov = rng.random() < 0.4
        prev_school_options = [s for s, sc in schools.items()
                                if (sc.location == "province") == prev_prov]
        prev_school = rng.choice(prev_school_options) if prev_school_options else rng.choice(school_ids)
        researchers[rid] = Researcher(
            researcher_id=rid,
            topic=rng.choice(topics),
            level=rng.choice(levels),
            can_travel=rng.random() < 0.6,
            first_participation=rng.random() < 0.3,
            previous_talk_province=prev_prov,
            previous_school=prev_school,
            max_talks=2 if rng.random() < 0.4 else 1,
        )

    return schools, talks, researchers


# ---------------------------------------------------------------------------
# Result display utilities
# ---------------------------------------------------------------------------

def _decode_chromosome(chrom, talks, researchers, schools):
    """Return a human-readable summary of the solution."""
    r_id_list = list(researchers.keys())
    rows = []
    for talk_id, idx in enumerate(chrom):
        talk = talks[talk_id]
        school = schools[talk.school_id]
        r_str = r_id_list[idx] if 0 <= idx < len(r_id_list) else "UNASSIGNED"
        r = researchers.get(r_str)
        rows.append({
            "talk_id": talk_id,
            "school": talk.school_id,
            "location": school.location,
            "topic": talk.topic,
            "level": talk.level,
            "researcher": r_str,
            "r_topic": r.topic if r else "-",
            "r_level": r.level if r else "-",
        })
    return rows


def _print_solution_summary(rows, best_fitness, elapsed):
    """Print a formatted solution table."""
    print("\n" + "=" * 75)
    print("  FINAL SOLUTION")
    print("=" * 75)
    hdr = f"{'Talk':>5} | {'School':<12} | {'Location':<10} | {'Topic':<18} | {'Researcher':<14} | {'Level':<18}"
    print(hdr)
    print("-" * 75)
    for r in rows:
        print(f"{r['talk_id']:>5} | {r['school']:<12} | {r['location']:<10} | {r['topic']:<18} | {r['researcher']:<14} | {r['level']:<18}")
    print("=" * 75)
    print(f"  Best Fitness (penalty): {best_fitness:.2f}")
    print(f"  Elapsed time:           {elapsed:.2f}s")
    print("=" * 75)


def _print_school_coverage(rows, schools):
    """Print which schools are covered and which are not."""
    covered = {r["school"] for r in rows if r["researcher"] != "UNASSIGNED"}
    print("\n  SCHOOL COVERAGE:")
    for sid, school in schools.items():
        status = "✓ ASSIGNED" if sid in covered else "✗ UNSERVED"
        print(f"    {sid:<12} | {school.location:<10} | {school.school_type:<10} | {status}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="CHC Talk Allocation – Practice 3")
    parser.add_argument("--schools",      type=str, default=None)
    parser.add_argument("--talks",        type=str, default=None)
    parser.add_argument("--researchers",  type=str, default=None)
    parser.add_argument("--pop-size",     type=int, default=50)
    parser.add_argument("--generations",  type=int, default=200)
    parser.add_argument("--mutation-rate",type=float, default=0.35)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--verbose",      action="store_true")
    # Synthetic instance params
    parser.add_argument("--num-schools",  type=int, default=10)
    parser.add_argument("--num-talks",    type=int, default=20)
    parser.add_argument("--num-researchers", type=int, default=15)
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n========= CHC TALK ALLOCATION – Practice 3 =========")

    # --- Load or generate data ---
    if args.schools and args.talks and args.researchers:
        print(f"  Loading instance from CSVs…")
        schools, talks, researchers, valid_map = load_instance(
            args.schools, args.talks, args.researchers
        )
    else:
        print(f"  No CSVs provided – generating synthetic instance "
              f"({args.num_schools} schools, {args.num_talks} talks, "
              f"{args.num_researchers} researchers)…")
        schools, talks, researchers = _generate_synthetic_instance(
            args.num_schools, args.num_talks, args.num_researchers, seed=args.seed
        )
        valid_map = build_valid_researchers_per_talk(talks, researchers, schools)

    T = len(talks)
    R = len(researchers)
    print(f"  Instance: T={T} talks | E={len(schools)} schools | R={R} researchers")

    # Summarise valid researchers per talk
    coverages = [len(v) for v in valid_map.values()]
    n_infeasible = sum(1 for c in coverages if c == 0)
    print(f"  Preprocessing: avg valid researchers/talk = {sum(coverages)/max(len(coverages),1):.1f} "
          f"| infeasible talks (no valid researcher) = {n_infeasible}")

    # --- Run CHC ---
    print(f"\n  Running CHC (pop={args.pop_size}, gen={args.generations}, "
          f"mutation_rate={args.mutation_rate}, seed={args.seed})…\n")
    t0 = time.time()
    best_chrom, best_fitness, convergence = chc(
        talks=talks,
        schools=schools,
        researchers=researchers,
        valid_map=valid_map,
        pop_size=args.pop_size,
        max_generations=args.generations,
        mutation_rate=args.mutation_rate,
        config=DEFAULT_CONFIG,
        seed=args.seed,
        verbose=args.verbose,
    )
    elapsed = time.time() - t0

    # --- Display results ---
    decoded = _decode_chromosome(best_chrom, talks, researchers, schools)
    _print_solution_summary(decoded, best_fitness, elapsed)
    _print_school_coverage(decoded, schools)

    # Save convergence data
    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    conv_path = out_dir / "convergence.json"
    with open(conv_path, "w") as f:
        json.dump({"convergence": convergence, "best_fitness": best_fitness}, f)
    print(f"\n  Convergence curve saved to {conv_path}")


if __name__ == "__main__":
    main()
