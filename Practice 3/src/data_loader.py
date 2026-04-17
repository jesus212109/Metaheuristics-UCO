"""
Data loading utilities: reads the three CSVs produced by the data generator
and returns School, Talk, and Researcher lists plus the preprocessing dict.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

from models import School, Talk, Researcher


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------

def load_schools(csv_path: str | Path) -> Dict[str, School]:
    """Load schools CSV and return a dict keyed by school_id."""
    df = pd.read_csv(csv_path)
    schools: Dict[str, School] = {}
    for _, row in df.iterrows():
        s = School(
            school_id=str(row["school"]),
            location=str(row["location"]),
            disadvantaged_area=bool(row["disadvantaged_area"]),
            school_type=str(row["school_type"]),
            first_year=bool(row["first_year"]),
        )
        schools[s.school_id] = s
    return schools


def load_talks(csv_path: str | Path, schools: Dict[str, School]) -> List[Talk]:
    """Load requested talks CSV and return an indexed list of Talk objects."""
    df = pd.read_csv(csv_path)
    talks: List[Talk] = []
    for idx, row in df.iterrows():
        t = Talk(
            talk_id=idx,
            topic=str(row["topic"]),
            level=str(row["talk_level"]),
            school_id=str(row["school"]),
        )
        talks.append(t)
    return talks


def load_researchers(csv_path: str | Path) -> Dict[str, Researcher]:
    """Load proposed talks CSV and return a dict keyed by researcher_id.

    Each researcher may appear multiple times (one row per talk they can give).
    We keep max_talks as the number of rows they appear in.
    """
    df = pd.read_csv(csv_path)
    researchers: Dict[str, Researcher] = {}
    talk_counts: Dict[str, int] = {}

    for _, row in df.iterrows():
        rid = str(row["researcher"])
        talk_counts[rid] = talk_counts.get(rid, 0) + 1
        if rid not in researchers:
            r = Researcher(
                researcher_id=rid,
                topic=str(row["topic"]),
                level=str(row["talk_level"]),
                can_travel=bool(row["travelling"]),
                first_participation=bool(row["first_participation"]),
                previous_talk_province=bool(row["previous_talk_province"]),
                previous_school=str(row["previous_school"]) if "previous_school" in row else None,
                max_talks=1,
            )
            researchers[rid] = r

    # Update max_talks from row counts
    for rid, count in talk_counts.items():
        researchers[rid].max_talks = count

    return researchers


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def build_valid_researchers_per_talk(
    talks: List[Talk],
    researchers: Dict[str, Researcher],
    schools: Dict[str, School],
) -> Dict[int, List[str]]:
    """
    For each talk, compute the list of researcher IDs that *can* cover it.

    Hard-constraint filters applied here:
      1. Topic match: researcher's topic == talk's topic OR talk topic is "any".
      2. Level match: researcher's level == talk's level.
      3. Location feasibility: if the school is in the province, the researcher
         must have can_travel=True.

    Returns:
        dict mapping talk_id -> [researcher_id, ...]
    """
    valid: Dict[int, List[str]] = {}
    for talk in talks:
        school = schools[talk.school_id]
        candidates: List[str] = []
        for r in researchers.values():
            # Hard constraint 1: topic
            if talk.topic != "any" and r.topic != talk.topic:
                continue
            # Hard constraint 2: level
            if r.level != talk.level:
                continue
            # Hard constraint 3: travel
            if school.location == "province" and not r.can_travel:
                continue
            candidates.append(r.researcher_id)
        valid[talk.talk_id] = candidates
    return valid


def load_instance(
    schools_csv: str | Path,
    talks_csv: str | Path,
    researchers_csv: str | Path,
) -> Tuple[Dict[str, School], List[Talk], Dict[str, Researcher], Dict[int, List[str]]]:
    """Convenience wrapper: load all CSVs and return (schools, talks, researchers, valid_map)."""
    schools = load_schools(schools_csv)
    talks = load_talks(talks_csv, schools)
    researchers = load_researchers(researchers_csv)
    valid_map = build_valid_researchers_per_talk(talks, researchers, schools)
    return schools, talks, researchers, valid_map
