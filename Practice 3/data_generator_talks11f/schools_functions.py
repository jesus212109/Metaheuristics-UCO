## Functions to generate random schools
## For use in the Metaheuristics course at the University of Cordoba, Spain
## Author: Aurora Ramirez
## HUMAINS Research Group

import numpy as np

def generate_random_school():
    """
    Generates random characteristics for a school.

    Returns:
        dict: A dictionary containing:
            - location: 'city' or 'province'
            - disadvantaged_area: True or False
            - school_type: 'private', 'public', or 'concerted'
            - first_year: True or False
    """
    location = np.random.choice(["city", "province"])
    school_type = np.random.choice(["private", "public", "concerted"])
    if school_type == "public":
        disadvantaged_area = np.random.choice([True, False])
    else:
        disadvantaged_area = False

    first_year = np.random.choice([True, False])

    return {
        "location": location,
        "disadvantaged_area": disadvantaged_area,
        "school_type": school_type,
        "first_year": first_year
    }