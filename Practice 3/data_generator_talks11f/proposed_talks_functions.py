## Functions to generate random proposed talks
## For use in the Metaheuristics course at the University of Cordoba, Spain
## Author: Aurora Ramirez
## HUMAINS Research Group

import numpy as np

def generate_random_repeat_talk():
    """
    Randomly generates whether a talk can be repeated or not.
    
    Returns:
        bool: True if the talk can be repeated, False otherwise.
    """
    return np.random.choice([True, False])

def generate_random_travelling():
    """
    Randomly generates whether a researcher can travel (go outside the city) or not.
    
    Returns:
        bool: True if the researcher can travel, False otherwise.
    """
    return np.random.choice([True, False])

def generate_random_first_participation():
    """
    Randomly generates whether a researcher is participating for the first time or not.
    
    Returns:
        bool: True if it's the researcher's first participation, False otherwise.
    """
    return np.random.choice([True, False])

def generate_random_previous_talk_province():
    """
    Randomly generates the province of a previous talk from a predefined list of options.
    
    Returns:
        bool: True if the researcher gave a previous talk somewhere in the province, False otherwise.
    """
    return np.random.choice([True, False])

def generate_random_previous_school(num_schools: int):
    """
    Randomly generates the school of a previous talk from a predefined list of options.
    
    Args:
        num_schools (int): The number of schools to choose from.
    
    Returns:
        int: A randomly selected school for a previous talk.
    """
    return np.random.choice(list(range(num_schools)))
