import numpy as np

def generate_random_topic():
    """
    Randomly generates a topic for a talk from a predefined list of options.
    
    Returns:
        str: A randomly selected topic.
    """
    topics = [
        "biology", "chemistry", "physics", "computer science", "maths",
        "education", "agronomy", "electronics", "psychology", "engineering",
        "medicine", "archeology", "language", "law", "history", "economics"
    ]
    return np.random.choice(topics)

def generate_random_talk_level():
    """
    Randomly generates the level of a talk from a predefined list of options.
    
    Returns:
        str: A randomly selected talk level.
    """
    levels = ["preschool", "primary", "secondary", "high school", "vocational training"]
    return np.random.choice(levels)