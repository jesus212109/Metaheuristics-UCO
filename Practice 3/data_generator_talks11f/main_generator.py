## Main data generator script for the Optimal11F problem
## For use in the Metaheuristics course at the University of Cordoba, Spain
## Author: Aurora Ramirez
## HUMAINS Research Group

import os
import pandas as pd
import numpy as np
from schools_functions import generate_random_school
from talks_functions import generate_random_topic, generate_random_talk_level
from proposed_talks_functions import (
    generate_random_repeat_talk,
    generate_random_travelling,
    generate_random_first_participation,
    generate_random_previous_talk_province
)

# Parameters
num_schools = 10
num_talks = 20
num_researchers = 8
max_talks_researchers = 2

# Probability of requesting a specific topic (e.g., "Mathematics") vs. any topic
# Higher values means more probability of requesting a specific topic (harder to satisfy)
prob_topics = 0.2

data_dir = "data/"
instance_name = f"instance_{num_schools}s_{num_talks}t_{num_researchers}r"

# Create random schools
schools_df = pd.DataFrame(columns=["school", "location", "disadvantaged_area", "school_type", "first_year"])
for i in range(num_schools): 
    school_data = generate_random_school()
    school_data["school"] = f"school{i+1}"
    schools_df = pd.concat([schools_df, pd.DataFrame([school_data])], ignore_index=True)

# Create data frame for requested talks
requested_talks_df = pd.DataFrame(
    columns=["topic", "talk_level", "school"]
)

# Fill the resquested_talks data frame with random values
n = 0
while (n < num_talks):
    random_school_index = np.random.randint(1, num_schools+1)
    new_talk = {
        "topic": generate_random_topic() if np.random.rand() < prob_topics else "any",
        "talk_level": generate_random_talk_level(),
        "school": f"school{random_school_index}"
    }
    # Check if the new talk is already in the data frame to avoid duplicates
    if not ((requested_talks_df["topic"] == new_talk["topic"]) & 
            (requested_talks_df["talk_level"] == new_talk["talk_level"]) & 
            (requested_talks_df["school"] == new_talk["school"])).any():
        requested_talks_df = pd.concat([requested_talks_df, pd.DataFrame([new_talk])], ignore_index=True)
        n += 1

# Create data frame for proposed talks
proposed_talks_df = pd.DataFrame(
    columns=["topic", "talk_level", "researcher", "travelling", "first_participation", "previous_talk_province", "previous_school"]
)

for i in range(num_researchers):
    repeat_talk = generate_random_repeat_talk()
    can_travel = generate_random_travelling()
    first_participation = generate_random_first_participation()
    previous_talk_province = generate_random_previous_talk_province()
    topic = generate_random_topic()
    
    if previous_talk_province:
        schools_province = schools_df[schools_df["location"] == "province"]["school"].tolist()
        if schools_province:
            previous_school = np.random.choice(schools_province)
        else:
            previous_talk_province = False
    
    if not previous_talk_province:
        schools_city = schools_df[schools_df["location"] == "city"]["school"].tolist()
        previous_school = np.random.choice(schools_city)

    if repeat_talk:
        num_talks_researcher = max_talks_researchers
    else:
        num_talks_researcher = 1
    
    for j in range(0, num_talks_researcher):
        new_proposed_talk = {
            "topic": topic,
            "talk_level": generate_random_talk_level(),
            "researcher": f"researcher{i+1}",
            "travelling": can_travel,
            "first_participation": first_participation,
            "previous_talk_province": previous_talk_province,
            "previous_school": previous_school
        }
        proposed_talks_df = pd.concat([proposed_talks_df, pd.DataFrame([new_proposed_talk])], ignore_index=True)


#save data frames to csv files
data_dir = os.path.join(data_dir, instance_name)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
schools_df.to_csv(os.path.join(data_dir, instance_name + "_schools.csv"), index=False)   
requested_talks_df.to_csv(os.path.join(data_dir, instance_name + "_requested_talks.csv"), index=False)
proposed_talks_df.to_csv(os.path.join(data_dir, instance_name + "_proposed_talks.csv"), index=False)