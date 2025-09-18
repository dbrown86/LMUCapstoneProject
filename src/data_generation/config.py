# Configuration file for data generation module
import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import date, datetime, timedelta
import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')

TOTAL_DONORS = 50000
OUTPUT_DIR = "synthetic_donor_dataset"
START_DATE = date(1940, 1, 1)
END_DATE = date(2025, 12, 31)

random.seed(42)
rng = np.random.default_rng(42)
fake = Faker()
Faker.seed(42)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generate unique random ID pool
def generate_random_ids(count, min_id=100000, max_id=999999):
    """Generate unique random IDs"""
    return random.sample(range(min_id, max_id), count)

# Generate all donor IDs upfront
DONOR_IDS = generate_random_ids(TOTAL_DONORS)

print(f"Configuration set for {TOTAL_DONORS:,} donors")
print(f"ID range: {min(DONOR_IDS)} to {max(DONOR_IDS)}")
print(f"Output directory: {OUTPUT_DIR}")


#Source: https://www.census.gov/topics/population/genealogy/data/2010_surnames.html
#  Top 50 most common surnames = ~35% of population
#  Maintains realistic ethnic diversity
#  Balances accuracy vs. computational efficiency

class DemographicsGenerator:
    def __init__(self):
        # US Census-weighted surnames (simplified version)
        self.surnames = {
            'Smith': 0.024, 'Johnson': 0.019, 'Williams': 0.016, 'Brown': 0.014,
            'Jones': 0.014, 'Garcia': 0.011, 'Miller': 0.011, 'Davis': 0.010,
            'Rodriguez': 0.010, 'Martinez': 0.009, 'Hernandez': 0.008, 'Lopez': 0.008,
            'Gonzalez': 0.008, 'Wilson': 0.007, 'Anderson': 0.007, 'Thomas': 0.007,
            'Taylor': 0.007, 'Moore': 0.006, 'Jackson': 0.006, 'Martin': 0.006,
            'Lee': 0.006, 'Perez': 0.005, 'Thompson': 0.005, 'White': 0.005,
            'Harris': 0.005, 'Sanchez': 0.005, 'Clark': 0.004, 'Ramirez': 0.004,
            'Lewis': 0.004, 'Robinson': 0.004, 'Walker': 0.004, 'Young': 0.004,
            'Allen': 0.004, 'King': 0.004, 'Wright': 0.004, 'Scott': 0.004,
            'Torres': 0.004, 'Nguyen': 0.004, 'Hill': 0.004, 'Flores': 0.003,
            'Green': 0.003, 'Adams': 0.003, 'Nelson': 0.003, 'Baker': 0.003,
            'Hall': 0.003, 'Rivera': 0.003, 'Campbell': 0.003, 'Mitchell': 0.003,
            'Carter': 0.003, 'Roberts': 0.003, 'Gomez': 0.003, 'Phillips': 0.003
        }
        
        self.first_names_male = [
            'James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard',
            'Joseph', 'Thomas', 'Christopher', 'Charles', 'Daniel', 'Matthew',
            'Anthony', 'Mark', 'Donald', 'Steven', 'Paul', 'Andrew', 'Joshua',
            'Kenneth', 'Kevin', 'Brian', 'George', 'Timothy', 'Ronald', 'Jason',
            'Edward', 'Jeffrey', 'Ryan', 'Jacob', 'Gary', 'Nicholas', 'Eric',
            'Jonathan', 'Stephen', 'Larry', 'Justin', 'Scott', 'Brandon',
            'Benjamin', 'Samuel', 'Frank', 'Gregory', 'Raymond', 'Alexander',
            'Patrick', 'Jack', 'Dennis', 'Jerry'
        ]
        
        self.first_names_female = [
            'Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara',
            'Susan', 'Jessica', 'Sarah', 'Karen', 'Nancy', 'Lisa', 'Betty',
            'Helen', 'Sandra', 'Donna', 'Carol', 'Ruth', 'Sharon', 'Michelle',
            'Laura', 'Sarah', 'Kimberly', 'Deborah', 'Dorothy', 'Amy',
            'Angela', 'Ashley', 'Brenda', 'Emma', 'Olivia', 'Cynthia',
            'Marie', 'Janet', 'Catherine', 'Frances', 'Christine', 'Samantha',
            'Debra', 'Rachel', 'Carolyn', 'Janet', 'Virginia', 'Maria',
            'Heather', 'Diane', 'Julie', 'Joyce', 'Victoria', 'Kelly'
        ]
        
        self.regions = [
            'Northeast', 'Southeast', 'Midwest', 'Southwest', 
            'West Coast', 'Mountain West', 'International'
        ]
        
        self.industries = [
            'Technology', 'Healthcare', 'Finance', 'Education', 'Legal',
            'Real Estate', 'Manufacturing', 'Retail', 'Consulting',
            'Non-Profit', 'Government', 'Energy', 'Media', 'Retired'
        ]
    
    def generate_name(self):
        gender = random.choice(['M', 'F'])
        first_name = random.choice(self.first_names_male if gender == 'M' else self.first_names_female)
        last_name = random.choices(list(self.surnames.keys()), 
                                  weights=list(self.surnames.values()))[0]
        full_name = f"{first_name} {last_name}"
        return first_name, last_name, full_name, gender
    
    def generate_demographics(self):
        return {
            'Geographic_Region': random.choice(self.regions),
            'Professional_Background': random.choice(self.industries)
        }

# Initialize generator
demo_gen = DemographicsGenerator()
print("Demographics generator initialized")

class ConstituentGenerator:
    def __init__(self):
        self.primary_weights = {
            'Alum': 0.55,
            'Parent': 0.20,
            'Friend': 0.15,
            'Foundation': 0.04,
            'Corporation': 0.03,
            'Trust': 0.01,
            'Trustee': 0.015,
            'Regent': 0.005
        }
        
        self.secondary_roles = ['Alum', 'Parent', 'Friend', None]
        
        self.advancement_staff = [
            'Sarah Chen', 'Michael Rodriguez', 'Jennifer Liu', 'David Park',
            'Lisa Thompson', 'James Wilson', 'Maria Gonzalez', 'Robert Kim',
            'Amanda Davis', 'Christopher Lee', 'Nicole Brown', 'Kevin Wu',
            'Rachel Johnson', 'Mark Williams', 'Emily Zhang'
        ]
        
        self.rating_distribution = {
            'A': 0.001, 'B': 0.002, 'C': 0.005, 'D': 0.008,
            'E': 0.012, 'F': 0.025, 'G': 0.040, 'H': 0.060,
            'I': 0.087, 'J': 0.110, 'K': 0.130, 'L': 0.150,
            'M': 0.140, 'N': 0.120, 'O': 0.080, 'P': 0.030
        }
        
        self.stage_distribution = {
            'Identification': 0.45,
            'Qualification': 0.25,
            'Cultivation': 0.18,
            'Solicitation': 0.08,
            'Stewardship': 0.04
        }
    
    def assign_constituent_types(self):
        primary = random.choices(list(self.primary_weights.keys()), 
                               weights=list(self.primary_weights.values()))[0]
        # 40% chance of having secondary role
        secondary = random.choice(self.secondary_roles) if random.random() < 0.4 else None
        return primary, secondary
    
    def generate_class_year(self, constituent_type, current_year=2025):
        if constituent_type not in ['Alum', 'Trustee', 'Regent']:
            return None
        # Normal distribution centered around 1995, std dev 20
        year = int(np.random.normal(1995, 20))
        return max(1950, min(current_year, year))
    
    def assign_manager(self):
        return random.choice(self.advancement_staff)
    
    def assign_rating(self):
        return random.choices(list(self.rating_distribution.keys()), 
                             weights=list(self.rating_distribution.values()))[0]
    
    def assign_prospect_stage(self):
        return random.choices(list(self.stage_distribution.keys()), 
                             weights=list(self.stage_distribution.values()))[0]

# Initialize generator
const_gen = ConstituentGenerator()
print("Constituent generator initialized")