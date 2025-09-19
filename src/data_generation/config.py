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
        year = int(rng.normal(1995, 20))
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

class GivingGenerator:
    def __init__(self):
        self.designations = [
            'Annual Fund', 'Scholarships', 'Men\'s Basketball', 'Library',
            'Endowment Fund', 'Capital Campaign', 'Faculty Support',
             'Greek Life', 'Theatre Arts', 'Aquatic Center',
            'DEI Initiatives', 'Women\'s Volleyball', 'Alumni Association', 'Study Abroad'
            'Men\'s Soccer', 'Women\'s Soccer', 'Student Health Center', 'Community Service Center',
            'Finance Club', 'Engineering Department', 'Club Sports', 'Career Development Center',
            'Veterans Center', 'Music Department', 'History Department', 'Debate Team', 
            'Computer Science Department'
        ]
    
    def generate_lifetime_giving(self):
        """Random lifetime giving using log-normal distribution"""
        # 40% chance of being a non-donor
        if random.random() < 0.4:
            return 0.0
        
        # Log-normal distribution parameters
        mu = 8.5  # log mean (roughly $5K median)
        sigma = 2.5  # log standard deviation
        
        lifetime_giving = rng.lognormal(mu, sigma)
        lifetime_giving = max(1, min(100000000, lifetime_giving))  # Cap at $100M
        return round(lifetime_giving, 2)
    
    def generate_last_gift_data(self, lifetime_giving):
        if lifetime_giving == 0:
            return 0, None, None
        
        # Last gift typically 5-50% of lifetime giving
        percentage = random.uniform(0.05, 0.5)
        last_gift = lifetime_giving * percentage
        
        designation = random.choice(self.designations)
        
        # Gift date (weighted toward recent years)
        years = list(range(1980, 2026))
        weights = [1 if year < 2010 else (year - 2005) for year in years]
        gift_year = random.choices(years, weights=weights)[0]
        
        start_date = date(gift_year, 1, 1)
        end_date = date(gift_year, 12, 31)
        days_diff = (end_date - start_date).days
        gift_date = start_date + timedelta(days=random.randint(0, days_diff))
        
        return round(last_gift, 2), designation, gift_date
    
    def generate_giving_patterns(self, lifetime_giving, class_year):
        if lifetime_giving == 0:
            return 0, 0
        
        # Estimate donor age for realistic patterns
        current_year = 2025
        estimated_age = current_year - (class_year or 1980) + 22
        giving_years_possible = max(1, estimated_age - 25)
        
        total_years = random.randint(0, min(50, giving_years_possible))
        consecutive_years = random.randint(0, min(total_years, 25))
        
        return consecutive_years, total_years
    
    def validate_rating_vs_giving_mismatch(self, rating, lifetime_giving):
        """Check for realistic mismatches"""
        if lifetime_giving == 0 and rating in ['A', 'B', 'C', 'D']:
            return "High-rated non-donor"
        elif lifetime_giving > 1000000 and rating in ['M', 'N', 'O', 'P']:
            return "Major donor, low rating"
        else:
            return "Aligned"

# Initialize generator
giving_gen = GivingGenerator()
print("Giving generator initialized")

def generate_family_assignments(num_donors, family_percentage=0.30):
    """Generate family assignments before creating donor records"""
    print(f"Pre-generating family assignments for {family_percentage*100:.0f}% of donors...")
    
    # Determine how many donors will be in families
    family_donor_count = int(num_donors * family_percentage)
    
    # Initialize family tracking
    family_assignments = {}  # donor_index -> {'Family_ID': int, 'Relationship_Type': str}
    family_counter = 10000
    assigned_indices = set()
    
    # Create family clusters
    while len(assigned_indices) < family_donor_count:
        available_count = family_donor_count - len(assigned_indices)
        if available_count <= 0:
            break
            
        # Determine family size (weighted toward smaller families)
        family_size = random.choices([2, 3, 4, 5], weights=[0.5, 0.3, 0.15, 0.05])[0]
        family_size = min(family_size, available_count)
        
        # Select random donor indices for this family
        available_indices = [i for i in range(num_donors) if i not in assigned_indices]
        if len(available_indices) < family_size:
            family_size = len(available_indices)
        
        family_members = random.sample(available_indices, family_size)
        family_id = family_counter
        family_counter += 1
        
        
        for i, donor_index in enumerate(family_members):
            if i == 0:
                rel_type = 'Head'
            elif i == 1 and family_size >= 2:
                rel_type = 'Spouse'
            else:
                rel_type = random.choice(['Child', 'Parent', 'Sibling'])
            
            family_assignments[donor_index] = {
                'Family_ID': family_id,
                'Relationship_Type': rel_type
            }
            
            assigned_indices.add(donor_index)
    
    print(f"Created {len(set(f['Family_ID'] for f in family_assignments.values()))} families")
    print(f"Assigned {len(family_assignments)} donors to families")
    
    return family_assignments

def generate_core_donors_with_families(num_donors, donor_ids):
    """Generate core donor information including family relationships"""
    print(f"Generating {num_donors:,} core donor records with family relationships...")
    
    # Pre-generate family assignments
    family_assignments = generate_family_assignments(num_donors)
    
    donors = []
    relationships = []  # Separate list to track relationships
    
    for i in tqdm(range(num_donors), desc="Creating donors with families"):
        donor_id = donor_ids[i]
        
        # Generate names and demographics
        first_name, last_name, full_name, gender = demo_gen.generate_name()
        demographics = demo_gen.generate_demographics()
        
        # Generate constituent information
        primary_type, secondary_type = const_gen.assign_constituent_types()
        class_year = const_gen.generate_class_year(primary_type)
        
        # Generate advancement data
        manager = const_gen.assign_manager()
        rating = const_gen.assign_rating()
        prospect_stage = const_gen.assign_prospect_stage()
        
        # Generate giving data (random, not rating-based)
        lifetime_giving = giving_gen.generate_lifetime_giving()
        last_gift, designation, gift_date = giving_gen.generate_last_gift_data(lifetime_giving)
        consecutive_years, total_years = giving_gen.generate_giving_patterns(lifetime_giving, class_year)
        
        # Check rating vs giving alignment
        alignment = giving_gen.validate_rating_vs_giving_mismatch(rating, lifetime_giving)
        
        # Get family information if assigned
        family_info = family_assignments.get(i, {})
        family_id = family_info.get('Family_ID', None)
        relationship_type = family_info.get('Relationship_Type', None)
        
        # Determine family giving potential
        if family_id is not None:
            if lifetime_giving > 50000:
                family_giving_potential = 'High'
            elif lifetime_giving > 10000:
                family_giving_potential = 'Medium'
            else:
                family_giving_potential = 'Low'
        else:
            family_giving_potential = 'Individual'
        
        # Create donor record with family fields included
        donor = {
            'ID': donor_id,
            'First_Name': first_name,
            'Last_Name': last_name,
            'Full_Name': full_name,
            'Gender': gender,
            'Primary_Constituent_Type': primary_type,
            'Constituent_Type_2': secondary_type,
            'Class_Year': class_year,
            'Primary_Manager': manager,
            'Rating': rating,
            'Prospect_Stage': prospect_stage,
            'Lifetime_Giving': lifetime_giving,
            'Last_Gift': last_gift,
            'Last_Gift_Designation': designation,
            'Last_Gift_Date': gift_date,
            'Consecutive_Yr_Giving_Count': consecutive_years,
            'Total_Yr_Giving_Count': total_years,
            'Rating_Giving_Alignment': alignment,
            'Family_ID': family_id,
            'Relationship_Type': relationship_type,
            'Family_Giving_Potential': family_giving_potential,
            **demographics
        }
        
        donors.append(donor)
        
        # If donor has family, add to relationships tracking
        if family_id is not None:
            relationships.append({
                'Donor_ID': donor_id,
                'Family_ID': family_id,
                'Relationship_Type': relationship_type
            })
    
    return pd.DataFrame(donors), pd.DataFrame(relationships)

# Generate core donor data with integrated family relationships
print("=" * 50)
print("STEP 3: GENERATING CORE DONOR DATA WITH FAMILY RELATIONSHIPS")
print("=" * 50)

donors_df, relationships_df = generate_core_donors_with_families(TOTAL_DONORS, DONOR_IDS)

print(f"\nGenerated {len(donors_df):,} donor records")
print(f"Non-donors: {(donors_df['Lifetime_Giving'] == 0).sum():,}")
print(f"Donors: {(donors_df['Lifetime_Giving'] > 0).sum():,}")
print(f"Major donors (>$100K): {(donors_df['Lifetime_Giving'] > 100000).sum():,}")
print(f"Donors in families: {donors_df['Family_ID'].notna().sum():,}")
print(f"Number of families: {donors_df['Family_ID'].nunique()}")