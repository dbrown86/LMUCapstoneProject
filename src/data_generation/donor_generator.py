# Main donor generation functions
import pandas as pd
import numpy as np
import random
from datetime import date, timedelta
from tqdm import tqdm
from .data_generation import DemographicsGenerator, ConstituentGenerator, GivingGenerator, ContactReportGenerator

def generate_random_ids(count, min_id=100000, max_id=999999):
    """Generate unique random IDs"""
    return random.sample(range(min_id, max_id), count)

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

def generate_core_donors_with_families(num_donors, donor_ids, rng):
    """Generate core donor information including family relationships"""
    print(f"Generating {num_donors:,} core donor records with family relationships...")
    
    # Initialize generators
    demo_gen = DemographicsGenerator()
    const_gen = ConstituentGenerator()
    giving_gen = GivingGenerator()
    
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
        class_year = const_gen.generate_class_year(primary_type, rng=rng)
        
        # Generate advancement data
        manager = const_gen.assign_manager()
        rating = const_gen.assign_rating()
        prospect_stage = const_gen.assign_prospect_stage()
        
        # Generate giving data (random, not rating-based)
        lifetime_giving = giving_gen.generate_lifetime_giving(rng)
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

def generate_giving_history(donors_df, rng):
    """Generate detailed giving history for donors"""
    print("Generating detailed giving history...")
    
    giving_gen = GivingGenerator()
    giving_history = []
    gift_id_counter = 1000000
    
    # Only generate history for actual donors
    donor_subset = donors_df[donors_df['Lifetime_Giving'] > 0].copy()
    
    for _, donor in tqdm(donor_subset.iterrows(), total=len(donor_subset), desc="Creating giving history"):
        total_years = donor['Total_Yr_Giving_Count']
        lifetime_giving = donor['Lifetime_Giving']
        
        if total_years == 0:
            continue
        
        # Generate giving years
        current_year = 2025
        class_year = int(donor['Class_Year']) if pd.notna(donor['Class_Year']) else 1980
        earliest_possible = max(1990, class_year + 5)  # Can't give before age ~23
        
        # Select random years for giving
        possible_years = list(range(earliest_possible, current_year + 1))
        giving_years = sorted(random.sample(possible_years, min(int(total_years), len(possible_years))))
        
        # Distribute lifetime giving across years
        remaining_amount = round(float(lifetime_giving), 2)
        
        for i, year in enumerate(giving_years):
            is_last = i == len(giving_years) - 1
            if is_last:
                # Last gift absorbs any rounding remainder to match lifetime total exactly
                gift_amount = round(remaining_amount, 2)
            else:
                # Random percentage of remaining (weighted toward smaller early gifts)
                percentage = random.uniform(0.05, 0.3)
                tentative_amount = round(remaining_amount * percentage, 2)

                # Enforce minimum for this gift and preserve minimums for remaining future gifts
                remaining_gifts = len(giving_years) - i - 1
                min_required_future = 25 * remaining_gifts
                # Maximum we can allocate now while reserving minimums later
                max_allowable_now = max(0.0, round(remaining_amount - min_required_future, 2))
                # Apply per-gift minimum of $25 for intermediate gifts when possible
                desired_amount = max(25.0, tentative_amount)
                if max_allowable_now > 0:
                    gift_amount = min(desired_amount, max_allowable_now)
                else:
                    # Not enough remaining to allocate minimums going forward; allocate as little as possible now
                    gift_amount = min(desired_amount, max(0.0, round(remaining_amount - 25 * (remaining_gifts - 1), 2))) if remaining_gifts > 0 else desired_amount

                gift_amount = round(gift_amount, 2)
                remaining_amount = round(remaining_amount - gift_amount, 2)

            # Generate gift details and clamp dates to today if needed
            gift_month = random.randint(1, 12)
            gift_day = random.randint(1, 28)  # Safe day for all months
            gift_dt = date(year, gift_month, gift_day)
            if gift_dt > date.today():
                gift_dt = date.today()
            
            giving_history.append({
                'Gift_ID': gift_id_counter,
                'Donor_ID': donor['ID'],
                'Gift_Date': gift_dt,
                'Gift_Amount': gift_amount,
                'Designation': random.choice(giving_gen.designations),
                'Gift_Type': random.choice(['Cash', 'Check', 'Credit Card', 'Stock', 'Online']),
                'Campaign_Year': year if random.random() < 0.3 else None,  # 30% are campaign gifts
                'Anonymous': random.random() < 0.05  # 5% anonymous
            })
            
            gift_id_counter += 1
    
    return pd.DataFrame(giving_history)

def generate_enhanced_fields(donors_df):
    """Generate enhanced fields for deep learning applications"""
    print("Generating enhanced fields for deep learning...")
    
    enhanced_data = []
    
    for _, donor in tqdm(donors_df.iterrows(), total=len(donors_df), desc="Adding enhanced fields"):
        # Interest keywords
        interests = [
            'Healthcare', 'Education', 'Arts', 'Athletics', 'Environment',
            'Social Justice', 'Technology', 'Research', 'Student Support',
            'Faculty Development', 'Infrastructure', 'Global Programs'
        ]
        num_interests = random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
        selected_interests = random.sample(interests, num_interests)
        
        # Calculate engagement score
        engagement_score = 20  # Base score
        
        if donor['Lifetime_Giving'] > 0:
            engagement_score += min(30, donor['Lifetime_Giving'] / 10000)
        
        if donor['Consecutive_Yr_Giving_Count'] > 0:
            engagement_score += min(25, donor['Consecutive_Yr_Giving_Count'] * 2)
        
        if donor['Primary_Constituent_Type'] in ['Trustee', 'Regent']:
            engagement_score += 25
        
        # Family bonus for engagement
        if donor['Family_ID'] is not None:
            engagement_score += 10
        
        engagement_score += random.randint(-10, 15)
        engagement_score = max(0, min(100, engagement_score))
        
        # Legacy intent probability
        age_estimate = 2025 - (donor['Class_Year'] or 1980) + 22
        legacy_prob = 0.1  # Base probability
        
        if age_estimate > 65:
            legacy_prob += 0.3
        elif age_estimate > 55:
            legacy_prob += 0.2
        
        if donor['Lifetime_Giving'] > 100000:
            legacy_prob += 0.2
        elif donor['Lifetime_Giving'] > 50000:
            legacy_prob += 0.1
        
        # Family history bonus for legacy intent
        if donor['Family_ID'] is not None and donor['Lifetime_Giving'] > 25000:
            legacy_prob += 0.15
        
        legacy_prob = min(0.95, legacy_prob)
        
        # Board affiliations
        board_affiliations = None
        if donor['Rating'] in ['A', 'B', 'C', 'D'] and random.random() < 0.4:
            boards = ['Hospital Board', 'Museum Board', 'Foundation Board', 'Corporate Board', 'Nonprofit Board']
            num_boards = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
            board_affiliations = ', '.join(random.sample(boards, num_boards))
        
        enhanced_data.append({
            'Donor_ID': donor['ID'],
            'Interest_Keywords': ', '.join(selected_interests),
            'Engagement_Score': round(engagement_score, 1),
            'Legacy_Intent_Probability': round(legacy_prob, 3),
            'Legacy_Intent_Binary': random.random() < legacy_prob,
            'Board_Affiliations': board_affiliations,
            'Estimated_Age': age_estimate
        })
    
    return pd.DataFrame(enhanced_data)

def generate_contact_reports(donors_df):
    """Generate contact reports for eligible donors"""
    print("Generating contact reports...")
    
    contact_gen = ContactReportGenerator()
    const_gen = ConstituentGenerator()
    contact_reports = []
    report_id_counter = 500000
    
    for _, donor in tqdm(donors_df.iterrows(), total=len(donors_df), desc="Creating contact reports"):
        # Determine if donor should have contact report
        probability = contact_gen.should_have_contact_report(
            donor['Primary_Constituent_Type'],
            donor['Prospect_Stage'],
            donor['Lifetime_Giving']
        )
        
        if random.random() < probability:
            # Generate contact report
            report_text = contact_gen.generate_report(
                donor['Full_Name'],
                donor['Prospect_Stage'],
                donor['Rating'],
                donor['Lifetime_Giving'],
                donor['Primary_Constituent_Type']
            )
            
            contact_date = contact_gen.generate_contact_date()
            author = const_gen.assign_manager()
            
            contact_reports.append({
                'Contact_Report_ID': report_id_counter,
                'Donor_ID': donor['ID'],
                'Contact_Date': contact_date,
                'Contact_Type': random.choice(['Meeting', 'Phone Call', 'Email', 'Event']),
                'Author': author,
                'Report_Text': report_text,
                'Outcome_Category': 'Positive' if any(pos in report_text.lower() for pos in ['interest', 'support', 'commit']) 
                                 else 'Unresponsive' if 'unresponsive' in report_text.lower() or 'not return' in report_text.lower()
                                 else 'Negative'
            })
            
            report_id_counter += 1
    
    return pd.DataFrame(contact_reports)
