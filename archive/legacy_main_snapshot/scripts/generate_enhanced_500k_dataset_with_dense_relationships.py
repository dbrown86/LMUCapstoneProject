#!/usr/bin/env python3
"""
Generate Enhanced 500K Synthetic Donor Dataset with Dense Relationships
Includes: Class Year, Parent Year, Event Name, Event Attendance, Dense Relationship Networks
"""

print("🔄 Script starting - loading modules...", flush=True)

import sys
import os

# Add src directory to path (use absolute path)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

print("📦 Loading pandas, numpy, and other libraries...", flush=True)
import pandas as pd
import numpy as np
import random
from datetime import date, timedelta
from tqdm import tqdm
import json
from pathlib import Path

print("📦 Loading data generation modules...", flush=True)
from data_generation.data_generation import DemographicsGenerator, ConstituentGenerator, GivingGenerator, ContactReportGenerator
from data_generation.donor_generator import generate_random_ids, generate_family_assignments

print("✅ All imports complete! Starting dataset generation...", flush=True)

class EnhancedConstituentGenerator(ConstituentGenerator):
    """Enhanced constituent generator with class year and parent year logic"""
    
    def __init__(self):
        super().__init__()
        # Event names for event attendance
        self.event_names = [
            'Alumni Reunion', 'Alumni Baseball Game', 'Alumni Happy Hour Mixer', 
            'New Parent BBQ', 'Family Weekend Mixer', 'Speaker Spotlight', 
            'Faculty Meet and Greet', 'Holiday Mixer', 'Campus Tour'
        ]
        
        # Event type probabilities
        self.event_type_probs = {
            'Alumni Reunion': 0.25,
            'Alumni Baseball Game': 0.15,
            'Alumni Happy Hour Mixer': 0.20,
            'New Parent BBQ': 0.10,
            'Family Weekend Mixer': 0.15,
            'Speaker Spotlight': 0.05,
            'Faculty Meet and Greet': 0.05,
            'Holiday Mixer': 0.03,
            'Campus Tour': 0.02
        }
    
    def generate_class_year(self, constituent_type, current_year=2025, rng=None):
        """Generate class year for alumni and parents"""
        if constituent_type not in ['Alum', 'Trustee', 'Regent']:
            return None
        
        # Alumni class years: 1950-2025, weighted toward more recent
        years = list(range(1950, 2026))
        # Weight more recent years higher
        weights = [1 if year < 1980 else (year - 1975) for year in years]
        
        year = random.choices(years, weights=weights)[0]
        return max(1950, min(current_year, year))
    
    def generate_parent_year(self, constituent_type, class_year=None, rng=None):
        """Generate parent year for parents and alumni who are also parents"""
        if constituent_type not in ['Parent'] and not (constituent_type in ['Alum', 'Trustee', 'Regent'] and random.random() < 0.3):
            return None
        
        # Parent year: 1990-2025, weighted toward more recent
        years = list(range(1990, 2026))
        # Weight more recent years higher
        weights = [1 if year < 2000 else (year - 1995) for year in years]
        
        parent_year = random.choices(years, weights=weights)[0]
        
        # If donor is also an alum, parent year should be after class year
        if class_year and parent_year <= class_year:
            parent_year = class_year + random.randint(1, 10)
            parent_year = min(2025, parent_year)
        
        return parent_year
    
    def generate_event_attendance(self, constituent_type, class_year=None, parent_year=None, rng=None):
        """Generate event attendance based on constituent type and years"""
        events_attended = []
        
        # Base probability of attending any event
        base_prob = 0.3
        
        # Adjust based on constituent type
        if constituent_type in ['Trustee', 'Regent']:
            base_prob = 0.8
        elif constituent_type == 'Alum':
            base_prob = 0.4
        elif constituent_type == 'Parent':
            base_prob = 0.5
        elif constituent_type == 'Friend':
            base_prob = 0.2
        
        # Generate events based on constituent type and years
        for event_name, prob in self.event_type_probs.items():
            # Adjust probability based on event type and constituent type
            adjusted_prob = prob * base_prob
            
            # Alumni-specific events
            if 'Alumni' in event_name and constituent_type not in ['Alum', 'Trustee', 'Regent']:
                adjusted_prob *= 0.1  # Very low for non-alumni
            
            # Parent-specific events
            if 'Parent' in event_name and constituent_type != 'Parent':
                adjusted_prob *= 0.1  # Very low for non-parents
            
            # Family events - higher for parents and alumni with families
            if 'Family' in event_name:
                if constituent_type == 'Parent':
                    adjusted_prob *= 1.5
                elif constituent_type in ['Alum', 'Trustee', 'Regent']:
                    adjusted_prob *= 0.8
            
            # Recent graduates more likely to attend certain events
            if class_year and class_year >= 2015:
                if 'Happy Hour' in event_name or 'Baseball' in event_name:
                    adjusted_prob *= 1.5
            
            # Older alumni more likely to attend reunion
            if class_year and class_year <= 1990:
                if 'Reunion' in event_name:
                    adjusted_prob *= 2.0
            
            # Cap probability
            adjusted_prob = min(0.9, adjusted_prob)
            
            if random.random() < adjusted_prob:
                # Generate attendance date (weighted toward recent years)
                attendance_year = random.choices(
                    list(range(2015, 2026)), 
                    weights=list(range(1, 12))
                )[0]
                
                # Generate specific date within the year
                month = random.randint(1, 12)
                day = random.randint(1, 28)
                attendance_date = date(attendance_year, month, day)
                
                # Don't allow future dates
                if attendance_date > date.today():
                    attendance_date = date.today() - timedelta(days=random.randint(1, 365))
                
                events_attended.append({
                    'Event_Name': event_name,
                    'Attendance_Date': attendance_date
                })
        
        return events_attended

class DenseRelationshipGenerator:
    """Generate dense relationship networks for GNN performance"""
    
    def __init__(self):
        self.relationship_types = {
            'Professional': ['Colleague', 'Business_Partner', 'Board_Member', 'Mentor_Mentee', 'Industry_Peer'],
            'Geographic': ['Neighbor', 'Same_City', 'Same_Region', 'Campus_Proximity'],
            'Alumni': ['Classmate', 'Same_Major', 'Greek_Life', 'Study_Abroad', 'Athletic_Team', 'Dorm_Mate'],
            'Activity': ['Event_Co_Attendee', 'Committee_Member', 'Volunteer_Partner', 'Fundraising_Team'],
            'Giving': ['Similar_Giving_Level', 'Same_Designation', 'Campaign_Participant', 'Matching_Gift'],
            'Social': ['Social_Connection', 'Mutual_Friend', 'Community_Leader', 'Cultural_Interest']
        }
        
        self.relationship_strengths = {
            'Family': 1.0,
            'Professional': 0.8,
            'Alumni': 0.7,
            'Geographic': 0.5,
            'Activity': 0.6,
            'Giving': 0.7,
            'Social': 0.4
        }
    
    def generate_professional_relationships(self, donors_df, target_edges=500000):
        """Generate professional relationships based on industry and location - VECTORIZED"""
        print("Generating professional relationships (VECTORIZED)...")
        relationships = []
        max_relationships = target_edges  # ENFORCE HARD CAP
        
        # Group by industry and geographic region
        industry_groups = donors_df.groupby(['Professional_Background', 'Geographic_Region'])
        
        for (industry, region), group in industry_groups:
            if len(relationships) >= max_relationships:
                break  # STOP when we hit the cap
            ids = group['ID'].values
            n = len(ids)
            
            if n < 2:
                continue
            
            # VECTORIZED: Calculate expected edges (REDUCED DENSITY)
            expected_edges = int(n * (n-1) / 2 * 0.005)  # REDUCED from 3% to 0.5%
            if expected_edges == 0:
                continue
            
            # VECTORIZED: Generate random pairs
            num_samples = min(expected_edges * 2, n * (n-1) // 2)
            i_idx = np.random.randint(0, n, size=num_samples)
            j_idx = np.random.randint(0, n, size=num_samples)
            
            # Filter valid pairs (i < j)
            mask = i_idx < j_idx
            i_idx, j_idx = i_idx[mask], j_idx[mask]
            
            if len(i_idx) == 0:
                continue
            
            # Remove duplicates
            pairs = np.unique(np.column_stack([i_idx, j_idx]), axis=0)
            if len(pairs) > expected_edges:
                pairs = pairs[np.random.choice(len(pairs), expected_edges, replace=False)]
            
            # MEMORY FIX: Process in TINY batches to prevent OOM
            batch_size = 1000  # DRASTICALLY REDUCED
            for batch_start in range(0, len(pairs), batch_size):
                if len(relationships) >= max_relationships:
                    break  # STOP IMMEDIATELY when cap reached
                batch_pairs = pairs[batch_start:batch_start + batch_size]
                rel_types = np.random.choice(self.relationship_types['Professional'], size=len(batch_pairs))
                strengths = np.random.uniform(0.3, 1.0, size=len(batch_pairs))
                
                relationships.extend([
                    {
                        'Donor_ID_1': ids[pair[0]],
                        'Donor_ID_2': ids[pair[1]],
                        'Relationship_Type': rel_type,
                        'Relationship_Category': 'Professional',
                        'Relationship_Strength': float(strength)
                    }
                    for pair, rel_type, strength in zip(batch_pairs, rel_types, strengths)
                ])
        
        print(f"   Generated {len(relationships):,} professional relationships")
        return relationships
    
    def generate_geographic_relationships(self, donors_df, target_edges=300000):
        """Generate geographic relationships based on location proximity - VECTORIZED with CHUNKING"""
        print("Generating geographic relationships (VECTORIZED with memory optimization)...")
        relationships = []
        max_relationships = target_edges  # ENFORCE HARD CAP
        
        # Group by geographic region
        region_groups = donors_df.groupby('Geographic_Region')
        
        for region, group in region_groups:
            if len(relationships) >= max_relationships:
                break  # STOP when we hit the cap
            ids = group['ID'].values
            n = len(ids)
            
            if n < 2:
                continue
            
            expected_edges = int(n * (n-1) / 2 * 0.003)  # REDUCED from 2% to 0.3%
            if expected_edges == 0:
                continue
            
            # MEMORY FIX: Limit chunk size to prevent OOM
            max_pairs_per_chunk = 5000  # REDUCED: Process only 5k pairs at a time
            
            if expected_edges > max_pairs_per_chunk:
                # Process in multiple chunks for large groups
                num_chunks = (expected_edges + max_pairs_per_chunk - 1) // max_pairs_per_chunk
                pairs_per_chunk = expected_edges // num_chunks
                
                for chunk_idx in range(num_chunks):
                    num_samples = min(pairs_per_chunk * 2, n * (n-1) // 2)
                    i_idx = np.random.randint(0, n, size=num_samples)
                    j_idx = np.random.randint(0, n, size=num_samples)
                    
                    mask = i_idx < j_idx
                    i_idx, j_idx = i_idx[mask], j_idx[mask]
                    
                    if len(i_idx) == 0:
                        continue
                    
                    pairs = np.unique(np.column_stack([i_idx, j_idx]), axis=0)
                    if len(pairs) > pairs_per_chunk:
                        pairs = pairs[np.random.choice(len(pairs), pairs_per_chunk, replace=False)]
                    
                    # MEMORY FIX: Process in TINY batches within chunks
                    batch_size = 500  # Ultra-small batches
                    for batch_start in range(0, len(pairs), batch_size):
                        if len(relationships) >= max_relationships:
                            break  # STOP IMMEDIATELY when cap reached
                        batch_pairs = pairs[batch_start:batch_start + batch_size]
                        rel_types = np.random.choice(self.relationship_types['Geographic'], size=len(batch_pairs))
                        strengths = np.random.uniform(0.2, 0.8, size=len(batch_pairs))
                        
                        relationships.extend([
                            {
                                'Donor_ID_1': ids[pair[0]],
                                'Donor_ID_2': ids[pair[1]],
                                'Relationship_Type': rel_type,
                                'Relationship_Category': 'Geographic',
                                'Relationship_Strength': float(strength)
                            }
                            for pair, rel_type, strength in zip(batch_pairs, rel_types, strengths)
                        ])
            else:
                # Small enough to process but still need batching
                num_samples = min(expected_edges * 2, n * (n-1) // 2)
                i_idx = np.random.randint(0, n, size=num_samples)
                j_idx = np.random.randint(0, n, size=num_samples)
                
                mask = i_idx < j_idx
                i_idx, j_idx = i_idx[mask], j_idx[mask]
                
                if len(i_idx) == 0:
                    continue
                
                pairs = np.unique(np.column_stack([i_idx, j_idx]), axis=0)
                if len(pairs) > expected_edges:
                    pairs = pairs[np.random.choice(len(pairs), expected_edges, replace=False)]
                
                # MEMORY FIX: Always batch, even for "small" groups
                batch_size = 500  # TINY batch for safety
                for batch_start in range(0, len(pairs), batch_size):
                    if len(relationships) >= max_relationships:
                        break  # STOP IMMEDIATELY when cap reached
                    batch_pairs = pairs[batch_start:batch_start + batch_size]
                    rel_types = np.random.choice(self.relationship_types['Geographic'], size=len(batch_pairs))
                    strengths = np.random.uniform(0.2, 0.8, size=len(batch_pairs))
                    
                    relationships.extend([
                        {
                            'Donor_ID_1': ids[pair[0]],
                            'Donor_ID_2': ids[pair[1]],
                            'Relationship_Type': rel_type,
                            'Relationship_Category': 'Geographic',
                            'Relationship_Strength': float(strength)
                        }
                        for pair, rel_type, strength in zip(batch_pairs, rel_types, strengths)
                    ])
        
        print(f"   Generated {len(relationships):,} geographic relationships")
        return relationships
    
    def generate_alumni_relationships(self, donors_df, target_edges=400000):
        """Generate alumni relationships based on class year and major - VECTORIZED"""
        print("Generating alumni relationships (VECTORIZED)...")
        relationships = []
        max_relationships = target_edges  # ENFORCE HARD CAP
        
        # Group by class year (within 5 years)
        for year in range(1950, 2026, 5):
            if len(relationships) >= max_relationships:
                break  # STOP when we hit the cap
            year_group = donors_df[
                (donors_df['Class_Year'] >= year) & 
                (donors_df['Class_Year'] < year + 5) &
                (donors_df['Primary_Constituent_Type'].isin(['Alum', 'Trustee', 'Regent']))
            ]
            
            ids = year_group['ID'].values
            n = len(ids)
            
            if n < 2:
                continue
            
            expected_edges = int(n * (n-1) / 2 * 0.01)  # REDUCED from 5% to 1%
            if expected_edges == 0:
                continue
            
            num_samples = min(expected_edges * 2, n * (n-1) // 2)
            i_idx = np.random.randint(0, n, size=num_samples)
            j_idx = np.random.randint(0, n, size=num_samples)
            
            mask = i_idx < j_idx
            i_idx, j_idx = i_idx[mask], j_idx[mask]
            
            if len(i_idx) == 0:
                continue
            
            pairs = np.unique(np.column_stack([i_idx, j_idx]), axis=0)
            if len(pairs) > expected_edges:
                pairs = pairs[np.random.choice(len(pairs), expected_edges, replace=False)]
            
            # MEMORY FIX: Process in TINY batches to prevent OOM
            batch_size = 1000  # DRASTICALLY REDUCED
            for batch_start in range(0, len(pairs), batch_size):
                if len(relationships) >= max_relationships:
                    break  # STOP IMMEDIATELY when cap reached
                batch_pairs = pairs[batch_start:batch_start + batch_size]
                rel_types = np.random.choice(self.relationship_types['Alumni'], size=len(batch_pairs))
                strengths = np.random.uniform(0.4, 1.0, size=len(batch_pairs))
                
                relationships.extend([
                    {
                        'Donor_ID_1': ids[pair[0]],
                        'Donor_ID_2': ids[pair[1]],
                        'Relationship_Type': rel_type,
                        'Relationship_Category': 'Alumni',
                        'Relationship_Strength': float(strength)
                    }
                    for pair, rel_type, strength in zip(batch_pairs, rel_types, strengths)
                ])
        
        print(f"   Generated {len(relationships):,} alumni relationships")
        return relationships
    
    def generate_activity_relationships(self, event_attendance_df, target_edges=200000):
        """Generate relationships based on event co-attendance - VECTORIZED"""
        print("Generating activity relationships (VECTORIZED)...")
        relationships = []
        max_relationships = target_edges  # ENFORCE HARD CAP
        
        # Group by event attendance
        event_groups = event_attendance_df.groupby('Event_Name')
        
        for event_name, attendees in event_groups:
            if len(relationships) >= max_relationships:
                break  # STOP when we hit the cap
            ids = attendees['Donor_ID'].values
            n = len(ids)
            
            if n < 2:
                continue
            
            expected_edges = int(n * (n-1) / 2 * 0.01)  # REDUCED from 8% to 1%
            if expected_edges == 0:
                continue
            
            num_samples = min(expected_edges * 2, n * (n-1) // 2)
            i_idx = np.random.randint(0, n, size=num_samples)
            j_idx = np.random.randint(0, n, size=num_samples)
            
            mask = i_idx < j_idx
            i_idx, j_idx = i_idx[mask], j_idx[mask]
            
            if len(i_idx) == 0:
                continue
            
            pairs = np.unique(np.column_stack([i_idx, j_idx]), axis=0)
            if len(pairs) > expected_edges:
                pairs = pairs[np.random.choice(len(pairs), expected_edges, replace=False)]
            
            # MEMORY FIX: Process in TINY batches to prevent OOM
            batch_size = 1000  # DRASTICALLY REDUCED
            for batch_start in range(0, len(pairs), batch_size):
                if len(relationships) >= max_relationships:
                    break  # STOP IMMEDIATELY when cap reached
                batch_pairs = pairs[batch_start:batch_start + batch_size]
                rel_types = np.random.choice(self.relationship_types['Activity'], size=len(batch_pairs))
                strengths = np.random.uniform(0.3, 0.9, size=len(batch_pairs))
                
                relationships.extend([
                    {
                        'Donor_ID_1': ids[pair[0]],
                        'Donor_ID_2': ids[pair[1]],
                        'Relationship_Type': rel_type,
                        'Relationship_Category': 'Activity',
                        'Relationship_Strength': float(strength)
                    }
                    for pair, rel_type, strength in zip(batch_pairs, rel_types, strengths)
                ])
        
        print(f"   Generated {len(relationships):,} activity relationships")
        return relationships
    
    def generate_giving_relationships(self, donors_df, target_edges=300000):
        """Generate relationships based on similar giving patterns - VECTORIZED"""
        print("Generating giving relationships (VECTORIZED)...")
        relationships = []
        max_relationships = target_edges  # ENFORCE HARD CAP
        
        # Group by giving level ranges
        giving_levels = [
            (0, 1000, 'Low'),
            (1000, 10000, 'Medium'),
            (10000, 100000, 'High'),
            (100000, float('inf'), 'Major')
        ]
        
        for min_giving, max_giving, level in giving_levels:
            if len(relationships) >= max_relationships:
                break  # STOP when we hit the cap
            level_group = donors_df[
                (donors_df['Lifetime_Giving'] >= min_giving) & 
                (donors_df['Lifetime_Giving'] < max_giving)
            ]
            
            ids = level_group['ID'].values
            n = len(ids)
            
            if n < 2:
                continue
            
            expected_edges = int(n * (n-1) / 2 * 0.03)
            if expected_edges == 0:
                continue
            
            num_samples = min(expected_edges * 2, n * (n-1) // 2)
            i_idx = np.random.randint(0, n, size=num_samples)
            j_idx = np.random.randint(0, n, size=num_samples)
            
            mask = i_idx < j_idx
            i_idx, j_idx = i_idx[mask], j_idx[mask]
            
            if len(i_idx) == 0:
                continue
            
            pairs = np.unique(np.column_stack([i_idx, j_idx]), axis=0)
            if len(pairs) > expected_edges:
                pairs = pairs[np.random.choice(len(pairs), expected_edges, replace=False)]
            
            # MEMORY FIX: Process in TINY batches to prevent OOM
            batch_size = 1000  # DRASTICALLY REDUCED
            for batch_start in range(0, len(pairs), batch_size):
                if len(relationships) >= max_relationships:
                    break  # STOP IMMEDIATELY when cap reached
                batch_pairs = pairs[batch_start:batch_start + batch_size]
                rel_types = np.random.choice(self.relationship_types['Giving'], size=len(batch_pairs))
                strengths = np.random.uniform(0.3, 0.8, size=len(batch_pairs))
                
                relationships.extend([
                    {
                        'Donor_ID_1': ids[pair[0]],
                        'Donor_ID_2': ids[pair[1]],
                        'Relationship_Type': rel_type,
                        'Relationship_Category': 'Giving',
                        'Relationship_Strength': float(strength)
                    }
                    for pair, rel_type, strength in zip(batch_pairs, rel_types, strengths)
                ])
        
        print(f"   Generated {len(relationships):,} giving relationships")
        return relationships
    
    def generate_social_relationships(self, donors_df, target_edges=200000):
        """Generate social relationships based on mutual connections - VECTORIZED"""
        print("Generating social relationships (VECTORIZED)...")
        relationships = []
        
        # Random social connections across all donors
        ids = donors_df['ID'].values
        n = len(ids)
        
        # Generate random edges
        expected_edges = int(n * 0.02)  # 2% of donors
        
        if expected_edges > 0:
            i_idx = np.random.randint(0, n, size=expected_edges)
            j_idx = np.random.randint(0, n, size=expected_edges)
            
            # Filter: i != j
            mask = i_idx != j_idx
            i_idx, j_idx = i_idx[mask], j_idx[mask]
            
            if len(i_idx) > 0:
                # MEMORY FIX: Process in TINY batches to prevent OOM
                batch_size = 1000  # DRASTICALLY REDUCED
                for batch_start in range(0, len(i_idx), batch_size):
                    batch_i = i_idx[batch_start:batch_start + batch_size]
                    batch_j = j_idx[batch_start:batch_start + batch_size]
                    rel_types = np.random.choice(self.relationship_types['Social'], size=len(batch_i))
                    strengths = np.random.uniform(0.2, 0.6, size=len(batch_i))
                    
                    relationships.extend([
                        {
                            'Donor_ID_1': ids[i],
                            'Donor_ID_2': ids[j],
                            'Relationship_Type': rel_type,
                            'Relationship_Category': 'Social',
                            'Relationship_Strength': float(strength)
                        }
                        for i, j, rel_type, strength in zip(batch_i, batch_j, rel_types, strengths)
                    ])
        
        print(f"   Generated {len(relationships):,} social relationships")
        return relationships
    
    def generate_dense_relationships(self, donors_df, event_attendance_df, output_dir='data/synthetic_donor_dataset_500k_dense'):
        """Generate all types of dense relationships - WRITE INCREMENTALLY TO DISK"""
        print("🌐 GENERATING DENSE RELATIONSHIP NETWORKS (MEMORY-OPTIMIZED: Writing to disk incrementally)")
        print("=" * 60)
        
        import csv
        output_path = Path(output_dir) / 'dense_relationships.csv'
        
        # Check if file already exists
        if output_path.exists():
            print(f"   ⏭️  Dense relationships already exist at {output_path}")
            return pd.read_csv(output_path)
        
        # Open CSV file for incremental writing
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Donor_ID_1', 'Donor_ID_2', 'Relationship_Type', 'Relationship_Category', 'Relationship_Strength']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Generate each relationship type and write immediately
            print("Generating professional relationships (STREAMING TO DISK)...")
            professional_rels = self.generate_professional_relationships(donors_df)
            writer.writerows(professional_rels)
            del professional_rels  # Free memory immediately
            print(f"   ✅ Professional relationships written to disk")
            
            print("Generating geographic relationships (STREAMING TO DISK)...")
            geographic_rels = self.generate_geographic_relationships(donors_df)
            writer.writerows(geographic_rels)
            del geographic_rels
            print(f"   ✅ Geographic relationships written to disk")
            
            print("Generating alumni relationships (STREAMING TO DISK)...")
            alumni_rels = self.generate_alumni_relationships(donors_df)
            writer.writerows(alumni_rels)
            del alumni_rels
            print(f"   ✅ Alumni relationships written to disk")
            
            print("Generating activity relationships (STREAMING TO DISK)...")
            activity_rels = self.generate_activity_relationships(event_attendance_df)
            writer.writerows(activity_rels)
            del activity_rels
            print(f"   ✅ Activity relationships written to disk")
            
            print("Generating giving relationships (STREAMING TO DISK)...")
            giving_rels = self.generate_giving_relationships(donors_df)
            writer.writerows(giving_rels)
            del giving_rels
            print(f"   ✅ Giving relationships written to disk")
            
            print("Generating social relationships (STREAMING TO DISK)...")
            social_rels = self.generate_social_relationships(donors_df)
            writer.writerows(social_rels)
            del social_rels
            print(f"   ✅ Social relationships written to disk")
        
        print(f"\n✅ All relationships written to {output_path}")
        
        # Read back for return
        relationships_df = pd.read_csv(output_path)
        
        # Add family relationships (existing)
        family_relationships = self.generate_family_relationships(donors_df)
        family_df = pd.DataFrame(family_relationships)
        
        # Combine all relationships
        all_relationships_df = pd.concat([relationships_df, family_df], ignore_index=True)
        
        print(f"\n✅ Total relationships generated: {len(all_relationships_df):,}")
        print(f"   Graph density: {len(all_relationships_df) / (len(donors_df) * (len(donors_df) - 1) / 2) * 100:.4f}%")
        
        # Relationship type breakdown
        print("\n📊 RELATIONSHIP TYPE BREAKDOWN:")
        rel_counts = all_relationships_df['Relationship_Category'].value_counts()
        for category, count in rel_counts.items():
            print(f"   {category}: {count:,} relationships")
        
        return all_relationships_df
    
    def generate_family_relationships(self, donors_df):
        """Generate family relationships (existing logic)"""
        print("Generating family relationships...")
        relationships = []
        
        # Get family groups
        family_groups = donors_df[donors_df['Family_ID'].notna()].groupby('Family_ID')
        
        for family_id, family_members in family_groups:
            if len(family_members) > 1:
                # Create connections between all family members
                for i, donor1 in family_members.iterrows():
                    for j, donor2 in family_members.iterrows():
                        if i < j:
                            relationships.append({
                                'Donor_ID_1': donor1['ID'],
                                'Donor_ID_2': donor2['ID'],
                                'Relationship_Type': donor1['Relationship_Type'],
                                'Relationship_Category': 'Family',
                                'Relationship_Strength': 1.0
                            })
        
        print(f"   Generated {len(relationships):,} family relationships")
        return relationships

def generate_family_assignments_fast(num_donors, family_percentage=0.20, seed=42):
    """Fast vectorized family assignment for large datasets.
    Creates contiguous family groups with minimal Python overhead.
    """
    import numpy as np
    print(f"Pre-generating family assignments for {family_percentage*100:.0f}% of donors (FAST)...")
    family_donor_count = int(num_donors * family_percentage)

    rng = np.random.default_rng(seed)
    indices = np.arange(num_donors, dtype=np.int64)
    rng.shuffle(indices)
    selected = indices[:family_donor_count]

    # Sample family sizes and pack contiguously
    size_choices = np.array([2, 3, 4, 5])
    size_probs = np.array([0.5, 0.3, 0.15, 0.05])
    est_families = max(1, family_donor_count // 3)
    sizes = rng.choice(size_choices, size=est_families, p=size_probs)

    families = []
    ptr = 0
    for fs in sizes:
        if ptr >= family_donor_count:
            break
        k = int(min(fs, family_donor_count - ptr))
        if k <= 0:
            break
        families.append(selected[ptr:ptr + k])
        ptr += k

    family_assignments = {}
    fam_id = 10000
    rel_cycle = ['Head', 'Spouse', 'Child', 'Parent', 'Sibling']
    for fam in families:
        for i, idx in enumerate(fam):
            rel = rel_cycle[min(i, 2 + (i % 3))]
            family_assignments[int(idx)] = {'Family_ID': fam_id, 'Relationship_Type': rel}
        fam_id += 1

    print(f"Created {len(families)} families")
    print(f"Assigned {len(family_assignments)} donors to families")
    return family_assignments

def generate_enhanced_500k_donors_with_dense_relationships():
    """Generate 500K donors with enhanced characteristics and dense relationships"""
    print("🚀 GENERATING ENHANCED 500K SYNTHETIC DONOR DATASET WITH DENSE RELATIONSHIPS", flush=True)
    print("=" * 80, flush=True)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    rng = np.random.RandomState(42)
    
    # Runtime controls via environment variables
    num_donors = int(os.environ.get('NUM_DONORS', '500000'))
    fast_mode = os.environ.get('FAST_MODE', '0') == '1'
    save_dense_csv = os.environ.get('SAVE_DENSE_CSV', '1') == '1'
    disable_tqdm = os.environ.get('DISABLE_TQDM', '0') == '1'

    print(f"Generating {num_donors:,} donors...", flush=True)
    
    # Generate unique donor IDs
    print("Generating unique donor IDs...", flush=True)
    donor_ids = generate_random_ids(num_donors)
    
    # Initialize generators
    demo_gen = DemographicsGenerator()
    const_gen = EnhancedConstituentGenerator()
    giving_gen = GivingGenerator()
    rel_gen = DenseRelationshipGenerator()
    
    # Pre-generate family assignments
    # Always use fast vectorized version for large datasets
    family_assignments = generate_family_assignments_fast(num_donors, family_percentage=0.20)
    
    # Generate core donor data
    print("Generating core donor data...", flush=True)
    donors_chunk = []
    relationships_chunk = []
    event_attendance_chunk = []

    # Output directories and checkpoint
    output_dir = 'data/synthetic_donor_dataset_500k_dense'
    os.makedirs(output_dir, exist_ok=True)
    parts_dir = os.path.join(output_dir, 'parts')
    os.makedirs(parts_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, 'checkpoints.json')

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
    else:
        checkpoint = {}

    start_index = int(checkpoint.get('next_index', 0))
    checkpoint_interval = int(os.environ.get('CHECKPOINT_INTERVAL', '50000'))
    print(f"Resume checkpoint: starting at index {start_index:,} of {num_donors:,}", flush=True)

    for i in tqdm(
        range(start_index, num_donors),
        initial=start_index,
        total=num_donors,
        desc="Creating enhanced donors",
        disable=disable_tqdm,
    ):
        donor_id = donor_ids[i]
        
        # Generate names and demographics
        first_name, last_name, full_name, gender = demo_gen.generate_name()
        demographics = demo_gen.generate_demographics()
        
        # Generate constituent information
        primary_type, secondary_type = const_gen.assign_constituent_types()
        class_year = const_gen.generate_class_year(primary_type, rng=rng)
        parent_year = const_gen.generate_parent_year(primary_type, class_year, rng=rng)
        
        # Generate advancement data
        manager = const_gen.assign_manager()
        rating = const_gen.assign_rating()
        prospect_stage = const_gen.assign_prospect_stage()
        
        # Generate giving data
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
        
        # Generate event attendance
        events = const_gen.generate_event_attendance(primary_type, class_year, parent_year, rng)
        
        # Adjust event attendance probability based on giving history
        if lifetime_giving > 0:
            # Higher giving = more likely to attend events
            for event in events:
                if random.random() < min(0.3, lifetime_giving / 100000):
                    # Add more events for high-value donors
                    additional_events = const_gen.generate_event_attendance(primary_type, class_year, parent_year, rng)
                    events.extend(additional_events)
        
        # Create donor record
        donor = {
            'ID': donor_id,
            'First_Name': first_name,
            'Last_Name': last_name,
            'Full_Name': full_name,
            'Gender': gender,
            'Primary_Constituent_Type': primary_type,
            'Constituent_Type_2': secondary_type,
            'Class_Year': class_year,
            'Parent_Year': parent_year,
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
        
        donors_chunk.append(donor)
        
        # Add to relationships if has family
        if family_id is not None:
            relationships_chunk.append({
                'Donor_ID': donor_id,
                'Family_ID': family_id,
                'Relationship_Type': relationship_type
            })
        
        # Add event attendance records
        for event in events:
            event_attendance_chunk.append({
                'Donor_ID': donor_id,
                'Event_Name': event['Event_Name'],
                'Attendance_Date': event['Attendance_Date']
            })

        # Flush chunk on interval
        if ((i + 1) % checkpoint_interval == 0) or (i + 1 == num_donors):
            start = i + 1 - len(donors_chunk)
            end = i + 1
            if donors_chunk:
                pd.DataFrame(donors_chunk).to_csv(os.path.join(parts_dir, f'donors_part_{start}_{end}.csv'), index=False)
                donors_chunk.clear()
            if relationships_chunk:
                pd.DataFrame(relationships_chunk).to_csv(os.path.join(parts_dir, f'family_relationships_part_{start}_{end}.csv'), index=False)
                relationships_chunk.clear()
            if event_attendance_chunk:
                pd.DataFrame(event_attendance_chunk).to_csv(os.path.join(parts_dir, f'event_attendance_part_{start}_{end}.csv'), index=False)
                event_attendance_chunk.clear()
            # Update checkpoint
            checkpoint['next_index'] = end
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f)
    
    print("\n📂 Loading existing donor data (this may take 2-3 minutes for large files)...", flush=True)
    
    # Merge parts into consolidated CSVs if not already merged
    def merge_parts(pattern_prefix, output_filename):
        part_files = sorted([fn for fn in os.listdir(parts_dir) if fn.startswith(pattern_prefix) and fn.endswith('.csv')],
                            key=lambda x: int(x.split('_')[-2]))
        if not part_files:
            return pd.DataFrame()
        out_path = os.path.join(output_dir, output_filename)
        if os.path.exists(out_path):
            try:
                print(f"      Loading {output_filename}...", flush=True)
                return pd.read_csv(out_path)
            except Exception:
                pass
        dfs = []
        for fn in part_files:
            dfs.append(pd.read_csv(os.path.join(parts_dir, fn)))
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(out_path, index=False)
        return df

    donors_df = merge_parts('donors_part_', 'donors.csv')
    print(f"   ✅ Loaded {len(donors_df):,} donors", flush=True)
    
    relationships_df = merge_parts('family_relationships_part_', 'family_relationships.csv')
    print(f"   ✅ Loaded {len(relationships_df):,} family relationships", flush=True)
    
    event_attendance_df = merge_parts('event_attendance_part_', 'event_attendance.csv')
    print(f"   ✅ Loaded {len(event_attendance_df):,} event attendance records", flush=True)
    
    # Generate additional data
    print("\nGenerating additional data...", flush=True)
    
    # Generate giving history (skip if exists)
    giving_history_path = os.path.join(output_dir, 'giving_history.csv')
    if os.path.exists(giving_history_path):
        print("   ⏭️  Skipping giving history (already exists)")
        giving_history_df = pd.read_csv(giving_history_path)
    else:
        print("\n⏳ Generating giving history (OPTIMIZED - this may take 1-2 minutes)...")
        giving_history_df = generate_giving_history(donors_df, rng)
        giving_history_df.to_csv(giving_history_path, index=False)
        print(f"   ✅ Generated {len(giving_history_df):,} giving history records")
    
    # Generate enhanced fields (skip if exists)
    enhanced_path = os.path.join(output_dir, 'enhanced_fields.csv')
    if os.path.exists(enhanced_path):
        print("   ⏭️  Skipping enhanced fields (already exists)")
        enhanced_df = pd.read_csv(enhanced_path)
    else:
        print("\n⏳ Generating enhanced fields (OPTIMIZED - this may take 30-60 seconds)...")
        enhanced_df = generate_enhanced_fields(donors_df)
        enhanced_df.to_csv(enhanced_path, index=False)
        print(f"   ✅ Generated {len(enhanced_df):,} enhanced field records")
    
    # Generate contact reports (skip if exists)
    contact_reports_path = os.path.join(output_dir, 'contact_reports.csv')
    if os.path.exists(contact_reports_path):
        print("   ⏭️  Skipping contact reports (already exists)")
        contact_reports_df = pd.read_csv(contact_reports_path)
    else:
        print("\n⏳ Generating contact reports (OPTIMIZED - this may take 30-60 seconds)...")
        contact_reports_df = generate_contact_reports(donors_df)
        contact_reports_df.to_csv(contact_reports_path, index=False)
        print(f"   ✅ Generated {len(contact_reports_df):,} contact reports")
    
    # Generate dense relationships (skip if exists)
    dense_parquet_path = os.path.join(output_dir, 'dense_relationships.parquet')
    dense_csv_path = os.path.join(output_dir, 'dense_relationships.csv')
    if os.path.exists(dense_parquet_path) or os.path.exists(dense_csv_path):
        print("   ⏭️  Skipping dense relationships (already exists)")
        try:
            dense_relationships_df = pd.read_parquet(dense_parquet_path)
        except Exception:
            dense_relationships_df = pd.read_csv(dense_csv_path)
    else:
        print("\n⏳ Generating dense relationships (this may take 10-15 minutes with VECTORIZED code)...")
        dense_relationships_df = rel_gen.generate_dense_relationships(donors_df, event_attendance_df)
    
    # Create output directory
    # output_dir already created above
    
    # Save datasets
    print(f"\nSaving datasets to {output_dir}/...")
    
    # donors.csv, family_relationships.csv, event_attendance.csv already written by merge_parts
    # Prefer Parquet for speed and smaller files; optionally skip CSV
    try:
        dense_relationships_df.to_parquet(f'{output_dir}/dense_relationships.parquet', engine='pyarrow', compression='snappy')
        if save_dense_csv:
            dense_relationships_df.to_csv(f'{output_dir}/dense_relationships.csv', index=False)
    except Exception as e:
        print(f"⚠️ Parquet save failed ({e}); falling back to CSV only.")
        dense_relationships_df.to_csv(f'{output_dir}/dense_relationships.csv', index=False)
    event_attendance_df.to_csv(f'{output_dir}/event_attendance.csv', index=False)
    giving_history_df.to_csv(f'{output_dir}/giving_history.csv', index=False)
    enhanced_df.to_csv(f'{output_dir}/enhanced_fields.csv', index=False)
    contact_reports_df.to_csv(f'{output_dir}/contact_reports.csv', index=False)

    # Mark completion in checkpoint
    checkpoint['completed'] = True
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f)
    
    print("✅ All datasets saved successfully!")
    
    # Generate summary statistics
    print("\n📊 DATASET SUMMARY:")
    print("-" * 50)
    print(f"Total Donors: {len(donors_df):,}")
    print(f"Alumni: {len(donors_df[donors_df['Primary_Constituent_Type'] == 'Alum']):,}")
    print(f"Parents: {len(donors_df[donors_df['Primary_Constituent_Type'] == 'Parent']):,}")
    print(f"Alumni with Parent Year: {len(donors_df[(donors_df['Primary_Constituent_Type'] == 'Alum') & (donors_df['Parent_Year'].notna())]):,}")
    print(f"Total Event Attendance: {len(event_attendance_df):,}")
    print(f"Unique Events: {event_attendance_df['Event_Name'].nunique()}")
    print(f"Donors with Events: {event_attendance_df['Donor_ID'].nunique():,}")
    print(f"Family Relationships: {len(relationships_df):,}")
    print(f"Dense Relationships: {len(dense_relationships_df):,}")
    print(f"Giving History Records: {len(giving_history_df):,}")
    print(f"Contact Reports: {len(contact_reports_df):,}")
    
    # Event attendance breakdown
    print("\n📅 EVENT ATTENDANCE BREAKDOWN:")
    print("-" * 50)
    event_counts = event_attendance_df['Event_Name'].value_counts()
    for event, count in event_counts.items():
        print(f"{event}: {count:,} attendances")
    
    return donors_df, relationships_df, dense_relationships_df, event_attendance_df, giving_history_df, enhanced_df, contact_reports_df

def generate_giving_history(donors_df, rng):
    """Generate detailed giving history for donors - OPTIMIZED"""
    print("Generating detailed giving history (OPTIMIZED - faster than iterrows)...")
    
    giving_gen = GivingGenerator()
    giving_history = []
    gift_id_counter = 1000000
    
    # Only generate history for actual donors
    donor_subset = donors_df[donors_df['Lifetime_Giving'] > 0].copy()
    
    # OPTIMIZED: Use itertuples instead of iterrows (5-10x faster)
    for donor in tqdm(donor_subset.itertuples(), total=len(donor_subset), desc="Creating giving history"):
        total_years = donor.Total_Yr_Giving_Count
        lifetime_giving = donor.Lifetime_Giving
        
        if total_years == 0:
            continue
        
        # Generate giving years
        current_year = 2025
        class_year = int(donor.Class_Year) if pd.notna(donor.Class_Year) else 1980
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
                'Donor_ID': donor.ID,
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
    """Generate enhanced fields for deep learning applications - OPTIMIZED"""
    print("Generating enhanced fields for deep learning (OPTIMIZED - faster than iterrows)...")
    
    enhanced_data = []
    
    # OPTIMIZED: Use itertuples instead of iterrows (5-10x faster)
    for donor in tqdm(donors_df.itertuples(), total=len(donors_df), desc="Adding enhanced fields"):
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
        
        if donor.Lifetime_Giving > 0:
            engagement_score += min(30, donor.Lifetime_Giving / 10000)
        
        if donor.Consecutive_Yr_Giving_Count > 0:
            engagement_score += min(25, donor.Consecutive_Yr_Giving_Count * 2)
        
        if donor.Primary_Constituent_Type in ['Trustee', 'Regent']:
            engagement_score += 25
        
        # Family bonus for engagement
        if pd.notna(donor.Family_ID):
            engagement_score += 10
        
        engagement_score += random.randint(-10, 15)
        engagement_score = max(0, min(100, engagement_score))
        
        # Legacy intent probability
        age_estimate = 2025 - (donor.Class_Year if pd.notna(donor.Class_Year) else 1980) + 22
        legacy_prob = 0.1  # Base probability
        
        if age_estimate > 65:
            legacy_prob += 0.3
        elif age_estimate > 55:
            legacy_prob += 0.2
        
        if donor.Lifetime_Giving > 100000:
            legacy_prob += 0.2
        elif donor.Lifetime_Giving > 50000:
            legacy_prob += 0.1
        
        # Family history bonus for legacy intent
        if pd.notna(donor.Family_ID) and donor.Lifetime_Giving > 25000:
            legacy_prob += 0.15
        
        legacy_prob = min(0.95, legacy_prob)
        
        # Board affiliations
        board_affiliations = None
        if donor.Rating in ['A', 'B', 'C', 'D'] and random.random() < 0.4:
            boards = ['Hospital Board', 'Museum Board', 'Foundation Board', 'Corporate Board', 'Nonprofit Board']
            num_boards = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
            board_affiliations = ', '.join(random.sample(boards, num_boards))
        
        enhanced_data.append({
            'Donor_ID': donor.ID,
            'Interest_Keywords': ', '.join(selected_interests),
            'Engagement_Score': round(engagement_score, 1),
            'Legacy_Intent_Probability': round(legacy_prob, 3),
            'Legacy_Intent_Binary': random.random() < legacy_prob,
            'Board_Affiliations': board_affiliations,
            'Estimated_Age': age_estimate
        })
    
    return pd.DataFrame(enhanced_data)

def generate_contact_reports(donors_df):
    """Generate contact reports for eligible donors - OPTIMIZED"""
    print("Generating contact reports (OPTIMIZED - faster than iterrows)...")
    
    contact_gen = ContactReportGenerator()
    const_gen = ConstituentGenerator()
    contact_reports = []
    report_id_counter = 500000
    
    # OPTIMIZED: Use itertuples instead of iterrows (5-10x faster)
    for donor in tqdm(donors_df.itertuples(), total=len(donors_df), desc="Creating contact reports"):
        # Determine if donor should have contact report
        probability = contact_gen.should_have_contact_report(
            donor.Primary_Constituent_Type,
            donor.Prospect_Stage,
            donor.Lifetime_Giving
        )
        
        if random.random() < probability:
            # Generate contact report
            report_text = contact_gen.generate_report(
                donor.Full_Name,
                donor.Prospect_Stage,
                donor.Rating,
                donor.Lifetime_Giving,
                donor.Primary_Constituent_Type
            )
            
            contact_date = contact_gen.generate_contact_date()
            author = const_gen.assign_manager()
            
            contact_reports.append({
                'Contact_Report_ID': report_id_counter,
                'Donor_ID': donor.ID,
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

if __name__ == "__main__":
    # Generate the enhanced 500K dataset with dense relationships
    donors_df, relationships_df, dense_relationships_df, event_attendance_df, giving_history_df, enhanced_df, contact_reports_df = generate_enhanced_500k_donors_with_dense_relationships()
    
    print("\n🎉 Enhanced 500K synthetic donor dataset with dense relationships generation complete!")
    print("Files saved to: data/synthetic_donor_dataset_500k_dense/")
