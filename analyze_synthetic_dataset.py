#!/usr/bin/env python3
"""Analyze synthetic donor dataset to extract key insights"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load datasets
data_dir = Path('data/synthetic_donor_dataset')
donors_df = pd.read_csv(data_dir / 'donors.csv')
relationships_df = pd.read_csv(data_dir / 'relationships.csv')

print("=" * 80)
print("SYNTHETIC DONOR DATASET - KEY INSIGHTS")
print("=" * 80)
print(f"\nTotal Donors: {len(donors_df):,}")
print(f"Total Relationships: {len(relationships_df):,}")

# 1. CONSTITUENT TYPES
print("\n" + "=" * 80)
print("1. CONSTITUENT TYPES (Primary)")
print("=" * 80)
const_types = donors_df['Primary_Constituent_Type'].value_counts()
for ct, count in const_types.items():
    pct = count / len(donors_df) * 100
    print(f"   {ct:20s}: {count:6,} ({pct:5.1f}%)")

# Secondary constituent types
print("\n   Secondary Constituent Types:")
if 'Constituent_Type_2' in donors_df.columns:
    sec_types = donors_df['Constituent_Type_2'].value_counts()
    for st, count in sec_types.items():
        if pd.notna(st):
            pct = count / len(donors_df) * 100
            print(f"   {st:20s}: {count:6,} ({pct:5.1f}%)")

# 2. RATING TIERS
print("\n" + "=" * 80)
print("2. RATING TIERS (16 tiers: A-P, where A = highest capacity)")
print("=" * 80)
ratings = donors_df['Rating'].value_counts().sort_index()
for r, count in ratings.items():
    pct = count / len(donors_df) * 100
    print(f"   Rating {r:2s}: {count:6,} ({pct:5.2f}%)")

# Rating summary
print(f"\n   Rating Summary:")
print(f"   Highest tier (A): {ratings.get('A', 0):,} donors")
print(f"   Lowest tier (P): {ratings.get('P', 0):,} donors")
print(f"   Most common: {ratings.idxmax()} ({ratings.max():,} donors)")

# 3. GIVING YEAR RANGE
print("\n" + "=" * 80)
print("3. GIVING YEAR RANGE")
print("=" * 80)
donors_df['Last_Gift_Date'] = pd.to_datetime(donors_df['Last_Gift_Date'], errors='coerce')
valid_dates = donors_df['Last_Gift_Date'].dropna()

if len(valid_dates) > 0:
    earliest = valid_dates.min()
    latest = valid_dates.max()
    span_years = (latest - earliest).days / 365.25
    
    print(f"   Earliest gift date: {earliest.strftime('%Y-%m-%d')}")
    print(f"   Latest gift date:   {latest.strftime('%Y-%m-%d')}")
    print(f"   Time span:          {span_years:.1f} years")
    print(f"   Donors with valid dates: {len(valid_dates):,} ({len(valid_dates)/len(donors_df)*100:.1f}%)")
    
    # Year distribution
    donors_df['Gift_Year'] = donors_df['Last_Gift_Date'].dt.year
    year_dist = donors_df['Gift_Year'].value_counts().sort_index()
    print(f"\n   Gift Years Range: {int(year_dist.index.min())} - {int(year_dist.index.max())}")
    print(f"   Most active year: {int(year_dist.idxmax())} ({year_dist.max():,} gifts)")

# 4. NETWORK/RELATIONSHIP FEATURES
print("\n" + "=" * 80)
print("4. NETWORK/RELATIONSHIP FEATURES")
print("=" * 80)

# Family relationships
has_family = donors_df['Family_ID'].notna()
print(f"   Donors with Family_ID: {has_family.sum():,} ({has_family.sum()/len(donors_df)*100:.1f}%)")
print(f"   Unique Family IDs: {donors_df['Family_ID'].nunique():,}")

# Relationship types in donors table
if 'Relationship_Type' in donors_df.columns:
    print(f"\n   Relationship Types (in donors table):")
    rel_types = donors_df['Relationship_Type'].value_counts()
    for rt, count in rel_types.items():
        if pd.notna(rt):
            pct = count / len(donors_df) * 100
            print(f"      {rt:15s}: {count:6,} ({pct:5.1f}%)")

# Relationships table analysis
print(f"\n   Relationships Table Analysis:")
print(f"   Total relationship records: {len(relationships_df):,}")
if 'Relationship_Type' in relationships_df.columns:
    rel_table_types = relationships_df['Relationship_Type'].value_counts()
    print(f"   Relationship Types:")
    for rt, count in rel_table_types.items():
        pct = count / len(relationships_df) * 100
        print(f"      {rt:15s}: {count:6,} ({pct:5.1f}%)")

if 'Family_ID' in relationships_df.columns:
    print(f"   Unique Family IDs in relationships: {relationships_df['Family_ID'].nunique():,}")

# Additional network insights
print(f"\n   Network Connectivity:")
if 'Family_ID' in donors_df.columns:
    family_sizes = donors_df.groupby('Family_ID').size()
    print(f"   Average family size: {family_sizes.mean():.2f} donors")
    print(f"   Largest family: {family_sizes.max()} donors")
    print(f"   Families with 2+ members: {(family_sizes >= 2).sum():,}")

# 5. ADDITIONAL INSIGHTS
print("\n" + "=" * 80)
print("5. ADDITIONAL DATASET CHARACTERISTICS")
print("=" * 80)

# Prospect stages
if 'Prospect_Stage' in donors_df.columns:
    stages = donors_df['Prospect_Stage'].value_counts()
    print(f"\n   Prospect Stages:")
    for stage, count in stages.items():
        pct = count / len(donors_df) * 100
        print(f"      {stage:15s}: {count:6,} ({pct:5.1f}%)")

# Geographic regions
if 'Geographic_Region' in donors_df.columns:
    regions = donors_df['Geographic_Region'].value_counts()
    print(f"\n   Geographic Regions:")
    for region, count in regions.items():
        pct = count / len(donors_df) * 100
        print(f"      {region:20s}: {count:6,} ({pct:5.1f}%)")

# Giving statistics
if 'Lifetime_Giving' in donors_df.columns:
    print(f"\n   Lifetime Giving Statistics:")
    print(f"      Total lifetime giving: ${donors_df['Lifetime_Giving'].sum():,.2f}")
    print(f"      Average lifetime giving: ${donors_df['Lifetime_Giving'].mean():,.2f}")
    print(f"      Median lifetime giving: ${donors_df['Lifetime_Giving'].median():,.2f}")
    print(f"      Max lifetime giving: ${donors_df['Lifetime_Giving'].max():,.2f}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

