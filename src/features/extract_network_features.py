#!/usr/bin/env python3
"""
Extract Network Features from Relationship Data
Converts graph structure into numerical features for tabular model
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def extract_relationship_features(donors_df, relationships_df):
    """
    Extract comprehensive relationship features for each donor
    """
    print("\n📊 Extracting relationship features...")
    
    # 1. BASIC NETWORK METRICS
    print("   Computing basic network metrics...")
    outgoing = relationships_df.groupby('Donor_ID_1').size().rename('outgoing_relationships')
    incoming = relationships_df.groupby('Donor_ID_2').size().rename('incoming_relationships')
    
    total_degree = pd.DataFrame({
        'total_relationships': outgoing.add(incoming, fill_value=0)
    })
    
    # 2. RELATIONSHIP DIVERSITY
    print("   Computing relationship diversity...")
    relationship_diversity = relationships_df.groupby('Donor_ID_1')['Relationship_Type'].nunique().rename('relationship_diversity')
    
    # 3. RELATIONSHIP STRENGTH
    print("   Computing relationship strength metrics...")
    if 'Strength' in relationships_df.columns:
        avg_strength = relationships_df.groupby('Donor_ID_1')['Strength'].mean().rename('avg_relationship_strength')
        max_strength = relationships_df.groupby('Donor_ID_1')['Strength'].max().rename('max_relationship_strength')
    else:
        avg_strength = pd.Series(dtype=float, name='avg_relationship_strength')
        max_strength = pd.Series(dtype=float, name='max_relationship_strength')
    
    # 4. RELATIONSHIP CATEGORIES
    print("   Computing relationship categories...")
    
    # Professional relationships
    professional_rels = relationships_df[
        relationships_df['Relationship_Type'].isin([
            'Colleague', 'Business_Partner', 'Industry_Peer', 'Mentor_Mentee'
        ])
    ].groupby('Donor_ID_1').size().rename('professional_network_size')
    
    # Social relationships
    social_rels = relationships_df[
        relationships_df['Relationship_Type'].isin([
            'Mutual_Friend', 'Social_Connection', 'Neighbor'
        ])
    ].groupby('Donor_ID_1').size().rename('social_network_size')
    
    # Alumni relationships
    alumni_rels = relationships_df[
        relationships_df['Relationship_Type'].isin([
            'Classmate', 'Same_Major', 'Dorm_Mate', 'Greek_Life', 'Study_Abroad', 'Athletic_Team'
        ])
    ].groupby('Donor_ID_1').size().rename('alumni_network_size')
    
    # Family relationships
    family_rels = relationships_df[
        relationships_df['Relationship_Type'].isin([
            'Spouse', 'Parent', 'Child', 'Head'
        ])
    ].groupby('Donor_ID_1').size().rename('family_network_size')
    
    # Philanthropic relationships
    philanthropic_rels = relationships_df[
        relationships_df['Relationship_Type'].isin([
            'Board_Member', 'Committee_Member', 'Volunteer_Partner', 
            'Fundraising_Team', 'Campaign_Participant'
        ])
    ].groupby('Donor_ID_1').size().rename('philanthropic_network_size')
    
    # Geographic relationships
    geographic_rels = relationships_df[
        relationships_df['Relationship_Type'].isin([
            'Same_City', 'Same_Region', 'Campus_Proximity'
        ])
    ].groupby('Donor_ID_1').size().rename('geographic_network_size')
    
    # 5. NETWORK GIVING POTENTIAL
    print("   Computing network giving metrics...")
    if 'Lifetime_Giving' in donors_df.columns:
        donor_giving = donors_df[['ID', 'Lifetime_Giving']].set_index('ID')
        
        network_giving = relationships_df.merge(
            donor_giving,
            left_on='Donor_ID_2',
            right_index=True,
            how='left'
        ).groupby('Donor_ID_1')['Lifetime_Giving'].agg([
            ('network_avg_giving', 'mean'),
            ('network_max_giving', 'max'),
            ('network_total_giving', 'sum')
        ])
        
        # High-value connections
        high_value_threshold = donors_df['Lifetime_Giving'].quantile(0.8)
        high_value_donors = set(donors_df[donors_df['Lifetime_Giving'] >= high_value_threshold]['ID'])
        
        high_value_connections = relationships_df[
            relationships_df['Donor_ID_2'].isin(high_value_donors)
        ].groupby('Donor_ID_1').size().rename('high_value_connections')
    else:
        network_giving = pd.DataFrame()
        high_value_connections = pd.Series(dtype=float, name='high_value_connections')
    
    # 6. COMBINE ALL FEATURES
    print("   Combining features...")
    relationship_features = pd.concat([
        total_degree,
        outgoing,
        incoming,
        relationship_diversity,
        avg_strength,
        max_strength,
        professional_rels,
        social_rels,
        alumni_rels,
        family_rels,
        philanthropic_rels,
        geographic_rels,
        network_giving,
        high_value_connections
    ], axis=1).fillna(0)
    
    # Merge with donors
    donors_with_network = donors_df.merge(
        relationship_features,
        left_on='ID',
        right_index=True,
        how='left'
    )
    
    # Fill NaN for donors with no relationships
    network_cols = relationship_features.columns
    donors_with_network[network_cols] = donors_with_network[network_cols].fillna(0)
    
    print(f"   ✅ Added {len(network_cols)} relationship features")
    
    return donors_with_network, network_cols.tolist()


def main():
    print("🔗 EXTRACTING NETWORK FEATURES FROM RELATIONSHIPS")
    print("=" * 60)
    
    # Load data
    parquet_dir = Path("data/parquet_export")
    
    print("\n📂 Loading data...")
    donors_df = pd.read_parquet(parquet_dir / 'donors_with_features.parquet')
    relationships_df = pd.read_parquet(parquet_dir / 'relationships.parquet')
    
    print(f"   Donors: {len(donors_df):,}")
    print(f"   Relationships: {len(relationships_df):,}")
    print(f"   Original features: {len(donors_df.columns)}")
    
    # Extract features
    donors_with_network, network_features = extract_relationship_features(donors_df, relationships_df)
    
    # Save enhanced dataset
    output_file = parquet_dir / 'donors_with_network_features.parquet'
    donors_with_network.to_parquet(output_file, index=False, compression='snappy')
    
    print(f"\n💾 Saved to {output_file}")
    print(f"   Total features: {len(donors_with_network.columns)}")
    print(f"   New network features: {len(network_features)}")
    print(f"   File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Show statistics
    print("\n📊 Network Feature Statistics:")
    print("=" * 60)
    
    for feature in network_features[:15]:  # Show first 15
        values = donors_with_network[feature]
        non_zero = (values > 0).sum()
        print(f"{feature:30s}: mean={values.mean():8.2f}, max={values.max():8.0f}, non-zero={non_zero:6,} ({non_zero/len(values)*100:5.1f}%)")
    
    if len(network_features) > 15:
        print(f"... and {len(network_features) - 15} more features")
    
    # Show sample
    print("\n📋 Sample of network features:")
    print(donors_with_network[['ID'] + network_features[:10]].head(10))
    
    print("\n✅ Network feature extraction complete!")
    print(f"\n💡 Next step: Use 'donors_with_network_features.parquet' for training")
    print(f"   This file includes all original features + {len(network_features)} network features")


if __name__ == "__main__":
    main()






