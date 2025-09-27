# Data validation and visualization functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

def validate_dataset(donors_df, relationships_df, contact_reports_df, giving_history_df):
    """Perform comprehensive data quality validation"""
    print("Performing data quality validation...")
    
    validation_results = {
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # 1. Check for duplicate IDs 
    if donors_df['ID'].duplicated().any():
        validation_results['errors'].append("Duplicate donor IDs found")
    
    # 2. Validate giving totals 
    if not giving_history_df.empty:
        giving_summary = giving_history_df.groupby('Donor_ID')['Gift_Amount'].sum()
        lifetime_giving = donors_df.set_index('ID')['Lifetime_Giving']
        
        mismatches = 0
        for donor_id in giving_summary.index:
            if donor_id in lifetime_giving.index:
                expected = lifetime_giving[donor_id]
                actual = giving_summary[donor_id]
                if abs(expected - actual) > 1.0:  # Allow for rounding differences
                    mismatches += 1
        
        if mismatches > 0:
            validation_results['warnings'].append(f"{mismatches} donors have giving total mismatches")
    
    # 3. Check logical constraints 
    invalid_last_gift = (donors_df['Last_Gift'] > donors_df['Lifetime_Giving']).sum()
    if invalid_last_gift > 0:
        validation_results['errors'].append(f"{invalid_last_gift} donors have last gift > lifetime giving")
    
    invalid_consecutive = (donors_df['Consecutive_Yr_Giving_Count'] > donors_df['Total_Yr_Giving_Count']).sum()
    if invalid_consecutive > 0:
        validation_results['errors'].append(f"{invalid_consecutive} donors have consecutive > total years")
    
    # 4. Check date ranges 
    if not giving_history_df.empty:
        future_gifts = giving_history_df[giving_history_df['Gift_Date'] > date.today()]
        if len(future_gifts) > 0:
            validation_results['errors'].append(f"{len(future_gifts)} gifts dated in the future")
    
    # 5. Validate family relationships 
    family_donors = donors_df['Family_ID'].notna().sum()
    relationship_records = len(relationships_df)
    
    if family_donors != relationship_records:
        validation_results['warnings'].append(f"Family field count ({family_donors}) doesn't match relationship records ({relationship_records})")
    
    # 6. Check family consistency 
    if not relationships_df.empty:
        # Every family should have at least 2 members
        family_sizes = relationships_df.groupby('Family_ID').size()
        single_member_families = (family_sizes == 1).sum()
        if single_member_families > 0:
            validation_results['warnings'].append(f"{single_member_families} families have only one member")
        
        # Check for orphaned family IDs
        donor_family_ids = set(donors_df['Family_ID'].dropna())
        relationship_family_ids = set(relationships_df['Family_ID'])
        if donor_family_ids != relationship_family_ids:
            validation_results['warnings'].append("Mismatched Family IDs between donors and relationships tables")
    
    # 7. Validate contact report consistency 
    if not contact_reports_df.empty:
        # Check for contact reports with invalid donor IDs
        invalid_contact_donors = set(contact_reports_df['Donor_ID']) - set(donors_df['ID'])
        if invalid_contact_donors:
            validation_results['errors'].append(f"{len(invalid_contact_donors)} contact reports reference non-existent donors")
        
        # Check contact report dates vs donor activity
        future_contacts = contact_reports_df[contact_reports_df['Contact_Date'] > date.today()]
        if len(future_contacts) > 0:
            validation_results['warnings'].append(f"{len(future_contacts)} contact reports dated in the future")
    
    # 8. Validate giving history consistency 
    if not giving_history_df.empty:
        # Check for gifts with invalid donor IDs
        invalid_gift_donors = set(giving_history_df['Donor_ID']) - set(donors_df['ID'])
        if invalid_gift_donors:
            validation_results['errors'].append(f"{len(invalid_gift_donors)} gifts reference non-existent donors")
        
        # Check for gifts to non-donors
        non_donor_ids = set(donors_df[donors_df['Lifetime_Giving'] == 0]['ID'])
        gifts_to_non_donors = giving_history_df[giving_history_df['Donor_ID'].isin(non_donor_ids)]
        if len(gifts_to_non_donors) > 0:
            validation_results['errors'].append(f"{len(gifts_to_non_donors)} gifts attributed to non-donors")
    
    # 9. Class year validation 
    invalid_class_years = donors_df[
        (donors_df['Class_Year'].notna()) & 
        ((donors_df['Class_Year'] < 1950) | (donors_df['Class_Year'] > 2025))
    ]
    if len(invalid_class_years) > 0:
        validation_results['warnings'].append(f"{len(invalid_class_years)} donors have implausible class years")
    
    # 10. Rating distribution validation 
    rating_counts = donors_df['Rating'].value_counts()
    total_donors = len(donors_df)
    
    # Check if rating distribution is reasonable (A-ratings should be rare)
    high_ratings = rating_counts.get('A', 0) + rating_counts.get('B', 0)
    if high_ratings / total_donors > 0.01:  # More than 1% in top ratings
        validation_results['warnings'].append(f"Unusually high percentage ({high_ratings/total_donors*100:.1f}%) of top-rated donors")
    
    # 11. Statistical consistency checks 
    # Check if non-donors have any giving-related fields filled incorrectly
    non_donors = donors_df[donors_df['Lifetime_Giving'] == 0]
    non_donors_with_last_gift = non_donors[non_donors['Last_Gift'] > 0]
    if len(non_donors_with_last_gift) > 0:
        validation_results['errors'].append(f"{len(non_donors_with_last_gift)} non-donors have last gift amounts")
    
    non_donors_with_dates = non_donors[non_donors['Last_Gift_Date'].notna()]
    if len(non_donors_with_dates) > 0:
        validation_results['errors'].append(f"{len(non_donors_with_dates)} non-donors have gift dates")
    
    # 12. Enhanced statistics 
    validation_results['stats'] = {
        'total_donors': len(donors_df),
        'non_donors': (donors_df['Lifetime_Giving'] == 0).sum(),
        'major_donors_100k': (donors_df['Lifetime_Giving'] > 100000).sum(),
        'donors_in_families': family_donors,
        'unique_families': donors_df['Family_ID'].nunique(),
        'avg_family_size': family_donors / donors_df['Family_ID'].nunique() if donors_df['Family_ID'].nunique() > 0 else 0,
        'contact_report_coverage': len(contact_reports_df) / len(donors_df) * 100 if not contact_reports_df.empty else 0,
        'total_gifts': len(giving_history_df) if not giving_history_df.empty else 0,
        'total_giving': donors_df['Lifetime_Giving'].sum(),
        'median_giving': donors_df[donors_df['Lifetime_Giving'] > 0]['Lifetime_Giving'].median() if (donors_df['Lifetime_Giving'] > 0).sum() > 0 else 0,
        'rating_distribution': donors_df['Rating'].value_counts().to_dict(),
        'constituent_distribution': donors_df['Primary_Constituent_Type'].value_counts().to_dict(),
        'prospect_stage_distribution': donors_df['Prospect_Stage'].value_counts().to_dict(),
        'alignment_issues': donors_df['Rating_Giving_Alignment'].value_counts().to_dict(),
        'class_year_range': (donors_df['Class_Year'].min(), donors_df['Class_Year'].max()) if donors_df['Class_Year'].notna().sum() > 0 else (None, None),
        'data_completeness': {
            'class_year_filled': donors_df['Class_Year'].notna().sum() / len(donors_df) * 100,
            'secondary_role_filled': donors_df['Constituent_Type_2'].notna().sum() / len(donors_df) * 100,
            'family_relationships': family_donors / len(donors_df) * 100
        }
    }
    
    return validation_results

def create_dataset_visualizations(donors_df, giving_history_df, contact_reports_df, output_dir):
    """Create visualizations for dataset analysis"""
    print("Creating dataset visualizations...")
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Synthetic Donor Dataset Analysis', fontsize=16, fontweight='bold')
    
    # 1. Rating Distribution
    rating_counts = donors_df['Rating'].value_counts().sort_index()
    
    # Create labels with rating bands for better visualization
    rating_labels = []
    rating_bands = {
        'A': '$100M+', 'B': '$50M-99.9M', 'C': '$25M-49.9M', 'D': '$10M-24.9M',
        'E': '$5M-9.9M', 'F': '$1M-4.9M', 'G': '$500K-999.9K', 'H': '$250K-499.9K',
        'I': '$100K-249.9K', 'J': '$50K-99.9K', 'K': '$25K-49.9K', 'L': '$10K-24.9K',
        'M': '$5K-9.9K', 'N': '$2.5K-4.9K', 'O': '$1K-2.4K', 'P': '<$1K'
    }
    
    for rating in rating_counts.index:
        rating_labels.append(f"{rating}\n{rating_bands[rating]}")
    
    axes[0,0].bar(range(len(rating_counts)), rating_counts.values, color='skyblue')
    axes[0,0].set_title('Wealth Rating Distribution')
    axes[0,0].set_xlabel('Rating (Wealth Capacity Bands)')
    axes[0,0].set_ylabel('Number of Donors')
    axes[0,0].set_xticks(range(len(rating_counts)))
    axes[0,0].set_xticklabels(rating_labels, rotation=45, ha='right', fontsize=8)
    
    # 2. Giving vs Non-Giving
    giving_status = ['Non-Donors', 'Donors']
    giving_counts = [(donors_df['Lifetime_Giving'] == 0).sum(), (donors_df['Lifetime_Giving'] > 0).sum()]
    axes[0,1].pie(giving_counts, labels=giving_status, autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'])
    axes[0,1].set_title('Donor vs Non-Donor Distribution')
    
    # 3. Lifetime Giving Distribution (log scale)
    giving_donors = donors_df[donors_df['Lifetime_Giving'] > 0]['Lifetime_Giving']
    axes[0,2].hist(np.log10(giving_donors), bins=30, color='gold', alpha=0.7)
    axes[0,2].set_title('Lifetime Giving Distribution (Log Scale)')
    axes[0,2].set_xlabel('Log10(Lifetime Giving)')
    axes[0,2].set_ylabel('Frequency')
    
    # 4. Constituent Type Distribution
    const_counts = donors_df['Primary_Constituent_Type'].value_counts()
    axes[1,0].bar(const_counts.index, const_counts.values, color='lightblue')
    axes[1,0].set_title('Constituent Type Distribution')
    axes[1,0].set_xlabel('Constituent Type')
    axes[1,0].set_ylabel('Number of Donors')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 5. Contact Report Outcomes
    if not contact_reports_df.empty:
        outcome_counts = contact_reports_df['Outcome_Category'].value_counts()
        axes[1,1].bar(outcome_counts.index, outcome_counts.values, 
                     color=['green', 'red', 'orange'])
        axes[1,1].set_title('Contact Report Outcomes')
        axes[1,1].set_xlabel('Outcome Type')
        axes[1,1].set_ylabel('Number of Reports')
    else:
        axes[1,1].text(0.5, 0.5, 'No Contact Reports', ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Contact Report Outcomes')
    
    # 6. Giving by Year (if giving history exists)
    if not giving_history_df.empty:
        giving_history_df['Year'] = pd.to_datetime(giving_history_df['Gift_Date']).dt.year
        yearly_giving = giving_history_df.groupby('Year')['Gift_Amount'].sum()
        axes[1,2].plot(yearly_giving.index, yearly_giving.values, marker='o', color='purple')
        axes[1,2].set_title('Total Giving by Year')
        axes[1,2].set_xlabel('Year')
        axes[1,2].set_ylabel('Total Giving ($)')
        axes[1,2].tick_params(axis='x', rotation=45)
    else:
        axes[1,2].text(0.5, 0.5, 'No Giving History', ha='center', va='center', transform=axes[1,2].transAxes)
        axes[1,2].set_title('Total Giving by Year')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary statistics table
    print("\n" + "="*60)
    print("DATASET SUMMARY STATISTICS")
    print("="*60)
    
    summary_stats = {
        'Total Records': len(donors_df),
        'Non-Donors': (donors_df['Lifetime_Giving'] == 0).sum(),
        'Donors': (donors_df['Lifetime_Giving'] > 0).sum(),
        'Major Donors (>$100K)': (donors_df['Lifetime_Giving'] > 100000).sum(),
        'Total Lifetime Giving': f"${donors_df['Lifetime_Giving'].sum():,.2f}",
        'Average Lifetime Giving': f"${donors_df['Lifetime_Giving'].mean():,.2f}",
        'Median Lifetime Giving': f"${donors_df['Lifetime_Giving'].median():,.2f}",
        'Contact Reports': len(contact_reports_df),
        'Gift Transactions': len(giving_history_df),
    }
    
    for key, value in summary_stats.items():
        print(f"{key:.<30} {value:>25}")
    
    print("\nRating Distribution:")
    
    # Rating band definitions
    rating_bands = {
        'A': '$100M+',
        'B': '$50M - $99.9M',
        'C': '$25M - $49.9M',
        'D': '$10M - $24.9M',
        'E': '$5M - $9.9M',
        'F': '$1M - $4.9M',
        'G': '$500K - $999.9K',
        'H': '$250K - $499.9K',
        'I': '$100K - $249.9K',
        'J': '$50K - $99.9K',
        'K': '$25K - $49.9K',
        'L': '$10K - $24.9K',
        'M': '$5K - $9.9K',
        'N': '$2.5K - $4.9K',
        'O': '$1K - $2.4K',
        'P': 'Less than $1K'
    }
    
    for rating in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']:
        count = (donors_df['Rating'] == rating).sum()
        percentage = count / len(donors_df) * 100
        band = rating_bands[rating]
        print(f"  {rating} ({band:.<20}): {count:,} ({percentage:.1f}%)")
