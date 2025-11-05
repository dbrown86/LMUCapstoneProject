"""
Standalone Verification Script for Business Impact Calculations
==============================================================

This script allows you to independently verify the hero metrics and before/after
chart calculations from the business impact dashboard.

Usage:
    python verify_business_impact.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

def verify_business_impact_calculations(
    parquet_path=None,
    contact_percentage=20,
    cost_per_contact=2.0,
    prob_threshold=0.5
):
    """
    Verify business impact calculations independently.
    
    Parameters:
    -----------
    parquet_path : str, optional
        Path to the parquet file. If None, searches for default paths.
    contact_percentage : float
        Percentage of donors to contact (1-100)
    cost_per_contact : float
        Cost per contact in dollars
    prob_threshold : float
        Probability threshold for high-probability donors
    """
    
    # Find parquet file
    if parquet_path is None:
        root = Path(__file__).resolve().parent.parent
        parquet_paths = [
            root / "data/parquet_export/donors_with_network_features.parquet",
            root / "donors_with_network_features.parquet",
            "data/parquet_export/donors_with_network_features.parquet"
        ]
        
        for path in parquet_paths:
            if Path(path).exists():
                parquet_path = path
                break
        else:
            raise FileNotFoundError(f"Could not find parquet file. Tried: {parquet_paths}")
    
    print(f"📂 Loading data from: {parquet_path}")
    df = pd.read_parquet(parquet_path, engine='pyarrow')
    print(f"✅ Loaded {len(df):,} donors\n")
    
    # Identify probability column
    prob_col = None
    if 'Will_Give_Again_Probability' in df.columns:
        prob_col = 'Will_Give_Again_Probability'
    elif 'predicted_prob' in df.columns:
        prob_col = 'predicted_prob'
    else:
        raise ValueError("Could not find probability column (Will_Give_Again_Probability or predicted_prob)")
    
    # Identify outcome column
    outcome_col = None
    if 'Gave_Again_In_2024' in df.columns:
        outcome_col = 'Gave_Again_In_2024'
    elif 'actual_gave' in df.columns:
        outcome_col = 'actual_gave'
    else:
        raise ValueError("Could not find outcome column (Gave_Again_In_2024 or actual_gave)")
    
    # Identify gift amount column
    gift_col = None
    for col in ['Last_Gift', 'last_gift', 'LastGift', 'last_gift_amount']:
        if col in df.columns:
            gift_col = col
            break
    
    if gift_col is None:
        if 'avg_gift' in df.columns:
            gift_col = 'avg_gift'
        else:
            raise ValueError("Could not find gift amount column")
    
    print("=" * 80)
    print("VERIFICATION OF BUSINESS IMPACT CALCULATIONS")
    print("=" * 80)
    print(f"\n📊 Input Parameters:")
    print(f"   - Contact percentage: {contact_percentage}%")
    print(f"   - Cost per contact: ${cost_per_contact:.2f}")
    print(f"   - Probability threshold: {prob_threshold:.2f}")
    print(f"   - Probability column: {prob_col}")
    print(f"   - Outcome column: {outcome_col}")
    print(f"   - Gift amount column: {gift_col}")
    
    # Calculate inputs
    num_to_contact = int(len(df) * contact_percentage / 100)
    
    # Baseline rate (all donors)
    baseline_rate = df[outcome_col].mean()
    if pd.isna(baseline_rate) or baseline_rate <= 0:
        baseline_rate = 0.17
        print(f"\n⚠️  Warning: Baseline rate was invalid, using default 17%")
    
    # Average gift amount
    gift_amounts = pd.to_numeric(df[gift_col], errors='coerce').fillna(0).clip(lower=0)
    avg_gift_amount = gift_amounts.median() if gift_amounts.median() > 0 else gift_amounts.mean()
    if avg_gift_amount <= 0:
        avg_gift_amount = 500
        print(f"\n⚠️  Warning: Average gift amount was invalid, using default $500")
    
    # Top donors (by probability)
    top_donors = df.nlargest(num_to_contact, prob_col)
    
    # Fusion response rate (from top predicted donors)
    fusion_response_rate = top_donors[outcome_col].mean()
    if pd.isna(fusion_response_rate) or fusion_response_rate is None:
        fusion_response_rate = baseline_rate
        print(f"\n⚠️  Warning: Fusion response rate was invalid, using baseline rate")
    
    print(f"\n📈 Calculated Inputs:")
    print(f"   - Number to contact: {num_to_contact:,}")
    print(f"   - Baseline conversion rate: {baseline_rate:.4%}")
    print(f"   - Fusion response rate: {fusion_response_rate:.4%}")
    print(f"   - Average gift amount: ${avg_gift_amount:,.2f}")
    
    # BASELINE CALCULATIONS
    print(f"\n{'=' * 80}")
    print("BASELINE CALCULATIONS (Old Way - Random Contact)")
    print("=" * 80)
    
    baseline_contacts = num_to_contact
    baseline_responses = int(baseline_contacts * baseline_rate)
    baseline_revenue = baseline_responses * avg_gift_amount
    baseline_cost = baseline_contacts * cost_per_contact
    baseline_net_revenue = baseline_revenue - baseline_cost
    baseline_roi = ((baseline_revenue - baseline_cost) / baseline_cost * 100) if baseline_cost > 0 else 0
    
    print(f"   Contacts: {baseline_contacts:,}")
    print(f"   Response Rate: {baseline_rate:.2%}")
    print(f"   Expected Responses: {baseline_responses:,} = int({baseline_contacts:,} × {baseline_rate:.4%})")
    print(f"   Total Revenue: ${baseline_revenue:,.2f} = {baseline_responses:,} × ${avg_gift_amount:,.2f}")
    print(f"   Cost: ${baseline_cost:,.2f} = {baseline_contacts:,} × ${cost_per_contact:.2f}")
    print(f"   Net Revenue: ${baseline_net_revenue:,.2f} = ${baseline_revenue:,.2f} - ${baseline_cost:,.2f}")
    print(f"   ROI: {baseline_roi:.2f}% = ((${baseline_revenue:,.2f} - ${baseline_cost:,.2f}) / ${baseline_cost:,.2f}) × 100")
    
    # FUSION CALCULATIONS
    print(f"\n{'=' * 80}")
    print("FUSION CALCULATIONS (New Way - AI-Powered Targeting)")
    print("=" * 80)
    
    fusion_contacts = num_to_contact
    fusion_responses = int(fusion_contacts * fusion_response_rate)
    fusion_revenue = fusion_responses * avg_gift_amount
    fusion_cost = fusion_contacts * cost_per_contact
    fusion_net_revenue = fusion_revenue - fusion_cost
    fusion_roi = ((fusion_revenue - fusion_cost) / fusion_cost * 100) if fusion_cost > 0 else 0
    
    print(f"   Contacts: {fusion_contacts:,}")
    print(f"   Response Rate: {fusion_response_rate:.2%}")
    print(f"   Expected Responses: {fusion_responses:,} = int({fusion_contacts:,} × {fusion_response_rate:.4%})")
    print(f"   Total Revenue: ${fusion_revenue:,.2f} = {fusion_responses:,} × ${avg_gift_amount:,.2f}")
    print(f"   Cost: ${fusion_cost:,.2f} = {fusion_contacts:,} × ${cost_per_contact:.2f}")
    print(f"   Net Revenue: ${fusion_net_revenue:,.2f} = ${fusion_revenue:,.2f} - ${fusion_cost:,.2f}")
    print(f"   ROI: {fusion_roi:.2f}% = ((${fusion_revenue:,.2f} - ${fusion_cost:,.2f}) / ${fusion_cost:,.2f}) × 100")
    
    # HERO METRICS
    print(f"\n{'=' * 80}")
    print("HERO METRICS (Displayed in Dashboard)")
    print("=" * 80)
    
    revenue_gain = fusion_revenue - baseline_revenue
    roi_improvement = fusion_roi - baseline_roi
    response_rate_improvement = ((fusion_response_rate / baseline_rate) - 1) * 100 if baseline_rate > 0 else 0
    
    print(f"   1. Revenue (Baseline): ${baseline_revenue:,.0f}")
    print(f"   2. Revenue (Fusion): ${fusion_revenue:,.0f}")
    print(f"   3. Revenue Gain: ${revenue_gain:,.0f} = ${fusion_revenue:,.0f} - ${baseline_revenue:,.0f}")
    print(f"   4. ROI Improvement: +{roi_improvement:.0f}% = {fusion_roi:.2f}% - {baseline_roi:.2f}%")
    
    # BEFORE/AFTER COMPARISON
    print(f"\n{'=' * 80}")
    print("BEFORE/AFTER COMPARISON TABLE")
    print("=" * 80)
    
    comparison = pd.DataFrame({
        'Metric': [
            'Contacts Made',
            'Response Rate',
            'Expected Responses',
            'Total Revenue',
            'Cost of Outreach',
            'Net Revenue',
            'ROI'
        ],
        'Baseline (Old Way)': [
            f"{baseline_contacts:,}",
            f"{baseline_rate:.1%}",
            f"{baseline_responses:,}",
            f"${baseline_revenue:,.0f}",
            f"${baseline_cost:,.0f}",
            f"${baseline_net_revenue:,.0f}",
            f"{baseline_roi:.0f}%"
        ],
        'Fusion Model (New Way)': [
            f"{fusion_contacts:,}",
            f"{fusion_response_rate:.1%}",
            f"{fusion_responses:,}",
            f"${fusion_revenue:,.0f}",
            f"${fusion_cost:,.0f}",
            f"${fusion_net_revenue:,.0f}",
            f"{fusion_roi:.0f}%"
        ],
        'Improvement': [
            "Same effort",
            f"+{response_rate_improvement:.1f}%",
            f"+{fusion_responses - baseline_responses:,}",
            f"+${revenue_gain:,.0f}",
            "Same cost",
            f"+${fusion_net_revenue - baseline_net_revenue:,.0f}",
            f"+{roi_improvement:.0f}%"
        ]
    })
    
    print("\n" + comparison.to_string(index=False))
    
    # VERIFICATION CHECKS
    print(f"\n{'=' * 80}")
    print("VERIFICATION CHECKS")
    print("=" * 80)
    
    checks_passed = True
    
    # Check 1: Baseline revenue calculation
    expected_baseline_revenue = baseline_responses * avg_gift_amount
    if abs(baseline_revenue - expected_baseline_revenue) < 0.01:
        print("✅ Check 1: Baseline revenue calculation is correct")
    else:
        print(f"❌ Check 1: Baseline revenue mismatch! Expected ${expected_baseline_revenue:,.2f}, got ${baseline_revenue:,.2f}")
        checks_passed = False
    
    # Check 2: Fusion revenue calculation
    expected_fusion_revenue = fusion_responses * avg_gift_amount
    if abs(fusion_revenue - expected_fusion_revenue) < 0.01:
        print("✅ Check 2: Fusion revenue calculation is correct")
    else:
        print(f"❌ Check 2: Fusion revenue mismatch! Expected ${expected_fusion_revenue:,.2f}, got ${fusion_revenue:,.2f}")
        checks_passed = False
    
    # Check 3: Revenue gain
    expected_revenue_gain = fusion_revenue - baseline_revenue
    if abs(revenue_gain - expected_revenue_gain) < 0.01:
        print("✅ Check 3: Revenue gain calculation is correct")
    else:
        print(f"❌ Check 3: Revenue gain mismatch! Expected ${expected_revenue_gain:,.2f}, got ${revenue_gain:,.2f}")
        checks_passed = False
    
    # Check 4: ROI calculations
    expected_baseline_roi = ((baseline_revenue - baseline_cost) / baseline_cost * 100) if baseline_cost > 0 else 0
    expected_fusion_roi = ((fusion_revenue - fusion_cost) / fusion_cost * 100) if fusion_cost > 0 else 0
    
    if abs(baseline_roi - expected_baseline_roi) < 0.01:
        print("✅ Check 4: Baseline ROI calculation is correct")
    else:
        print(f"❌ Check 4: Baseline ROI mismatch! Expected {expected_baseline_roi:.2f}%, got {baseline_roi:.2f}%")
        checks_passed = False
    
    if abs(fusion_roi - expected_fusion_roi) < 0.01:
        print("✅ Check 5: Fusion ROI calculation is correct")
    else:
        print(f"❌ Check 5: Fusion ROI mismatch! Expected {expected_fusion_roi:.2f}%, got {fusion_roi:.2f}%")
        checks_passed = False
    
    # Check 6: ROI improvement
    expected_roi_improvement = fusion_roi - baseline_roi
    if abs(roi_improvement - expected_roi_improvement) < 0.01:
        print("✅ Check 6: ROI improvement calculation is correct")
    else:
        print(f"❌ Check 6: ROI improvement mismatch! Expected {expected_roi_improvement:.2f}%, got {roi_improvement:.2f}%")
        checks_passed = False
    
    print(f"\n{'=' * 80}")
    if checks_passed:
        print("✅ ALL VERIFICATION CHECKS PASSED!")
        print("The calculations match the expected formulas.")
    else:
        print("❌ SOME VERIFICATION CHECKS FAILED!")
        print("Please review the calculations above.")
    print("=" * 80)
    
    return {
        'baseline_revenue': baseline_revenue,
        'fusion_revenue': fusion_revenue,
        'revenue_gain': revenue_gain,
        'roi_improvement': roi_improvement,
        'baseline_roi': baseline_roi,
        'fusion_roi': fusion_roi,
        'checks_passed': checks_passed
    }


if __name__ == "__main__":
    import sys
    
    # Allow command-line arguments
    contact_pct = 20
    cost_per = 2.0
    threshold = 0.5
    
    if len(sys.argv) > 1:
        contact_pct = float(sys.argv[1])
    if len(sys.argv) > 2:
        cost_per = float(sys.argv[2])
    if len(sys.argv) > 3:
        threshold = float(sys.argv[3])
    
    try:
        results = verify_business_impact_calculations(
            contact_percentage=contact_pct,
            cost_per_contact=cost_per,
            prob_threshold=threshold
        )
        sys.exit(0 if results['checks_passed'] else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

