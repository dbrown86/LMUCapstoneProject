# Data generation module for synthetic donor dataset
import pandas as pd
import numpy as np
import random
# from faker import Faker  # Not used - commented out to prevent hanging
from datetime import date, datetime, timedelta
import os
from collections import defaultdict
# import matplotlib  # Not used - commented out to prevent hanging
# matplotlib.use('Agg')  # Use non-GUI backend (prevents hanging)
# import matplotlib.pyplot as plt  # Not used - commented out to prevent hanging
# import seaborn as sns  # Not used - commented out to prevent hanging
from tqdm import tqdm  # Use regular tqdm (not notebook version)
import warnings
warnings.filterwarnings('ignore')

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

    def generate_class_year(self, constituent_type, current_year=2025, rng=None):
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

class GivingGenerator:
    def __init__(self):
        self.designations = [
            'Annual Fund', 'Scholarships', 'Men\'s Basketball', 'Library',
            'Endowment Fund', 'Capital Campaign', 'Faculty Support',
             'Greek Life', 'Theatre Arts', 'Aquatic Center',
            'DEI Initiatives', 'Women\'s Volleyball', 'Alumni Association', 'Study Abroad',
            'Men\'s Soccer', 'Women\'s Soccer', 'Student Health Center', 'Community Service Center',
            'Finance Club', 'Engineering Department', 'Club Sports', 'Career Development Center',
            'Veterans Center', 'Music Department', 'History Department', 'Debate Team', 
            'Computer Science Department'
        ]
    
    def generate_lifetime_giving(self, rng):
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
        
        # Gift date (weighted toward recent years, but not in future)
        years = list(range(1980, 2026))
        weights = [1 if year < 2010 else (year - 2005) for year in years]
        # Filter out future years
        valid_years = [y for y in years if y <= 2025]
        valid_weights = [w for y, w in zip(years, weights) if y <= 2025]
        gift_year = random.choices(valid_years, weights=valid_weights)[0]
        
        start_date = date(gift_year, 1, 1)
        # Do not allow future dates
        year_end = date(gift_year, 12, 31)
        today = date.today()
        end_date = min(year_end, today)
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

class ContactReportGenerator:
    def __init__(self):
        self.subjects = [
            'scholarship support', 'capital campaign', 'annual giving',
            'planned giving', 'faculty chair', 'building project',
            'student programs', 'research funding', 'athletics',
            'library expansion', 'endowment fund'
        ]
        
        self.positive_outcomes = [
            'expressed strong interest', 'requested proposal',
            'wants to think about it', 'indicated support',
            'asked for more information', 'committed to gift',
            'interested in legacy giving', 'will discuss with spouse'
        ]
        
        self.negative_outcomes = [
            'declined to support at this time',
            'not interested in this initiative',
            'unable to make a commitment this year',
            'prefers to support other organizations',
            'cited economic concerns',
            'does not wish to be contacted further',
            'has other philanthropic priorities',
            'feels disconnected from the university',
            'requested removal from solicitation lists',
            'expressed disappointment with recent university decisions'
        ]
        
        self.unresponsive_outcomes = [
            'has not returned multiple phone calls',
            'did not respond to email outreach',
            'was unreachable for scheduled meeting',
            'has been unresponsive to recent communications',
            'missed scheduled phone appointment without notice',
            'did not reply to invitation for campus visit',
            'has not responded to follow-up attempts',
            'was unable to connect despite multiple attempts',
            'requested to reschedule but did not follow through',
            'phone calls have gone unanswered'
        ]
        
        self.templates = {
            'positive_meeting': [
                "Met with {donor_name} at {location} to discuss {subject}. {donor_name} {outcome}.",
                "Had lunch meeting with {donor_name} regarding {subject}. Discussion focused on {focus_area}. {donor_name} {outcome}.",
                "Office visit with {donor_name} to review {subject} proposal. {donor_name} asked questions about {detail} and {outcome}."
            ],
            'positive_phone': [
                "Phone call with {donor_name} to discuss {subject}. Conversation lasted {duration} minutes. {donor_name} {outcome}.",
                "Brief check-in call with {donor_name}. Discussed {subject}. {donor_name} {outcome}."
            ],
            'positive_event': [
                "Spoke with {donor_name} at {event_name}. Discussed {subject}. {donor_name} {outcome}.",
                "Connected with {donor_name} during {event_name}. Brief conversation about {subject}. {donor_name} {outcome}."
            ],
            'solicitation': [
                "Formal solicitation meeting with {donor_name} for ${amount} gift to {subject}. {donor_name} {outcome}.",
                "Presented {subject} proposal to {donor_name}. Request amount: ${amount}. {donor_name} {outcome}."
            ],
            'stewardship': [
                "Stewardship call with {donor_name} to thank for recent gift to {subject}. {donor_name} {outcome}.",
                "Sent impact report to {donor_name} regarding their {subject} support. {donor_name} {outcome}."
            ],
            'negative': [
                "Met with {donor_name} to discuss {subject}. Unfortunately, {donor_name} {outcome}.",
                "Phone conversation with {donor_name} about potential support for {subject}. {donor_name} {outcome}.",
                "Solicitation meeting with {donor_name} for {subject} support. {donor_name} {outcome}. Will respect their decision."
            ],
            'unresponsive': [
                "Attempted to contact {donor_name} regarding {subject}. {donor_name} {outcome}. Will try alternative approach.",
                "Multiple outreach attempts to {donor_name} about {subject}. {donor_name} {outcome}. Considering pause in contact.",
                "Follow-up with {donor_name} on {subject} proposal. {donor_name} {outcome}. May need to reassess approach."
            ]
        }
    
    def should_have_contact_report(self, constituent_type, prospect_stage, lifetime_giving):
        """Determine probability of having a contact report"""
        base_probability = 0.3  # 30% base chance
        
        # Adjust based on constituent type
        if constituent_type in ['Trustee', 'Regent']:
            base_probability += 0.6
        elif constituent_type in ['Alum', 'Parent']:
            base_probability += 0.2
        elif constituent_type == 'Friend':
            base_probability += 0.1
        
        # Adjust based on prospect stage
        stage_multipliers = {
            'Identification': 0.5, 'Qualification': 1.2,
            'Cultivation': 1.8, 'Solicitation': 2.0, 'Stewardship': 1.5
        }
        base_probability *= stage_multipliers.get(prospect_stage, 1.0)
        
        # Donors more likely to have contact reports
        if lifetime_giving > 0:
            base_probability += 0.3
        if lifetime_giving > 100000:
            base_probability += 0.4
        
        return min(0.95, base_probability)
    
    def generate_report(self, donor_name, prospect_stage, rating, lifetime_giving, constituent_type):
        """Generate contact report with realistic outcomes"""
        # Determine outcome probabilities
        negative_prob = 0.20
        unresponsive_prob = 0.15
        
        if lifetime_giving == 0:
            negative_prob += 0.25
            unresponsive_prob += 0.20
        
        if prospect_stage in ['Identification', 'Qualification']:
            unresponsive_prob += 0.15
            negative_prob += 0.10
        
        if lifetime_giving > 100000:
            negative_prob -= 0.15
            unresponsive_prob -= 0.10
        
        if prospect_stage == 'Stewardship':
            negative_prob = 0.05
            unresponsive_prob = 0.02
        
        # Cap probabilities
        negative_prob = max(0.05, min(0.5, negative_prob))
        unresponsive_prob = max(0.02, min(0.4, unresponsive_prob))
        
        # Choose outcome type
        outcome_roll = random.random()
        if outcome_roll < unresponsive_prob:
            template_type = 'unresponsive'
            outcome = random.choice(self.unresponsive_outcomes)
        elif outcome_roll < (unresponsive_prob + negative_prob):
            template_type = 'negative'
            outcome = random.choice(self.negative_outcomes)
        else:
            # Choose positive template type based on stage
            if prospect_stage == 'Stewardship':
                template_type = 'stewardship'
            elif prospect_stage == 'Solicitation':
                template_type = random.choice(['solicitation', 'positive_meeting'])
            else:
                template_type = random.choice(['positive_meeting', 'positive_phone', 'positive_event'])
            outcome = random.choice(self.positive_outcomes)
        
        template = random.choice(self.templates[template_type])
        
        # Fill template variables
        variables = {
            'donor_name': donor_name,
            'subject': random.choice(self.subjects),
            'outcome': outcome,
            'location': random.choice(['their office', 'campus', 'restaurant', 'their home']),
            'focus_area': random.choice(['implementation timeline', 'naming opportunities', 'impact metrics']),
            'detail': random.choice(['budget breakdown', 'project timeline', 'recognition benefits']),
            'duration': random.randint(15, 60),
            'event_name': random.choice(['Alumni Reception', 'Donor Gala', 'President\'s Circle Event']),
            'amount': f"{random.randint(10, 500):,}" if lifetime_giving > 10000 else f"{random.randint(1, 25):,}"
        }
        
        try:
            return template.format(**variables)
        except KeyError:
            return f"Met with {donor_name} to discuss {variables['subject']}. {outcome}."
    
    def generate_contact_date(self):
        """Generate weighted random date (more recent = higher weight)"""
        start_date = date(2015, 1, 1)
        end_date = date.today()  # Use today as the end date, not future
        
        days_range = (end_date - start_date).days
        weights = [np.exp(-((days_range - i) / 365) * 0.3) for i in range(days_range)]
        day_offset = random.choices(range(days_range), weights=weights)[0]
        
        return start_date + timedelta(days=day_offset)
