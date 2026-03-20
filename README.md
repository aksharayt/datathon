# Food Waste Policy Intelligence Platform

GW Data Science Association Hackathon 2026



## Live Application

Open the platform here: https://huggingface.co/spaces/aksharatarikere/food-waste-policy-platform-pro

The application runs fully in your browser. No installation needed to view it.

\---

## Overview

This is a 23-page interactive data platform built on 5 ReFED Food Waste Monitor datasets covering US food surplus from 2010 to 2024. The goal is to answer one question: where exactly is food being lost, why, and what would a targeted policy intervention recover in tons, dollars, and meals?

The platform is not just charts. Every number is live, every filter changes every chart in real time, and the policy recommendations are generated directly from the filtered data.

\---

## Problem Statement

The US generates roughly 70 million tons of food surplus every year, worth over $297 billion, while 47 million Americans face food insecurity. These are not two separate problems. They are the same problem from opposite ends of the supply chain. The gap between wasted food and hungry people is entirely a logistics and incentives problem, not a production problem.

\---

## Key Findings from the Data

These five findings come directly from the ReFED datasets and contradict what most people assume about food waste:

**1. Households waste more than restaurants.**
Residential is the single largest wasting sector, generating more surplus than Foodservice, Manufacturing, and Retail combined. Most food waste policy targets restaurants. The data says target households first.

**2. Foodservice sends 77% of its surplus to landfill.**
This is the worst diversion rate of any sector by a large margin. Retail, by comparison, donates nearly 20% of its surplus. This gap is regulatory, not operational. A mandate would close it.

**3. Prepared Foods wastes over half its entire supply.**
A 55%+ waste rate on a high-volume food type indicates structural overproduction, not consumer behavior. Demand forecasting is the most direct fix.

**4. Processing byproducts are the number one cause of surplus.**
Trimmings and manufacturing byproducts exceed every other cause group including unharvested fields. The problem is embedded in how food is processed before it reaches consumers.

**5. California alone generates 17% of national surplus.**
Geographic concentration creates real policy leverage. A single state-level pilot in the highest-surplus state produces measurable national results within three years.

\---

## Application Structure

The platform has 23 pages organized into 6 sections. Navigation and all filters are at the top of the app.

**Problem section:** Executive Dashboard, Supply Chain Funnel, Temporal Analysis

**Diagnosis section:** Sector Deep Dive, Sub-Sector Intel, Food Type and Category, Root Cause Engine, Hidden Waste, Waste Flow Sankey Diagram

**Geographic section:** State Intelligence, State Clustering, Zero Waste Index

**Impact section:** Environmental Analysis, Waste x Hunger Nexus

**Solutions section:** Innovation Hub, Policy Simulator, ROI Calculator, Policy Recommendations, State Action Plan

**Forward section:** Trend Forecast, The One Law, State vs State, Cost of Inaction

**Global filters that apply to every page:**

* Year range (2010 to 2024)
* Sector (Farm, Foodservice, Manufacturing, Residential, Retail)
* Food type
* State

\---

## Notable Pages

**Waste x Hunger Nexus:** Uses Pearson correlation and Ridge regression to identify states where high food waste and high food insecurity overlap at the same time. Includes a redirection calculator where you can set what percentage of critical-state surplus gets rerouted to food banks and see exactly how many people could be fed year-round.

**The One Law:** Scores 8 federal policy interventions across four dimensions, impact scale at 35%, political feasibility at 30%, speed to results at 20%, and ROI at 15%. Produces a single ranked recommendation backed by data, not opinion.

**State vs State:** Compare any two states across every metric with green and red winner cards. Produces a specific, quotable finding such as "If Texas matched California's landfill rate, it would divert X tons per year."

**Cost of Inaction:** A live counter starts running from the moment you open the page, showing meals wasted, dollars lost, and CO2e emitted in real time. All rates are pulled from the filtered dataset.

**Policy ROI Calculator:** Sliders for five interventions. Calculates return on investment using food value recovery, carbon credit revenue, and water conservation value combined.

\---

## Machine Learning Models Used

Linear Regression for forecasting surplus trends by sector up to 5 years forward.

KMeans Clustering to group all 50 states into policy archetypes based on waste behavior.

PCA (Principal Component Analysis) to visualize those state clusters in two dimensions.

Ridge Regression to model the correlation between food waste intensity and food insecurity rates across states.

StandardScaler with z-score thresholding to detect states that are statistical anomalies relative to their peer group.

\---

## Datasets

Five ReFED Food Waste Monitor datasets are used. The app detects each one automatically by column structure, so no filenames need to match exactly.

US Food Surplus Summary: 4,350 rows covering sector, food type, surplus tons, landfill, donations, GHG, and water use.

US Food Surplus Detail: 13,560 rows, adds food category breakdown.

US Food Surplus Cause Summary: 16,455 rows, covers 31 individual waste causes across 9 cause groups.

US State Food Surplus Summary: 216,930 rows, all of the above broken out by state.

US State Food Surplus Detail: 294,273 rows, adds state plus food category.

Two supplementary datasets are also used, both permitted by hackathon rules: USDA ERS Household Food Security Survey 2022 for state food insecurity rates, and US Census Bureau 2022 for state population figures.

\---

## Running Locally

Clone the repository:

&#x20;   git clone https://github.com/\[your-username]/food-waste-policy-platform
    cd food-waste-policy-platform


Install dependencies:

&#x20;   pip install -r requirements.txt


Place all 5 ReFED CSV files in the same folder as app.py, then run:

&#x20;   streamlit run app.py


The app will open in your browser at localhost:8501.

\---

## Dependencies

&#x20;   streamlit
    pandas
    numpy
    plotly
    scikit-learn
    scipy


\---

## Tech Stack

Streamlit for the app framework and UI. Pandas and NumPy for data processing. Plotly Express and Plotly Graph Objects for all visualizations. scikit-learn for all machine learning models. SciPy for statistical analysis. Deployed on Hugging Face Spaces.

\---

## Policy Recommendation Summary

The analysis supports five legislative actions under what I call the Smart Food Waste Reduction Act:

First, a Residential Smart Label and Composting Act establishing a single federal date label standard and subsidizing household composting. Estimated reduction of 8% of residential surplus within 1 to 2 years.

Second, a Commercial Foodservice Landfill Diversion Mandate requiring operators above 1 ton per week to divert organics, modeled on the Massachusetts 2014 law. Estimated reduction of 15% of foodservice surplus within 1 to 3 years.

Third, a Prepared Foods Demand Intelligence Initiative providing federal tax incentives for predictive demand planning and dynamic markdown requirements. Estimated reduction of 5% of national surplus within 3 to 5 years.

Fourth, a Processing Byproduct Valorization Program offering tax credits for converting manufacturing byproducts to animal feed or biofuel, with EPA reporting mandates. Estimated reduction of 8% of national surplus within 3 to 7 years.

Fifth, a High-Impact State Federal Innovation Pilot designating the highest-surplus state as a Federal Food Waste Innovation Zone. Estimated reduction of 10% of that state's surplus within 1 to 2 years.

Combined projected impact is approximately 33% reduction in national surplus and enough recovered meals to feed millions of food-insecure Americans year-round, without producing a single additional pound of food.

\---

Data source: ReFED Food Waste Monitor at refed.org, analysis period 2010 to 2024.

