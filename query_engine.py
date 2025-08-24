# --- UNIFIED BRAND INFLUENCE API SERVER ---
# This single Flask application combines the functionality of four separate scripts:
# 1. Influencer: A comprehensive query engine for dashboards and influencer analytics.
# 2. Gold Tier: A specific endpoint for ranking influencers into Gold, Silver, and Bronze tiers.
# 3. Target: A dashboard endpoint focused on retrieving and calculating target vs. actual KPIs.
# 4. Monthly View: A detailed breakdown of influencer performance for a specific month.
#
# INSTRUCTIONS:
# 1. Ensure you have a .env file with SUPABASE_URL and SUPABASE_KEY.
# 2. Run this file: python your_app_name.py
# 3. Access the documented endpoints below.

import os
import sys
import traceback
import math
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
from loguru import logger
from typing import List, Dict, Any

# --- 1. GLOBAL CONFIGURATION & INITIALIZATION ---

# --- Loguru Configuration ---
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True
)

# --- Environment & Supabase Client Setup ---
load_dotenv()
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

if not url or not key:
    logger.critical("Supabase URL and Key must be set in the .env file.")
    sys.exit(1)

try:
    supabase: Client = create_client(url, key)
    logger.success("Successfully connected to Supabase.")
except Exception as e:
    logger.critical(f"Failed to connect to Supabase. Check credentials. Error: {e}")
    sys.exit(1)

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Shared Constants ---
MARKET_TARGET_TABLES = [
    'uk_target', 'france_target', 'sweden_target', 'norway_target', 'denmark_target'
]
INFLUENCER_CAMPAIGN_TABLES = [
    'france_2023_influencers', 'france_2024_influencers', 'france_2025_influencers',
    'uk_2023_influencers', 'uk_2024_influencers', 'uk_2025_influencers',
    'nordics_2025_influencers'
]
DASHBOARD_REGIONS_CONFIG = [
    {'table': 'uk_target', 'region': 'UK', 'currency': 'GBP'},
    {'table': 'france_target', 'region': 'France', 'currency': 'EUR'},
    {'table': 'sweden_target', 'region': 'Sweden', 'currency': 'SEK'},
    {'table': 'norway_target', 'region': 'Norway', 'currency': 'NOK'},
    {'table': 'denmark_target', 'region': 'Denmark', 'currency': 'DKK'},
]
NORDIC_COUNTRIES = ['Sweden', 'Norway', 'Denmark']
HARDCODED_RATES = {"EUR": 1.0, "GBP": 0.85, "SEK": 11.30, "NOK": 11.50, "DKK": 7.46}
MONTH_ORDER = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

# --- 2. SHARED HELPER FUNCTIONS ---

def convert_to_eur(amount, currency):
    """Converts a given amount from a specified currency to EUR."""
    if amount is None: return 0.0
    rate = HARDCODED_RATES.get(str(currency).upper(), 1.0)
    return float(amount) / rate if rate != 0 else 0

def get_market_from_table_name(table_name):
    """Extracts market name from a table name string."""
    if 'uk' in table_name: return 'UK'
    if 'france' in table_name: return 'France'
    if 'sweden' in table_name: return 'Sweden'
    if 'norway' in table_name: return 'Norway'
    if 'denmark' in table_name: return 'Denmark'
    if 'nordics' in table_name: return 'Nordics'
    return 'Unknown'

def safe_float(value: Any) -> float:
    """Safely converts a value to float, returning 0.0 on failure."""
    if value is None: return 0.0
    try: return float(str(value).replace(',', ''))
    except (ValueError, TypeError): return 0.0

def safe_int(value: Any) -> int:
    """Safely converts a value to int, returning 0 on failure."""
    if value is None: return 0
    try: return int(safe_float(value))
    except (ValueError, TypeError): return 0

def fetch_table_data(table_name: str) -> List[Dict[str, Any]]:
    """Generic function to fetch all data from a single Supabase table."""
    try:
        response = supabase.from_(table_name).select("*").execute()
        return response.data if response.data else []
    except Exception as e:
        logger.error(f"Error fetching data from '{table_name}': {e}")
        return []

# --- 3. ENDPOINT DEFINITIONS ---

# === Endpoint 1: Comprehensive Influencer API (from 'Influencer' script) ===

@app.route('/api/influencer/query', methods=['POST'])
def handle_influencer_query():
    """
    Endpoint: /api/influencer/query
    Description: A comprehensive query engine for influencer marketing data.
                 It can return high-level dashboard KPIs, detailed monthly breakdowns,
                 influencer summaries, discovery tiers, or deep-dive into a single influencer's profile.
    Method: POST
    Payload:
    {
      "source": "dashboard" | "influencer_analytics",
      "filters": {
        "market": "All" | "UK" | "France" | "Nordics" | "Sweden" | "Norway" | "Denmark",
        "year": "All" | 2023 | 2024 | 2025,
        "month": "All" | "Jan" | "Feb" | ... ,
        "influencer_name": "Specific Influencer Name", // for deep-dive
        "tier": "gold" | "silver" | "bronze" // Optional: for view="discovery_tiers" to get only one tier
      },
      "view": "summary" | "discovery_tiers" | "monthly_breakdown" // only for source: "influencer_analytics"
    }
    """
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"error": "Invalid JSON payload"}), 400
        
        source = payload.get("source")
        logger.info(f"Routing request for source: '{source}'")

        if source == "dashboard":
            result = _influencer_get_dashboard_data(payload)
        elif source == "influencer_analytics":
            result = _influencer_get_analytics_data(payload)
        else:
            logger.warning(f"Invalid source '{source}' received.")
            result = {"error": f"Invalid 'source'. Must be 'dashboard' or 'influencer_analytics'."}
        
        if "error" in result:
            return jsonify(result), 400
            
        return jsonify(result)

    except Exception as e:
        logger.critical(f"An unhandled exception occurred during /api/influencer/query: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

# --- Logic for Endpoint 1 ---
def _influencer_get_dashboard_data(payload: dict):
    logger.info("Starting 'get_dashboard_data' processing for Influencer API.")
    try:
        filters = payload.get("filters", {})
        market_filter, year_filter = filters.get("market"), filters.get("year")
        tables_to_query = MARKET_TARGET_TABLES
        if market_filter == "Nordics":
            tables_to_query = ['sweden_target', 'norway_target', 'denmark_target']
        elif market_filter and market_filter != "All":
            table_name = next((t for t in MARKET_TARGET_TABLES if get_market_from_table_name(t) == market_filter), None)
            tables_to_query = [table_name] if table_name else []
        
        all_data = []
        for table in tables_to_query:
            query = supabase.from_(table).select('*')
            if year_filter and year_filter != "All":
                query = query.eq('year', int(year_filter))
            res = query.execute()
            if res.data:
                for row in res.data: row['table_source'] = table
                all_data.extend(res.data)
        
        if not all_data: return {"kpi_summary": {}, "monthly_detail": []}
        df = pd.DataFrame(all_data)
        df['region'] = df['table_source'].apply(get_market_from_table_name)
        df['currency'] = df['region'].map({'UK': 'GBP', 'France': 'EUR', 'Sweden': 'SEK', 'Norway': 'NOK', 'Denmark': 'DKK'})
        numeric_cols = ['year', 'target_budget_clean', 'actual_spend_clean', 'target_conversions_clean', 'actual_conversions_clean']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        if market_filter == "Nordics":
            df['target_budget_eur'] = df.apply(lambda row: convert_to_eur(row['target_budget_clean'], row['currency']), axis=1)
            df['actual_spend_eur'] = df.apply(lambda row: convert_to_eur(row['actual_spend_clean'], row['currency']), axis=1)
            monthly_agg = df.groupby('month').agg(
                target_budget_clean=('target_budget_eur', 'sum'), actual_spend_clean=('actual_spend_eur', 'sum'),
                target_conversions_clean=('target_conversions_clean', 'sum'), actual_conversions_clean=('actual_conversions_clean', 'sum')
            ).reset_index()
            monthly_agg['region'], monthly_agg['currency'] = 'Nordics', 'EUR'
            df = monthly_agg

        kpi = {'target_budget': int(df['target_budget_clean'].sum()), 'actual_spend': int(df['actual_spend_clean'].sum()),
               'target_conversions': int(df['target_conversions_clean'].sum()), 'actual_conversions': int(df['actual_conversions_clean'].sum())}
        kpi['actual_cac'] = float(kpi['actual_spend'] / kpi['actual_conversions'] if kpi['actual_conversions'] else 0)
        df['target_cac'] = df['target_budget_clean'] / df['target_conversions_clean']
        df['actual_cac'] = df['actual_spend_clean'] / df['actual_conversions_clean']
        df.fillna(0, inplace=True); df.replace([float('inf'), -float('inf')], 0, inplace=True)
        df['month_order'] = df['month'].map(MONTH_ORDER)
        df = df.sort_values('month_order').drop(columns=['month_order'])
        return {"source": "dashboard", "kpi_summary": kpi, "monthly_detail": df.to_dict(orient='records')}
    except Exception as e:
        logger.error(f"Dashboard query for Influencer API failed: {e}\n{traceback.format_exc()}")
        return {"error": f"Dashboard query failed: {str(e)}"}

def _influencer_get_analytics_data(payload: dict):
    logger.info("Starting 'get_influencer_analytics_data' processing.")
    try:
        all_campaigns = []
        for table in INFLUENCER_CAMPAIGN_TABLES:
            res = fetch_table_data(table)
            if res:
                try: table_year = int(table.split('_')[1])
                except (IndexError, ValueError): table_year = 0
                for row in res:
                    row['table_source'] = table
                    if 'year' not in row or not row['year']: row['year'] = table_year
                all_campaigns.extend(res)
        
        if not all_campaigns: return {"items": [], "count": 0}
        df = pd.DataFrame(all_campaigns)
        df['market'] = df.apply(lambda row: row.get('market') or get_market_from_table_name(row.get('table_source', '')), axis=1)
        df['currency'] = df['market'].map({'UK': 'GBP', 'France': 'EUR', 'Sweden': 'SEK', 'Norway': 'NOK', 'Denmark': 'DKK'}).fillna('EUR')
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
        numeric_cols = ['total_budget_clean', 'actual_conversions_clean', 'views_clean', 'views', 'clicks_clean', 'clicks', 'ctr_clean', 'cvr_clean']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df['views'] = df['views_clean'].where(df['views_clean'] > 0, df['views'])
        df['clicks'] = df['clicks_clean'].where(df['clicks_clean'] > 0, df['clicks'])
        df['influencer_name'] = df['influencer_name'].astype(str).str.strip()

        filters = payload.get("filters", {})

        if influencer_name := filters.get("influencer_name"):
            df = df[df['influencer_name'].str.lower() == influencer_name.lower()]

        if year_str := filters.get("year"):
            if year_str != "All":
                df = df[df['year'] == int(year_str)]

        if market := filters.get("market"):
            if market != "All":
                markets_to_filter = NORDIC_COUNTRIES if market == "Nordics" else [market]
                df = df[df['market'].isin(markets_to_filter)]
        
        if month := filters.get("month"):
            if month != "All":
                df = df[df['month'] == month]

        if "influencer_name" in filters:
            return _influencer_process_profile(df, filters.get("influencer_name"))
        
        view = payload.get("view", "summary")
        if view == "summary": return _influencer_process_summary(df, payload)
        
        # --- MODIFICATION START: Pass the payload to the discovery tiers function ---
        if view == "discovery_tiers": return _influencer_process_discovery_tiers(df, payload)
        # --- MODIFICATION END ---
        
        if view == "monthly_breakdown": return _influencer_process_monthly_breakdown(df)
        
        return {"error": f"Invalid view '{view}'."}
        
    except Exception as e:
        logger.error(f"Influencer Analytics query failed: {e}\n{traceback.format_exc()}")
        return {"error": f"Influencer Analytics query failed: {str(e)}"}

def _influencer_process_summary(df: pd.DataFrame, payload: dict):
    grouped = df.groupby('influencer_name').apply(lambda x: pd.Series({
        'campaign_count': len(x), 'total_conversions': x['actual_conversions_clean'].sum(), 'total_views': x['views'].sum(),
        'total_clicks': x['clicks'].sum(), 'markets': list(x['market'].unique()), 'assets': list(x['asset'].dropna().unique()),
        'total_spend_eur': sum(convert_to_eur(row['total_budget_clean'], row['currency']) for _, row in x.iterrows()),
        'avg_ctr': x[x['ctr_clean'] > 0]['ctr_clean'].mean(), 'avg_cvr': x[x['cvr_clean'] > 0]['cvr_clean'].mean()
    })).reset_index()
    grouped['effective_cac_eur'] = (grouped['total_spend_eur'] / grouped['total_conversions']).fillna(0).replace([float('inf'), -float('inf')], 0)
    grouped.fillna({'avg_ctr': 0, 'avg_cvr': 0}, inplace=True)
    if sort_config := payload.get("sort"):
        grouped = grouped.sort_values(by=sort_config.get("by", "total_spend_eur"), ascending=sort_config.get("order", "desc") == "asc")
    return {"source": "influencer_summary", "count": len(grouped), "items": grouped.to_dict(orient='records')}

# --- MODIFICATION START: Update the discovery tiers function to handle a specific tier request ---
def _influencer_process_discovery_tiers(df: pd.DataFrame, payload: dict):
    """
    Calculates Gold, Silver, and Bronze tiers. If a specific tier is requested in the
    payload filters, it returns only that tier's data. Otherwise, it returns all three.
    """
    summary_result = _influencer_process_summary(df, {})
    if not summary_result.get("items"): 
        return {"gold": [], "silver": [], "bronze": []}
        
    grouped = pd.DataFrame(summary_result["items"])
    if grouped.empty: 
        return {"gold": [], "silver": [], "bronze": []}

    # Separate influencers with zero or positive CAC for ranking
    zero_cac = grouped[grouped['effective_cac_eur'] <= 0]
    ranked = grouped[grouped['effective_cac_eur'] > 0].sort_values(by='effective_cac_eur', ascending=True)
    
    # Calculate tier boundaries
    count = len(ranked)
    top_third_index = math.ceil(count / 3)
    mid_third_index = math.ceil(count * 2 / 3)
    
    # Create DataFrames for each tier
    gold_df = ranked.iloc[:top_third_index]
    silver_df = ranked.iloc[top_third_index:mid_third_index]
    bronze_df = pd.concat([ranked.iloc[mid_third_index:], zero_cac])
    
    # Store all tiers in a dictionary for easy access
    all_tiers = {
        "gold": gold_df.to_dict(orient='records'),
        "silver": silver_df.to_dict(orient='records'),
        "bronze": bronze_df.to_dict(orient='records')
    }

    # Check if a specific tier was requested in the filters
    filters = payload.get("filters", {})
    requested_tier = filters.get("tier")

    if requested_tier and requested_tier in all_tiers:
        # If a valid tier is requested, return only that tier's data in a specific format
        return {
            "source": "discovery_tier_specific",
            "tier": requested_tier,
            "items": all_tiers[requested_tier]
        }
    else:
        # Otherwise, return all tiers for backward compatibility
        return {
            "source": "discovery_tiers",
            "gold": all_tiers["gold"],
            "silver": all_tiers["silver"],
            "bronze": all_tiers["bronze"]
        }
# --- MODIFICATION END ---

def _influencer_process_monthly_breakdown(df: pd.DataFrame):
    if df.empty or 'month' not in df.columns: return {"monthly_data": []}
    df = df.dropna(subset=['month'])
    results = []
    for month_name, month_df in df.groupby('month'):
        total_spend_eur = sum(convert_to_eur(row['total_budget_clean'], row['currency']) for _, row in month_df.iterrows())
        total_conversions = month_df['actual_conversions_clean'].sum()
        summary = {'total_spend_eur': total_spend_eur, 'total_conversions': total_conversions, 'avg_cac_eur': total_spend_eur / total_conversions if total_conversions else 0, 'influencer_count': month_df['influencer_name'].nunique()}
        details = month_df[['influencer_name', 'market', 'currency', 'total_budget_clean', 'actual_conversions_clean']].rename(columns={'total_budget_clean': 'budget_local', 'actual_conversions_clean': 'conversions'})
        details['cac_local'] = (details['budget_local'] / details['conversions']).fillna(0).replace([float('inf'), -float('inf')], 0)
        results.append({'month': month_name, 'summary': summary, 'details': details.to_dict(orient='records')})
    results.sort(key=lambda x: MONTH_ORDER.get(x['month'], 99))
    return {"source": "monthly_breakdown", "monthly_data": results}

def _influencer_process_profile(df: pd.DataFrame, influencer_name: str):
    influencer_df = df.copy()

    if influencer_df.empty: 
        return {"error": f"No data found for influencer '{influencer_name}' matching the specified filters."}
    
    influencer_df['cac_local'] = (influencer_df['total_budget_clean'] / influencer_df['actual_conversions_clean']).fillna(0).replace([float('inf'), -float('inf')], 0)
    influencer_df['ctr'] = (influencer_df['clicks'] / influencer_df['views']).fillna(0).replace([float('inf'), -float('inf')], 0)
    if 'month' in influencer_df.columns:
        influencer_df['month_order'] = influencer_df['month'].map(MONTH_ORDER)
        influencer_df = influencer_df.sort_values(by=['year', 'month_order'])
    
    return {"source": "influencer_detail", "campaigns": influencer_df.to_dict(orient='records')}

# === Endpoint 2: Gold Tier Discovery (from 'Gold Tier' script) ===

@app.route('/api/discovery', methods=['POST'])
def get_discovery_data():
    """
    Endpoint: /api/discovery
    Description: Fetches all influencer campaign data, calculates performance metrics (CAC, Spend/Campaign),
                 and returns a ranked list of influencers for a specified tier (gold, silver, bronze).
    Method: POST
    Payload:
    {
      "filters": {
        "market": "All" | "UK" | "France" | "Nordics",
        "year": 2023 | 2024 | 2025,
        "tier": "gold" | "silver" | "bronze"
      }
    }
    """
    try:
        payload = request.get_json()
        filters = payload.get("filters", {})
        tier = filters.get("tier", "gold").lower()
        all_campaign_data = _discovery_fetch_and_process_influencer_data()
        tiers = _discovery_calculate_rankings(filters, all_campaign_data)
        
        if tier not in tiers: return jsonify({"error": "Invalid tier. Use 'gold', 'silver', or 'bronze'."}), 400
        response_data = []
        for influencer in tiers[tier]:
            response_data.append({
                "influencerName": influencer["influencerName"], "totalSpendEUR": round(influencer["totalSpendEUR"], 2),
                "totalConversions": influencer["totalConversions"], "campaignCount": influencer["campaignCount"],
                "effectiveCAC": round(influencer["effectiveCAC"], 2), "averageSpendPerCampaign": round(influencer["averageSpendPerCampaign"], 2)
            })
        return jsonify({"filters_applied": filters, "tier": tier, "influencer_count": len(response_data), "influencers": response_data}), 200
    except Exception as e:
        logger.error(f"Discovery endpoint error: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

# --- Logic for Endpoint 2 ---
def _discovery_standardize_influencer_data(raw_data: List[Dict[str, Any]], table_name: str) -> List[Dict[str, Any]]:
    standardized = []
    market_prefix, year_str = table_name.split('_', 1); year_str = year_str.split('_')[0]
    for d in raw_data:
        market, currency = 'Unknown', 'EUR'
        prefix = market_prefix.lower()
        if prefix == 'uk': market, currency = 'UK', 'GBP'
        elif prefix == 'france': market, currency = 'France', 'EUR'
        elif prefix == 'nordics' and d.get('market'):
            market = d['market']
            if market == 'Sweden': currency = 'SEK'
            elif market == 'Norway': currency = 'NOK'
            elif market == 'Denmark': currency = 'DKK'
        standardized.append({
            'influencer_name': d.get('influencer_name'), 'market': market, 'year': int(year_str), 'currency': currency,
            'total_budget': safe_float(d.get('total_budget_clean')), 'actual_conversions': safe_int(d.get('actual_conversions_clean')),
        })
    return standardized

def _discovery_fetch_and_process_influencer_data() -> List[Dict[str, Any]]:
    all_campaign_data = []
    for table_name in INFLUENCER_CAMPAIGN_TABLES:
        raw_data = fetch_table_data(table_name)
        all_campaign_data.extend(_discovery_standardize_influencer_data(raw_data, table_name))
    return all_campaign_data

def _discovery_calculate_rankings(filters: Dict[str, Any], all_campaign_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    market, year = filters.get("market"), filters.get("year")
    filtered_campaigns = all_campaign_data
    if year: filtered_campaigns = [c for c in filtered_campaigns if c['year'] == int(year)]
    if market and market != "All":
        markets_to_filter = NORDIC_COUNTRIES if market == "Nordics" else [market]
        filtered_campaigns = [c for c in filtered_campaigns if c['market'] in markets_to_filter]
    
    summary = {}
    for campaign in filtered_campaigns:
        name = campaign['influencer_name']
        if not name: continue
        if name not in summary:
            summary[name] = {'influencerName': name, 'totalSpendEUR': 0, 'totalConversions': 0, 'campaignCount': 0}
        summary[name]['totalSpendEUR'] += convert_to_eur(campaign['total_budget'], campaign['currency'])
        summary[name]['totalConversions'] += campaign['actual_conversions']
        summary[name]['campaignCount'] += 1
    
    summary_with_metrics = []
    for s in summary.values():
        s['effectiveCAC'] = s['totalSpendEUR'] / s['totalConversions'] if s['totalConversions'] > 0 else 0
        s['averageSpendPerCampaign'] = s['totalSpendEUR'] / s['campaignCount'] if s['campaignCount'] > 0 else 0
        summary_with_metrics.append(s)
    
    zero_cac = [inf for inf in summary_with_metrics if inf['effectiveCAC'] == 0]
    ranked = sorted([inf for inf in summary_with_metrics if inf['effectiveCAC'] > 0], key=lambda x: x['effectiveCAC'])
    count = len(ranked); top_third_index, mid_third_index = count // 3, (count * 2) // 3
    tiers = {"gold": ranked[:top_third_index], "silver": ranked[top_third_index:mid_third_index], "bronze": ranked[mid_third_index:] + zero_cac}
    logger.info(f"Tiering complete: Gold({len(tiers['gold'])}), Silver({len(tiers['silver'])}), Bronze({len(tiers['bronze'])})")
    return tiers


# === Endpoint 3: Dashboard Targets (from 'Target' script) ===

@app.route('/api/dashboard/targets', methods=['POST'])
def get_dashboard_targets_data():
    """
    Endpoint: /api/dashboard/targets
    Description: Fetches marketing target data, applies filters, and calculates high-level KPIs
                 like total target budget, actual spend, conversions, and CAC.
    Method: POST
    Payload:
    {
      "filters": {
        "market": "All" | "UK" | "France" | "Nordics",
        "year": 2023 | 2024 | 2025,
        "month": "Jan" | "Feb" | ...
      }
    }
    """
    try:
        payload = request.get_json()
        filters = payload.get("filters", {})
        market, year, month = filters.get("market"), filters.get("year"), filters.get("month")
        
        all_data = _target_fetch_and_process_all_data()
        
        filtered_data = all_data
        if market: filtered_data = [d for d in filtered_data if d['region'] == market]
        if year: filtered_data = [d for d in filtered_data if d['year'] == int(year)]
        if month: filtered_data = [d for d in filtered_data if d['month'] == month]
        
        total_target_budget = sum(d['target_budget'] for d in filtered_data)
        total_actual_spend = sum(d['actual_spend'] for d in filtered_data)
        total_actual_conversions = sum(d['actual_conversions'] for d in filtered_data)
        actual_cac = total_actual_spend / total_actual_conversions if total_actual_conversions > 0 else 0
        kpis = {"total_target_budget": round(total_target_budget, 2), "total_actual_spend": round(total_actual_spend, 2),
                "total_actual_conversions": total_actual_conversions, "actual_cac": round(actual_cac, 2)}
        
        kpis_original = {"total_target_budget": round(total_target_budget, 2)}
        
        return jsonify({"filters_applied": filters, "record_count": len(filtered_data), "kpis": kpis_original}), 200
    except Exception as e:
        logger.error(f"Dashboard targets endpoint error: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

# --- Logic for Endpoint 3 ---
def _target_standardize_data(raw_data: list, region_info: dict) -> list:
    return [{'region': region_info['region'], 'currency': region_info['currency'], 'month': d.get('month'), 'year': safe_int(d.get('year')),
             'target_budget': safe_float(d.get('target_budget_clean')), 'actual_spend': safe_float(d.get('actual_spend_clean')),
             'target_conversions': safe_int(d.get('target_conversions_clean')), 'actual_conversions': safe_int(d.get('actual_conversions_clean'))} for d in raw_data]

def _target_fetch_and_process_all_data() -> list:
    all_regions_data = []
    for region_info in DASHBOARD_REGIONS_CONFIG:
        raw_data = fetch_table_data(region_info['table'])
        all_regions_data.extend(_target_standardize_data(raw_data, region_info))
    nordics_by_month_year = {}
    for d in all_regions_data:
        if d['region'] in NORDIC_COUNTRIES:
            key = f"{d['year']}-{d['month']}"
            if key not in nordics_by_month_year:
                nordics_by_month_year[key] = {'region': 'Nordics', 'currency': 'EUR', 'month': d['month'], 'year': d['year'], 'target_budget': 0, 'actual_spend': 0, 'target_conversions': 0, 'actual_conversions': 0}
            nordics_by_month_year[key]['target_budget'] += convert_to_eur(d['target_budget'], d['currency'])
            nordics_by_month_year[key]['actual_spend'] += convert_to_eur(d['actual_spend'], d['currency'])
            nordics_by_month_year[key]['target_conversions'] += d['target_conversions']
            nordics_by_month_year[key]['actual_conversions'] += d['actual_conversions']
    return all_regions_data + list(nordics_by_month_year.values())


# === Endpoint 4: Monthly Breakdown (from 'monthly view' script) ===

@app.route('/api/monthly_breakdown', methods=['POST'])
def get_monthly_breakdown():
    """
    Endpoint: /api/monthly_breakdown
    Description: Provides a detailed performance breakdown for a specific month and year,
                 including aggregated metrics and a list of participating influencers.
    Method: POST
    Payload:
    {
      "filters": {
        "market": "All" | "UK" | "France" | "Nordics",
        "year": 2024,
        "month": "Mar"
      }
    }
    """
    try:
        req_data = request.get_json()
        filters = req_data.get('filters', {})
        market_filter, month_filter, year_filter = filters.get('market', 'All'), filters.get('month'), filters.get('year')
        if not month_filter or not year_filter:
            return jsonify({"error": "Missing required filters: 'month' and 'year'."}), 400
    except Exception:
        return jsonify({"error": "Could not parse request body."}), 400

    all_campaigns = _monthly_fetch_and_standardize_all_campaigns()
    if not all_campaigns: return jsonify({"error": "Failed to fetch any campaign data."}), 500

    filtered_campaigns = [c for c in all_campaigns if c['year'] == year_filter and str(c['month']).lower() == month_filter.lower() and (market_filter.lower() == 'all' or (market_filter.lower() == 'nordics' and c['market'] in NORDIC_COUNTRIES) or c['market'].lower() == market_filter.lower())]
    if not filtered_campaigns:
        return jsonify({"message": "No campaigns found for the specified filters.", "metrics": {"budget_spent_eur": 0, "conversions": 0, "average_cac_eur": 0, "influencer_count": 0}, "influencers": []}), 200

    total_spend_eur = sum(convert_to_eur(c['totalBudget'], c['currency']) for c in filtered_campaigns)
    total_conversions = sum(c['actualConversions'] for c in filtered_campaigns)
    unique_influencers = {c['influencerName'] for c in filtered_campaigns}
    average_cac_eur = (total_spend_eur / total_conversions) if total_conversions > 0 else 0
    
    metrics_data = {"budget_spent_eur": round(total_spend_eur, 2), "conversions": int(total_conversions),
                    "average_cac_eur": round(average_cac_eur, 2), "influencer_count": len(unique_influencers)}
    
    influencers_list = []
    for c in filtered_campaigns:
        budget_local = c['totalBudget']
        conversions = c['actualConversions']
        currency = c['currency']
        
        cac_local = (budget_local / conversions) if conversions > 0 else 0
        budget_eur = convert_to_eur(budget_local, currency)
        cac_eur = (budget_eur / conversions) if conversions > 0 else 0
        
        influencers_list.append({
            "name": c['influencerName'],
            "market": c['market'],
            "currency": currency,
            "budget_local": round(budget_local, 2),
            "conversions": conversions,
            "cac_local": round(cac_local, 2),
            "cac_eur": round(cac_eur, 2)
        })
        
    return jsonify({"metrics": metrics_data, "influencers": influencers_list}), 200

# --- Logic for Endpoint 4 ---
def _monthly_fetch_and_standardize_all_campaigns():
    all_campaigns_standardized = []
    for table_name in INFLUENCER_CAMPAIGN_TABLES:
        try:
            response = fetch_table_data(table_name)
            if response:
                market_prefix, year_str, _ = table_name.split('_')
                for d in response:
                    market, currency = 'Unknown', 'EUR'
                    if market_prefix == 'uk': market, currency = 'UK', 'GBP'
                    elif market_prefix == 'france': market, currency = 'France', 'EUR'
                    elif market_prefix == 'nordics' and d.get('market'):
                        market = d['market']
                        if market == 'Sweden': currency = 'SEK'
                        elif market == 'Norway': currency = 'NOK'
                        elif market == 'Denmark': currency = 'DKK'
                    all_campaigns_standardized.append({
                        'influencerName': d.get('influencer_name'), 'market': market, 'year': int(d.get('year', year_str)),
                        'currency': currency, 'month': d.get('month'), 'totalBudget': safe_float(d.get('total_budget_clean')),
                        'actualConversions': safe_int(d.get('actual_conversions_clean')),
                    })
        except Exception as e:
            logger.warning(f"Could not fetch or process table '{table_name}' for monthly view. Error: {e}")
            continue
    return all_campaigns_standardized

# --- 4. HEALTH CHECK & SERVER EXECUTION ---

@app.route('/')
def health_check():
    """A simple health check endpoint."""
    return jsonify({"status": "healthy", "message": "Unified Brand Influence Query API is running."})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting Unified Flask API server on host 0.0.0.0 and port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
