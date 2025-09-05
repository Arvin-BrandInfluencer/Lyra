# --- MEMORY-OPTIMIZED UNIFIED BRAND INFLUENCE API SERVER ---
# This refactored version eliminates memory issues by:
# 1. Pushing all filtering to the database level instead of loading everything into memory
# 2. Using targeted queries with specific filters
# 3. Minimizing DataFrame creation and operations
# 4. Optimizing data types for memory efficiency

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
from typing import List, Dict, Any, Optional

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

def fetch_filtered_table_data(table_name: str, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """
    Memory-optimized function to fetch filtered data from a single Supabase table.
    Applies filters at the database level instead of loading everything into memory.
    """
    try:
        query = supabase.from_(table_name).select("*")
        
        if filters:
            # Apply year filter
            if year_filter := filters.get("year"):
                if year_filter != "All":
                    query = query.eq('year', int(year_filter))
            
            # Apply month filter
            if month_filter := filters.get("month"):
                if month_filter != "All":
                    query = query.eq('month', month_filter)
            
            # Apply influencer name filter (case-insensitive)
            if influencer_filter := filters.get("influencer_name"):
                query = query.ilike('influencer_name', f'%{influencer_filter}%')
        
        response = query.execute()
        return response.data if response.data else []
    except Exception as e:
        logger.error(f"Error fetching filtered data from '{table_name}': {e}")
        return []

def get_relevant_tables(table_list: List[str], filters: Dict) -> List[str]:
    """
    Returns only the tables that are relevant based on the provided filters.
    This reduces the number of database queries needed.
    """
    market_filter = filters.get("market")
    year_filter = filters.get("year")
    
    relevant_tables = table_list.copy()
    
    # Filter by year if specified
    if year_filter and year_filter != "All":
        relevant_tables = [t for t in relevant_tables if str(year_filter) in t]
    
    # Filter by market if specified
    if market_filter and market_filter != "All":
        if market_filter == "Nordics":
            relevant_tables = [t for t in relevant_tables if any(country.lower() in t.lower() for country in ['nordics', 'sweden', 'norway', 'denmark'])]
        else:
            relevant_tables = [t for t in relevant_tables if market_filter.lower() in t.lower()]
    
    return relevant_tables

def create_optimized_dataframe(data: List[Dict], optimize_dtypes: bool = True) -> pd.DataFrame:
    """
    Creates a DataFrame with optimized data types to reduce memory usage.
    """
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    
    if optimize_dtypes and not df.empty:
        # Optimize categorical columns
        categorical_cols = ['market', 'month', 'currency', 'region', 'asset']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # Optimize numeric columns
        numeric_cols = ['year', 'total_budget_clean', 'actual_spend_clean', 'target_budget_clean', 
                       'actual_conversions_clean', 'target_conversions_clean', 'views_clean', 'clicks_clean']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                # Use smaller integer types where possible
                if col == 'year':
                    df[col] = pd.to_numeric(df[col], downcast='integer')
    
    return df

# --- 3. ENDPOINT DEFINITIONS ---

# === Endpoint 1: Comprehensive Influencer API ===

@app.route('/api/influencer/query', methods=['POST'])
def handle_influencer_query():
    """Memory-optimized influencer query endpoint"""
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

def _influencer_get_dashboard_data(payload: dict):
    """Memory-optimized dashboard data fetching"""
    logger.info("Starting memory-optimized 'get_dashboard_data' processing.")
    try:
        filters = payload.get("filters", {})
        market_filter, year_filter = filters.get("market"), filters.get("year")
        
        # Determine relevant tables based on market filter
        if market_filter == "Nordics":
            tables_to_query = ['sweden_target', 'norway_target', 'denmark_target']
        elif market_filter and market_filter != "All":
            table_name = next((t for t in MARKET_TARGET_TABLES if get_market_from_table_name(t) == market_filter), None)
            tables_to_query = [table_name] if table_name else []
        else:
            tables_to_query = MARKET_TARGET_TABLES
        
        all_data = []
        for table in tables_to_query:
            # Apply filters at database level
            table_filters = {}
            if year_filter and year_filter != "All":
                table_filters["year"] = year_filter
            
            filtered_data = fetch_filtered_table_data(table, table_filters)
            if filtered_data:
                for row in filtered_data: 
                    row['table_source'] = table
                all_data.extend(filtered_data)
        
        if not all_data: 
            return {"kpi_summary": {}, "monthly_detail": []}
        
        # Create optimized DataFrame from smaller, filtered dataset
        df = create_optimized_dataframe(all_data)
        df['region'] = df['table_source'].apply(get_market_from_table_name)
        df['currency'] = df['region'].map({'UK': 'GBP', 'France': 'EUR', 'Sweden': 'SEK', 'Norway': 'NOK', 'Denmark': 'DKK'})
        
        # Ensure numeric columns are properly typed
        numeric_cols = ['target_budget_clean', 'actual_spend_clean', 'target_conversions_clean', 'actual_conversions_clean']
        for col in numeric_cols: 
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Handle Nordics aggregation if needed
        if market_filter == "Nordics":
            df['target_budget_eur'] = df.apply(lambda row: convert_to_eur(row['target_budget_clean'], row['currency']), axis=1)
            df['actual_spend_eur'] = df.apply(lambda row: convert_to_eur(row['actual_spend_clean'], row['currency']), axis=1)
            monthly_agg = df.groupby('month').agg(
                target_budget_clean=('target_budget_eur', 'sum'), 
                actual_spend_clean=('actual_spend_eur', 'sum'),
                target_conversions_clean=('target_conversions_clean', 'sum'), 
                actual_conversions_clean=('actual_conversions_clean', 'sum')
            ).reset_index()
            monthly_agg['region'], monthly_agg['currency'] = 'Nordics', 'EUR'
            df = monthly_agg

        # Calculate KPIs
        kpi = {
            'target_budget': int(df['target_budget_clean'].sum()), 
            'actual_spend': int(df['actual_spend_clean'].sum()),
            'target_conversions': int(df['target_conversions_clean'].sum()), 
            'actual_conversions': int(df['actual_conversions_clean'].sum())
        }
        kpi['actual_cac'] = float(kpi['actual_spend'] / kpi['actual_conversions'] if kpi['actual_conversions'] else 0)
        
        # Calculate monthly details
        df['target_cac'] = df['target_budget_clean'] / df['target_conversions_clean']
        df['actual_cac'] = df['actual_spend_clean'] / df['actual_conversions_clean']
        df.fillna(0, inplace=True)
        df.replace([float('inf'), -float('inf')], 0, inplace=True)
        df['month_order'] = df['month'].map(MONTH_ORDER)
        df = df.sort_values('month_order').drop(columns=['month_order'])
        
        return {
            "source": "dashboard", 
            "kpi_summary": kpi, 
            "monthly_detail": df.to_dict(orient='records')
        }
        
    except Exception as e:
        logger.error(f"Dashboard query failed: {e}\n{traceback.format_exc()}")
        return {"error": f"Dashboard query failed: {str(e)}"}

def _influencer_get_analytics_data(payload: dict):
    """Memory-optimized influencer analytics data fetching"""
    logger.info("Starting memory-optimized 'get_influencer_analytics_data' processing.")
    try:
        filters = payload.get("filters", {})
        market_filter = filters.get("market")
        year_filter = filters.get("year")
        month_filter = filters.get("month")
        influencer_filter = filters.get("influencer_name")

        # Get only relevant tables based on filters
        relevant_tables = get_relevant_tables(INFLUENCER_CAMPAIGN_TABLES, filters)
        logger.info(f"Querying {len(relevant_tables)} relevant tables out of {len(INFLUENCER_CAMPAIGN_TABLES)} total tables")

        all_campaigns = []
        for table in relevant_tables:
            # Build table-specific filters
            table_filters = {k: v for k, v in filters.items() if v and v != "All"}
            
            # Handle market filtering for nordics table
            if 'nordics' in table.lower() and market_filter and market_filter != "All":
                if market_filter == "Nordics":
                    # Don't add market filter, fetch all nordics data
                    table_filters.pop("market", None)
                elif market_filter in NORDIC_COUNTRIES:
                    # Keep market filter for specific nordic country
                    table_filters["market"] = market_filter
                else:
                    # Skip this table if market doesn't match
                    continue
            elif market_filter and market_filter != "All" and market_filter != "Nordics":
                # For non-nordics tables, check if table matches market
                table_market = get_market_from_table_name(table)
                if table_market != market_filter:
                    continue
            
            try:
                filtered_data = fetch_filtered_table_data(table, table_filters)
                if filtered_data:
                    try: 
                        table_year = int(table.split('_')[1])
                    except (IndexError, ValueError): 
                        table_year = 0
                    
                    for row in filtered_data:
                        row['table_source'] = table
                        if 'year' not in row or not row['year']: 
                            row['year'] = table_year
                    all_campaigns.extend(filtered_data)
                    
            except Exception as e:
                logger.warning(f"Failed to query table {table}: {e}")
                continue
        
        if not all_campaigns: 
            return {"items": [], "count": 0}

        # Create optimized DataFrame from filtered data
        df = create_optimized_dataframe(all_campaigns)
        
        # Add derived columns efficiently
        df['market'] = df.apply(lambda row: row.get('market') or get_market_from_table_name(row.get('table_source', '')), axis=1)
        df['currency'] = df['market'].map({'UK': 'GBP', 'France': 'EUR', 'Sweden': 'SEK', 'Norway': 'NOK', 'Denmark': 'DKK'}).fillna('EUR')
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
        
        # Optimize numeric columns
        numeric_cols = ['total_budget_clean', 'actual_conversions_clean', 'views_clean', 'views', 'clicks_clean', 'clicks', 'ctr_clean', 'cvr_clean']
        for col in numeric_cols: 
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        df['views'] = df['views_clean'].where(df['views_clean'] > 0, df['views'])
        df['clicks'] = df['clicks_clean'].where(df['clicks_clean'] > 0, df['clicks'])
        df['influencer_name'] = df['influencer_name'].astype(str).str.strip()

        # Route to appropriate processing function
        if "influencer_name" in filters:
            return _influencer_process_profile(df, filters.get("influencer_name"))
        
        view = payload.get("view", "summary")
        if view == "summary": 
            return _influencer_process_summary(df, payload)
        if view == "discovery_tiers": 
            return _influencer_process_discovery_tiers(df, payload)
        if view == "monthly_breakdown": 
            return _influencer_process_monthly_breakdown(df)
        
        return {"error": f"Invalid view '{view}'."}
        
    except Exception as e:
        logger.error(f"Influencer Analytics query failed: {e}\n{traceback.format_exc()}")
        return {"error": f"Influencer Analytics query failed: {str(e)}"}

def _influencer_process_summary(df: pd.DataFrame, payload: dict):
    """Process summary data efficiently"""
    grouped = df.groupby('influencer_name').apply(lambda x: pd.Series({
        'campaign_count': len(x), 
        'total_conversions': x['actual_conversions_clean'].sum(), 
        'total_views': x['views'].sum(),
        'total_clicks': x['clicks'].sum(), 
        'markets': list(x['market'].unique()), 
        'assets': list(x['asset'].dropna().unique()),
        'total_spend_eur': sum(convert_to_eur(row['total_budget_clean'], row['currency']) for _, row in x.iterrows()),
        'avg_ctr': x[x['ctr_clean'] > 0]['ctr_clean'].mean(), 
        'avg_cvr': x[x['cvr_clean'] > 0]['cvr_clean'].mean()
    })).reset_index()
    
    grouped['effective_cac_eur'] = (grouped['total_spend_eur'] / grouped['total_conversions']).fillna(0).replace([float('inf'), -float('inf')], 0)
    grouped.fillna({'avg_ctr': 0, 'avg_cvr': 0}, inplace=True)
    
    if sort_config := payload.get("sort"):
        grouped = grouped.sort_values(by=sort_config.get("by", "total_spend_eur"), ascending=sort_config.get("order", "desc") == "asc")
    
    return {"source": "influencer_summary", "count": len(grouped), "items": grouped.to_dict(orient='records')}

def _influencer_process_discovery_tiers(df: pd.DataFrame, payload: dict):
    """Calculate discovery tiers efficiently"""
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
        return {
            "source": "discovery_tier_specific",
            "tier": requested_tier,
            "items": all_tiers[requested_tier]
        }
    else:
        return {
            "source": "discovery_tiers",
            "gold": all_tiers["gold"],
            "silver": all_tiers["silver"],
            "bronze": all_tiers["bronze"]
        }

def _influencer_process_monthly_breakdown(df: pd.DataFrame):
    """Process monthly breakdown efficiently"""
    if df.empty or 'month' not in df.columns: 
        return {"monthly_data": []}
    
    df = df.dropna(subset=['month'])
    results = []
    
    for month_name, month_df in df.groupby('month'):
        total_spend_eur = sum(convert_to_eur(row['total_budget_clean'], row['currency']) for _, row in month_df.iterrows())
        total_conversions = month_df['actual_conversions_clean'].sum()
        summary = {
            'total_spend_eur': total_spend_eur, 
            'total_conversions': total_conversions, 
            'avg_cac_eur': total_spend_eur / total_conversions if total_conversions else 0, 
            'influencer_count': month_df['influencer_name'].nunique()
        }
        details = month_df[['influencer_name', 'market', 'currency', 'total_budget_clean', 'actual_conversions_clean']].rename(columns={'total_budget_clean': 'budget_local', 'actual_conversions_clean': 'conversions'})
        details['cac_local'] = (details['budget_local'] / details['conversions']).fillna(0).replace([float('inf'), -float('inf')], 0)
        results.append({'month': month_name, 'summary': summary, 'details': details.to_dict(orient='records')})
    
    results.sort(key=lambda x: MONTH_ORDER.get(x['month'], 99))
    return {"source": "monthly_breakdown", "monthly_data": results}

def _influencer_process_profile(df: pd.DataFrame, influencer_name: str):
    """Process individual influencer profile efficiently"""
    influencer_df = df.copy()

    if influencer_df.empty: 
        return {"error": f"No data found for influencer '{influencer_name}' matching the specified filters."}
    
    influencer_df['cac_local'] = (influencer_df['total_budget_clean'] / influencer_df['actual_conversions_clean']).fillna(0).replace([float('inf'), -float('inf')], 0)
    influencer_df['ctr'] = (influencer_df['clicks'] / influencer_df['views']).fillna(0).replace([float('inf'), -float('inf')], 0)
    
    if 'month' in influencer_df.columns:
        influencer_df['month_order'] = influencer_df['month'].map(MONTH_ORDER)
        influencer_df = influencer_df.sort_values(by=['year', 'month_order'])
    
    return {"source": "influencer_detail", "campaigns": influencer_df.to_dict(orient='records')}

# === Endpoint 2: Discovery Tiers ===

@app.route('/api/discovery', methods=['POST'])
def get_discovery_data():
    """Memory-optimized discovery endpoint"""
    try:
        payload = request.get_json()
        filters = payload.get("filters", {})
        tier = filters.get("tier", "gold").lower()
        
        # Use the optimized analytics data function
        analytics_payload = {"source": "influencer_analytics", "view": "discovery_tiers", "filters": filters}
        tiers_result = _influencer_get_analytics_data(analytics_payload)
        
        if "error" in tiers_result:
            return jsonify(tiers_result), 400
        
        if tier not in ["gold", "silver", "bronze"]:
            return jsonify({"error": "Invalid tier. Use 'gold', 'silver', or 'bronze'."}), 400
        
        tier_data = tiers_result.get(tier, [])
        
        response_data = []
        for influencer in tier_data:
            response_data.append({
                "influencerName": influencer["influencer_name"], 
                "totalSpendEUR": round(influencer["total_spend_eur"], 2),
                "totalConversions": influencer["total_conversions"], 
                "campaignCount": influencer["campaign_count"],
                "effectiveCAC": round(influencer["effective_cac_eur"], 2), 
                "averageSpendPerCampaign": round(influencer["total_spend_eur"] / influencer["campaign_count"], 2)
            })
        
        return jsonify({
            "filters_applied": filters, 
            "tier": tier, 
            "influencer_count": len(response_data), 
            "influencers": response_data
        }), 200
        
    except Exception as e:
        logger.error(f"Discovery endpoint error: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

# === Endpoint 3: Dashboard Targets ===

@app.route('/api/dashboard/targets', methods=['POST'])
def get_dashboard_targets_data():
    """Memory-optimized dashboard targets endpoint"""
    try:
        payload = request.get_json()
        filters = payload.get("filters", {})
        market, year, month = filters.get("market"), filters.get("year"), filters.get("month")
        
        # Determine relevant tables and apply filters at database level
        if market == "Nordics":
            relevant_configs = [c for c in DASHBOARD_REGIONS_CONFIG if c['region'] in NORDIC_COUNTRIES]
        elif market and market != "All":
            relevant_configs = [c for c in DASHBOARD_REGIONS_CONFIG if c['region'] == market]
        else:
            relevant_configs = DASHBOARD_REGIONS_CONFIG
        
        all_data = []
        for region_info in relevant_configs:
            # Build filters for this table
            table_filters = {}
            if year: table_filters["year"] = year
            if month: table_filters["month"] = month
            
            filtered_data = fetch_filtered_table_data(region_info['table'], table_filters)
            
            # Process filtered data
            for d in filtered_data:
                processed_row = {
                    'region': region_info['region'], 
                    'currency': region_info['currency'], 
                    'month': d.get('month'), 
                    'year': safe_int(d.get('year')),
                    'target_budget': safe_float(d.get('target_budget_clean')), 
                    'actual_spend': safe_float(d.get('actual_spend_clean')),
                    'target_conversions': safe_int(d.get('target_conversions_clean')), 
                    'actual_conversions': safe_int(d.get('actual_conversions_clean'))
                }
                all_data.append(processed_row)
        
        # Handle Nordics aggregation if needed
        if market == "Nordics":
            nordics_by_month_year = {}
            for d in all_data:
                if d['region'] in NORDIC_COUNTRIES:
                    key = f"{d['year']}-{d['month']}"
                    if key not in nordics_by_month_year:
                        nordics_by_month_year[key] = {
                            'region': 'Nordics', 'currency': 'EUR', 'month': d['month'], 'year': d['year'], 
                            'target_budget': 0, 'actual_spend': 0, 'target_conversions': 0, 'actual_conversions': 0
                        }
                    nordics_by_month_year[key]['target_budget'] += convert_to_eur(d['target_budget'], d['currency'])
                    nordics_by_month_year[key]['actual_spend'] += convert_to_eur(d['actual_spend'], d['currency'])
                    nordics_by_month_year[key]['target_conversions'] += d['target_conversions']
                    nordics_by_month_year[key]['actual_conversions'] += d['actual_conversions']
            all_data = list(nordics_by_month_year.values())
        
        # Calculate KPIs from filtered data
        total_target_budget = sum(d['target_budget'] for d in all_data)
        total_actual_spend = sum(d['actual_spend'] for d in all_data)
        total_actual_conversions = sum(d['actual_conversions'] for d in all_data)
        actual_cac = total_actual_spend / total_actual_conversions if total_actual_conversions > 0 else 0
        
        kpis = {"total_target_budget": round(total_target_budget, 2)}
        
        return jsonify({
            "filters_applied": filters, 
            "record_count": len(all_data), 
            "kpis": kpis
        }), 200
        
    except Exception as e:
        logger.error(f"Dashboard targets endpoint error: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

# === Endpoint 4: Monthly Breakdown ===

@app.route('/api/monthly_breakdown', methods=['POST'])
def get_monthly_breakdown():
    """Memory-optimized monthly breakdown endpoint"""
    try:
        req_data = request.get_json()
        filters = req_data.get('filters', {})
        market_filter, month_filter, year_filter = filters.get('market', 'All'), filters.get('month'), filters.get('year')
        
        if not month_filter or not year_filter:
            return jsonify({"error": "Missing required filters: 'month' and 'year'."}), 400
    except Exception:
        return jsonify({"error": "Could not parse request body."}), 400

    # Get relevant tables based on market filter
    relevant_tables = get_relevant_tables(INFLUENCER_CAMPAIGN_TABLES, filters)
    logger.info(f"Monthly breakdown: Querying {len(relevant_tables)} relevant tables")
    
    all_campaigns = []
    for table_name in relevant_tables:
        try:
            # Apply filters at database level
            table_filters = {
                "year": year_filter,
                "month": month_filter
            }
            
            # Handle market filtering for nordics table
            if 'nordics' in table_name.lower() and market_filter and market_filter != "All":
                if market_filter == "Nordics":
                    # Don't add market filter, fetch all nordics data
                    pass
                elif market_filter in NORDIC_COUNTRIES:
                    # Add market filter for specific nordic country
                    table_filters["market"] = market_filter
                else:
                    # Skip this table if market doesn't match
                    continue
            elif market_filter and market_filter != "All" and market_filter != "Nordics":
                # For non-nordics tables, check if table matches market
                table_market = get_market_from_table_name(table_name)
                if table_market != market_filter:
                    continue
            
            filtered_data = fetch_filtered_table_data(table_name, table_filters)
            
            if filtered_data:
                market_prefix, year_str, _ = table_name.split('_')
                for d in filtered_data:
                    market, currency = 'Unknown', 'EUR'
                    if market_prefix == 'uk': 
                        market, currency = 'UK', 'GBP'
                    elif market_prefix == 'france': 
                        market, currency = 'France', 'EUR'
                    elif market_prefix == 'nordics' and d.get('market'):
                        market = d['market']
                        if market == 'Sweden': currency = 'SEK'
                        elif market == 'Norway': currency = 'NOK'
                        elif market == 'Denmark': currency = 'DKK'
                    
                    campaign = {
                        'influencerName': d.get('influencer_name'), 
                        'market': market, 
                        'year': int(d.get('year', year_str)),
                        'currency': currency, 
                        'month': d.get('month'), 
                        'totalBudget': safe_float(d.get('total_budget_clean')),
                        'actualConversions': safe_int(d.get('actual_conversions_clean'))
                    }
                    all_campaigns.append(campaign)
                    
        except Exception as e:
            logger.warning(f"Could not fetch or process table '{table_name}' for monthly view. Error: {e}")
            continue

    if not all_campaigns:
        return jsonify({
            "message": "No campaigns found for the specified filters.",
            "metrics": {"budget_spent_eur": 0, "conversions": 0, "average_cac_eur": 0, "influencer_count": 0},
            "influencers": []
        }), 200

    # Calculate metrics from filtered data
    total_spend_eur = sum(convert_to_eur(c['totalBudget'], c['currency']) for c in all_campaigns)
    total_conversions = sum(c['actualConversions'] for c in all_campaigns)
    unique_influencers = {c['influencerName'] for c in all_campaigns}
    average_cac_eur = (total_spend_eur / total_conversions) if total_conversions > 0 else 0
    
    metrics_data = {
        "budget_spent_eur": round(total_spend_eur, 2), 
        "conversions": int(total_conversions),
        "average_cac_eur": round(average_cac_eur, 2), 
        "influencer_count": len(unique_influencers)
    }
    
    influencers_list = []
    for c in all_campaigns:
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

# --- 4. HEALTH CHECK & SERVER EXECUTION ---

@app.route('/')
def health_check():
    """A simple health check endpoint."""
    return jsonify({"status": "healthy", "message": "Memory-Optimized Unified Brand Influence Query API is running."})

@app.route('/api/health', methods=['GET'])
def detailed_health_check():
    """Detailed health check with memory optimization info."""
    try:
        # Test database connection
        test_query = supabase.from_('uk_target').select("count", count="exact").limit(1).execute()
        db_status = "healthy" if test_query else "unhealthy"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return jsonify({
        "status": "healthy",
        "message": "Memory-Optimized Unified Brand Influence Query API",
        "database_connection": db_status,
        "optimization_features": [
            "Database-level filtering",
            "Optimized DataFrame creation", 
            "Selective table querying",
            "Memory-efficient data types",
            "Reduced in-memory operations"
        ],
        "endpoints": [
            "/api/influencer/query",
            "/api/discovery", 
            "/api/dashboard/targets",
            "/api/monthly_breakdown"
        ]
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting Memory-Optimized Flask API server on host 0.0.0.0 and port {port}")
    logger.info("Memory optimizations enabled:")
    logger.info("- Database-level filtering")
    logger.info("- Selective table querying") 
    logger.info("- Optimized DataFrame operations")
    logger.info("- Memory-efficient data types")
    app.run(host="0.0.0.0", port=port, debug=False)
