# --- UNIFIED BRAND INFLUENCE API SERVER (MEMORY-EFFICIENT & CONCURRENCY-SAFE) ---
# This version uses a Supabase cache table as a persistent memory store for filtering.
# It is designed to handle multiple simultaneous requests without data collision.
#
# INSTRUCTIONS:
# 1. Ensure you have a .env file with SUPABASE_URL and SUPABASE_KEY.
# 2. Ensure you have run the setup SQL script in your Supabase project.
# 3. Run this file: python app.py

import os
import sys
import traceback
import math
import pandas as pd
import uuid  # Import the uuid library for generating unique request IDs
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
    colorize=True,
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
INFLUENCER_CAMPAIGN_TABLES = [
    'france_2023_influencers', 'france_2024_influencers', 'france_2025_influencers',
    'uk_2023_influencers', 'uk_2024_influencers', 'uk_2025_influencers',
    'nordics_2025_influencers'
]
CACHE_TABLE_NAME = 'processed_influencer_data_cache'

MARKET_TARGET_TABLES = [
    'uk_target', 'france_target', 'sweden_target', 'norway_target', 'denmark_target'
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
    if amount is None: return 0.0
    rate = HARDCODED_RATES.get(str(currency).upper(), 1.0)
    return float(amount) / rate if rate != 0 else 0

def get_market_from_table_name(table_name):
    if 'uk' in table_name: return 'UK'
    if 'france' in table_name: return 'France'
    if 'sweden' in table_name: return 'Sweden'
    if 'norway' in table_name: return 'Norway'
    if 'denmark' in table_name: return 'Denmark'
    if 'nordics' in table_name: return 'Nordics'
    return 'Unknown'

def safe_float(value: Any) -> float:
    if value is None: return 0.0
    try: return float(str(value).replace(',', ''))
    except (ValueError, TypeError): return 0.0

def safe_int(value: Any) -> int:
    if value is None: return 0
    try: return int(safe_float(value))
    except (ValueError, TypeError): return 0

def fetch_table_data(table_name: str) -> List[Dict[str, Any]]:
    try:
        response = supabase.from_(table_name).select("*").execute()
        return response.data if response.data else []
    except Exception as e:
        logger.error(f"Error fetching data from '{table_name}': {e}")
        return []

# --- 3. ENDPOINT DEFINITIONS ---

@app.route('/api/influencer/query', methods=['POST'])
def handle_influencer_query():
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"error": "Invalid JSON payload"}), 400
        
        source = payload.get("source")
        logger.info(f"Routing request for source: '{source}'")

        if source == "dashboard":
            result = _influencer_get_dashboard_data(payload)
        elif source == "influencer_analytics":
            result = _influencer_get_analytics_data_load_then_fetch(payload)
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
    logger.info("Starting 'get_dashboard_data' processing.")
    # This logic is kept as is, assuming target data is small and doesn't need the caching mechanism.
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
        logger.error(f"Dashboard query failed: {e}\n{traceback.format_exc()}")
        return {"error": f"Dashboard query failed: {str(e)}"}

def _influencer_get_analytics_data_load_then_fetch(payload: dict):
    request_id = str(uuid.uuid4())
    logger.info(f"Starting analytics request with ID: {request_id}")

    try:
        logger.info(f"[{request_id}] Fetching all raw data from source tables...")
        all_campaigns = []
        for table in INFLUENCER_CAMPAIGN_TABLES:
            res = fetch_table_data(table)
            if res:
                table_year = int(table.split('_')[1])
                for row in res:
                    row['table_source'] = table
                    if 'year' not in row or not row['year']: row['year'] = table_year
                all_campaigns.extend(res)
        
        if not all_campaigns:
            logger.warning(f"[{request_id}] No raw campaign data found.")
            return {"items": [], "count": 0}

        df_processed = pd.DataFrame(all_campaigns)
        df_processed['market'] = df_processed.apply(lambda row: row.get('market') or get_market_from_table_name(row.get('table_source', '')), axis=1)
        df_processed['currency'] = df_processed['market'].map({'UK': 'GBP', 'France': 'EUR', 'Sweden': 'SEK', 'Norway': 'NOK', 'Denmark': 'DKK'}).fillna('EUR')
        df_processed['year'] = pd.to_numeric(df_processed['year'], errors='coerce').fillna(0).astype(int)
        numeric_cols = ['total_budget_clean', 'actual_conversions_clean', 'views_clean', 'views', 'clicks_clean', 'clicks', 'ctr_clean', 'cvr_clean']
        for col in numeric_cols: df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)
        df_processed['views'] = df_processed['views_clean'].where(df_processed['views_clean'] > 0, df_processed['views'])
        df_processed['clicks'] = df_processed['clicks_clean'].where(df_processed['clicks_clean'] > 0, df_processed['clicks'])
        df_processed['influencer_name'] = df_processed['influencer_name'].astype(str).str.strip()

        df_processed['request_id'] = request_id
        df_for_upload = df_processed[[
            'request_id', 'influencer_name', 'market', 'year', 'currency', 'month', 'asset',
            'total_budget_clean', 'actual_conversions_clean', 'views_clean',
            'views', 'clicks_clean', 'clicks', 'ctr_clean', 'cvr_clean'
        ]].copy()
        
        logger.info(f"[{request_id}] Appending {len(df_for_upload)} processed records to Supabase cache...")
        upload_data = df_for_upload.to_dict(orient='records')
        supabase.from_(CACHE_TABLE_NAME).insert(upload_data).execute()
        logger.success(f"[{request_id}] Append to cache table complete.")

        del df_processed
        del df_for_upload
        del all_campaigns

        logger.info(f"[{request_id}] Querying the cache table with user filters...")
        filters = payload.get("filters", {})
        query = supabase.from_(CACHE_TABLE_NAME).select("*").eq('request_id', request_id)

        if influencer_name := filters.get("influencer_name"):
            query = query.ilike('influencer_name', f'%{influencer_name.strip()}%')
        if year_str := filters.get("year", "All"):
            if year_str != "All":
                query = query.eq('year', int(year_str))
        if market := filters.get("market", "All"):
            if market != "All":
                markets_to_filter = NORDIC_COUNTRIES if market == "Nordics" else [market]
                query = query.in_('market', markets_to_filter)
        if month := filters.get("month", "All"):
            if month != "All":
                query = query.eq('month', month)

        response = query.execute()
        filtered_data = response.data
        if not filtered_data:
            logger.warning(f"[{request_id}] No data found in cache matching filters.")
            return {"items": [], "count": 0}
            
        logger.success(f"[{request_id}] Fetched {len(filtered_data)} filtered records from cache.")
        df = pd.DataFrame(filtered_data)

        if "influencer_name" in filters:
            return _influencer_process_profile(df, filters.get("influencer_name"))
        
        view = payload.get("view", "summary")
        if view == "summary": return _influencer_process_summary(df, payload)
        if view == "discovery_tiers": return _influencer_process_discovery_tiers(df, payload)
        if view == "monthly_breakdown": return _influencer_process_monthly_breakdown(df)
        
        return {"error": f"Invalid view '{view}'."}
        
    except Exception as e:
        logger.error(f"[{request_id}] Analytics request failed: {e}\n{traceback.format_exc()}")
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

def _influencer_process_discovery_tiers(df: pd.DataFrame, payload: dict):
    summary_result = _influencer_process_summary(df, {})
    if not summary_result.get("items"): 
        return {"gold": [], "silver": [], "bronze": []}
    grouped = pd.DataFrame(summary_result["items"])
    if grouped.empty: 
        return {"gold": [], "silver": [], "bronze": []}
    zero_cac = grouped[grouped['effective_cac_eur'] <= 0]
    ranked = grouped[grouped['effective_cac_eur'] > 0].sort_values(by='effective_cac_eur', ascending=True)
    count = len(ranked)
    top_third_index = math.ceil(count / 3)
    mid_third_index = math.ceil(count * 2 / 3)
    gold_df = ranked.iloc[:top_third_index]
    silver_df = ranked.iloc[top_third_index:mid_third_index]
    bronze_df = pd.concat([ranked.iloc[mid_third_index:], zero_cac])
    all_tiers = {
        "gold": gold_df.to_dict(orient='records'),
        "silver": silver_df.to_dict(orient='records'),
        "bronze": bronze_df.to_dict(orient='records')
    }
    filters = payload.get("filters", {})
    requested_tier = filters.get("tier")
    if requested_tier and requested_tier in all_tiers:
        return {"source": "discovery_tier_specific", "tier": requested_tier, "items": all_tiers[requested_tier]}
    else:
        return {"source": "discovery_tiers", **all_tiers}

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
    if df.empty: 
        return {"error": f"No data found for influencer '{influencer_name}' matching the specified filters."}
    influencer_df = df.copy()
    influencer_df['cac_local'] = (influencer_df['total_budget_clean'] / influencer_df['actual_conversions_clean']).fillna(0).replace([float('inf'), -float('inf')], 0)
    influencer_df['ctr'] = (influencer_df['clicks'] / influencer_df['views']).fillna(0).replace([float('inf'), -float('inf')], 0)
    if 'month' in influencer_df.columns:
        influencer_df['month_order'] = influencer_df['month'].map(MONTH_ORDER)
        influencer_df = influencer_df.sort_values(by=['year', 'month_order'])
    return {"source": "influencer_detail", "campaigns": influencer_df.to_dict(orient='records')}

# --- REMOVED REDUNDANT ENDPOINTS ---
# The logic from the original /api/discovery, /api/dashboard/targets, and /api/monthly_breakdown 
# endpoints is now fully handled by the comprehensive /api/influencer/query endpoint.

# --- 4. HEALTH CHECK & SERVER EXECUTION ---
@app.route('/')
def health_check():
    return jsonify({"status": "healthy", "message": "Unified Brand Influence Query API is running."})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting Unified Flask API server on host 0.0.0.0 and port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
