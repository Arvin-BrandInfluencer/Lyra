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

# --- Loguru Configuration ---
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True
)

# --- Initialization & Configuration ---
load_dotenv()
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

try:
    supabase: Client = create_client(url, key)
    logger.success("Successfully connected to Supabase.")
except Exception as e:
    logger.critical(f"Failed to connect to Supabase. Check credentials. Error: {e}")
    sys.exit(1)

# --- CONFIGURATION ---
MARKET_TARGET_TABLES = [
    'uk_target', 'france_target', 'sweden_target', 'norway_target', 'denmark_target'
]
INFLUENCER_CAMPAIGN_TABLES = [
    'france_2023_influencers', 'france_2024_influencers', 'france_2025_influencers',
    'uk_2023_influencers', 'uk_2024_influencers', 'uk_2025_influencers',
    'nordics_2025_influencers'
]

# --- Helper Functions ---
HARDCODED_RATES = { "EUR": 1.0, "GBP": 0.85, "SEK": 11.30, "NOK": 11.50, "DKK": 7.46 }
NORDIC_COUNTRIES = ['Sweden', 'Norway', 'Denmark']
MONTH_ORDER = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

def convert_to_eur(amount, currency):
    rate = HARDCODED_RATES.get(str(currency).upper(), 1.0)
    return amount / rate if rate != 0 else 0

def get_market_from_table_name(table_name):
    if 'uk' in table_name: return 'UK'
    if 'france' in table_name: return 'France'
    if 'sweden' in table_name: return 'Sweden'
    if 'norway' in table_name: return 'Norway'
    if 'denmark' in table_name: return 'Denmark'
    if 'nordics' in table_name: return 'Nordics'
    return 'Unknown'

# --- Main Query Router ---
def execute_query(payload: dict):
    source = payload.get("source")
    logger.info(f"Routing request for source: '{source}'")
    if source == "dashboard":
        return get_dashboard_data(payload)
    elif source == "influencer_analytics":
        return get_influencer_analytics_data(payload)
    else:
        logger.warning(f"Invalid source '{source}' received.")
        return {"error": f"Invalid 'source'. Must be 'dashboard' or 'influencer_analytics'."}

# --- Data Processing Functions ---

def get_dashboard_data(payload: dict):
    logger.info("Starting 'get_dashboard_data' processing.")
    try:
        filters = payload.get("filters", {})
        market_filter = filters.get("market")
        year_filter = filters.get("year")
        logger.info(f"Filters applied: market='{market_filter}', year='{year_filter}'")

        if market_filter == "Nordics":
            tables_to_query = ['sweden_target', 'norway_target', 'denmark_target']
        elif market_filter and market_filter != "All":
            table_name = next((t for t in MARKET_TARGET_TABLES if get_market_from_table_name(t) == market_filter), None)
            tables_to_query = [table_name] if table_name else []
        else:
            tables_to_query = MARKET_TARGET_TABLES
        
        logger.info(f"Querying Supabase tables: {tables_to_query}")
        all_data = []
        for table in tables_to_query:
            query = supabase.from_(table).select('*')
            if year_filter and year_filter != "All":
                query = query.eq('year', int(year_filter))
            
            res = query.execute()
            if res.data:
                for row in res.data:
                    row['table_source'] = table
                all_data.extend(res.data)
        
        if not all_data:
            logger.warning("No data found for the given filters. Returning empty response.")
            return {"kpi_summary": {}, "monthly_detail": []}

        df = pd.DataFrame(all_data)
        df['region'] = df['table_source'].apply(get_market_from_table_name)
        df['currency'] = df['region'].map({'UK': 'GBP', 'France': 'EUR', 'Sweden': 'SEK', 'Norway': 'NOK', 'Denmark': 'DKK'})
        numeric_cols = ['year', 'target_budget_clean', 'actual_spend_clean', 'target_conversions_clean', 'actual_conversions_clean']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        if market_filter == "Nordics":
            logger.info("Aggregating data for Nordics market.")
            df['target_budget_eur'] = df.apply(lambda row: convert_to_eur(row['target_budget_clean'], row['currency']), axis=1)
            df['actual_spend_eur'] = df.apply(lambda row: convert_to_eur(row['actual_spend_clean'], row['currency']), axis=1)
            
            monthly_agg = df.groupby('month').agg(
                target_budget_clean=pd.NamedAgg(column='target_budget_eur', aggfunc='sum'),
                actual_spend_clean=pd.NamedAgg(column='actual_spend_eur', aggfunc='sum'),
                target_conversions_clean=pd.NamedAgg(column='target_conversions_clean', aggfunc='sum'),
                actual_conversions_clean=pd.NamedAgg(column='actual_conversions_clean', aggfunc='sum')
            ).reset_index()
            monthly_agg['region'] = 'Nordics'
            monthly_agg['currency'] = 'EUR'
            df = monthly_agg

        kpi = {
            'target_budget': int(df['target_budget_clean'].sum()),
            'actual_spend': int(df['actual_spend_clean'].sum()),
            'target_conversions': int(df['target_conversions_clean'].sum()),
            'actual_conversions': int(df['actual_conversions_clean'].sum()),
        }
        kpi['actual_cac'] = float(kpi['actual_spend'] / kpi['actual_conversions'] if kpi['actual_conversions'] else 0)
        
        df['target_cac'] = df['target_budget_clean'] / df['target_conversions_clean']
        df['actual_cac'] = df['actual_spend_clean'] / df['actual_conversions_clean']
        df.fillna(0, inplace=True)
        df.replace([float('inf'), -float('inf')], 0, inplace=True)
        
        df['month_order'] = df['month'].map(MONTH_ORDER)
        df = df.sort_values('month_order').drop(columns=['month_order'])

        logger.success("Dashboard data processing complete.")
        return { "source": "dashboard", "kpi_summary": kpi, "monthly_detail": df.to_dict(orient='records') }
    except Exception as e:
        logger.error(f"Dashboard query failed: {e}\n{traceback.format_exc()}")
        return {"error": f"Dashboard query failed: {str(e)}"}

def get_influencer_analytics_data(payload: dict):
    logger.info("Starting 'get_influencer_analytics_data' processing.")
    try:
        all_campaigns = []
        for table in INFLUENCER_CAMPAIGN_TABLES:
            try: table_year = int(table.split('_')[1])
            except (IndexError, ValueError): table_year = 0 
            res = supabase.from_(table).select('*').execute()
            if res.data:
                for row in res.data:
                    row['table_source'] = table
                    if 'year' not in row or not row['year']: row['year'] = table_year
                all_campaigns.extend(res.data)
        
        if not all_campaigns: return { "items": [], "count": 0 }
        df = pd.DataFrame(all_campaigns)
        if df.empty: return {"count": 0, "items": []}
        
        df['market'] = df.apply(lambda row: row.get('market') or get_market_from_table_name(row.get('table_source', '')), axis=1)
        df['currency'] = df['market'].map({'UK': 'GBP', 'France': 'EUR', 'Sweden': 'SEK', 'Norway': 'NOK', 'Denmark': 'DKK'}).fillna('EUR')
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
        numeric_cols = ['total_budget_clean', 'actual_conversions_clean', 'views_clean', 'views', 'clicks_clean', 'clicks', 'ctr_clean', 'cvr_clean']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df['views'] = df['views_clean'].where(df['views_clean'] > 0, df['views'])
        df['clicks'] = df['clicks_clean'].where(df['clicks_clean'] > 0, df['clicks'])
        df['influencer_name'] = df['influencer_name'].astype(str).str.strip()

        filters = payload.get("filters", {})
        
        # --- NEW LOGIC: Handle single influencer deep dive before other filters ---
        if influencer_name := filters.get("influencer_name"):
            logger.info(f"Routing to influencer detail view for: '{influencer_name}'")
            return _process_influencer_profile(df, influencer_name)
        # --- END NEW LOGIC ---

        if year_str := filters.get("year"):
            if year_str != "All":
                df = df[df['year'] == int(year_str)]
        if market := filters.get("market"):
            if market != "All":
                markets_to_filter = NORDIC_COUNTRIES if market == "Nordics" else [market]
                df = df[df['market'].isin(markets_to_filter)]

        view = payload.get("view", "summary")
        logger.info(f"Routing to view: '{view}'")

        if view == "summary":
            return _process_influencer_summary(df, payload)
        elif view == "discovery_tiers":
            return _process_discovery_tiers(df, payload)
        elif view == "monthly_breakdown":
            return _process_monthly_breakdown(df, payload)
        else:
            return {"error": f"Invalid view '{view}'."}

    except Exception as e:
        logger.error(f"Influencer Analytics query failed: {e}\n{traceback.format_exc()}")
        return {"error": f"Influencer Analytics query failed: {str(e)}"}

def _process_influencer_summary(df: pd.DataFrame, payload: dict):
    logger.info("Processing influencer summary...")
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
        sort_by = sort_config.get("by", "total_spend_eur")
        is_ascending = sort_config.get("order", "desc") == "asc"
        grouped = grouped.sort_values(by=sort_by, ascending=is_ascending)

    logger.success(f"Influencer summary processing complete. Found {len(grouped)} unique influencers.")
    return { "source": "influencer_summary", "count": len(grouped), "items": grouped.to_dict(orient='records') }

def _process_discovery_tiers(df: pd.DataFrame, payload: dict):
    logger.info("Processing discovery tiers...")
    summary_result = _process_influencer_summary(df, {})
    if not summary_result.get("items"):
        return {"gold": [], "silver": [], "bronze": []}
    grouped = pd.DataFrame(summary_result["items"])
    if grouped.empty: return {"gold": [], "silver": [], "bronze": []}

    zero_cac = grouped[grouped['effective_cac_eur'] <= 0]
    ranked = grouped[grouped['effective_cac_eur'] > 0].sort_values(by='effective_cac_eur', ascending=True)

    count = len(ranked)
    top_third_index = math.ceil(count / 3)
    mid_third_index = math.ceil(count * 2 / 3)

    gold = ranked.iloc[:top_third_index]
    silver = ranked.iloc[top_third_index:mid_third_index]
    bronze = pd.concat([ranked.iloc[mid_third_index:], zero_cac])

    logger.success(f"Discovery tiering complete. Gold: {len(gold)}, Silver: {len(silver)}, Bronze: {len(bronze)}.")
    return { "source": "discovery_tiers", "gold": gold.to_dict(orient='records'), "silver": silver.to_dict(orient='records'), "bronze": bronze.to_dict(orient='records') }

def _process_monthly_breakdown(df: pd.DataFrame, payload: dict):
    logger.info("Processing monthly breakdown...")
    if df.empty or 'month' not in df.columns: return {"monthly_data": []}
    df = df.dropna(subset=['month'])
    
    results = []
    for month_name, month_df in df.groupby('month'):
        total_spend_eur = sum(convert_to_eur(row['total_budget_clean'], row['currency']) for _, row in month_df.iterrows())
        total_conversions = month_df['actual_conversions_clean'].sum()
        
        summary = { 'total_spend_eur': total_spend_eur, 'total_conversions': total_conversions, 'avg_cac_eur': total_spend_eur / total_conversions if total_conversions else 0, 'influencer_count': month_df['influencer_name'].nunique() }
        
        details = month_df[['influencer_name', 'market', 'currency', 'total_budget_clean', 'actual_conversions_clean']].rename(columns={'total_budget_clean': 'budget_local', 'actual_conversions_clean': 'conversions'})
        details['cac_local'] = (details['budget_local'] / details['conversions']).fillna(0).replace([float('inf'), -float('inf')], 0)
        
        results.append({ 'month': month_name, 'summary': summary, 'details': details.to_dict(orient='records') })
    
    results.sort(key=lambda x: MONTH_ORDER.get(x['month'], 99))
    logger.success(f"Monthly breakdown processing complete for {len(results)} months.")
    return {"source": "monthly_breakdown", "monthly_data": results}

def _process_influencer_profile(df: pd.DataFrame, influencer_name: str):
    logger.info(f"Processing influencer profile for '{influencer_name}'")
    influencer_df = df[df['influencer_name'].str.lower() == influencer_name.lower()].copy()

    if influencer_df.empty:
        logger.warning(f"No data found for influencer '{influencer_name}'.")
        return {"error": f"No data found for influencer '{influencer_name}'."}

    logger.info(f"Found {len(influencer_df)} total campaigns for '{influencer_name}'.")
    influencer_df['cac_local'] = (influencer_df['total_budget_clean'] / influencer_df['actual_conversions_clean']).fillna(0).replace([float('inf'), -float('inf')], 0)
    influencer_df['ctr'] = (influencer_df['clicks'] / influencer_df['views']).fillna(0).replace([float('inf'), -float('inf')], 0)
    influencer_df['month_order'] = influencer_df['month'].map(MONTH_ORDER)
    influencer_df = influencer_df.sort_values(by=['year', 'month_order'])

    logger.success(f"Successfully processed profile for '{influencer_name}'.")
    return { "source": "influencer_detail", "campaigns": influencer_df.to_dict(orient='records') }

# --- Flask Web Server ---
app = Flask(__name__)
CORS(app)

@app.route('/')
def health_check():
    return jsonify({"status": "healthy", "message": "Brand Influence Query API is running."})

@app.route('/query', methods=['POST'])
def handle_query():
    try:
        payload = request.get_json()
        if not payload:
            logger.error("Request received with invalid or empty JSON payload.")
            return jsonify({"error": "Invalid JSON payload"}), 400
        
        result = execute_query(payload)
        
        if "error" in result:
            return jsonify(result), 400
            
        # Custom JSON serializer to handle pandas/numpy types
        def convert(o):
            if isinstance(o, (pd.np.int64, pd.np.int32)): return int(o)
            if isinstance(o, (pd.np.float64, pd.np.float32)): return float(o)
            if pd.isna(o): return None
            raise TypeError

        # This part seems to cause issues, using Flask's native jsonify is safer
        # and we handled type conversion within the functions themselves.
        return jsonify(result)

    except Exception as e:
        logger.critical(f"An unhandled exception occurred during a request: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    # Get port from environment variable or default to 10000 (Render's default)
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting Flask API server on host 0.0.0.0 and port {port}")
    
    # Bind to 0.0.0.0 and the PORT environment variable as required by Render
    app.run(host="0.0.0.0", port=port, debug=False)
