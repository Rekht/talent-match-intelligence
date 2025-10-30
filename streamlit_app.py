import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from supabase import create_client
import numpy as np
import os
from dotenv import load_dotenv
from datetime import datetime

# ======================================
# LOAD ENVIRONMENT VARIABLES
# ======================================
load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
A4F_API_KEY = os.getenv('A4F_API_KEY')
MODEL = os.getenv('A4F_MODEL', 'provider-3/gpt-5-nano')
API_ENDPOINT = "https://api.a4f.co/v1/chat/completions"

# Validasi konfigurasi
if not all([SUPABASE_URL, SUPABASE_KEY, A4F_API_KEY]):
    st.error("Missing environment variables. Please check your .env file")
    st.stop()

# ======================================
# INISIALISASI SUPABASE
# ======================================
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error(f"Failed to initialize Supabase: {e}")
    supabase = None

# ======================================
# STREAMLIT PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="Talent Match Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================
# PROFESSIONAL CSS STYLING - LIGHT THEME
# ======================================

# ======================================
# SESSION STATE INITIALIZATION
# ======================================
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

if 'form_data' not in st.session_state:
    st.session_state.form_data = {
        'role_name': '',
        'job_level': '',
        'role_purpose': '',
        'benchmark_ids': [],
        'job_vacancy_id': None
    }

if 'search_query' not in st.session_state:
    st.session_state.search_query = ''

# ======================================
# HEADER
# ======================================
st.markdown('<div class="main-header">Talent Match Intelligence System</div>', unsafe_allow_html=True)

# ======================================
# DATABASE QUERY FUNCTIONS - UNLIMITED (PAGINATION)
# ======================================
@st.cache_data(ttl=300)
def get_employees():
    """Fetch ALL employees from Supabase using pagination"""
    if supabase is None:
        return pd.DataFrame()
    
    try:
        all_data = []
        page_size = 1000
        offset = 0
        
        while True:
            response = supabase.table('employees')\
                .select('*')\
                .range(offset, offset + page_size - 1)\
                .execute()
            
            if not response.data:
                break
            
            all_data.extend(response.data)
            
            # If we got less than page_size, we're done
            if len(response.data) < page_size:
                break
            
            offset += page_size
        
        return pd.DataFrame(all_data) if all_data else pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching employees: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_performance_data():
    """Fetch ALL performance data using pagination"""
    if supabase is None:
        return pd.DataFrame()
    
    try:
        all_data = []
        page_size = 1000
        offset = 0
        
        while True:
            response = supabase.table('performance_yearly')\
                .select('*')\
                .range(offset, offset + page_size - 1)\
                .execute()
            
            if not response.data:
                break
            
            all_data.extend(response.data)
            
            if len(response.data) < page_size:
                break
            
            offset += page_size
        
        return pd.DataFrame(all_data) if all_data else pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching performance data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_psychometric_data():
    """Fetch ALL psychometric profiles using pagination"""
    if supabase is None:
        return pd.DataFrame()
    
    try:
        all_data = []
        page_size = 1000
        offset = 0
        
        while True:
            response = supabase.table('profiles_psych')\
                .select('*')\
                .range(offset, offset + page_size - 1)\
                .execute()
            
            if not response.data:
                break
            
            all_data.extend(response.data)
            
            if len(response.data) < page_size:
                break
            
            offset += page_size
        
        return pd.DataFrame(all_data) if all_data else pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching psychometric data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_competencies_data():
    """Fetch ALL competencies using pagination"""
    if supabase is None:
        return pd.DataFrame()
    
    try:
        all_data = []
        page_size = 1000
        offset = 0
        
        while True:
            response = supabase.table('competencies_yearly')\
                .select('*')\
                .range(offset, offset + page_size - 1)\
                .execute()
            
            if not response.data:
                break
            
            all_data.extend(response.data)
            
            if len(response.data) < page_size:
                break
            
            offset += page_size
        
        return pd.DataFrame(all_data) if all_data else pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching competencies: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_papi_scores():
    """Fetch ALL PAPI scores using pagination"""
    if supabase is None:
        return pd.DataFrame()
    
    try:
        all_data = []
        page_size = 1000
        offset = 0
        
        while True:
            response = supabase.table('papi_scores')\
                .select('*')\
                .range(offset, offset + page_size - 1)\
                .execute()
            
            if not response.data:
                break
            
            all_data.extend(response.data)
            
            if len(response.data) < page_size:
                break
            
            offset += page_size
        
        return pd.DataFrame(all_data) if all_data else pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching PAPI scores: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_strengths_data():
    """Fetch ALL strengths using pagination"""
    if supabase is None:
        return pd.DataFrame()
    
    try:
        all_data = []
        page_size = 1000
        offset = 0
        
        while True:
            response = supabase.table('strengths')\
                .select('*')\
                .range(offset, offset + page_size - 1)\
                .execute()
            
            if not response.data:
                break
            
            all_data.extend(response.data)
            
            if len(response.data) < page_size:
                break
            
            offset += page_size
        
        return pd.DataFrame(all_data) if all_data else pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching strengths: {e}")
        return pd.DataFrame()

def search_employees(query, employees_df):
    """
    Enhanced search function that matches employee ID or name
    Searches for exact sequential matches in employee_id
    """
    if query == '' or employees_df.empty:
        return employees_df
    
    query = query.strip().upper()
    
    # Convert employee_id to string for searching
    employees_df['employee_id_str'] = employees_df['employee_id'].astype(str).str.upper()
    employees_df['fullname_str'] = employees_df['fullname'].astype(str).str.upper()
    
    # Search logic: exact sequential match in ID or name contains
    mask = (
        employees_df['employee_id_str'].str.contains(query, na=False, regex=False) |
        employees_df['fullname_str'].str.contains(query, na=False, regex=False)
    )
    
    # For numeric queries, prioritize sequential matches
    if query.isdigit():
        # Find IDs that contain the exact sequence
        sequential_mask = employees_df['employee_id_str'].str.contains(query, na=False, regex=False)
        result = employees_df[sequential_mask].copy()
    else:
        result = employees_df[mask].copy()
    
    # Clean up temporary columns
    result = result.drop(columns=['employee_id_str', 'fullname_str'])
    
    return result

def get_high_performers():
    """Get employees with rating 5 (high performers)"""
    try:
        performance_df = get_performance_data()
        employees_df = get_employees()
        
        if performance_df.empty or employees_df.empty:
            return []
        
        # Filter high performers (rating 5)
        if 'rating' in performance_df.columns:
            high_performers = performance_df[performance_df['rating'] == 5]
            
            # Get latest year for each employee
            if 'year' in high_performers.columns:
                high_performers = high_performers.sort_values('year', ascending=False).drop_duplicates('employee_id')
            
            merged_df = high_performers.merge(employees_df, on='employee_id', how='inner')
            
            if merged_df.empty:
                return []
            
            result = merged_df[['employee_id', 'fullname', 'position_id']].to_dict('records')
            return result
        else:
            return []
            
    except Exception as e:
        st.sidebar.error(f"Error getting high performers: {e}")
        return []

# ======================================
# SAVE JOB VACANCY FUNCTION
# ======================================
def save_job_vacancy(role_name, job_level, role_purpose, benchmark_ids, weights_config=None):
    """Save job vacancy to talent_benchmarks table"""
    if supabase is None:
        st.error("Supabase connection not available")
        return None
    
    try:
        vacancy_data = {
            'role_name': role_name,
            'job_level': job_level,
            'role_purpose': role_purpose,
            'selected_talent_ids': benchmark_ids,
            'weights_config': weights_config or {},
            'created_at': datetime.now().isoformat()
        }
        
        response = supabase.table('talent_benchmarks').insert(vacancy_data).execute()
        
        if response.data:
            job_vacancy_id = response.data[0].get('job_vacancy_id') or response.data[0].get('id')
            st.success(f"Job vacancy saved successfully (ID: {job_vacancy_id})")
            return job_vacancy_id
        else:
            st.warning("Vacancy data saved but ID not returned")
            return None
            
    except Exception as e:
        st.error(f"Error saving vacancy: {e}")
        return None

# ======================================
# AI GENERATION FUNCTION
# ======================================
def generate_job_profile(role_name, job_level, role_purpose):
    """Generate job profile using A4F API"""
    try:
        headers = {
            "Authorization": f"Bearer {A4F_API_KEY}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""
Create a professional job profile for a {role_name} position at {job_level} level.

Role Purpose: {role_purpose}

Please provide the response in this exact JSON format:
{{
    "job_requirements": ["requirement1", "requirement2", "requirement3", "requirement4", "requirement5"],
    "job_description": "A comprehensive description of the role, its responsibilities, and impact on the organization",
    "key_competencies": ["competency1", "competency2", "competency3", "competency4"]
}}

Make it realistic, detailed, and suitable for corporate use.
"""
        
        data = {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1500
        }
        
        response = requests.post(API_ENDPOINT, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            
            # Extract JSON from response
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            parsed_content = json.loads(content)
            return parsed_content
        else:
            st.error(f"AI API Error: {response.status_code}")
            return None
            
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse AI response: {e}")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timeout. Please try again.")
        return None
    except Exception as e:
        st.error(f"Error generating job profile: {str(e)}")
        return None

# ======================================
# DETAILED TV BREAKDOWN FUNCTION
# ======================================
def get_detailed_tv_breakdown(employee_id, baselines, psych_df, comp_df, papi_df):
    """
    Get detailed Talent Variable breakdown for an employee
    Returns DataFrame with: tgv_name, tv_name, baseline_score, user_score, tv_match_rate
    """
    breakdown_data = []
    
    employee_id = str(employee_id)
    
    # Cognitive TVs
    if not psych_df.empty:
        candidate_psych = psych_df[psych_df['employee_id'] == employee_id]
        
        if not candidate_psych.empty:
            for metric in ['iq', 'gtq_total', 'pauli', 'faxtor']:
                baseline_key = f'cognitive_{metric}'
                if baseline_key in baselines:
                    user_score = candidate_psych[metric].values[0] if metric in candidate_psych.columns else None
                    baseline_score = baselines[baseline_key]
                    
                    if pd.notna(user_score) and baseline_score > 0:
                        tv_match = (user_score / baseline_score) * 100
                        tv_match = min(tv_match, 150)
                        
                        breakdown_data.append({
                            'tgv_name': 'Cognitive',
                            'tv_name': metric.upper(),
                            'baseline_score': round(baseline_score, 2),
                            'user_score': round(user_score, 2),
                            'tv_match_rate': round(tv_match, 1)
                        })
    
    # Leadership TVs
    if not comp_df.empty:
        candidate_comp = comp_df[comp_df['employee_id'] == employee_id]
        
        if not candidate_comp.empty:
            latest_year = candidate_comp['year'].max()
            candidate_comp = candidate_comp[candidate_comp['year'] == latest_year]
            
            for pillar in ['GDR', 'CEX', 'IDS']:
                baseline_key = f'leadership_{pillar}'
                if baseline_key in baselines:
                    pillar_data = candidate_comp[candidate_comp['pillar_code'] == pillar]['score']
                    if len(pillar_data) > 0:
                        user_score = pillar_data.values[0]
                        baseline_score = baselines[baseline_key]
                        
                        if baseline_score > 0:
                            tv_match = (user_score / baseline_score) * 100
                            tv_match = min(tv_match, 150)
                            
                            breakdown_data.append({
                                'tgv_name': 'Leadership',
                                'tv_name': pillar,
                                'baseline_score': round(baseline_score, 2),
                                'user_score': round(user_score, 2),
                                'tv_match_rate': round(tv_match, 1)
                            })
    
    # Teamwork TVs
    if not papi_df.empty:
        candidate_papi = papi_df[papi_df['employee_id'] == employee_id]
        
        if not candidate_papi.empty:
            for scale in ['Papi_B', 'Papi_O', 'Papi_R']:
                baseline_key = f'teamwork_{scale}'
                if baseline_key in baselines:
                    scale_data = candidate_papi[candidate_papi['scale_code'] == scale]['score']
                    if len(scale_data) > 0:
                        user_score = scale_data.values[0]
                        baseline_score = baselines[baseline_key]
                        
                        if baseline_score > 0:
                            tv_match = (user_score / baseline_score) * 100
                            tv_match = min(tv_match, 150)
                            
                            breakdown_data.append({
                                'tgv_name': 'Teamwork',
                                'tv_name': scale,
                                'baseline_score': round(baseline_score, 2),
                                'user_score': round(user_score, 2),
                                'tv_match_rate': round(tv_match, 1)
                            })
    
    return pd.DataFrame(breakdown_data)

# ======================================
# REAL MATCHING ALGORITHM
# ======================================
def calculate_match_scores(benchmark_ids, all_employees, weights_config=None):
    """
    Real SQL-based matching algorithm implementing TGV/TV framework
    """
    
    if not benchmark_ids:
        return pd.DataFrame(), {}
    
    # Fetch data
    employees_df = get_employees()
    psych_df = get_psychometric_data()
    comp_df = get_competencies_data()
    papi_df = get_papi_scores()
    strengths_df = get_strengths_data()
    
    # Default weights
    if weights_config is None:
        weights_config = {
            'tgv_weights': {
                'cognitive': 0.30,
                'leadership': 0.25,
                'teamwork': 0.25,
                'technical': 0.20
            },
            'tv_weights': {
                'cognitive_iq': 0.3,
                'cognitive_gtq_total': 0.3,
                'cognitive_pauli': 0.2,
                'cognitive_faxtor': 0.2,
                # ... tambahkan weights untuk TV lainnya
            }
        }
    
    benchmark_ids = [str(bid) for bid in benchmark_ids]
    
    # STEP 1: Calculate Benchmark Baselines (MEDIAN)
    baselines = {}
    
    # 1.1 Cognitive Baselines
    if not psych_df.empty:
        benchmark_psych = psych_df[psych_df['employee_id'].astype(str).isin(benchmark_ids)]
        if not benchmark_psych.empty:
            for metric in ['iq', 'gtq_total', 'pauli', 'faxtor']:
                if metric in benchmark_psych.columns:
                    values = benchmark_psych[metric].dropna()
                    if len(values) > 0:
                        baselines[f'cognitive_{metric}'] = values.median()
    
    # 1.2 Leadership Baselines (dari competency pillars)
    if not comp_df.empty:
        benchmark_comp = comp_df[comp_df['employee_id'].astype(str).isin(benchmark_ids)]
        if not benchmark_comp.empty:
            latest_year = benchmark_comp['year'].max()
            benchmark_comp_latest = benchmark_comp[benchmark_comp['year'] == latest_year]
            
            leadership_pillars = ['GDR', 'CEX', 'IDS', 'STO']  # Expanded pillars
            for pillar in leadership_pillars:
                pillar_data = benchmark_comp_latest[benchmark_comp_latest['pillar_code'] == pillar]['score']
                if len(pillar_data) > 0:
                    baselines[f'leadership_{pillar}'] = pillar_data.median()
    
    # 1.3 Teamwork Baselines (PAPI scales)
    if not papi_df.empty:
        benchmark_papi = papi_df[papi_df['employee_id'].astype(str).isin(benchmark_ids)]
        if not benchmark_papi.empty:
            # Scales yang "higher is better" untuk teamwork
            teamwork_scales = ['Papi_B', 'Papi_O', 'Papi_R']  # Cooperation, Organization, Relations
            for scale in teamwork_scales:
                scale_data = benchmark_papi[benchmark_papi['scale_code'] == scale]['score']
                if len(scale_data) > 0:
                    baselines[f'teamwork_{scale}'] = scale_data.median()
    
    # 1.4 Technical Baselines (dari competency pillars technical)
    if not comp_df.empty:
        technical_pillars = ['QDD', 'SEA', 'VCU', 'FTC', 'CSI']  # Technical pillars
        for pillar in technical_pillars:
            pillar_data = benchmark_comp_latest[benchmark_comp_latest['pillar_code'] == pillar]['score']
            if len(pillar_data) > 0:
                baselines[f'technical_{pillar}'] = pillar_data.median()
    
    # STEP 2: Calculate TV Match Rates dengan formula yang benar
    match_results = []
    
    for emp in all_employees[:200]:  # Increased limit
        emp_id = str(emp.get('employee_id', ''))
        
        if emp_id in benchmark_ids:
            continue
        
        tv_matches = {}
        
        # 2.1 Cognitive TV Matches (higher is better)
        if not psych_df.empty:
            candidate_psych = psych_df[psych_df['employee_id'] == emp_id]
            if not candidate_psych.empty:
                for metric in ['iq', 'gtq_total', 'pauli', 'faxtor']:
                    baseline_key = f'cognitive_{metric}'
                    if baseline_key in baselines:
                        user_score = candidate_psych[metric].values[0] if metric in candidate_psych.columns and pd.notna(candidate_psych[metric].values[0]) else None
                        baseline_score = baselines[baseline_key]
                        
                        if user_score is not None and baseline_score > 0:
                            # HIGHER IS BETTER formula
                            tv_match = (user_score / baseline_score) * 100
                            tv_match = min(max(tv_match, 0), 100)  # Bound 0-100%
                            tv_matches[baseline_key] = tv_match
        
        # 2.2 Leadership TV Matches (higher is better)
        if not comp_df.empty:
            candidate_comp = comp_df[comp_df['employee_id'] == emp_id]
            if not candidate_comp.empty:
                latest_year = candidate_comp['year'].max()
                candidate_comp_latest = candidate_comp[candidate_comp['year'] == latest_year]
                
                for pillar in ['GDR', 'CEX', 'IDS', 'STO']:
                    baseline_key = f'leadership_{pillar}'
                    if baseline_key in baselines:
                        pillar_data = candidate_comp_latest[candidate_comp_latest['pillar_code'] == pillar]['score']
                        if len(pillar_data) > 0 and pd.notna(pillar_data.values[0]):
                            user_score = pillar_data.values[0]
                            baseline_score = baselines[baseline_key]
                            
                            if baseline_score > 0:
                                tv_match = (user_score / baseline_score) * 100
                                tv_match = min(max(tv_match, 0), 100)
                                tv_matches[baseline_key] = tv_match
        
        # 2.3 Teamwork TV Matches (higher is better untuk scales cooperation)
        if not papi_df.empty:
            candidate_papi = papi_df[papi_df['employee_id'] == emp_id]
            if not candidate_papi.empty:
                for scale in ['Papi_B', 'Papi_O', 'Papi_R']:
                    baseline_key = f'teamwork_{scale}'
                    if baseline_key in baselines:
                        scale_data = candidate_papi[candidate_papi['scale_code'] == scale]['score']
                        if len(scale_data) > 0 and pd.notna(scale_data.values[0]):
                            user_score = scale_data.values[0]
                            baseline_score = baselines[baseline_key]
                            
                            if baseline_score > 0:
                                tv_match = (user_score / baseline_score) * 100
                                tv_match = min(max(tv_match, 0), 100)
                                tv_matches[baseline_key] = tv_match
        
        # 2.4 Technical TV Matches (higher is better)
        if not comp_df.empty:
            for pillar in ['QDD', 'SEA', 'VCU', 'FTC', 'CSI']:
                baseline_key = f'technical_{pillar}'
                if baseline_key in baselines:
                    pillar_data = candidate_comp_latest[candidate_comp_latest['pillar_code'] == pillar]['score']
                    if len(pillar_data) > 0 and pd.notna(pillar_data.values[0]):
                        user_score = pillar_data.values[0]
                        baseline_score = baselines[baseline_key]
                        
                        if baseline_score > 0:
                            tv_match = (user_score / baseline_score) * 100
                            tv_match = min(max(tv_match, 0), 100)
                            tv_matches[baseline_key] = tv_match
        
        if not tv_matches:
            continue
        
        # STEP 3: Aggregate TV → TGV dengan weights
        tgv_scores = {}
        tgv_weights = weights_config.get('tgv_weights', {})
        tv_weights = weights_config.get('tv_weights', {})
        
        # Cognitive TGV
        cognitive_tvs = {k: v for k, v in tv_matches.items() if k.startswith('cognitive_')}
        if cognitive_tvs:
            total_weight = 0
            weighted_sum = 0
            for tv, score in cognitive_tvs.items():
                weight = tv_weights.get(tv, 1.0)  # Default weight 1.0 jika tidak ada
                weighted_sum += score * weight
                total_weight += weight
            tgv_scores['cognitive'] = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Leadership TGV
        leadership_tvs = {k: v for k, v in tv_matches.items() if k.startswith('leadership_')}
        if leadership_tvs:
            total_weight = 0
            weighted_sum = 0
            for tv, score in leadership_tvs.items():
                weight = tv_weights.get(tv, 1.0)
                weighted_sum += score * weight
                total_weight += weight
            tgv_scores['leadership'] = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Teamwork TGV
        teamwork_tvs = {k: v for k, v in tv_matches.items() if k.startswith('teamwork_')}
        if teamwork_tvs:
            total_weight = 0
            weighted_sum = 0
            for tv, score in teamwork_tvs.items():
                weight = tv_weights.get(tv, 1.0)
                weighted_sum += score * weight
                total_weight += weight
            tgv_scores['teamwork'] = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Technical TGV
        technical_tvs = {k: v for k, v in tv_matches.items() if k.startswith('technical_')}
        if technical_tvs:
            total_weight = 0
            weighted_sum = 0
            for tv, score in technical_tvs.items():
                weight = tv_weights.get(tv, 1.0)
                weighted_sum += score * weight
                total_weight += weight
            tgv_scores['technical'] = weighted_sum / total_weight if total_weight > 0 else 0
        
        # STEP 4: Final Match Rate dengan TGV weights
        final_match = 0
        total_tgv_weight = 0
        
        for tgv, weight in tgv_weights.items():
            if tgv in tgv_scores:
                final_match += tgv_scores[tgv] * weight
                total_tgv_weight += weight
        
        # Normalize jika total weight tidak 1.0
        if total_tgv_weight > 0:
            final_match = final_match / total_tgv_weight
        else:
            final_match = np.mean(list(tgv_scores.values())) if tgv_scores else 0
        
        match_results.append({
            'employee_id': emp_id,
            'fullname': emp.get('fullname', 'Unknown'),
            'position': emp.get('position_id', 'N/A'),
            'directorate': emp.get('directorate_id', 'N/A'),
            'grade': emp.get('grade_id', 'N/A'),
            'final_match_rate': round(final_match, 1),
            'cognitive_match': round(tgv_scores.get('cognitive', 0), 1),
            'leadership_match': round(tgv_scores.get('leadership', 0), 1),
            'teamwork_match': round(tgv_scores.get('teamwork', 0), 1),
            'technical_match': round(tgv_scores.get('technical', 0), 1),
            'tv_matches_count': len(tv_matches)
        })
    
    if match_results:
        results_df = pd.DataFrame(match_results)
        # Filter hanya kandidat dengan cukup data TV
        results_df = results_df[results_df['tv_matches_count'] >= 3]  # Minimal 3 TV matches
        return results_df.sort_values('final_match_rate', ascending=False), baselines
    else:
        return pd.DataFrame(), {}


# ======================================
# SIDEBAR - DATABASE STATUS
# ======================================
with st.sidebar:
    st.markdown("### System Configuration")
    
    with st.expander("Database Status", expanded=True):
        if supabase:
            st.success("Supabase Connected")
            
            # Clear cache button
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
            
            # Fetch data with updated functions
            emp_df = get_employees()
            perf_df = get_performance_data()
            psych_df = get_psychometric_data()
            comp_df = get_competencies_data()
            papi_df = get_papi_scores()
            
            st.markdown("---")
            st.write(f"**Employees:** {len(emp_df):,} rows")
            st.write(f"**Performance:** {len(perf_df):,} rows")
            st.write(f"**Psychometric:** {len(psych_df):,} rows")
            st.write(f"**Competencies:** {len(comp_df):,} rows")
            st.write(f"**PAPI Scores:** {len(papi_df):,} rows")
            
            # Show warning if hitting limit
            if len(emp_df) >= 10000:
                st.warning("⚠️ Employee limit (10k) reached. Data may be incomplete.")
            if len(comp_df) >= 50000:
                st.warning("⚠️ Competencies limit (50k) reached. Data may be incomplete.")
            if len(papi_df) >= 50000:
                st.warning("⚠️ PAPI limit (50k) reached. Data may be incomplete.")
                
        else:
            st.error("Supabase Not Connected")
    
    st.markdown("---")
    
    if st.button("🔄 Reset Analysis", use_container_width=True):
        st.session_state.analysis_complete = False
        st.session_state.form_data = {
            'role_name': '',
            'job_level': '',
            'role_purpose': '',
            'benchmark_ids': [],
            'job_vacancy_id': None
        }
        st.rerun()

# ======================================
# MAIN FORM INPUT
# ======================================
st.markdown("## Job Vacancy Information")
st.markdown("Complete the form below to create a new job vacancy and find the best matching candidates.")

with st.container():
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    
    st.subheader("Role Details")
    
    # Full width layout - vertical stacking
    st.markdown('<div class="input-label">Role Name</div>', unsafe_allow_html=True)
    role_name = st.text_input(
        "Role Name",
        placeholder="e.g., Data Analyst, Marketing Manager",
        label_visibility="collapsed",
        key="role_name_input"
    )
    st.markdown('<div class="example-text">Example: Data Analyst</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="input-label">Job Level</div>', unsafe_allow_html=True)
    job_level = st.selectbox(
        "Job Level",
        ["", "Junior", "Middle", "Senior", "Executive"],
        label_visibility="collapsed",
        key="job_level_input"
    )
    st.markdown('<div class="example-text">Select the appropriate job level</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="input-label">Role Purpose</div>', unsafe_allow_html=True)
    role_purpose = st.text_area(
        "Role Purpose",
        placeholder="Describe the main purpose and objectives of this role...",
        height=120,
        label_visibility="collapsed",
        key="role_purpose_input"
    )
    st.markdown('<div class="example-text">Example: Analyze business data to provide insights for strategic decision-making</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("Benchmark Employee Selection")
    st.write("Select or enter Employee IDs from top performers (rating 5) to serve as benchmark references:")
    
    high_performers = get_high_performers()
    
    if high_performers:
        benchmark_options = {f"{emp['employee_id']} - {emp['fullname']}": emp['employee_id'] for emp in high_performers}
        
        st.markdown('<div class="input-label">Select Benchmark Employees (maximum 3)</div>', unsafe_allow_html=True)
        selected_benchmarks = st.multiselect(
            "Select Benchmark Employees",
            options=list(benchmark_options.keys()),
            max_selections=3,
            label_visibility="collapsed"
        )
        
        benchmark_ids = [benchmark_options[bm] for bm in selected_benchmarks]
        
        st.markdown('<div class="input-label">Or Enter Benchmark IDs Manually</div>', unsafe_allow_html=True)
        manual_benchmark = st.text_input(
            "Manual Benchmark IDs",
            placeholder="e.g., 312, 335, 175",
            label_visibility="collapsed",
            key="manual_benchmark_input"
        )
        st.markdown('<div class="example-text">Enter IDs separated by commas (e.g., 312, 335, 175)</div>', unsafe_allow_html=True)
        
        if manual_benchmark:
            manual_ids = [id_str.strip() for id_str in manual_benchmark.split(',') if id_str.strip()]
            benchmark_ids.extend(manual_ids)
            benchmark_ids = list(set(benchmark_ids))
        
        if len(benchmark_ids) > 3:
            st.warning("Maximum 3 benchmark employees allowed. Using first 3 IDs.")
            benchmark_ids = benchmark_ids[:3]
        
        if benchmark_ids:
            st.markdown("**Selected Benchmarks:**")
            for bid in benchmark_ids:
                st.write(f"• Employee ID: {bid}")
        
    else:
        st.info("No high performers found in database. Please enter benchmark IDs manually.")
        st.markdown('<div class="input-label">Enter Benchmark Employee IDs</div>', unsafe_allow_html=True)
        manual_benchmark = st.text_input(
            "Benchmark IDs",
            placeholder="e.g., 312, 335, 175",
            label_visibility="collapsed",
            key="manual_benchmark_only"
        )
        st.markdown('<div class="example-text">Enter 1-3 employee IDs separated by commas</div>', unsafe_allow_html=True)
        
        benchmark_ids = []
        if manual_benchmark:
            benchmark_ids = [id_str.strip() for id_str in manual_benchmark.split(',') if id_str.strip()]
            if len(benchmark_ids) > 3:
                st.warning("Maximum 3 benchmark employees allowed. Using first 3 IDs.")
                benchmark_ids = benchmark_ids[:3]
    
    st.markdown('</div>', unsafe_allow_html=True)

# ======================================
# GENERATE BUTTON
# ======================================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate_btn = st.button(
        "Generate Talent Match Analysis", 
        type="primary", 
        use_container_width=True,
        disabled=not all([role_name, job_level, role_purpose, benchmark_ids])
    )

# ======================================
# PROCESS RESULTS
# ======================================
if generate_btn:
    if not all([role_name, job_level, role_purpose, benchmark_ids]):
        st.error("Please complete all required fields!")
    else:
        # Save to database
        job_vacancy_id = save_job_vacancy(role_name, job_level, role_purpose, benchmark_ids)
        
        st.session_state.form_data = {
            'role_name': role_name,
            'job_level': job_level,
            'role_purpose': role_purpose,
            'benchmark_ids': benchmark_ids,
            'job_vacancy_id': job_vacancy_id
        }
        st.session_state.analysis_complete = True
        st.rerun()

elif st.session_state.analysis_complete:
    role_name = st.session_state.form_data['role_name']
    job_level = st.session_state.form_data['job_level']
    role_purpose = st.session_state.form_data['role_purpose']
    benchmark_ids = st.session_state.form_data['benchmark_ids']
    job_vacancy_id = st.session_state.form_data.get('job_vacancy_id')
    
    st.markdown("## Talent Match Analysis Report")
    st.markdown("Below are the analysis results based on your input parameters.")

    with st.spinner("Generating talent analysis..."):
        
        # AI Job Profile
        st.markdown("---")
        st.markdown('<div class="section-header">AI-Generated Job Profile</div>', unsafe_allow_html=True)
        
        if job_vacancy_id:
            st.markdown(f'<div class="info-box">Job Vacancy ID: <strong>{job_vacancy_id}</strong></div>', unsafe_allow_html=True)
        
        job_profile = generate_job_profile(role_name, job_level, role_purpose)
        
        if job_profile:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="analysis-container">', unsafe_allow_html=True)
                st.subheader("Job Requirements")
                requirements = job_profile.get('job_requirements', [])
                if isinstance(requirements, list):
                    for req in requirements:
                        st.write(f"• {req}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="analysis-container">', unsafe_allow_html=True)
                st.subheader("Job Description")
                description = job_profile.get('job_description', '')
                if description:
                    st.write(description)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="analysis-container">', unsafe_allow_html=True)
                st.subheader("Key Competencies")
                competencies = job_profile.get('key_competencies', [])
                if isinstance(competencies, list):
                    for comp in competencies:
                        st.write(f"• {comp}")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("Failed to generate job profile. Please check API connection or try again.")
        
        # Talent Matches
        st.markdown("---")
        st.markdown('<div class="section-header">Best Matching Candidates</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-box">
        <strong>Analysis Summary</strong><br>
        Role: {role_name} ({job_level})<br>
        Benchmarks: {', '.join(benchmark_ids)}<br>
        Purpose: {role_purpose}
        </div>
        """, unsafe_allow_html=True)
        
        all_employees = get_employees().to_dict('records')
        match_results, baselines = calculate_match_scores(benchmark_ids, all_employees)
        
        if not match_results.empty and 'final_match_rate' in match_results.columns:
            # Metrics Cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                avg_match = match_results['final_match_rate'].mean()
                st.metric("Average Match", f"{avg_match:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                top_match = match_results['final_match_rate'].max()
                st.metric("Highest Match", f"{top_match:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                qualified_count = len(match_results[match_results['final_match_rate'] >= 70])
                st.metric("Qualified Candidates", f"{qualified_count}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                total_candidates = len(match_results)
                st.metric("Total Candidates", f"{total_candidates}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Top 10 Matches
            st.markdown("---")
            st.subheader("Top 10 Matching Candidates")
            
            top_matches = match_results.head(10)
            
            for idx, match in top_matches.iterrows():
                match_rate = match['final_match_rate']
                
                if match_rate >= 80:
                    css_class = "match-high"
                    label = "EXCELLENT MATCH"
                elif match_rate >= 70:
                    css_class = "match-medium" 
                    label = "GOOD MATCH"
                else:
                    css_class = "match-low"
                    label = "NEEDS DEVELOPMENT"
                
                st.markdown(f"""
                <div class="{css_class}">
                    <h4>{match['fullname']} - {match_rate}% Match ({label})</h4>
                    <p><strong>Position:</strong> {match['position']} | <strong>Employee ID:</strong> {match['employee_id']} | <strong>Grade:</strong> {match['grade']}</p>
                    <small>
                        <strong>TGV Competency Breakdown:</strong><br>
                        • Cognitive: {match['cognitive_match']}% | 
                        • Leadership: {match['leadership_match']}%<br>
                        • Teamwork: {match['teamwork_match']}% | 
                        • Technical: {match['technical_match']}%
                    </small>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed TV Breakdown for Top Candidate
            st.markdown("---")
            st.markdown('<div class="section-header">Detailed Talent Variable Breakdown - Top Candidate</div>', unsafe_allow_html=True)
            
            if baselines and not top_matches.empty:
                top_candidate_id = top_matches.iloc[0]['employee_id']
                top_candidate_name = top_matches.iloc[0]['fullname']
                
                st.write(f"**Candidate:** {top_candidate_name} (ID: {top_candidate_id})")
                
                psych_df = get_psychometric_data()
                comp_df = get_competencies_data()
                papi_df = get_papi_scores()
                
                tv_breakdown = get_detailed_tv_breakdown(
                    top_candidate_id, 
                    baselines, 
                    psych_df, 
                    comp_df, 
                    papi_df
                )
                
                if not tv_breakdown.empty:
                    st.dataframe(
                        tv_breakdown.style.background_gradient(
                            subset=['tv_match_rate'],
                            cmap='RdYlGn',
                            vmin=50,
                            vmax=150
                        ).format({
                            'baseline_score': '{:.2f}',
                            'user_score': '{:.2f}',
                            'tv_match_rate': '{:.1f}%'
                        }),
                        use_container_width=True
                    )
                else:
                    st.info("Detailed TV breakdown not available for this candidate.")
            
            # Visualizations
            st.markdown("---")
            st.markdown('<div class="section-header">Data Visualizations</div>', unsafe_allow_html=True)
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Match Rate Distribution
                fig_dist = px.histogram(
                    match_results, 
                    x='final_match_rate',
                    nbins=20,
                    title='Match Rate Distribution',
                    labels={'final_match_rate': 'Match Rate (%)', 'count': 'Number of Candidates'},
                    color_discrete_sequence=['#3b82f6']
                )
                fig_dist.update_layout(
                    showlegend=False,
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with viz_col2:
                # Top 10 Candidates Bar Chart
                top10_data = match_results.head(10)
                fig_top10 = px.bar(
                    top10_data,
                    x='final_match_rate',
                    y='fullname',
                    orientation='h',
                    title='Top 10 Candidates by Match Rate',
                    labels={'final_match_rate': 'Match Rate (%)', 'fullname': 'Candidate'},
                    color='final_match_rate',
                    color_continuous_scale='RdYlGn'
                )
                fig_top10.update_layout(
                    showlegend=False,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig_top10, use_container_width=True)
            
            # TGV Competency Analysis
            st.markdown("---")
            st.subheader("TGV Competency Analysis - Top 5 Candidates")
            
            top5 = match_results.head(5)
            
            # Radar Chart for TGV comparison
            fig_comp = go.Figure()
            
            tgv_labels = ['Cognitive', 'Leadership', 'Teamwork', 'Technical']
            
            for idx, row in top5.iterrows():
                fig_comp.add_trace(go.Scatterpolar(
                    r=[
                        row['cognitive_match'],
                        row['leadership_match'],
                        row['teamwork_match'],
                        row['technical_match']
                    ],
                    theta=tgv_labels,
                    fill='toself',
                    name=row['fullname']
                ))
            
            fig_comp.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=True,
                title='TGV Competency Radar Chart - Top 5 Candidates',
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # TGV Breakdown Table
            st.markdown("---")
            st.subheader("Detailed TGV Breakdown - Top 10")
            
            tgv_breakdown = top_matches[['employee_id', 'fullname', 'position', 'grade', 'final_match_rate', 
                                         'cognitive_match', 'leadership_match', 'teamwork_match', 'technical_match']].copy()
            tgv_breakdown.columns = ['Employee ID', 'Candidate', 'Position', 'Grade', 'Final Match %', 
                                     'Cognitive %', 'Leadership %', 'Teamwork %', 'Technical %']
            
            st.dataframe(
                tgv_breakdown.style.background_gradient(
                    subset=['Final Match %', 'Cognitive %', 'Leadership %', 'Teamwork %', 'Technical %'],
                    cmap='RdYlGn',
                    vmin=50,
                    vmax=100
                ).format({
                    'Final Match %': '{:.1f}',
                    'Cognitive %': '{:.1f}',
                    'Leadership %': '{:.1f}',
                    'Teamwork %': '{:.1f}',
                    'Technical %': '{:.1f}'
                }),
                use_container_width=True
            )
            
            # Average TGV Comparison
            st.markdown("---")
            st.subheader("Average TGV Scores Comparison")
            
            avg_tgv = {
                'TGV': ['Cognitive', 'Leadership', 'Teamwork', 'Technical'],
                'Average Score': [
                    match_results['cognitive_match'].mean(),
                    match_results['leadership_match'].mean(),
                    match_results['teamwork_match'].mean(),
                    match_results['technical_match'].mean()
                ]
            }
            
            fig_avg = px.bar(
                avg_tgv,
                x='TGV',
                y='Average Score',
                title='Average TGV Scores Across All Candidates',
                color='Average Score',
                color_continuous_scale='Viridis',
                text='Average Score'
            )
            fig_avg.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_avg.update_layout(
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white',
                yaxis_range=[0, 100]
            )
            st.plotly_chart(fig_avg, use_container_width=True)
            
            # Download Results
            st.markdown("---")
            st.subheader("Download Results")
            
            csv = match_results.to_csv(index=False)
            st.download_button(
                label="Download Full Results (CSV)",
                data=csv,
                file_name=f"talent_match_results_{role_name.replace(' ', '_')}.csv",
                mime="text/csv"
            )
            
        else:
            st.error("No analysis results available. Please check your database connection.")

else:
    # Instructions
    st.markdown("---")
    st.markdown("""
    ### How to Use the System
    
    **Step 1: Fill Complete Form**
    - **Role Name**: Position name you're looking for
    - **Job Level**: Job level (Junior/Middle/Senior/Executive)  
    - **Role Purpose**: Main purpose and responsibilities
    - **Benchmark IDs**: Select or enter 1-3 high performer employee IDs
    
    **Step 2: Generate Analysis**
    Click the "Generate Talent Match Analysis" button
    
    **Step 3: Review Results**
    Check AI-generated profile, candidate rankings, and insights
    
    ### Example Input
    
    | Field | Example Value |
    |-------|---------------|
    | Role Name | Data Analyst |
    | Job Level | Middle |
    | Role Purpose | Analyze business data for strategic decision-making |
    | Benchmark IDs | 312, 335, 175 |
    
    ### Key Features
    
    - **AI-Powered Job Profiling**: Automatically generates comprehensive job descriptions
    - **Real SQL Matching Algorithm**: Implements TGV/TV framework with actual Supabase data
    - **Competency Analysis**: Detailed breakdown of Cognitive, Leadership, Teamwork, and Technical skills
    - **Visual Analytics**: Interactive charts and graphs for better insights
    - **Download Results**: Export matching results to CSV
    
    ### TGV/TV Framework
    
    **Talent Group Variables (TGV):**
    - **Cognitive**: IQ, GTQ, Pauli, Faxtor scores
    - **Leadership**: Competency pillars (GDR, CEX, IDS)
    - **Teamwork**: PAPI scales (collaboration, relations)
    - **Technical**: Role-specific skills and capabilities
    
    **Matching Process:**
    1. Calculate median baseline from benchmark employees
    2. Compare each candidate's scores to baseline
    3. Aggregate individual metrics into TGV scores
    4. Compute weighted final match rate
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6b7280; font-size: 0.9rem;'>"
    "Talent Match Intelligence System | Powered by AI & Data Analytics<br>"
    "© 2024 | Built with Streamlit, Supabase & A4F API<br>"
    "Implements TGV/TV Matching Algorithm with Real SQL Data"
    "</div>", 
    unsafe_allow_html=True
)