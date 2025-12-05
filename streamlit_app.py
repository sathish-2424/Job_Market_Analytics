import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import itertools
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# ===== STREAMLIT CONFIG =====
st.set_page_config(
    page_title="TalentScout | Market Intelligence",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Custom CSS for dashboard styling
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stMetric {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #333;
    }
    </style>
""", unsafe_allow_html=True)

# ===== SYNTHETIC DATA GENERATOR =====
@st.cache_data
def generate_dummy_data(rows=500):
    """Generates realistic dummy data for demo purposes"""
    np.random.seed(42)
    
    titles = ['Data Scientist', 'Software Engineer', 'Product Manager', 'UX Designer', 'DevOps Engineer', 'Data Analyst', 'Frontend Dev', 'Backend Dev']
    companies = ['TechCorp', 'DataFlow', 'InnovateX', 'CloudSys', 'SoftServe', 'AlphaWave', 'CyberNet', 'FutureScale']
    countries = ['USA', 'UK', 'Germany', 'Canada', 'India', 'Australia', 'France']
    cities = ['San Francisco', 'London', 'Berlin', 'Toronto', 'Bangalore', 'Sydney', 'Paris', 'New York', 'Austin']
    modes = ['Remote', 'Hybrid', 'On-site']
    contracts = ['Full-time', 'Contract', 'Internship', 'Part-time']
    
    skill_sets = [
        "Python, SQL, Machine Learning, AWS",
        "React, JavaScript, CSS, HTML, Node.js",
        "Python, Django, PostgreSQL, Docker",
        "Java, Spring Boot, AWS, Microservices",
        "Figma, Adobe XD, User Research, Prototyping",
        "Kubernetes, Docker, Jenkins, Terraform, AWS",
        "Excel, Tableau, SQL, PowerBI",
        "C++, Python, Linux, Git"
    ]
    
    data = {
        'job_title': np.random.choice(titles, rows),
        'company': np.random.choice(companies, rows),
        'country': np.random.choice(countries, rows),
        'city': np.random.choice(cities, rows),
        'date_posted': [datetime.today() - timedelta(days=np.random.randint(0, 60)) for _ in range(rows)],
        'mode': np.random.choice(modes, rows, p=[0.3, 0.4, 0.3]),
        'contract': np.random.choice(contracts, rows, p=[0.7, 0.15, 0.1, 0.05]),
        'linkedin_skills': [np.random.choice(skill_sets) + ", " + np.random.choice(["Git", "Agile", "Jira", "Communication"]) for _ in range(rows)]
    }
    
    return pd.DataFrame(data)

# ===== DATA LOADING =====
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip().str.lower()
            
            # Date Parsing
            if 'date_posted' in df.columns:
                df['date_posted'] = pd.to_datetime(df['date_posted'], errors='coerce')
                
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    return generate_dummy_data()

# ===== HELPER FUNCTIONS =====
def extract_skills_list(df):
    """Flattens comma-separated skills into a single list"""
    if 'linkedin_skills' not in df.columns:
        return []
    
    all_skills = []
    for skill_str in df['linkedin_skills'].dropna():
        # Split by comma, strip whitespace, and normalize case
        skills = [s.strip() for s in str(skill_str).split(',') if s.strip()]
        all_skills.extend(skills)
    return all_skills

def plot_interactive_bar(df, col, title, color_seq, top_n=15):
    """Generic interactive bar chart"""
    counts = df[col].value_counts().head(top_n).reset_index()
    counts.columns = [col, 'count']
    
    fig = px.bar(
        counts, x='count', y=col, orientation='h',
        title=title, text='count',
        color='count', color_continuous_scale=color_seq,
        template='plotly_dark'
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=400)
    return fig

# ===== SIDEBAR =====
st.sidebar.title("⚙️ TalentScout")
uploaded_file = st.sidebar.file_uploader("Upload Job Data (CSV)", type=['csv'])

if not uploaded_file:
    st.sidebar.info("ℹ️ Using synthetic demo data.")

df_raw = load_data(uploaded_file)

if df_raw is None:
    st.stop()

# --- FILTER LOGIC ---
st.sidebar.divider()
st.sidebar.subheader("🔍 Filters")

# Date Filter
if 'date_posted' in df_raw.columns:
    min_date = df_raw['date_posted'].min().date()
    max_date = df_raw['date_posted'].max().date()
    date_range = st.sidebar.date_input("Date Range", [min_date, max_date])
    
    if len(date_range) == 2:
        mask = (df_raw['date_posted'].dt.date >= date_range[0]) & (df_raw['date_posted'].dt.date <= date_range[1])
        df = df_raw.loc[mask]
    else:
        df = df_raw.copy()
else:
    df = df_raw.copy()

# Categorical Filters
if 'country' in df.columns:
    countries = st.sidebar.multiselect("Country", df['country'].unique())
    if countries: df = df[df['country'].isin(countries)]

if 'job_title' in df.columns:
    jobs = st.sidebar.multiselect("Job Title", df['job_title'].unique())
    if jobs: df = df[df['job_title'].isin(jobs)]

st.sidebar.markdown(f"**Showing {len(df)} records**")

# ===== MAIN LAYOUT =====
st.title("📊 Global Job Market Intelligence")

# KPI Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Active Jobs", len(df))
m2.metric("Unique Companies", df['company'].nunique() if 'company' in df.columns else 0)
m3.metric("Locations", df['city'].nunique() if 'city' in df.columns else 0)
m4.metric("Avg Skills per Job", int(df['linkedin_skills'].str.count(',').mean() + 1) if 'linkedin_skills' in df.columns else 0)

# TABS
tab1, tab2, tab3 = st.tabs(["🌎 Market Overview", "🧠 Skill Ecosystem", "🔬 Deep Dive Analysis"])

# --- TAB 1: MARKET OVERVIEW ---
with tab1:
    c1, c2 = st.columns([2, 1])
    
    with c1:
        if 'job_title' in df.columns:
            st.plotly_chart(plot_interactive_bar(df, 'job_title', 'Top In-Demand Roles', 'Viridis'), use_container_width=True)
            
        if 'company' in df.columns:
            st.plotly_chart(plot_interactive_bar(df, 'company', 'Top Hiring Companies', 'Magma'), use_container_width=True)
            
    with c2:
        if 'mode' in df.columns:
            st.subheader("Work Mode")
            counts = df['mode'].value_counts()
            fig_pie = px.pie(names=counts.index, values=counts.values, hole=0.4, template='plotly_dark', color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        if 'contract' in df.columns:
            st.subheader("Contract Type")
            counts = df['contract'].value_counts()
            fig_pie2 = px.pie(names=counts.index, values=counts.values, hole=0.4, template='plotly_dark', color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig_pie2, use_container_width=True)

# --- TAB 2: SKILL ECOSYSTEM ---
with tab2:
    st.subheader("🔥 Skill Demand Analysis")
    
    all_skills = extract_skills_list(df)
    
    if all_skills:
        # Top Skills Bar Chart
        skill_counts = pd.Series(all_skills).value_counts().head(20).reset_index()
        skill_counts.columns = ['Skill', 'Count']
        
        fig_skills = px.bar(
            skill_counts, x='Skill', y='Count',
            color='Count', title='Most Frequent Skills in Job Descriptions',
            template='plotly_dark', color_continuous_scale='Teal'
        )
        st.plotly_chart(fig_skills, use_container_width=True)
        
        # Word Cloud
        col_wc, col_net = st.columns([1, 1])
        
        with col_wc:
            st.subheader("☁️ Skill Cloud")
            try:
                text = " ".join(skill.replace(" ", "_") for skill in all_skills) # Underscore to keep phrases together
                wc = WordCloud(width=800, height=400, background_color='#0E1117', colormap='viridis', max_words=100).generate(text)
                
                fig_wc, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                fig_wc.patch.set_facecolor('#0E1117') # Match Streamlit Dark Theme
                st.pyplot(fig_wc)
            except Exception as e:
                st.warning("Could not generate WordCloud")
        
        with col_net:
            st.subheader("🔗 Skill Co-occurrence Heatmap")
            # Analyze pairs
            if 'linkedin_skills' in df.columns:
                pair_counts = Counter()
                for skills_str in df['linkedin_skills'].dropna():
                    skills = sorted([s.strip() for s in str(skills_str).split(',') if s.strip()])
                    pair_counts.update(itertools.combinations(skills, 2))
                
                if pair_counts:
                    top_pairs = pair_counts.most_common(15)
                    pairs_df = pd.DataFrame(top_pairs, columns=['Pair', 'Count'])
                    pairs_df['Skill A'] = pairs_df['Pair'].apply(lambda x: x[0])
                    pairs_df['Skill B'] = pairs_df['Pair'].apply(lambda x: x[1])
                    
                    # Transform to matrix for heatmap
                    pivot = pairs_df.pivot(index='Skill A', columns='Skill B', values='Count').fillna(0)
                    fig_heat = px.imshow(pivot, text_auto=True, color_continuous_scale='RdBu_r', template='plotly_dark')
                    st.plotly_chart(fig_heat, use_container_width=True)

# --- TAB 3: DEEP DIVE ---
with tab3:
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("📊 Market Concentration (HHI Index)")
        st.markdown("Is the market dominated by a few players or highly competitive?")
        
        if 'company' in df.columns:
            counts = df['company'].value_counts()
            shares = (counts / len(df)) ** 2
            hhi = shares.sum()
            
            # Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = hhi,
                title = {'text': "Company Concentration (HHI)"},
                delta = {'reference': 0.15, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                gauge = {
                    'axis': {'range': [None, 0.5]},
                    'bar': {'color': "#F63366"},
                    'steps': [
                        {'range': [0, 0.15], 'color': "#00CC96"}, # Competitive
                        {'range': [0.15, 0.25], 'color': "#FFB302"}, # Moderate
                        {'range': [0.25, 0.5], 'color': "#FF6692"}], # Concentrated
                    'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': hhi}
                }
            ))
            fig_gauge.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            if hhi < 0.15: st.success("🟢 Highly Competitive Market")
            elif hhi < 0.25: st.warning("🟡 Moderately Concentrated")
            else: st.error("🔴 Highly Concentrated Market")

    with c2:
        st.subheader("🌍 Geographic Hotspots")
        if 'country' in df.columns:
            geo_counts = df['country'].value_counts().reset_index()
            geo_counts.columns = ['Country', 'Jobs']
            
            fig_geo = px.choropleth(
                geo_counts, 
                locations='Country', 
                locationmode='country names',
                color='Jobs',
                color_continuous_scale='Plasma',
                template='plotly_dark'
            )
            st.plotly_chart(fig_geo, use_container_width=True)
        else:
            st.info("Country data not available for map visualization")