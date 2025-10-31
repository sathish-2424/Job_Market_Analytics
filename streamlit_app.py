import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import itertools
import warnings
warnings.filterwarnings('ignore')

# ===== STREAMLIT CONFIG =====
st.set_page_config(page_title="Job Market Dashboard", layout="wide", page_icon="📊")
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ===== CACHE DATA LOADING =====
@st.cache_data
def load_data(uploaded_file):
    """Load and validate CSV data"""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ Loaded {uploaded_file.name}")
        else:
            df = pd.read_csv('dataset.csv')
            st.sidebar.success("✅ Loaded default dataset.csv")
        
        # Basic validation
        if df.empty:
            st.error("❌ Dataset is empty")
            return None
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower()
        
        return df
    
    except FileNotFoundError:
        st.error("❌ No dataset.csv found. Please upload a CSV file.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

# ===== HELPER FUNCTIONS =====
def safe_head(df, n=5):
    """Safely display first n rows"""
    return df.head(n)

def safe_column_check(df, columns):
    """Check if columns exist in dataframe"""
    return [col for col in columns if col in df.columns]

def plot_top_counts(df, column, title, top_n=15, palette='viridis'):
    """Bar plot of top N values in a column"""
    if column not in df.columns:
        st.warning(f"Column '{column}' not found in dataset")
        return None
    
    counts = df[column].value_counts().head(top_n)
    if counts.empty:
        st.info(f"No data available for '{column}'")
        return None
    
    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.3)))
    sns.barplot(x=counts.values, y=counts.index, palette=palette, ax=ax)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Count', fontsize=11)
    ax.set_ylabel(column.title(), fontsize=11)
    plt.tight_layout()
    
    return fig

def plot_pie(series, title):
    """Pie chart for categorical distribution"""
    counts = series.value_counts()
    if counts.empty:
        st.info("No data available for pie chart")
        return None
    
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = sns.color_palette("Set3", len(counts))
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, 
           colors=colors, textprops={'fontsize': 10})
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    return fig

def generate_wordcloud(text, width=1200, height=600):
    """Generate word cloud from text"""
    if not text or len(text.strip()) == 0:
        st.info("No text data available for word cloud")
        return None
    
    try:
        wc = WordCloud(width=width, height=height, background_color="white",
                      colormap="viridis", max_words=200).generate(text)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis('off')
        return fig
    except Exception as e:
        st.warning(f"Could not generate word cloud: {str(e)}")
        return None

def analyze_correlations_streamlit(df):
    """Analyze correlations between categorical variables"""
    try:
        required_cols = ['country', 'mode', 'contract']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            st.warning(f"Missing columns for correlation: {', '.join(missing)}")
            return None, None
        
        le_country = LabelEncoder()
        le_mode = LabelEncoder()
        le_contract = LabelEncoder()

        df_encoded = df.copy()
        df_encoded['country_encoded'] = le_country.fit_transform(df['country'].astype(str))
        df_encoded['mode_encoded'] = le_mode.fit_transform(df['mode'].astype(str))
        df_encoded['contract_encoded'] = le_contract.fit_transform(df['contract'].astype(str))

        corr_matrix = df_encoded[['country_encoded', 'mode_encoded', 'contract_encoded']].corr()
        
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5, ax=ax, fmt='.2f')
        ax.set_title('Correlation: Country, Mode, Contract', fontsize=12, fontweight='bold')
        
        return corr_matrix, fig
    
    except Exception as e:
        st.error(f"Error in correlation analysis: {str(e)}")
        return None, None

def calculate_market_concentration(df, column):
    """Calculate Herfindahl-Hirschman Index (HHI)"""
    if column not in df.columns:
        return None, "Column not found"
    
    counts = df[column].value_counts()
    if counts.empty:
        return None, "No data"
    
    market_shares = counts / counts.sum()
    hhi = (market_shares ** 2).sum()
    
    if hhi > 0.25:
        interpretation = '🔴 Highly Concentrated'
    elif hhi > 0.15:
        interpretation = '🟡 Moderately Concentrated'
    else:
        interpretation = '🟢 Competitive'
    
    return hhi, interpretation

def analyze_skill_combinations_streamlit(df, min_support=5, top_n=15):
    """Analyze co-occurring skill pairs"""
    if 'linkedin_skills' not in df.columns:
        return {}, None
    
    try:
        skills_lists = df['linkedin_skills'].dropna().apply(
            lambda x: [s.strip() for s in str(x).split(',') if s.strip()]
        )
        
        if len(skills_lists) == 0:
            return {}, None
        
        skill_pairs = []
        for skills in skills_lists:
            if len(skills) >= 2:
                skill_pairs.extend(list(itertools.combinations(sorted(skills), 2)))
        
        pair_counts = Counter(skill_pairs)
        frequent_pairs = {pair: count for pair, count in pair_counts.items() 
                         if count >= min_support}
        
        if not frequent_pairs:
            return frequent_pairs, None
        
        pairs_df = pd.DataFrame(list(frequent_pairs.items()), 
                               columns=['skill_pair', 'count'])
        pairs_df['pair_label'] = pairs_df['skill_pair'].apply(
            lambda x: f"{x[0]} + {x[1]}"
        )
        
        fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
        top_pairs = pairs_df.nlargest(top_n, 'count')
        sns.barplot(data=top_pairs, x='count', y='pair_label', palette='viridis', ax=ax)
        ax.set_title('Top Skill Combinations', fontsize=12, fontweight='bold')
        ax.set_xlabel('Frequency', fontsize=11)
        ax.set_ylabel('Skill Pair', fontsize=11)
        plt.tight_layout()
        
        return frequent_pairs, fig
    
    except Exception as e:
        st.warning(f"Error analyzing skill combinations: {str(e)}")
        return {}, None

def create_interactive_country_chart(df):
    """Create interactive Plotly bar chart for job distribution"""
    if 'country' not in df.columns:
        st.warning("Column 'country' not found")
        return None
    
    try:
        country_data = df['country'].value_counts().head(15).reset_index()
        country_data.columns = ['country', 'job_count']
        
        fig = px.bar(country_data, x='job_count', y='country', orientation='h',
                    title='Job Distribution by Country (Top 15)',
                    labels={'job_count': 'Number of Jobs', 'country': 'Country'},
                    color='job_count', color_continuous_scale='Viridis')
        fig.update_layout(height=500, showlegend=False)
        
        return fig
    
    except Exception as e:
        st.warning(f"Error creating interactive chart: {str(e)}")
        return None

def extract_skills(skills_str):
    """Safely extract skills from comma-separated string"""
    if pd.isna(skills_str):
        return []
    return [skill.strip() for skill in str(skills_str).split(',') if skill.strip()]

# ===== SIDEBAR =====
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("Upload your job market dataset (CSV) or use default 'dataset.csv'")

uploaded_file = st.sidebar.file_uploader("📁 Upload CSV", type=['csv'])
df = load_data(uploaded_file)

if df is None:
    st.stop()

st.sidebar.markdown("---")

# Page navigation
page = st.sidebar.radio("📍 Navigation", 
    ["Data Overview", "Analysis", "Results"], 
    index=0
)

# Filters
st.sidebar.subheader("🔍 Filters")

if 'date_posted' in df.columns:
    try:
        df['date_posted'] = pd.to_datetime(df['date_posted'], errors='coerce')
        valid_dates = df['date_posted'].dropna()
        
        if len(valid_dates) > 0:
            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()
            date_range = st.sidebar.date_input("📅 Date Range", 
                value=[min_date, max_date],
                min_value=min_date,
                max_value=max_date
            )
            
            if len(date_range) == 2:
                start_date, end_date = pd.to_datetime(date_range)
                df = df[(df['date_posted'] >= start_date) & (df['date_posted'] <= end_date)]
                st.sidebar.info(f"📊 Filtered: {len(df)} records")
    except Exception as e:
        st.sidebar.warning(f"Date filter error: {str(e)}")

# Additional filters
if 'country' in df.columns:
    countries = sorted(df['country'].dropna().unique().tolist())
    selected_countries = st.sidebar.multiselect("🌍 Countries", countries)
    if selected_countries:
        df = df[df['country'].isin(selected_countries)]

if 'job_title' in df.columns:
    job_titles = sorted(df['job_title'].dropna().unique().tolist())
    selected_jobs = st.sidebar.multiselect("💼 Job Titles", job_titles[:20])
    if selected_jobs:
        df = df[df['job_title'].isin(selected_jobs)]

st.sidebar.markdown("---")
st.sidebar.caption("App version: 2.0 | Enhanced job market analysis")

# ===== MAIN HEADER =====
st.title("📊 Job Market Analysis Dashboard")
st.markdown("Explore hiring trends, skills demand, and market insights.")

col1, col2, col3 = st.columns(3)
col1.metric("📈 Total Jobs", len(df))
if 'company' in df.columns:
    col2.metric("🏢 Companies", df['company'].nunique())
if 'country' in df.columns:
    col3.metric("🌍 Countries", df['country'].nunique())

st.divider()

# ===== PAGE: DATA OVERVIEW =====
if page == 'Data Overview':
    st.header("📋 Data Overview")
    
    with st.expander("🔎 Dataset Preview & Info", expanded=True):
        st.subheader("Sample Data")
        st.dataframe(safe_head(df, 10), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Shape")
            st.write(f"**Rows:** {df.shape[0]:,}")
            st.write(f"**Columns:** {df.shape[1]}")
        
        with col2:
            st.subheader("Missing Data")
            missing = df.isnull().sum()
            missing_pct = (missing / len(df) * 100)
            missing_df = pd.DataFrame({
                'Column': missing.index,
                'Missing': missing.values,
                'Percentage': missing_pct.values
            }).query('Missing > 0').sort_values('Missing', ascending=False)
            
            if not missing_df.empty:
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("✅ No missing values")
    
    # Statistics
    st.subheader("📊 Statistical Summary")
    st.dataframe(df.describe(include='all').round(2), use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🏆 Top Job Titles")
        fig = plot_top_counts(df, 'job_title', 'Top 15 Job Titles', top_n=15, palette='viridis')
        if fig:
            st.pyplot(fig)
        
        st.subheader("🌆 Top Cities")
        fig = plot_top_counts(df, 'city', 'Top 15 Cities', top_n=15, palette='coolwarm')
        if fig:
            st.pyplot(fig)
    
    with col2:
        st.subheader("📌 Work Mode")
        if 'mode' in df.columns:
            fig = plot_pie(df['mode'], 'Work Mode Distribution')
            if fig:
                st.pyplot(fig)
        
        st.subheader("📝 Contract Types")
        if 'contract' in df.columns:
            fig = plot_top_counts(df, 'contract', 'Contract Distribution', 
                                 top_n=10, palette='Set2')
            if fig:
                st.pyplot(fig)
    
    # Skills analysis
    if 'linkedin_skills' in df.columns:
        st.divider()
        st.subheader("💡 Skills Analysis")
        
        all_skills = []
        for skills_str in df['linkedin_skills'].dropna():
            all_skills.extend(extract_skills(skills_str))
        
        if all_skills:
            skill_counts = pd.Series(all_skills).value_counts().head(20)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=skill_counts.values, y=skill_counts.index, palette='viridis', ax=ax)
            ax.set_title('Top 20 Skills in Demand', fontsize=13, fontweight='bold')
            ax.set_xlabel('Frequency', fontsize=11)
            plt.tight_layout()
            st.pyplot(fig)
            

        else:
            st.info("No skills data available")

# ===== PAGE: ANALYSIS =====
elif page == 'Analysis':
    st.header("🔬 Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔗 Correlation Matrix")
        if all(col in df.columns for col in ['country', 'mode', 'contract']):
            corr_matrix, corr_fig = analyze_correlations_streamlit(df)
            if corr_fig:
                st.pyplot(corr_fig)
                if corr_matrix is not None:
                    st.dataframe(corr_matrix.round(3), use_container_width=True)
        else:
            st.info("Requires columns: country, mode, contract")
    
    with col2:
        st.subheader("📦 Skill Combinations")
        min_support = st.slider("Minimum Support (co-occurrence count)", 2, 10, 5)
        frequent_pairs, pairs_fig = analyze_skill_combinations_streamlit(df, min_support=min_support)
        
        if pairs_fig:
            st.pyplot(pairs_fig)
        else:
            st.info(f"No skill pairs found with min_support ≥ {min_support}")
    
    st.divider()
    
    # Market concentration
    st.subheader("📊 Market Concentration (HHI Index)")
    cols_to_check = [c for c in ['country', 'company', 'sector'] if c in df.columns]
    
    if cols_to_check:
        hhi_results = {}
        cols_display = st.columns(len(cols_to_check))
        
        for idx, col in enumerate(cols_to_check):
            hhi, interp = calculate_market_concentration(df, col)
            hhi_results[col] = {
                'hhi_score': float(hhi) if hhi else None,
                'interpretation': interp
            }
            
            with cols_display[idx]:
                st.metric(col.title(), f"{hhi:.3f}" if hhi else "N/A")
                st.write(interp)
        
        with st.expander("ℹ️ What is HHI?"):
            st.markdown("""
            **Herfindahl-Hirschman Index (HHI):**
            - **< 0.15**: Competitive market 🟢
            - **0.15 - 0.25**: Moderately concentrated 🟡
            - **> 0.25**: Highly concentrated 🔴
            """)
    else:
        st.info("No suitable columns for market concentration analysis")

# ===== PAGE: RESULTS =====
elif page == 'Results':
    st.header("📈 Results & Summary")
    
    # Top companies
    if 'company' in df.columns:
        st.subheader("🏢 Top Hiring Companies")
        top_companies = df['company'].value_counts().head(15)
        
        fig = plot_top_counts(df, 'company', 'Top 15 Companies Hiring', top_n=15, palette='magma')
        if fig:
            st.pyplot(fig)
        
        st.subheader("Company Details")
        company_table = top_companies.reset_index()
        company_table.columns = ['Company', 'Job Postings']
        st.dataframe(company_table.head(10), use_container_width=True)
    
    # Interactive chart
    st.divider()
    st.subheader("🌐 Interactive: Jobs by Country")
    fig = create_interactive_country_chart(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Skills per company
    st.divider()
    if 'company' in df.columns and 'linkedin_skills' in df.columns:
        st.subheader("💼 Top Skills by Company")
        top_companies = df['company'].value_counts().head(5).index
        
        for company in top_companies:
            with st.expander(f"📍 {company}", expanded=False):
                comp_skills = df[df['company'] == company]['linkedin_skills'].dropna()
                
                if len(comp_skills) == 0:
                    st.info("No skills data available")
                    continue
                
                all_sk = []
                for skills_str in comp_skills:
                    all_sk.extend(extract_skills(skills_str))
                
                if all_sk:
                    skill_counts = pd.Series(all_sk).value_counts().head(10)
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.barplot(x=skill_counts.values, y=skill_counts.index, 
                              palette='viridis', ax=ax)
                    ax.set_title(f'Top Skills Required - {company}', 
                               fontsize=12, fontweight='bold')
                    ax.set_xlabel('Frequency', fontsize=10)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("No skills data for this company")

# ===== FOOTER =====
st.divider()