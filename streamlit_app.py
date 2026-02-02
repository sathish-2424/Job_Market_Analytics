import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="TalentScout | Job Market Intelligence",
    layout="wide",
    page_icon="📊"
)

# ================= WHITE PROFESSIONAL UI =================
st.markdown("""
<style>
.stApp {
    background-color: #F8FAFC;
    font-family: Inter, system-ui, sans-serif;
}
.block-container {
    padding: 1.8rem 2.2rem;
    max-width: 1500px;
}
h1, h2, h3 {
    color: #020617;
}
.metric-card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 22px;
    border: 1px solid #EEF2F7;
    box-shadow: 0 6px 20px rgba(2,6,23,0.04);
}
.metric-card h2 {
    font-size: 28px;
    font-weight: 700;
}
.metric-card p {
    color: #64748B;
    font-size: 14px;
}
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #EEF2F7;
}
</style>
""", unsafe_allow_html=True)

def metric_card(title, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <h2>{value}</h2>
            <p>{title}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ================= DATA =================
@st.cache_data
def generate_data(rows=500):
    np.random.seed(42)
    titles = ['Data Scientist','Software Engineer','Product Manager',
              'UX Designer','DevOps Engineer','Data Analyst',
              'Frontend Dev','Backend Dev']
    companies = ['TechCorp','DataFlow','InnovateX','CloudSys',
                 'SoftServe','AlphaWave','CyberNet','FutureScale']
    countries = ['USA','UK','Germany','Canada','India','Australia','France']
    modes = ['Remote','Hybrid','On-site']
    contracts = ['Full-time','Contract','Internship','Part-time']
    skills = [
        "Python, SQL, AWS, ML",
        "React, JavaScript, CSS",
        "Java, Spring, Microservices",
        "Docker, Kubernetes, Jenkins",
        "Excel, PowerBI, Tableau"
    ]

    df = pd.DataFrame({
        "job_title": np.random.choice(titles, rows),
        "company": np.random.choice(companies, rows),
        "country": np.random.choice(countries, rows),
        "mode": np.random.choice(modes, rows, p=[0.4,0.35,0.25]),
        "contract": np.random.choice(contracts, rows, p=[0.7,0.15,0.1,0.05]),
        "date_posted": [datetime.today() - timedelta(days=np.random.randint(0,60)) for _ in range(rows)],
        "linkedin_skills": np.random.choice(skills, rows)
    })
    return df

@st.cache_data
def load_data(file):
    if file:
        df = pd.read_csv(file)
        df.columns = df.columns.str.lower()
        return df
    return generate_data()

# ================= SIDEBAR =================
with st.sidebar:
    st.title("TalentScout")
    file = st.file_uploader("Upload Job Data (CSV)", type=["csv"])
    st.caption("Demo data used if no file is uploaded")

df = load_data(file)

# ================= MAIN PAGE =================
st.title("📊 Global Job Market Intelligence")

# -------- KPIs --------
c1,c2,c3,c4 = st.columns(4)
with c1: metric_card("Active Jobs", len(df))
with c2: metric_card("Companies", df["company"].nunique())
with c3: metric_card("Countries", df["country"].nunique())
with c4: metric_card("Avg Skills / Job", int(df["linkedin_skills"].str.count(",").mean()+1))

st.divider()

# -------- ROLE & COMPANY --------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Top In-Demand Roles")
    role_df = df["job_title"].value_counts().head(10).reset_index()
    role_df.columns = ["Role","Count"]
    fig_roles = px.bar(
        role_df, x="Count", y="Role",
        orientation="h", template="plotly_white", color="Count"
    )
    st.plotly_chart(fig_roles, use_container_width=True)

with col2:
    st.subheader("Top Hiring Companies")
    comp_df = df["company"].value_counts().head(10).reset_index()
    comp_df.columns = ["Company","Count"]
    fig_comp = px.bar(
        comp_df, x="Count", y="Company",
        orientation="h", template="plotly_white", color="Count"
    )
    st.plotly_chart(fig_comp, use_container_width=True)

st.divider()

# -------- WORK MODE & CONTRACT --------
c1, c2 = st.columns(2)

with c1:
    st.subheader("Work Mode Distribution")
    fig_mode = px.pie(
        df, names="mode",
        template="plotly_white", hole=0.4
    )
    st.plotly_chart(fig_mode, use_container_width=True)

with c2:
    st.subheader("Contract Type")
    fig_contract = px.pie(
        df, names="contract",
        template="plotly_white", hole=0.4
    )
    st.plotly_chart(fig_contract, use_container_width=True)

st.divider()

# -------- TOP SKILLS --------
st.subheader("🔥 Most In-Demand Skills")
skills = df["linkedin_skills"].str.split(",").explode().str.strip()
skill_df = skills.value_counts().head(15).reset_index()
skill_df.columns = ["Skill","Count"]

fig_skill = px.bar(
    skill_df, x="Skill", y="Count",
    template="plotly_white", color="Count"
)
st.plotly_chart(fig_skill, use_container_width=True)

st.divider()

# -------- GEOGRAPHY --------
st.subheader("🌍 Job Distribution by Country")
geo = df["country"].value_counts().reset_index()
geo.columns = ["Country","Jobs"]

fig_geo = px.choropleth(
    geo, locations="Country", locationmode="country names",
    color="Jobs", template="plotly_white",
    color_continuous_scale="Blues"
)
st.plotly_chart(fig_geo, use_container_width=True)
