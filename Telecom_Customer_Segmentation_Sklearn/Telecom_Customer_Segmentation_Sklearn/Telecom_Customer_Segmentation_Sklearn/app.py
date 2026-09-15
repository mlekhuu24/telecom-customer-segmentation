import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

# Set Page Config
st.set_page_config(
    page_title="Telecom Customer Segmentation",
    page_icon="📡",
    layout="wide"
)

# Robust Base Directory Resolution
BASE_DIR = Path(__file__).resolve().parent
if not (BASE_DIR / "telecom_segmentation_model.pkl").exists() and (BASE_DIR / "Telecom_Customer_Segmentation_Sklearn" / "telecom_segmentation_model.pkl").exists():
    BASE_DIR = BASE_DIR / "Telecom_Customer_Segmentation_Sklearn"

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0369A1;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .cluster-vip {
        background: linear-gradient(135deg, #8B5CF6 0%, #6D28D9 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(139, 92, 246, 0.3);
    }
    .cluster-streamer {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(2, 132, 199, 0.3);
    }
    .cluster-loyal {
        background: linear-gradient(135deg, #10B981 0%, #047857 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.3);
    }
    .cluster-risk {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(245, 158, 11, 0.3);
    }
    .cluster-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.3rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📡 Telecom Customer Segmentation & CRM Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Unsupervised customer profiling powered by <b>K-Means Clustering</b> with feature standardization to drive targeted retention and upsell campaigns.</div>', unsafe_allow_html=True)

model_path = BASE_DIR / "telecom_segmentation_model.pkl"
csv_path = BASE_DIR / "segmented_customers.csv"
chart_path = BASE_DIR / "customer_segments.png"

if not model_path.exists():
    st.error(f"Model file not found at `{model_path}`! Please run 'python train_model.py' first.")
    st.stop()

@st.cache_resource
def get_bundle():
    return joblib.load(str(model_path))

bundle = get_bundle()
scaler = bundle["scaler"]
model = bundle["model"]
features = bundle["features"]

SEGMENT_PROFILES = {
    0: {
        "title": "🚀 High-Volume Talkers & Streamers",
        "css": "cluster-streamer",
        "desc": "Heavy phone & mobile data usage with growing tenure. High customer engagement and strong ARPU potential.",
        "action": "Offer premium unlimited 5G bundle upgrades and international calling packages."
    },
    1: {
        "title": "💎 VIP / Premium Power Users",
        "css": "cluster-vip",
        "desc": "Highest monthly billing accounts with long loyal tenure. Key contributors to telecom revenue.",
        "action": "Assign dedicated concierge support, priority bandwidth, and executive retention incentives."
    },
    2: {
        "title": "🛡️ Long-Term Budget Loyalists",
        "css": "cluster-loyal",
        "desc": "Maximum tenure, price-sensitive consumers with steady, modest usage and high subscription stability.",
        "action": "Maintain simple transparent billing and reward loyalty with anniversary perks; avoid unexpected fees."
    },
    3: {
        "title": "⚠️ High-Friction / Churn-Risk Users",
        "css": "cluster-risk",
        "desc": "Elevated support call frequencies, low call usage, and shorter contracts signaling frustration.",
        "action": "Trigger proactive customer success outreach, audit billing complaints, and resolve connectivity tickets."
    }
}

tab1, tab2, tab3 = st.tabs(["🔮 Live Customer Profiling & Segmentation", "📈 Cluster Visualization & Centroids", "📋 Segmented Subscriber Database"])

with tab1:
    col_input, col_result = st.columns([1.1, 0.9])
    
    with col_input:
        st.subheader("Subscriber Usage & Account Profile")
        
        preset = st.selectbox(
            "⚡ Quick Subscriber Persona Preset",
            ["Custom Account", "💎 VIP Enterprise Executive", "🚀 Young Heavy Mobile Streamer", "🛡️ Retired Long-Term Budget User", "⚠️ Dissatisfied At-Risk Subscriber"]
        )
        
        if preset == "💎 VIP Enterprise Executive":
            def_ten, def_mon, def_tot, def_data, def_min, def_supp = 52.0, 125.0, 6500.0, 85.0, 1500.0, 3.0
        elif preset == "🚀 Young Heavy Mobile Streamer":
            def_ten, def_mon, def_tot, def_data, def_min, def_supp = 18.0, 92.0, 1650.0, 90.0, 1850.0, 2.0
        elif preset == "🛡️ Retired Long-Term Budget User":
            def_ten, def_mon, def_tot, def_data, def_min, def_supp = 60.0, 35.0, 2100.0, 10.0, 1200.0, 2.0
        elif preset == "⚠️ Dissatisfied At-Risk Subscriber":
            def_ten, def_mon, def_tot, def_data, def_min, def_supp = 10.0, 75.0, 750.0, 15.0, 450.0, 9.0
        else:
            def_ten, def_mon, def_tot, def_data, def_min, def_supp = 36.0, 75.0, 2700.0, 40.0, 1100.0, 4.0
            
        with st.form("telecom_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                tenure = st.slider("Account Tenure (Months)", 1.0, 72.0, float(def_ten), step=1.0)
                monthly_charge = st.slider("Monthly Billing (₹ / $)", 15.0, 150.0, float(def_mon), step=1.0)
                total_charge = st.slider("Cumulative Lifetime Billed", 50.0, 9000.0, float(def_tot), step=50.0)
            with col_b:
                data_usage = st.slider("Monthly Data Usage (GB)", 1.0, 120.0, float(def_data), step=1.0)
                call_minutes = st.slider("Monthly Voice Minutes", 50.0, 3000.0, float(def_min), step=25.0)
                support_calls = st.slider("Customer Support Inquiries", 0.0, 15.0, float(def_supp), step=1.0)
                
            submit_btn = st.form_submit_button("🚀 Classify Subscriber Segment", use_container_width=True)
            
    with col_result:
        st.subheader("Cluster Classification")
        if submit_btn:
            sample = pd.DataFrame([{
                "tenure_months": tenure,
                "monthly_charge": monthly_charge,
                "total_charge": total_charge,
                "data_usage_gb": data_usage,
                "call_minutes": call_minutes,
                "support_calls": support_calls
            }])[features]
            
            scaled = scaler.transform(sample)
            seg_id = int(model.predict(scaled)[0])
            info = SEGMENT_PROFILES.get(seg_id, SEGMENT_PROFILES[0])
            
            st.markdown(f"""
            <div class="{info['css']}">
                <div style="font-size: 0.95rem; opacity: 0.9;">Assigned K-Means Cluster {seg_id}</div>
                <div class="cluster-title">{info['title']}</div>
                <div style="font-size: 1.05rem; opacity: 0.95;">{info['desc']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            st.markdown("### 🎯 Recommended CRM Strategy")
            st.info(f"💡 **Recommended Playbook:** {info['action']}")
            
            st.markdown("### 📊 Metric Quick Summary")
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Tenure", f"{tenure:.0f} mos")
            col_m2.metric("Monthly ARPU", f"${monthly_charge:.2f}")
            col_m3.metric("Data Usage", f"{data_usage:.0f} GB")
            
            with st.expander("🔍 Scaled Model Features Vector"):
                st.json({k: round(float(v), 3) for k, v in zip(features, scaled[0])})
        else:
            st.info("👈 Set subscriber metrics and click **'Classify Subscriber Segment'**.")

with tab2:
    st.subheader("Customer Segments Visual Mapping")
    if chart_path.exists():
        st.image(str(chart_path), caption="2D Cluster Projection (Monthly Charge vs Data Usage)", use_container_width=True)
    else:
        st.info("Cluster plot will appear after running train_model.py")

with tab3:
    st.subheader("Segmented Subscribers (segmented_customers.csv)")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Subscribers", f"{len(df):,}")
        col2.metric("Clusters Formed", f"{df['segment'].nunique()}")
        col3.metric("Avg Monthly Charge", f"${df['monthly_charge'].mean():.2f}")
        
        sel_seg = st.multiselect("Filter by Segment ID", options=sorted(df['segment'].unique()), default=sorted(df['segment'].unique()))
        st.dataframe(df[df['segment'].isin(sel_seg)].head(100), use_container_width=True)
    else:
        st.warning("Dataset not found.")
