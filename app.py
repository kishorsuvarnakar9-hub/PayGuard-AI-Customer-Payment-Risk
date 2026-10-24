from pathlib import Path
import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================================================
# PAYGUARD | ENTERPRISE PAYMENT RISK INTELLIGENCE
# =========================================================

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "customer_payment_risk_dataset.csv"
MODEL_PATH = ROOT / "models" / "payment_risk_model.joblib"
METRICS_PATH = ROOT / "reports" / "model_metrics.json"

st.set_page_config(
    page_title="PayGuard | Enterprise Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- DESIGN SYSTEM -------------------------

st.markdown("""
<style>
.stApp {
    background: #F4F7FB;
    color: #17243A;
}

.block-container {
    max-width: 100% !important;
    padding: 1.4rem 2rem 3rem !important;
}

[data-testid="stHeader"] {
    background: rgba(244,247,251,.95);
}

[data-testid="stSidebar"] {
    background: #101D33;
    border-right: 1px solid #253653;
}

[data-testid="stSidebar"] * {
    color: #E8EFFB;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #9CAFCB !important;
}

h1, h2, h3 {
    color: #17243A !important;
    letter-spacing: -0.025em;
}

p, label {
    color: #47566D;
}

/* BRAND */
.brand-icon {
    background: linear-gradient(135deg,#4778F5,#13A89E);
    width: 48px;
    height: 48px;
    border-radius: 15px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 26px;
    font-weight: 900;
    margin-bottom: 14px;
    box-shadow: 0 7px 20px #315FE733;
}

.brand-title {
    font-size: 1.15rem;
    font-weight: 900;
    letter-spacing: .12em;
    color: #FFFFFF;
}

.brand-subtitle {
    color: #9CAFCB;
    font-size: .68rem;
    letter-spacing: .12em;
    margin-top: 5px;
}

.side-label {
    color: #91A5C3;
    font-size: .68rem;
    font-weight: 800;
    letter-spacing: .14em;
    margin: 27px 0 10px;
}

/* HERO */
.hero {
    background: linear-gradient(115deg,#FFFFFF,#F1F6FF);
    border: 1px solid #E0E8F4;
    border-radius: 22px;
    padding: 28px 30px;
    margin: 0 0 25px;
    box-shadow: 0 7px 25px rgba(24,45,79,.04);
}

.hero-tag {
    display: inline-block;
    background: #E8F0FF;
    color: #2457D6;
    border: 1px solid #D7E4FF;
    padding: 6px 11px;
    border-radius: 30px;
    font-size: .68rem;
    font-weight: 850;
    letter-spacing: .12em;
}

.hero h1 {
    font-size: 2.15rem;
    margin: 13px 0 7px;
    line-height: 1.15;
}

.hero p {
    color: #68778D;
    margin: 0;
    font-size: .94rem;
}

/* KPI CARDS */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E9F2;
    border-radius: 16px;
    padding: 20px;
    min-height: 115px;
    box-shadow: 0 5px 18px rgba(24,45,79,.035);
}

[data-testid="stMetricLabel"] {
    color: #728096 !important;
    font-size: .83rem !important;
}

[data-testid="stMetricValue"] {
    color: #17243A !important;
    font-size: 1.7rem !important;
    font-weight: 850 !important;
}

[data-testid="stMetricDelta"] {
    font-size: .78rem !important;
}

/* NAVIGATION BUTTONS */
.stButton > button {
    width: 100%;
    min-height: 46px;
    border-radius: 11px;
    border: 1px solid #2B3D5B;
    background: #192943;
    color: #E4ECFA;
    font-weight: 750;
    text-align: left;
    transition: all .18s ease;
}

.stButton > button:hover {
    background: #263E62;
    color: #FFFFFF;
    border-color: #76A0FF;
    transform: translateY(-1px);
}

.nav-active .stButton > button {
    background: linear-gradient(100deg,#315FE7,#147F91) !important;
    border-color: #6AA9F5 !important;
    color: #FFFFFF !important;
    box-shadow: 0 5px 16px #315FE733;
}

/* CHARTS */
[data-testid="stPlotlyChart"] {
    background: #FFFFFF;
    border: 1px solid #E3EAF3;
    border-radius: 17px;
    padding: 8px;
    box-shadow: 0 5px 18px rgba(24,45,79,.025);
}

/* TABLES */
[data-testid="stDataFrame"] {
    border: 1px solid #E1E8F2;
    border-radius: 13px;
    overflow: hidden;
}

/* FORM */
[data-testid="stForm"] {
    background: #FFFFFF;
    border: 1px solid #E1E8F2;
    border-radius: 16px;
    padding: 20px;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-radius: 10px;
    background: #FFFFFF;
}

.section-label {
    color: #77869C;
    font-size: .7rem;
    font-weight: 850;
    letter-spacing: .13em;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.section-note {
    color: #77869C;
    font-size: .88rem;
    margin-bottom: 18px;
}

.footer {
    color: #8996A9;
    font-size: .76rem;
    border-top: 1px solid #E1E8F2;
    margin-top: 35px;
    padding-top: 18px;
}

hr {
    border-color: #293A55;
}

@media(max-width: 900px) {
    .block-container {
        padding: 1rem !important;
    }
    .hero {
        padding: 20px;
    }
    .hero h1 {
        font-size: 1.6rem;
    }
}
</style>
""", unsafe_allow_html=True)


# ------------------------- HELPERS -------------------------

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None

    data = pd.read_csv(DATA_PATH)

    for col in ["invoice_date", "due_date", "payment_date"]:
        if col in data.columns:
            data[col] = pd.to_datetime(
                data[col], dayfirst=True, errors="coerce"
            )

    return data


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


def money(value):
    return f"₹{value:,.0f}"


def style_chart(fig, height=350):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(family="Arial", color="#47566D", size=12),
        title=dict(
            font=dict(size=16, color="#17243A"),
            x=0.02,
            xanchor="left",
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(gridcolor="#EDF1F6", zeroline=False),
    )
    return fig


# ------------------------- LOAD DATA -------------------------

df = load_data()
bundle = load_model()

PAGES = {
    "Executive Overview": ("▦", "Executive Overview"),
    "Risk Prediction": ("◎", "Risk Prediction"),
    "Model Performance": ("⌁", "Model Performance"),
    "Data Quality": ("▤", "Data Quality"),
}

if "active_page" not in st.session_state:
    st.session_state.active_page = "Executive Overview"


# ------------------------- SIDEBAR -------------------------

with st.sidebar:
    st.markdown("""
    <div class="brand-icon">◈</div>
    <div class="brand-title">PAYGUARD</div>
    <div class="brand-subtitle">PAYMENT RISK INTELLIGENCE</div>
    <div class="side-label">WORKSPACE NAVIGATION</div>
    """, unsafe_allow_html=True)

    for page_name, (icon, label) in PAGES.items():
        active = st.session_state.active_page == page_name

        if active:
            st.markdown('<div class="nav-active">', unsafe_allow_html=True)

        if st.button(
            f"{icon}   {label}",
            key=f"nav_{page_name}",
            use_container_width=True,
        ):
            st.session_state.active_page = page_name
            st.rerun()

        if active:
            st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="side-label">SYSTEM STATUS</div>',
                unsafe_allow_html=True)

    if bundle is not None:
        st.success("Prediction model loaded")
    else:
        st.warning("Model not trained")

    st.caption("Finance analytics workspace")
    st.caption("Predictions support decisions; they do not guarantee outcomes.")


# ------------------------- MAIN HEADER -------------------------

page = st.session_state.active_page

st.markdown("""
<div class="hero">
    <span class="hero-tag">ENTERPRISE FINANCE ANALYTICS</span>
    <h1>Customer Payment Risk</h1>
    <p>Monitor payment behaviour, assess invoice risk and support accounts-receivable decisions.</p>
</div>
""", unsafe_allow_html=True)


if df is None:
    st.error(
        "Dataset not found. Place customer_payment_risk_dataset.csv "
        "inside data/raw/."
    )
    st.code("data/raw/customer_payment_risk_dataset.csv")
    st.stop()

if "late_payment" not in df.columns:
    st.error("Required target column 'late_payment' is missing.")
    st.stop()

target = pd.to_numeric(df["late_payment"], errors="coerce")
valid = target.isin([0, 1])

work = df.loc[valid].copy()
work["late_payment"] = target.loc[valid].astype(int)

if work.empty:
    st.error("No valid 0/1 values were found in late_payment.")
    st.stop()

total = len(work)
late_count = int(work["late_payment"].sum())
on_time_count = total - late_count
late_rate = late_count / total * 100

invoice_values = pd.to_numeric(
    work.get("invoice_amount", pd.Series(0, index=work.index)),
    errors="coerce"
).fillna(0)

outstanding_values = pd.to_numeric(
    work.get("outstanding_amount", pd.Series(0, index=work.index)),
    errors="coerce"
).fillna(0)

invoice_total = invoice_values.sum()
outstanding_total = outstanding_values.sum()


# =========================================================
# PAGE 1 — EXECUTIVE OVERVIEW
# =========================================================

if page == "Executive Overview":

    st.markdown(
        '<div class="section-label">Portfolio snapshot</div>',
        unsafe_allow_html=True
    )
    st.subheader("Executive Overview")
    st.markdown(
        '<div class="section-note">A consolidated view of invoice volume, '
        'payment behaviour and receivables exposure.</div>',
        unsafe_allow_html=True
    )

    k1, k2, k3, k4 = st.columns(4, gap="medium")

    k1.metric("Total Invoices", f"{total:,}")
    k2.metric(
        "Late-Payment Rate",
        f"{late_rate:.1f}%",
        f"{late_count:,} late invoices"
    )
    k3.metric("Invoice Value", money(invoice_total))
    k4.metric("Outstanding Amount", money(outstanding_total))

    st.markdown("### Payment Portfolio")

    left, right = st.columns([1, 1.35], gap="large")

    with left:
        counts = pd.DataFrame({
            "Payment Status": ["On Time", "Late"],
            "Invoices": [on_time_count, late_count],
        })

        fig = px.pie(
            counts,
            names="Payment Status",
            values="Invoices",
            hole=0.68,
            color="Payment Status",
            color_discrete_map={
                "On Time": "#149B8A",
                "Late": "#E8872B",
            },
            title="Payment Outcome Mix",
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent",
            marker=dict(line=dict(color="#FFFFFF", width=3)),
        )

        fig.add_annotation(
            text=f"<b>{total:,}</b><br><span style='font-size:11px'>Invoices</span>",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20, color="#17243A"),
        )

        style_chart(fig, 360)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        if "invoice_amount" in work.columns:
            chart_df = work[["invoice_amount", "late_payment"]].copy()

            chart_df["invoice_amount"] = pd.to_numeric(
                chart_df["invoice_amount"], errors="coerce"
            )

            chart_df = chart_df.dropna(subset=["invoice_amount"])
            chart_df["Payment Status"] = chart_df["late_payment"].map({
                0: "On Time",
                1: "Late",
            })

            fig = px.box(
                chart_df,
                x="Payment Status",
                y="invoice_amount",
                color="Payment Status",
                points=False,
                color_discrete_map={
                    "On Time": "#3569DF",
                    "Late": "#E8872B",
                },
                title="Invoice Amount Distribution",
            )

            style_chart(fig, 360)
            fig.update_layout(
                showlegend=False,
                xaxis_title="",
                yaxis_title="Invoice Amount (₹)",
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("invoice_amount column is not available.")

    if "previous_late_payments" in work.columns:
        st.markdown("### Historical Payment Behaviour")

        behaviour = work.copy()
        behaviour["previous_late_payments"] = pd.to_numeric(
            behaviour["previous_late_payments"], errors="coerce"
        )
        behaviour = behaviour.dropna(subset=["previous_late_payments"])

        if not behaviour.empty:
            agg = behaviour.groupby(
                "previous_late_payments", as_index=False
            )["late_payment"].mean()

            agg["Late-Payment Rate"] = agg["late_payment"] * 100

            fig = px.bar(
                agg,
                x="previous_late_payments",
                y="Late-Payment Rate",
                color="Late-Payment Rate",
                color_continuous_scale=[
                    "#149B8A", "#4778F5", "#E8872B"
                ],
                title="Late-Payment Rate by Previous Late Payments",
                labels={
                    "previous_late_payments": "Previous Late Payments"
                },
            )

            fig.update_traces(
                marker_line_width=0,
                hovertemplate=(
                    "Previous late payments: %{x}"
                    "<br>Late-payment rate: %{y:.1f}%<extra></extra>"
                ),
            )

            style_chart(fig, 340)
            fig.update_layout(
                coloraxis_showscale=False,
                xaxis_title="Previous Late Payments",
                yaxis_title="Late-Payment Rate (%)",
            )

            st.plotly_chart(fig, use_container_width=True)

    with st.expander("View Invoice Records"):
        st.dataframe(work.head(100), use_container_width=True, hide_index=True)


# =========================================================
# PAGE 2 — RISK PREDICTION
# =========================================================

elif page == "Risk Prediction":

    st.markdown(
        '<div class="section-label">Decision support</div>',
        unsafe_allow_html=True
    )
    st.subheader("Invoice Risk Assessment")
    st.markdown(
        '<div class="section-note">Estimate late-payment risk for a new invoice.</div>',
        unsafe_allow_html=True
    )

    if bundle is None:
        st.warning("Model is not trained. Run python train.py first.")
        st.stop()

    with st.form("risk_form", border=True):

        left, right = st.columns(2, gap="large")

        with left:
            invoice_amount = st.number_input(
                "Invoice Amount (₹)",
                min_value=0.0,
                value=25000.0,
                step=1000.0,
            )

            payment_terms = st.selectbox(
                "Payment Terms (Days)",
                [15, 30, 45, 60, 90],
                index=1,
            )

        with right:
            previous_late_payments = st.number_input(
                "Previous Late Payments",
                min_value=0,
                max_value=100,
                value=2,
                step=1,
            )

            invoice_date = st.date_input("Invoice Date")

        st.info(
            "Payment date and outstanding amount are excluded because "
            "they may reveal information unavailable at prediction time."
        )

        submitted = st.form_submit_button(
            "Assess Payment Risk",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        dt = pd.Timestamp(invoice_date)

        row = pd.DataFrame([{
            "invoice_amount": float(invoice_amount),
            "payment_terms": int(payment_terms),
            "previous_late_payments": int(previous_late_payments),
            "invoice_month": dt.month,
            "invoice_quarter": dt.quarter,
        }])

        try:
            model = (
                bundle["model"]
                if isinstance(bundle, dict) and "model" in bundle
                else bundle
            )

            probability = float(model.predict_proba(row)[0, 1])
            threshold = (
                float(bundle.get("threshold", 0.5))
                if isinstance(bundle, dict)
                else 0.5
            )

            high_risk = probability >= threshold

            r1, r2, r3 = st.columns(3)

            r1.metric("Late-Payment Probability", f"{probability:.1%}")
            r2.metric(
                "Risk Category",
                "HIGH RISK" if high_risk else "LOWER RISK",
            )
            r3.metric("Model Threshold", f"{threshold:.0%}")

            st.progress(min(max(probability, 0.0), 1.0))

            if high_risk:
                st.warning(
                    "Suggested action: review payment history and consider "
                    "prioritising a customer follow-up."
                )
            else:
                st.success(
                    "Suggested action: continue standard monitoring. "
                    "This estimate is not a guarantee."
                )

        except Exception as exc:
            st.error(
                "Prediction failed. Check that train.py and app.py use "
                f"the same feature columns. Details: {exc}"
            )


# =========================================================
# PAGE 3 — MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="section-label">Model governance</div>',
        unsafe_allow_html=True
    )
    st.subheader("Model Performance")
    st.markdown(
        '<div class="section-note">Compare evaluation metrics before using '
        'a model for operational decisions.</div>',
        unsafe_allow_html=True
    )

    if not METRICS_PATH.exists():
        st.warning("Metrics file not found. Run python train.py first.")
        st.stop()

    try:
        report = json.loads(
            METRICS_PATH.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Could not read model metrics: {exc}")
        st.stop()

    st.caption(
        f"Selected model: {report.get('best_model', 'Not specified')} "
        f"· Selection metric: {report.get('selection_metric', 'Not specified')}"
    )

    metrics_df = pd.DataFrame(report.get("all_model_metrics", []))

    if metrics_df.empty:
        st.info("No model metrics are available in model_metrics.json.")
    else:
        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True,
        )

        metric_cols = [
            col for col in ["precision", "recall", "f1", "roc_auc"]
            if col in metrics_df.columns
        ]

        if "model" in metrics_df.columns and metric_cols:
            long_df = metrics_df.melt(
                id_vars="model",
                value_vars=metric_cols,
                var_name="Metric",
                value_name="Score",
            )

            fig = px.bar(
                long_df,
                x="model",
                y="Score",
                color="Metric",
                barmode="group",
                color_discrete_sequence=[
                    "#3569DF", "#149B8A", "#7656D6", "#E8872B"
                ],
                title="Evaluation Metric Comparison",
            )

            fig.update_yaxes(range=[0, 1.05])
            style_chart(fig, 390)
            fig.update_layout(
                xaxis_title="",
                yaxis_title="Score",
            )

            st.plotly_chart(fig, use_container_width=True)

    st.info(
        report.get(
            "warning",
            "Validate model performance on a later time period before production use.",
        )
    )


# =========================================================
# PAGE 4 — DATA QUALITY
# =========================================================

elif page == "Data Quality":

    st.markdown(
        '<div class="section-label">Data observability</div>',
        unsafe_allow_html=True
    )
    st.subheader("Dataset Quality Monitor")
    st.markdown(
        '<div class="section-note">Inspect missing values, duplicate rows, '
        'column types and the source data.</div>',
        unsafe_allow_html=True
    )

    q1, q2, q3 = st.columns(3)

    q1.metric("Total Rows", f"{len(df):,}")
    q2.metric("Total Columns", f"{len(df.columns)}")
    q3.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")

    quality = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(value) for value in df.dtypes],
        "Missing Values": [
            int(value) for value in df.isna().sum()
        ],
        "Missing (%)": [
            round(float(value) / max(len(df), 1) * 100, 2)
            for value in df.isna().sum()
        ],
        "Unique Values": [
            int(df[col].nunique(dropna=True))
            for col in df.columns
        ],
    })

    st.dataframe(
        quality,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download Data Quality Report (CSV)",
        quality.to_csv(index=False).encode("utf-8"),
        file_name="data_quality_report.csv",
        mime="text/csv",
    )

    st.markdown("### Dataset Preview")

    st.dataframe(
        df.head(100),
        use_container_width=True,
        hide_index=True,
    )


# ------------------------- FOOTER -------------------------

st.markdown("""
<div class="footer">
    PAYGUARD · Customer Payment Risk Intelligence
    · Enterprise Finance Analytics
</div>
""", unsafe_allow_html=True)