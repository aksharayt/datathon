import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score as r2s
from scipy import stats
import os
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Food Waste Policy Intelligence Platform",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.stApp { background:#F4F6F9; color:#1C2833; font-family:'Segoe UI',Arial,sans-serif; }
header[data-testid="stHeader"], header.stAppHeader, .stAppHeader { display:none !important; height:0 !important; }
[data-testid="stDecoration"], [data-testid="stToolbar"] { display:none !important; }
#MainMenu, footer { visibility:hidden !important; }
[data-testid="stSidebar"] { background:#1B3A5C !important; }
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] label { color:#ECEFF1 !important; }
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stCaption { color:#90CAF9 !important; }
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background:#FFFFFF !important; border:1px solid #90CAF9 !important;
    border-radius:5px !important; }
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div > div { color:#1B3A5C !important; }
[data-testid="stSidebar"] .stSlider > div > div > div > div { background:#90CAF9 !important; }
[data-testid="stSidebar"] .stSlider > div > div > div > div > div {
    background:#FFFFFF !important; border:3px solid #1B3A5C !important;
    width:16px !important; height:16px !important; border-radius:50% !important; }
[data-testid="stSidebar"] .stSlider p { color:#90CAF9 !important; font-size:11px !important; }
[data-testid="stSidebar"] .stRadio > label { color:#ECEFF1 !important; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    color:#ECEFF1 !important; font-size:13px !important;
    padding:3px 6px !important; border-radius:5px !important; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background:rgba(255,255,255,0.1) !important; }
[data-testid="metric-container"] {
    background:#FFFFFF; border:1px solid #DDE3EC;
    border-radius:10px; padding:16px 20px; box-shadow:0 2px 8px rgba(0,0,0,.06); }
h1,h2,h3 { color:#1B3A5C !important; }
.block-container { padding-top:0.5rem !important; }
[data-testid="stSelectbox"] > div > div { background:#FFFFFF !important; border:1px solid #90A4AE !important; border-radius:5px !important; }
[data-testid="stSelectbox"] > div > div > div { color:#1B3A5C !important; }
[data-testid="stSelectbox"] label { color:#334155 !important; font-weight:600 !important; }
.insight { background:#E3F2FD; border-left:5px solid #1B3A5C; padding:12px 16px; border-radius:5px; margin:8px 0; font-size:14px; color:#0D2137; }
.warn    { background:#FFF3E0; border-left:5px solid #E65100; padding:12px 16px; border-radius:5px; margin:8px 0; font-size:14px; color:#6D2600; }
.ok      { background:#E8F5E9; border-left:5px solid #2E7D32; padding:12px 16px; border-radius:5px; margin:8px 0; font-size:14px; color:#1B5E20; }
.critical{ background:#FFEBEE; border-left:5px solid #B71C1C; padding:12px 16px; border-radius:5px; margin:8px 0; font-size:14px; color:#7F0000; }
.sol-tag { display:inline-block; background:#E3F2FD; color:#1B3A5C; font-size:11px; font-weight:700; padding:3px 9px; border-radius:20px; margin:2px 3px 6px 0; }
.stTabs [data-baseweb="tab"] { background:#FFFFFF; border-radius:7px 7px 0 0; padding:9px 18px; border:1px solid #DDE3EC; font-weight:600; color:#566573; }
.stTabs [aria-selected="true"] { background:#1B3A5C !important; color:white !important; }
</style>
""", unsafe_allow_html=True)

COLORS = ["#1B3A5C","#2E7D32","#E65100","#6A1B9A","#0277BD","#558B2F","#AD1457","#F9A825","#00838F","#4E342E"]

STATE_ABBREV = {
    "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA",
    "Colorado":"CO","Connecticut":"CT","Delaware":"DE","Florida":"FL","Georgia":"GA",
    "Hawaii":"HI","Idaho":"ID","Illinois":"IL","Indiana":"IN","Iowa":"IA",
    "Kansas":"KS","Kentucky":"KY","Louisiana":"LA","Maine":"ME","Maryland":"MD",
    "Massachusetts":"MA","Michigan":"MI","Minnesota":"MN","Mississippi":"MS",
    "Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV","New Hampshire":"NH",
    "New Jersey":"NJ","New Mexico":"NM","New York":"NY","North Carolina":"NC",
    "North Dakota":"ND","Ohio":"OH","Oklahoma":"OK","Oregon":"OR","Pennsylvania":"PA",
    "Rhode Island":"RI","South Carolina":"SC","South Dakota":"SD","Tennessee":"TN",
    "Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA","Washington":"WA",
    "West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY"
}

FOOD_INSECURITY = {
    "Alabama":15.0,"Alaska":11.7,"Arizona":13.1,"Arkansas":17.0,"California":10.5,
    "Colorado":9.8,"Connecticut":8.5,"Delaware":10.1,"Florida":12.8,"Georgia":13.5,
    "Hawaii":9.3,"Idaho":12.1,"Illinois":11.4,"Indiana":12.3,"Iowa":9.9,
    "Kansas":11.5,"Kentucky":15.7,"Louisiana":17.8,"Maine":10.7,"Maryland":8.7,
    "Massachusetts":8.4,"Michigan":12.8,"Minnesota":8.8,"Mississippi":18.9,
    "Missouri":13.2,"Montana":11.8,"Nebraska":10.4,"Nevada":12.7,"New Hampshire":7.1,
    "New Jersey":8.9,"New Mexico":16.5,"New York":11.2,"North Carolina":13.9,
    "North Dakota":9.8,"Ohio":13.1,"Oklahoma":16.2,"Oregon":12.5,"Pennsylvania":11.0,
    "Rhode Island":10.3,"South Carolina":14.1,"South Dakota":11.3,"Tennessee":15.2,
    "Texas":14.8,"Utah":9.2,"Vermont":10.1,"Virginia":9.5,"Washington":10.8,
    "West Virginia":16.7,"Wisconsin":9.7,"Wyoming":10.5
}

STATE_POP = {
    "Alabama":5073977,"Alaska":733583,"Arizona":7359197,"Arkansas":3045637,
    "California":39029342,"Colorado":5839926,"Connecticut":3626205,"Delaware":1018396,
    "Florida":22610726,"Georgia":10912876,"Hawaii":1440196,"Idaho":1939033,
    "Illinois":12582032,"Indiana":6833037,"Iowa":3200517,"Kansas":2937150,
    "Kentucky":4512310,"Louisiana":4590241,"Maine":1385340,"Maryland":6164660,
    "Massachusetts":6981974,"Michigan":10034113,"Minnesota":5717184,"Mississippi":2940057,
    "Missouri":6177957,"Montana":1122867,"Nebraska":1967923,"Nevada":3177772,
    "New Hampshire":1395231,"New Jersey":9261699,"New Mexico":2113344,"New York":19677151,
    "North Carolina":10698973,"North Dakota":779261,"Ohio":11756058,"Oklahoma":4019800,
    "Oregon":4240137,"Pennsylvania":12972008,"Rhode Island":1093734,
    "South Carolina":5282634,"South Dakota":909824,"Tennessee":7051339,
    "Texas":30029572,"Utah":3380800,"Vermont":647464,"Virginia":8683619,
    "Washington":7785786,"West Virginia":1775156,"Wisconsin":5892539,"Wyoming":581381
}

DEST_BASE = [
    "tons_landfill","tons_donations","tons_composting","tons_animal_feed",
    "tons_anaerobic_digestion","tons_incineration","tons_land_application",
    "tons_sewer","tons_dumping","tons_not_harvested","tons_industrial_uses"
]

# Navigation structure
NAV_GROUPS = {
    "Problem":    ["01 Executive Dashboard","02 Supply Chain Funnel","03 Temporal Analysis"],
    "Diagnosis":  ["04 Sector Deep Dive","05 Sub-Sector Intel","06 Food Type & Category",
                   "07 Root Cause Engine","08 Hidden Waste","09 Waste Flow (Sankey)"],
    "Geographic": ["10 State Intelligence","11 State Clustering","12 Zero Waste Index"],
    "Impact":     ["13 Environmental","14 Waste x Hunger"],
    "Solutions":  ["15 Innovation Hub","16 Policy Simulator","17 ROI Calculator",
                   "18 Policy Recs","19 State Action Plan"],
    "Forward":    ["20 Trend Forecast","22 The One Law",
                   "23 State vs State","24 Cost of Inaction"],
}
ALL_PAGES = [p for pages in NAV_GROUPS.values() for p in pages]

def dcols(df): return [c for c in DEST_BASE if c in df.columns]
def dlabel(c): return c.replace("tons_","").replace("_"," ").title()

def fmt(n, pre=""):
    n = float(n)
    if abs(n)>=1e9: return f"{pre}{n/1e9:.2f}B"
    if abs(n)>=1e6: return f"{pre}{n/1e6:.2f}M"
    if abs(n)>=1e3: return f"{pre}{n/1e3:.1f}K"
    return f"{pre}{n:.0f}"

def sdiv(a, b, fill=0.0):
    a,b = np.array(a,dtype=float), np.array(b,dtype=float)
    return np.where(b!=0, a/b, fill)

def cl(fig, title=None, h=None):
    upd = dict(plot_bgcolor="white", paper_bgcolor="white",
               font=dict(family="Segoe UI",size=12,color="#1C2833"),
               margin=dict(l=10,r=10,t=45 if title else 15,b=10))
    if title: upd["title"]=dict(text=title,font=dict(size=14,color="#1B3A5C"))
    if h:     upd["height"]=h
    fig.update_layout(**upd)
    return fig

def mpt_calc(df_s, df_d=None):
    """Compute meals per ton wasted from available data."""
    src = df_s if df_d is None else df_d
    tw = src["tons_waste"].sum() if "tons_waste" in src.columns else 0
    tm = src["meals_wasted"].sum() if "meals_wasted" in src.columns else 0
    return tm/tw if tw>0 else 1500

@st.cache_data(show_spinner=False)
def load_data():
    search_dirs = [".","/app","/content","/content/drive/MyDrive","/content/drive/My Drive"]
    paths=[]
    for r in search_dirs:
        if os.path.exists(r):
            for dp,_,fns in os.walk(r):
                for fn in fns:
                    if fn.lower().endswith(".csv"):
                        full=os.path.join(dp,fn)
                        if full not in paths: paths.append(full)
    if not paths:
        st.error("No CSV files found. Upload all 5 ReFED datasets to your Space files.")
        st.stop()
    s=d=c=ss=sd=None
    for p in paths:
        try:
            df=pd.read_csv(p,low_memory=False)
            cols=set(df.columns.str.lower())
            has_state="state" in cols
            has_cause="cause_name" in cols or "cause_group" in cols
            has_cat="food_category" in cols
            if has_cause:               c=df
            elif has_state and has_cat: sd=df
            elif has_state:             ss=df
            elif has_cat:               d=df
            else:                       s=df
        except Exception: pass
    missing=[(n,v) for n,v in [("US Summary",s),("US Detail",d),("Cause",c),
              ("State Summary",ss),("State Detail",sd)] if v is None]
    if missing:
        st.error(f"Missing datasets: {[m[0] for m in missing]}. Found {len(paths)} CSV(s).")
        st.stop()
    return s,d,c,ss,sd

def engineer(s,d,c,ss,sd):
    def add(df):
        df=df.copy()
        if "tons_surplus" not in df.columns: return df
        t=df["tons_surplus"].values
        for col in DEST_BASE+["tons_waste","tons_uneaten","tons_not_fit_for_human_consumption",
                               "tons_inedible_parts","tons_supply"]:
            if col in df.columns:
                df[col.replace("tons_","")+"_rate"]=sdiv(df[col].values,t)
        for col,nm in [("us_dollars_surplus","value_per_ton"),
                       ("surplus_total_100_year_mtco2e_footprint","co2e_per_ton"),
                       ("surplus_upstream_100_year_mtco2e_footprint","co2e_upstream_per_ton"),
                       ("surplus_downstream_100_year_mtco2e_footprint","co2e_downstream_per_ton"),
                       ("surplus_total_100_year_mtch4_footprint","ch4_per_ton"),
                       ("gallons_water_footprint","water_per_ton"),
                       ("meals_wasted","meals_per_ton")]:
            if col in df.columns: df[nm]=sdiv(df[col].values,t)
        if "tons_landfill" in df.columns:
            df["diversion_score"]=1-sdiv(df["tons_landfill"].values,t)
        return df
    c2=c.copy()
    if "tons_surplus_due_to_cause" in c2.columns:
        tot=c2["tons_surplus_due_to_cause"].sum()
        c2["pct_of_total"]=sdiv(c2["tons_surplus_due_to_cause"].values,[tot]*len(c2))*100
        if "us_dollars_surplus_due_to_cause" in c2.columns:
            c2["value_per_cause_ton"]=sdiv(c2["us_dollars_surplus_due_to_cause"].values,
                                            c2["tons_surplus_due_to_cause"].values)
    return add(s),add(d),c2,add(ss),add(sd)

def F(df,yr,sec=None,ft=None):
    m=(df["year"]>=yr[0])&(df["year"]<=yr[1])
    df=df[m].copy()
    if sec and sec!="All" and "sector" in df.columns:    df=df[df["sector"]==sec]
    if ft  and ft !="All" and "food_type" in df.columns: df=df[df["food_type"]==ft]
    return df

def FS(df,yr,state=None):
    m=(df["year"]>=yr[0])&(df["year"]<=yr[1])
    df=df[m].copy()
    if state and state!="All" and "state" in df.columns: df=df[df["state"]==state]
    return df

# -
# SIDEBAR - FILTERS ONLY (no page navigation)
# -
def build_sidebar(s, ss):
    if "nav_section" not in st.session_state:
        st.session_state["nav_section"] = list(NAV_GROUPS.keys())[0]
    if "page" not in st.session_state:
        st.session_state["page"] = NAV_GROUPS[st.session_state["nav_section"]][0]

    r1c1, r1c2, r1c3 = st.columns([2, 3, 3])

    with r1c1:
        section = st.selectbox(
            "Section", list(NAV_GROUPS.keys()),
            index=list(NAV_GROUPS.keys()).index(st.session_state["nav_section"]),
            key="top_section", label_visibility="collapsed"
        )
        if section != st.session_state["nav_section"]:
            st.session_state["nav_section"] = section
            st.session_state["page"] = NAV_GROUPS[section][0]

    with r1c2:
        pages_in_sec  = NAV_GROUPS[st.session_state["nav_section"]]
        page_labels   = [p.split(" ", 1)[1] if " " in p else p for p in pages_in_sec]
        label_to_page = dict(zip(page_labels, pages_in_sec))
        cur_page  = st.session_state["page"]
        if cur_page not in pages_in_sec:
            cur_page = pages_in_sec[0]
            st.session_state["page"] = cur_page
        cur_label = cur_page.split(" ", 1)[1] if " " in cur_page else cur_page
        chosen = st.selectbox(
            "Page", page_labels,
            index=page_labels.index(cur_label) if cur_label in page_labels else 0,
            key="top_page", label_visibility="collapsed"
        )
        st.session_state["page"] = label_to_page[chosen]

    with r1c3:
        yrs = sorted(s["year"].unique())
        yr  = st.slider("Year", int(min(yrs)), int(max(yrs)),
                        (int(min(yrs)), int(max(yrs))), key="top_yr",
                        label_visibility="collapsed")

    r2c1, r2c2, r2c3, r2c4 = st.columns([2, 2, 2, 2])

    with r2c1:
        sec = st.selectbox("Sector",
                           ["All"] + sorted(s["sector"].dropna().unique().tolist()),
                           key="top_sec", label_visibility="visible")
    with r2c2:
        ft  = st.selectbox("Food Type",
                           ["All"] + sorted(s["food_type"].dropna().unique().tolist()),
                           key="top_ft", label_visibility="visible")
    with r2c3:
        state = st.selectbox("State",
                             ["All"] + sorted(ss["state"].dropna().unique().tolist()),
                             key="top_state", label_visibility="visible")
    with r2c4:
        st.write("")

    st.markdown('<hr style="margin:6px 0 10px 0;border-color:#DDE3EC;">', unsafe_allow_html=True)

    return yr, sec, ft, state


def render_top_nav():
    return st.session_state.get("page", list(NAV_GROUPS.values())[0][0])


def pg_executive(fs, fss):
    st.title("Executive Dashboard")
    st.markdown(
        '<div style="background:#1B3A5C;color:#ECEFF1;border-radius:10px;'
        'padding:16px 22px;margin-bottom:16px;font-size:14px;line-height:1.7;">'
        '"Food waste and hunger are not two separate problems. They are the same problem '
        'viewed from opposite ends of the supply chain. This platform identifies exactly '
        'where food is being lost, why, and what a targeted policy intervention would recover '
        '-- in tons, in dollars, and in meals."'
        '</div>', unsafe_allow_html=True)

    col_p1,col_p2,col_p3=st.columns(3)
    for col,title,body,bg,border in [
        (col_p1,"LAYER 1: PREVENTION","Date labels, demand forecasting, ugly produce.","#E8F5E9","#2E7D32"),
        (col_p2,"LAYER 2: RECOVERY","Cold chain, Good Samaritan, Last Hour Economy.","#FFF3E0","#E65100"),
        (col_p3,"LAYER 3: DIVERSION","Composting, anaerobic digestion, biogas.","#E3F2FD","#1B3A5C"),
    ]:
        col.markdown(
            f'<div style="background:{bg};border-left:5px solid {border};padding:10px 14px;border-radius:5px;text-align:center;">'
            f'<b style="color:{border};font-size:12px;">{title}</b><br>'
            f'<span style="font-size:12px;color:#334155;">{body}</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    ts=fs["tons_surplus"].sum(); tl=fs["tons_landfill"].sum()
    tv=fs["us_dollars_surplus"].sum(); td=fs["tons_donations"].sum()
    tm=fs["meals_wasted"].sum(); tco=fs["surplus_total_100_year_mtco2e_footprint"].sum()
    tw=fs["gallons_water_footprint"].sum()

    c=st.columns(4)
    c[0].metric("Total Surplus",       fmt(ts)+" t")
    c[1].metric("Economic Value Lost", fmt(tv,"$"))
    c[2].metric("Goes to Landfill",    fmt(tl)+" t", delta=f"{tl/ts*100:.1f}% of surplus" if ts else "N/A", delta_color="inverse")
    c[3].metric("Meals Wasted",        fmt(tm))
    c=st.columns(4)
    c[0].metric("Total Donations",     fmt(td)+" t")
    c[1].metric("Donation Rate",       f"{td/ts*100:.1f}%" if ts else "N/A")
    c[2].metric("GHG Footprint",       f"{tco/1e6:.2f}M MTCO2e")
    c[3].metric("Water Footprint",     fmt(tw)+" gal")

    st.markdown("---")
    col1,col2=st.columns(2)
    with col1:
        sa=fs.groupby("sector")["tons_surplus"].sum().reset_index().sort_values("tons_surplus")
        fig=px.bar(sa,x="tons_surplus",y="sector",orientation="h",color="sector",
                   color_discrete_sequence=COLORS,labels={"tons_surplus":"Tons","sector":""})
        fig.update_layout(showlegend=False)
        st.plotly_chart(cl(fig,"Surplus by Sector"),width='stretch')
    with col2:
        dc=dcols(fs); ds=fs[dc].sum(); ds.index=[dlabel(i) for i in ds.index]; ds=ds[ds>0]
        fig2=px.pie(values=ds.values,names=ds.index,color_discrete_sequence=COLORS)
        fig2.update_traces(textposition="inside",textinfo="percent+label")
        st.plotly_chart(cl(fig2,"Waste Destination Mix"),width='stretch')

    trend=fs.groupby("year")[["tons_surplus","tons_waste","tons_donations","tons_landfill"]].sum().reset_index()
    fig3=px.line(trend,x="year",y=["tons_surplus","tons_waste","tons_donations","tons_landfill"],
                 color_discrete_sequence=COLORS,labels={"value":"Tons","variable":"Metric"})
    fig3.update_traces(mode="lines+markers")
    st.plotly_chart(cl(fig3,"National Surplus Trend Over Time"),width='stretch')

    if len(trend)>=2:
        lat=trend.iloc[-1]; prv=trend.iloc[-2]
        yoy=(lat["tons_surplus"]-prv["tons_surplus"])/prv["tons_surplus"]*100 if prv["tons_surplus"] else 0
        st.markdown(f'<div class="insight"><b>YoY Change:</b> Surplus changed by <b>{yoy:+.1f}%</b> from {int(prv["year"])} to {int(lat["year"])}.</div>', unsafe_allow_html=True)


def pg_sector(fs):
    st.title("Sector Deep Dive")
    dc=dcols(fs)
    sa=fs.groupby("sector").agg(
        tons_surplus=("tons_surplus","sum"), tons_landfill=("tons_landfill","sum"),
        tons_donations=("tons_donations","sum"), tons_composting=("tons_composting","sum"),
        tons_waste=("tons_waste","sum"), us_dollars_surplus=("us_dollars_surplus","sum"),
        meals_wasted=("meals_wasted","sum"),
        co2e=("surplus_total_100_year_mtco2e_footprint","sum"),
        ch4=("surplus_total_100_year_mtch4_footprint","sum"),
        water=("gallons_water_footprint","sum"),
    ).reset_index()
    sa["landfill_rate"] =sdiv(sa["tons_landfill"].values,  sa["tons_surplus"].values)
    sa["donation_rate"] =sdiv(sa["tons_donations"].values, sa["tons_surplus"].values)
    sa["value_per_ton"] =sdiv(sa["us_dollars_surplus"].values, sa["tons_surplus"].values)
    sa["diversion_score"]=1-sa["landfill_rate"]

    c1,c2=st.columns(2)
    with c1:
        fig=px.bar(sa.sort_values("landfill_rate"),x="sector",y="landfill_rate",
                   color="sector",color_discrete_sequence=COLORS)
        fig.update_yaxes(tickformat=".0%"); fig.update_layout(showlegend=False)
        st.plotly_chart(cl(fig,"Landfill Rate by Sector"),width='stretch')
    with c2:
        fig2=px.bar(sa.sort_values("donation_rate"),x="sector",y="donation_rate",
                    color="sector",color_discrete_sequence=COLORS)
        fig2.update_yaxes(tickformat=".0%"); fig2.update_layout(showlegend=False)
        st.plotly_chart(cl(fig2,"Donation Rate by Sector"),width='stretch')

    c3,c4=st.columns([1,3])
    with c3:
        met=st.selectbox("Compare By",["tons_surplus","tons_landfill","tons_donations","tons_composting",
            "tons_waste","us_dollars_surplus","meals_wasted","co2e","ch4","water",
            "landfill_rate","donation_rate","diversion_score","value_per_ton"])
    with c4:
        fig3=px.bar(sa.sort_values(met,ascending=False),x="sector",y=met,color="sector",color_discrete_sequence=COLORS)
        fig3.update_layout(showlegend=False)
        if "rate" in met or "score" in met: fig3.update_yaxes(tickformat=".0%")
        st.plotly_chart(cl(fig3,met.replace("_"," ").title()+" by Sector"),width='stretch')

    dst=fs.groupby("sector")[dc].sum().reset_index()
    dm=dst.melt(id_vars="sector",value_vars=dc,var_name="Dest",value_name="Tons")
    dm["Dest"]=dm["Dest"].apply(dlabel)
    fig4=px.bar(dm,x="sector",y="Tons",color="Dest",barmode="stack",color_discrete_sequence=COLORS)
    st.plotly_chart(cl(fig4,"Destination Stack by Sector"),width='stretch')
    st.dataframe(sa.rename(columns=lambda c:c.replace("_"," ").title()),width='stretch')
    top_l=sa.nlargest(1,"landfill_rate")
    st.markdown(f'<div class="warn"><b>Critical:</b> {top_l["sector"].values[0]} has a {top_l["landfill_rate"].values[0]:.1%} landfill rate. A mandatory diversion law for this sector is the highest-leverage single policy action.</div>', unsafe_allow_html=True)


def pg_supply_funnel(fs, fd):
    st.title("Supply Chain Loss Funnel")
    ts_supply=fs["tons_supply"].sum(); ts_surplus=fs["tons_surplus"].sum()
    ts_waste=fs["tons_waste"].sum(); ts_lf=fs["tons_landfill"].sum()
    ts_donated=fs["tons_donations"].sum(); ts_compost=fs["tons_composting"].sum()
    ts_ad=fs["tons_anaerobic_digestion"].sum(); ts_af=fs["tons_animal_feed"].sum()
    ts_ind=fs["tons_industrial_uses"].sum(); ts_incin=fs["tons_incineration"].sum()
    ts_land=fs["tons_land_application"].sum()
    ts_sewer=fs["tons_sewer"].sum(); ts_dump=fs["tons_dumping"].sum()

    c=st.columns(4)
    c[0].metric("Total Food Supply",  fmt(ts_supply)+" t")
    c[1].metric("Becomes Surplus",    fmt(ts_surplus)+" t", delta=f"{ts_surplus/ts_supply*100:.1f}% of supply" if ts_supply else "N/A", delta_color="inverse")
    c[2].metric("Goes to Waste",      fmt(ts_waste)+" t")
    c[3].metric("To Landfill",        fmt(ts_lf)+" t")
    st.markdown("---")

    funnel_stages=[
        ("Total Food Supply",       ts_supply),
        ("Reaches Market",          ts_supply-ts_surplus),
        ("Total Surplus",           ts_surplus),
        ("Donated",                 ts_donated),
        ("Composted",               ts_compost),
        ("Anaerobic Digestion",     ts_ad),
        ("Animal Feed",             ts_af),
        ("Industrial Uses",         ts_ind),
        ("Incinerated",             ts_incin),
        ("Land Application",        ts_land),
        ("Sewer",                   ts_sewer),
        ("Dumping",                 ts_dump),
        ("Landfilled",              ts_lf),
    ]
    fdf=pd.DataFrame([s for s in funnel_stages if s[1]>0],columns=["Stage","Tons"])
    fig=px.funnel(fdf,x="Tons",y="Stage",color_discrete_sequence=["#1B3A5C"])
    st.plotly_chart(cl(fig,"National Food Supply Chain Loss Funnel",h=520),width='stretch')

    c1,c2=st.columns(2)
    with c1:
        sec_s=fs.groupby("sector").agg(tons_supply=("tons_supply","sum"),tons_surplus=("tons_surplus","sum")).reset_index()
        sec_s["efficiency"]=1-sdiv(sec_s["tons_surplus"].values,sec_s["tons_supply"].values)
        fig2=px.bar(sec_s.sort_values("efficiency"),x="efficiency",y="sector",orientation="h",
                    color="efficiency",color_continuous_scale="RdYlGn",labels={"efficiency":"Supply Efficiency","sector":""})
        fig2.update_xaxes(tickformat=".0%"); fig2.update_layout(coloraxis_showscale=False)
        st.plotly_chart(cl(fig2,"Supply Efficiency by Sector"),width='stretch')
    with c2:
        if "food_category" in fd.columns and "tons_supply" in fd.columns:
            cat_l=fd.groupby("food_category").agg(tons_supply=("tons_supply","sum"),tons_surplus=("tons_surplus","sum"),tons_donations=("tons_donations","sum"),tons_landfill=("tons_landfill","sum")).reset_index()
            cat_l["surplus_rate"]=sdiv(cat_l["tons_surplus"].values,cat_l["tons_supply"].values)
            cat_l["recovery_rate"]=sdiv(cat_l["tons_donations"].values,cat_l["tons_surplus"].values)
            cat_l=cat_l[(cat_l["food_category"]!="Not Applicable")&(cat_l["tons_supply"]>cat_l["tons_supply"].quantile(0.3))].nlargest(20,"surplus_rate")
            fig3=px.scatter(cat_l,x="surplus_rate",y="recovery_rate",size="tons_surplus",color="food_category",hover_name="food_category",color_discrete_sequence=COLORS,size_max=40)
            fig3.update_xaxes(tickformat=".0%"); fig3.update_yaxes(tickformat=".0%")
            st.plotly_chart(cl(fig3,"Loss Rate vs Recovery Rate by Category"),width='stretch')


def pg_temporal(s_df):
    st.title("Temporal Analysis")
    sec_yr=s_df.groupby(["year","sector"]).agg(surplus=("tons_surplus","sum"),landfill=("tons_landfill","sum"),donations=("tons_donations","sum")).reset_index()
    sec_yr["landfill_rate"]=sdiv(sec_yr["landfill"].values,sec_yr["surplus"].values)
    sec_yr["donation_rate"]=sdiv(sec_yr["donations"].values,sec_yr["surplus"].values)
    sec_yr["surplus_yoy"]=sec_yr.groupby("sector")["surplus"].pct_change()*100

    c1,c2=st.columns(2)
    with c1:
        fig=px.line(sec_yr,x="year",y="landfill_rate",color="sector",markers=True,color_discrete_sequence=COLORS)
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(cl(fig,"Landfill Rate Trend (Are We Improving?)"),width='stretch')
    with c2:
        fig2=px.line(sec_yr,x="year",y="donation_rate",color="sector",markers=True,color_discrete_sequence=COLORS)
        fig2.update_yaxes(tickformat=".0%")
        st.plotly_chart(cl(fig2,"Donation Rate Trend"),width='stretch')

    nat_yr=s_df.groupby("year").agg(surplus=("tons_surplus","sum"),landfill=("tons_landfill","sum")).reset_index().sort_values("year")
    nat_yr["surplus_yoy"]=nat_yr["surplus"].pct_change()*100
    nat_yr["rolling_avg"]=nat_yr["surplus_yoy"].rolling(3,center=True).mean()
    nat_yr["deviation"]=nat_yr["surplus_yoy"]-nat_yr["rolling_avg"]
    nat_yr["is_anomaly"]=nat_yr["deviation"].abs()>(nat_yr["deviation"].std()*1.5)

    fig3=go.Figure()
    fig3.add_trace(go.Bar(x=nat_yr["year"],y=nat_yr["surplus_yoy"],
                          marker_color=["#B71C1C" if r["is_anomaly"] and r["surplus_yoy"]>0
                                        else "#2E7D32" if r["is_anomaly"] and r["surplus_yoy"]<0
                                        else "#94A3B8" for _,r in nat_yr.iterrows()],name="YoY Change %"))
    fig3.add_trace(go.Scatter(x=nat_yr["year"],y=nat_yr["rolling_avg"],mode="lines",name="3yr Rolling Avg",line=dict(color="#1B3A5C",width=2,dash="dash")))
    fig3.add_hline(y=0,line_color="black",opacity=0.3,line_dash="dot")
    fig3.update_layout(yaxis_title="YoY Change (%)",legend=dict(orientation="h"))
    st.plotly_chart(cl(fig3,"National Surplus YoY Change (Anomalies Highlighted)"),width='stretch')


def pg_food(fd):
    st.title("Food Type & Category Analysis")
    ft=fd.groupby("food_type").agg(tons_surplus=("tons_surplus","sum"),tons_waste=("tons_waste","sum"),
        tons_landfill=("tons_landfill","sum"),tons_donations=("tons_donations","sum"),
        us_dollars_surplus=("us_dollars_surplus","sum"),meals_wasted=("meals_wasted","sum"),
        co2e=("surplus_total_100_year_mtco2e_footprint","sum"),water=("gallons_water_footprint","sum")).reset_index()
    ft["waste_rate"]=sdiv(ft["tons_waste"].values,ft["tons_surplus"].values)
    ft["landfill_rate"]=sdiv(ft["tons_landfill"].values,ft["tons_surplus"].values)
    ft["donation_rate"]=sdiv(ft["tons_donations"].values,ft["tons_surplus"].values)

    c1,c2=st.columns(2)
    with c1:
        fig=px.bar(ft.sort_values("waste_rate"),x="waste_rate",y="food_type",orientation="h",
                   color="food_type",color_discrete_sequence=COLORS,labels={"waste_rate":"Waste Rate","food_type":""})
        fig.update_xaxes(tickformat=".0%"); fig.update_layout(showlegend=False)
        st.plotly_chart(cl(fig,"Waste Rate by Food Type"),width='stretch')
    with c2:
        fig2=px.scatter(ft,x="tons_surplus",y="waste_rate",size="us_dollars_surplus",color="food_type",
                        color_discrete_sequence=COLORS,hover_name="food_type")
        fig2.update_yaxes(tickformat=".0%")
        st.plotly_chart(cl(fig2,"Volume vs Waste Rate"),width='stretch')

    if "food_category" in fd.columns:
        c3,c4=st.columns([1,5])
        with c3: n=st.slider("Top N",5,30,15)
        with c4:
            fc=fd.groupby("food_category").agg(tons_surplus=("tons_surplus","sum"),tons_waste=("tons_waste","sum"),
                us_dollars_surplus=("us_dollars_surplus","sum"),meals_wasted=("meals_wasted","sum")).reset_index()
            fc["waste_rate"]=sdiv(fc["tons_waste"].values,fc["tons_surplus"].values)
            fc=fc[fc["food_category"]!="Not Applicable"]
            fig3=px.bar(fc.nlargest(n,"tons_surplus").sort_values("tons_surplus"),
                        x="tons_surplus",y="food_category",orientation="h",
                        color="waste_rate",color_continuous_scale="RdYlGn_r",
                        labels={"tons_surplus":"Tons","food_category":"","waste_rate":"Waste Rate"})
            st.plotly_chart(cl(fig3,f"Top {n} Food Categories (color = waste rate)"),width='stretch')

        st.dataframe(fc.sort_values("tons_surplus",ascending=False).reset_index(drop=True)
                     .rename(columns=lambda c:c.replace("_"," ").title()),width='stretch')

    top_wf=ft.nlargest(1,"waste_rate")
    st.markdown(f'<div class="insight"><b>Insight:</b> {top_wf["food_type"].values[0]} has a {top_wf["waste_rate"].values[0]:.1%} waste rate. Demand forecasting and dynamic pricing are the most direct fixes.</div>', unsafe_allow_html=True)


def pg_causes(fc_df):
    st.title("Root Cause Engine")
    cg=fc_df.groupby("cause_group").agg(tons=("tons_surplus_due_to_cause","sum"),
        usd=("us_dollars_surplus_due_to_cause","sum"),inedible=("tons_inedible_parts","sum"),
        not_fit=("tons_not_fit_for_human_consumption","sum")).reset_index()
    cg["pct"]=sdiv(cg["tons"].values,[cg["tons"].sum()]*len(cg))*100

    c1,c2=st.columns(2)
    with c1:
        fig=px.pie(cg,names="cause_group",values="tons",color_discrete_sequence=COLORS)
        fig.update_traces(textposition="inside",textinfo="percent+label")
        st.plotly_chart(cl(fig,"Surplus by Cause Group"),width='stretch')
    with c2:
        fig2=px.bar(cg.sort_values("tons"),x="tons",y="cause_group",orientation="h",
                    color="cause_group",color_discrete_sequence=COLORS,labels={"tons":"Tons","cause_group":""})
        fig2.update_layout(showlegend=False)
        st.plotly_chart(cl(fig2,"Tons by Cause Group"),width='stretch')

    c3,c4=st.columns([1,5])
    with c3: nc=st.slider("Top causes",5,31,15)
    with c4:
        cn=fc_df.groupby("cause_name")["tons_surplus_due_to_cause"].sum().reset_index().sort_values("tons_surplus_due_to_cause",ascending=False)
        fig3=px.bar(cn.head(nc).sort_values("tons_surplus_due_to_cause"),x="tons_surplus_due_to_cause",y="cause_name",
                    orientation="h",color="tons_surplus_due_to_cause",color_continuous_scale="Blues",
                    labels={"tons_surplus_due_to_cause":"Tons","cause_name":""})
        st.plotly_chart(cl(fig3,f"Top {nc} Causes"),width='stretch')

    available_dims=[d for d in ["sector","food_type","sub_sector"] if d in fc_df.columns]
    if available_dims:
        c5,c6=st.columns([1,5])
        with c5: dim=st.selectbox("Heatmap By",available_dims)
        with c6:
            hm=fc_df.groupby([dim,"cause_group"])["tons_surplus_due_to_cause"].sum().reset_index()
            pv=hm.pivot(index="cause_group",columns=dim,values="tons_surplus_due_to_cause").fillna(0)
            fig4=px.imshow(pv,color_continuous_scale="YlOrRd",labels={"color":"Tons"},aspect="auto")
            st.plotly_chart(cl(fig4,f"Cause Intensity Heatmap by {dim.title()}",h=400),width='stretch')

    if "tons_inedible_parts" in fc_df.columns:
        c1b,c2b=st.columns(2)
        c1b.metric("Inedible Parts",fmt(fc_df["tons_inedible_parts"].sum())+" t")
        c2b.metric("Not Fit for Human Consumption",fmt(fc_df["tons_not_fit_for_human_consumption"].sum())+" t")

    top_c=cg.nlargest(1,"tons")
    st.markdown(f'<div class="warn"><b>Insight:</b> Top cause group is <b>{top_c["cause_group"].values[0]}</b> at {fmt(top_c["tons"].values[0])} tons. Structural supply chain reform is as critical as field recovery.</div>', unsafe_allow_html=True)


def pg_hidden(fc_df, fd):
    st.title("Hidden Waste Detector")

    c1,c2=st.columns(2)
    with c1:
        st.subheader("1. Structural Causes")
        cg_s=fc_df.groupby("cause_group")["tons_surplus_due_to_cause"].sum().reset_index()
        fig1=px.bar(cg_s.sort_values("tons_surplus_due_to_cause"),x="tons_surplus_due_to_cause",
                    y="cause_group",orientation="h",color="tons_surplus_due_to_cause",
                    color_continuous_scale="Blues",labels={"tons_surplus_due_to_cause":"Tons","cause_group":""})
        st.plotly_chart(cl(fig1),width='stretch')
    with c2:
        st.subheader("2. Outlier High-Waste Categories")
        if "food_category" in fd.columns:
            fc2=fd.groupby("food_category").agg(tons_surplus=("tons_surplus","sum"),tons_waste=("tons_waste","sum")).reset_index()
            fc2["waste_rate"]=sdiv(fc2["tons_waste"].values,fc2["tons_surplus"].values)
            fc2=fc2[(fc2["food_category"]!="Not Applicable")&(fc2["tons_surplus"]>fc2["tons_surplus"].quantile(0.25))]
            fig3=px.bar(fc2.nlargest(15,"waste_rate").sort_values("waste_rate"),
                        x="waste_rate",y="food_category",orientation="h",
                        color="waste_rate",color_continuous_scale="Reds",
                        labels={"waste_rate":"Waste Rate","food_category":""})
            fig3.update_xaxes(tickformat=".0%")
            st.plotly_chart(cl(fig3,"Top 15 Hidden High-Waste Categories"),width='stretch')

    c3,c4=st.columns(2)
    with c3:
        st.subheader("3. Sewer & Dumping (Invisible Streams)")
        sd_c=[c for c in ["tons_sewer","tons_dumping"] if c in fd.columns]
        if sd_c and "sector" in fd.columns:
            sd_agg=fd.groupby("sector")[sd_c].sum().reset_index()
            dm=sd_agg.melt(id_vars="sector",value_vars=sd_c,var_name="Type",value_name="Tons")
            dm["Type"]=dm["Type"].apply(dlabel)
            fig4=px.bar(dm,x="sector",y="Tons",color="Type",barmode="group",color_discrete_sequence=["#1B3A5C","#E65100"])
            st.plotly_chart(cl(fig4),width='stretch')
    with c4:
        st.subheader("4. Supply vs Surplus Gap")
        if "tons_supply" in fd.columns:
            sup_agg=fd.groupby("food_type").agg(tons_supply=("tons_supply","sum"),tons_surplus=("tons_surplus","sum")).reset_index()
            sup_agg["surplus_pct"]=sdiv(sup_agg["tons_surplus"].values,sup_agg["tons_supply"].values)*100
            fig5=px.bar(sup_agg.sort_values("surplus_pct",ascending=False),x="food_type",y="surplus_pct",
                        color="food_type",color_discrete_sequence=COLORS,labels={"surplus_pct":"% of Supply Lost","food_type":""})
            fig5.update_yaxes(ticksuffix="%"); fig5.update_layout(showlegend=False)
            st.plotly_chart(cl(fig5),width='stretch')

    st.markdown('<div class="warn"><b>Regulatory Gap:</b> Sewer disposal bypasses every existing diversion program. Mandatory reporting must cover these streams.</div>', unsafe_allow_html=True)


def pg_sankey(fs):
    st.title("Waste Destination Flow (Sankey)")
    dc=dcols(fs); agg=fs.groupby("sector")[dc].sum().reset_index()
    secs=agg["sector"].tolist(); dests=[dlabel(c) for c in dc]
    nodes=secs+dests; nm={n:i for i,n in enumerate(nodes)}
    src,tgt,val=[],[],[]
    for _,row in agg.iterrows():
        for col,dl in zip(dc,dests):
            v=row[col]
            if v>0: src.append(nm[row["sector"]]); tgt.append(nm[dl]); val.append(v)
    fig=go.Figure(go.Sankey(
        node=dict(pad=15,thickness=20,label=nodes,color=["#1B3A5C"]*len(secs)+["#2E7D32"]*len(dests),line=dict(color="white",width=0.3)),
        link=dict(source=src,target=tgt,value=val,color=["rgba(46,125,50,0.25)"]*len(src))
    ))
    fig.update_layout(height=580,paper_bgcolor="white",title=dict(text="Surplus Flow: Sector to Destination",font=dict(size=15,color="#1B3A5C")))
    st.plotly_chart(fig,width='stretch')

    agg_ft=fs.groupby("food_type")[dc].sum().reset_index()
    dm=agg_ft.melt(id_vars="food_type",value_vars=dc,var_name="Dest",value_name="Tons")
    dm["Dest"]=dm["Dest"].apply(dlabel)
    fig2=px.bar(dm,x="food_type",y="Tons",color="Dest",barmode="stack",color_discrete_sequence=COLORS)
    st.plotly_chart(cl(fig2,"Destination Mix by Food Type"),width='stretch')


def pg_states(fss, sel):
    st.title("State Intelligence")
    sa=fss.groupby("state").agg(tons_surplus=("tons_surplus","sum"),tons_landfill=("tons_landfill","sum"),
        tons_donations=("tons_donations","sum"),tons_composting=("tons_composting","sum"),
        tons_waste=("tons_waste","sum"),us_dollars_surplus=("us_dollars_surplus","sum"),
        meals_wasted=("meals_wasted","sum"),co2e=("surplus_total_100_year_mtco2e_footprint","sum"),
        water=("gallons_water_footprint","sum")).reset_index()
    sa["landfill_rate"]=sdiv(sa["tons_landfill"].values,sa["tons_surplus"].values)
    sa["donation_rate"]=sdiv(sa["tons_donations"].values,sa["tons_surplus"].values)
    sa["abbrev"]=sa["state"].map(STATE_ABBREV)

    c1,c2=st.columns([2,2])
    with c1:
        mm=st.selectbox("Map Metric",["tons_surplus","tons_landfill","tons_donations","landfill_rate","donation_rate","meals_wasted","co2e","water"])
    with c2:
        st.write("")

    fig=px.choropleth(sa,locations="abbrev",locationmode="USA-states",color=mm,scope="usa",
                      color_continuous_scale="YlOrRd" if "rate" not in mm else "RdYlGn_r",
                      hover_name="state",hover_data={"tons_surplus":True,"tons_landfill":True,"landfill_rate":True,"abbrev":False})
    fig.update_layout(geo=dict(bgcolor="white"),paper_bgcolor="white",margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig,width='stretch')

    c1,c2=st.columns(2)
    with c1:
        fig2=px.bar(sa.nlargest(10,"tons_surplus").sort_values("tons_surplus"),x="tons_surplus",y="state",orientation="h",color="state",color_discrete_sequence=COLORS)
        fig2.update_layout(showlegend=False)
        st.plotly_chart(cl(fig2,"Top 10 States by Surplus"),width='stretch')
    with c2:
        fig3=px.scatter(sa,x="landfill_rate",y="donation_rate",size="tons_surplus",color="state",hover_name="state",color_discrete_sequence=COLORS)
        fig3.update_xaxes(tickformat=".0%"); fig3.update_yaxes(tickformat=".0%"); fig3.update_layout(showlegend=False)
        st.plotly_chart(cl(fig3,"Landfill vs Donation Rate"),width='stretch')

    if sel!="All":
        st.subheader(f"Deep Dive: {sel}")
        sd2=fss[fss["state"]==sel]
        c3,c4=st.columns(2)
        with c3:
            tr=sd2.groupby("year")[["tons_surplus","tons_landfill","tons_donations"]].sum().reset_index()
            fig4=px.line(tr,x="year",y=["tons_surplus","tons_landfill","tons_donations"],color_discrete_sequence=COLORS)
            fig4.update_traces(mode="lines+markers")
            st.plotly_chart(cl(fig4,f"{sel} Trend"),width='stretch')
        with c4:
            sec_b=sd2.groupby("sector")["tons_surplus"].sum().reset_index()
            fig5=px.pie(sec_b,names="sector",values="tons_surplus",color_discrete_sequence=COLORS)
            fig5.update_traces(textposition="inside",textinfo="percent+label")
            st.plotly_chart(cl(fig5,f"{sel} by Sector"),width='stretch')

    st.dataframe(sa.drop(columns=["abbrev"]).sort_values("tons_surplus",ascending=False).reset_index(drop=True).rename(columns=lambda c:c.replace("_"," ").title()).style.format({"Landfill Rate":"{:.1%}","Donation Rate":"{:.1%}"}),width='stretch')


def pg_clustering(ss_df):
    st.title("State Segmentation: Policy Archetypes")
    latest=ss_df[ss_df["year"]==ss_df["year"].max()]
    sf=latest.groupby("state").agg(tons_surplus=("tons_surplus","sum"),tons_landfill=("tons_landfill","sum"),
        tons_donations=("tons_donations","sum"),tons_composting=("tons_composting","sum"),
        meals_wasted=("meals_wasted","sum"),co2e=("surplus_total_100_year_mtco2e_footprint","sum"),
        water=("gallons_water_footprint","sum")).reset_index()
    sf["landfill_rate"]=sdiv(sf["tons_landfill"].values,sf["tons_surplus"].values)
    sf["donation_rate"]=sdiv(sf["tons_donations"].values,sf["tons_surplus"].values)
    sf["composting_rate"]=sdiv(sf["tons_composting"].values,sf["tons_surplus"].values)
    feat_cols=["tons_surplus","landfill_rate","donation_rate","composting_rate","meals_wasted","co2e","water"]
    X=StandardScaler().fit_transform(sf[feat_cols].fillna(0))

    c1,c2=st.columns([1,5])
    with c1: k=st.slider("Clusters (k)",2,7,4)
    sf["cluster"]=KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(X).astype(str)
    sf["abbrev"]=sf["state"].map(STATE_ABBREV)

    stats_c=sf.groupby("cluster")[feat_cols].mean()
    labels={}
    for cid in stats_c.index:
        row=stats_c.loc[cid]
        if row["landfill_rate"]>stats_c["landfill_rate"].median():
            labels[cid]=(f"Cluster {cid}: High Volume, High Landfill" if row["tons_surplus"]>stats_c["tons_surplus"].median() else f"Cluster {cid}: Low Volume, High Landfill")
        else:
            labels[cid]=(f"Cluster {cid}: Active Donors" if row["donation_rate"]>stats_c["donation_rate"].median() else f"Cluster {cid}: High Diversion")
    sf["cluster_label"]=sf["cluster"].map(labels)

    fig=px.choropleth(sf,locations="abbrev",locationmode="USA-states",color="cluster_label",scope="usa",
                      hover_name="state",color_discrete_sequence=COLORS)
    fig.update_layout(geo=dict(bgcolor="white"),paper_bgcolor="white",legend=dict(orientation="h",yanchor="bottom",y=-0.2))
    st.plotly_chart(fig,width='stretch')

    c3,c4=st.columns(2)
    with c3:
        fig2=px.scatter(sf,x="landfill_rate",y="donation_rate",color="cluster_label",size="tons_surplus",hover_name="state",color_discrete_sequence=COLORS)
        fig2.update_xaxes(tickformat=".0%"); fig2.update_yaxes(tickformat=".0%")
        st.plotly_chart(cl(fig2,"Cluster Scatter"),width='stretch')
    with c4:
        pca=PCA(n_components=2,random_state=42); X_pca=pca.fit_transform(X)
        sf["PCA_1"]=X_pca[:,0]; sf["PCA_2"]=X_pca[:,1]
        var1=pca.explained_variance_ratio_[0]*100; var2=pca.explained_variance_ratio_[1]*100
        fig_pca=px.scatter(sf,x="PCA_1",y="PCA_2",color="cluster_label",hover_name="state",size="tons_surplus",color_discrete_sequence=COLORS,size_max=30)
        fig_pca.update_layout(legend=dict(orientation="h",yanchor="bottom",y=-0.3))
        st.plotly_chart(cl(fig_pca,f"PCA (PC1+PC2 = {var1+var2:.1f}% variance)"),width='stretch')

    st.subheader("Cluster Profiles")
    cp=sf.groupby("cluster_label")[feat_cols].mean().reset_index()
    st.dataframe(cp.rename(columns=lambda c:c.replace("_"," ").title()),width='stretch')
    for lbl in sorted(sf["cluster_label"].unique()):
        sts=sf[sf["cluster_label"]==lbl]["state"].tolist()
        st.markdown(f"**{lbl}:** {', '.join(sts)}")


def pg_zero_waste_index(ss_df):
    st.title("Zero Waste Efficiency Index")
    latest_yr=ss_df["year"].max()
    sa=ss_df[ss_df["year"]==latest_yr].groupby("state").agg(
        tons_surplus=("tons_surplus","sum"),tons_landfill=("tons_landfill","sum"),
        tons_donations=("tons_donations","sum"),tons_composting=("tons_composting","sum"),
        tons_anaerobic_digestion=("tons_anaerobic_digestion","sum"),
        tons_animal_feed=("tons_animal_feed","sum"),tons_industrial_uses=("tons_industrial_uses","sum"),
        tons_waste=("tons_waste","sum"),co2e=("surplus_total_100_year_mtco2e_footprint","sum"),
        water=("gallons_water_footprint","sum")).reset_index()
    sa["landfill_rate"]=sdiv(sa["tons_landfill"].values,sa["tons_surplus"].values)
    sa["donation_rate"]=sdiv(sa["tons_donations"].values,sa["tons_surplus"].values)
    sa["composting_rate"]=sdiv(sa["tons_composting"].values,sa["tons_surplus"].values)
    sa["diversion_rate"]=sdiv((sa["tons_donations"]+sa["tons_composting"]+sa["tons_animal_feed"]+sa["tons_anaerobic_digestion"]+sa["tons_industrial_uses"]).values,sa["tons_surplus"].values)
    sa["co2e_intensity"]=sdiv(sa["co2e"].values,sa["tons_surplus"].values)
    sa["abbrev"]=sa["state"].map(STATE_ABBREV)

    feat_cols=["landfill_rate","diversion_rate","donation_rate","co2e_intensity","composting_rate"]
    X_sc=StandardScaler().fit_transform(sa[feat_cols].fillna(0))
    X_sc[:,0]=-X_sc[:,0]; X_sc[:,3]=-X_sc[:,3]
    raw=X_sc@np.array([0.40,0.25,0.15,0.10,0.10])
    mn,mx=raw.min(),raw.max()
    sa["zero_waste_index"]=(raw-mn)/(mx-mn+1e-9)*100
    sa["anomaly_flag"]=(np.abs((sa["landfill_rate"]-sa["landfill_rate"].mean())/(sa["landfill_rate"].std()+1e-9))>1.5)

    c1,c2=st.columns(2)
    with c1:
        fig=px.choropleth(sa,locations="abbrev",locationmode="USA-states",color="zero_waste_index",scope="usa",color_continuous_scale="RdYlGn",hover_name="state")
        fig.update_layout(geo=dict(bgcolor="white"),paper_bgcolor="white")
        st.plotly_chart(fig,width='stretch')
    with c2:
        rank_df=pd.concat([sa.nlargest(5,"zero_waste_index"),sa.nsmallest(5,"zero_waste_index")])
        rank_df["label"]=rank_df["state"]+" ("+rank_df["zero_waste_index"].map(lambda x:f"{x:.0f}")+")"
        fig2=px.bar(rank_df.sort_values("zero_waste_index"),x="zero_waste_index",y="label",orientation="h",color="zero_waste_index",color_continuous_scale="RdYlGn")
        fig2.update_layout(coloraxis_showscale=False)
        st.plotly_chart(cl(fig2,"Top 5 and Bottom 5 States"),width='stretch')

    anomalies=sa[sa["anomaly_flag"]].sort_values("zero_waste_index")
    if not anomalies.empty:
        st.subheader("Anomaly Detection: States Below Peer Performance")
        st.dataframe(anomalies[["state","zero_waste_index","landfill_rate","donation_rate","diversion_rate"]].reset_index(drop=True).rename(columns=lambda c:c.replace("_"," ").title()).style.format({"Landfill Rate":"{:.1%}","Donation Rate":"{:.1%}","Diversion Rate":"{:.1%}","Zero Waste Index":"{:.1f}"}),width='stretch')
        st.markdown(f'<div class="critical"><b>{len(anomalies)} states</b> are statistical anomalies -- more than 1.5 standard deviations below national peer performance on landfill rate. Highest urgency for intervention.</div>', unsafe_allow_html=True)


def pg_environment(fs, fss):
    st.title("Environmental Impact Analysis")
    tc=fs["surplus_total_100_year_mtco2e_footprint"].sum(); tm=fs["surplus_total_100_year_mtch4_footprint"].sum()
    tw=fs["gallons_water_footprint"].sum(); tc_up=fs["surplus_upstream_100_year_mtco2e_footprint"].sum()
    c=st.columns(4)
    c[0].metric("Total CO2e (100-yr)",f"{tc/1e6:.2f}M MTCO2e"); c[1].metric("Upstream CO2e",f"{tc_up/1e6:.2f}M MTCO2e")
    c[2].metric("Total CH4 (100-yr)",f"{tm:,.0f} MT"); c[3].metric("Water Footprint",fmt(tw)+" gal")

    ea=fs.groupby("sector").agg(upstream=("surplus_upstream_100_year_mtco2e_footprint","sum"),downstream=("surplus_downstream_100_year_mtco2e_footprint","sum"),ch4_up=("surplus_upstream_100_year_mtch4_footprint","sum"),ch4_dn=("surplus_downstream_100_year_mtch4_footprint","sum"),water=("gallons_water_footprint","sum")).reset_index()
    fig=go.Figure([go.Bar(name="Upstream CO2e",x=ea["sector"],y=ea["upstream"],marker_color="#1B3A5C"),go.Bar(name="Downstream CO2e",x=ea["sector"],y=ea["downstream"],marker_color="#2E7D32")])
    fig.update_layout(barmode="group",yaxis_title="MTCO2e")
    st.plotly_chart(cl(fig,"Upstream vs Downstream CO2e by Sector"),width='stretch')

    c1,c2=st.columns(2)
    with c1:
        wa=fs.groupby("food_type")["gallons_water_footprint"].sum().reset_index().sort_values("gallons_water_footprint")
        fig2=px.bar(wa,x="gallons_water_footprint",y="food_type",orientation="h",color="food_type",color_discrete_sequence=COLORS,labels={"gallons_water_footprint":"Gallons","food_type":""})
        fig2.update_layout(showlegend=False)
        st.plotly_chart(cl(fig2,"Water Footprint by Food Type"),width='stretch')
    with c2:
        c5,c6=st.columns([1,3])
        with c5: em=st.selectbox("State Map",["surplus_total_100_year_mtco2e_footprint","gallons_water_footprint","surplus_total_100_year_mtch4_footprint"])
        with c6:
            if em in fss.columns:
                se=fss.groupby("state")[em].sum().reset_index(); se["abbrev"]=se["state"].map(STATE_ABBREV)
                fig4=px.choropleth(se,locations="abbrev",locationmode="USA-states",color=em,scope="usa",color_continuous_scale="Blues",hover_name="state")
                fig4.update_layout(geo=dict(bgcolor="white"),paper_bgcolor="white")
                st.plotly_chart(fig4,width='stretch')


def pg_nexus(ss_df):
    st.title("Waste x Hunger Nexus")
    st.caption("Correlation model identifying states where high food waste and high hunger overlap")
    st.markdown('<div class="insight"><b>Data:</b> USDA ERS 2022 Food Security Survey + US Census 2022 populations.</div>', unsafe_allow_html=True)

    sa=ss_df.groupby("state").agg(tons_surplus=("tons_surplus","sum"),tons_landfill=("tons_landfill","sum"),
        tons_donations=("tons_donations","sum"),tons_waste=("tons_waste","sum"),
        meals_wasted=("meals_wasted","sum"),co2e=("surplus_total_100_year_mtco2e_footprint","sum")).reset_index()
    sa["food_insecurity_rate"]=sa["state"].map(FOOD_INSECURITY)
    sa["population"]=sa["state"].map(STATE_POP)
    sa["tons_per_capita"]=sdiv(sa["tons_surplus"].values,sa["population"].values)
    sa["landfill_rate"]=sdiv(sa["tons_landfill"].values,sa["tons_surplus"].values)
    sa["donation_rate"]=sdiv(sa["tons_donations"].values,sa["tons_surplus"].values)
    sa["meals_wasted_per_capita"]=sdiv(sa["meals_wasted"].values,sa["population"].values)
    sa["insecure_people"]=sa["food_insecurity_rate"]/100*sa["population"]
    sa["abbrev"]=sa["state"].map(STATE_ABBREV)
    sa=sa.dropna(subset=["food_insecurity_rate","population"])

    # --- FIX: define mpt here ---
    tw_nexus = sa["tons_waste"].sum()
    tm_nexus = sa["meals_wasted"].sum()
    mpt = tm_nexus / tw_nexus if tw_nexus > 0 else 1500

    c1,c2=st.columns([2,2])
    with c1:
        x_met=st.selectbox("Waste Metric (X-axis)",["tons_per_capita","landfill_rate","meals_wasted_per_capita","donation_rate","tons_surplus"])
    x_vals=sa[x_met].values; y_vals=sa["food_insecurity_rate"].values
    valid=np.isfinite(x_vals)&np.isfinite(y_vals)
    if valid.sum()>=5:
        r,p=stats.pearsonr(x_vals[valid],y_vals[valid]); r2=r**2
        sig="Statistically significant" if p<0.05 else "Not significant"
    else:
        r,p,r2,sig=0.0,1.0,0.0,"Insufficient data"
    m_reg,b_reg=np.polyfit(x_vals[valid],y_vals[valid],1) if valid.sum()>=2 else (0,0)
    x_line=np.linspace(x_vals[valid].min(),x_vals[valid].max(),100) if valid.sum()>=2 else np.array([])
    y_line=m_reg*x_line+b_reg

    c3,c4,c5=st.columns(3)
    c3.metric("Pearson r",f"{r:.3f}"); c4.metric("R2",f"{r2:.3f}"); c5.metric("P-value",f"{p:.4f}",delta=sig)

    fig=go.Figure()
    fig.add_trace(go.Scatter(x=x_vals,y=y_vals,mode="markers+text",text=sa["abbrev"],textposition="top center",
        marker=dict(size=10,color=sa["tons_surplus"],colorscale="YlOrRd",showscale=True,colorbar=dict(title="Surplus (t)")),
        hovertemplate="<b>%{text}</b><br>"+x_met.replace("_"," ").title()+": %{x:.3f}<br>Food Insecurity: %{y:.1f}%<extra></extra>",name="States"))
    if len(x_line): fig.add_trace(go.Scatter(x=x_line,y=y_line,mode="lines",line=dict(color="#1B3A5C",width=2,dash="dash"),name=f"Regression (r={r:.2f})"))
    fig.update_layout(xaxis_title=x_met.replace("_"," ").title(),yaxis_title="Food Insecurity Rate (%)")
    st.plotly_chart(cl(fig,"Food Waste vs Food Insecurity by State"),width='stretch')

    st.subheader("Critical Zone Quadrant Map")
    waste_med=sa["tons_per_capita"].median(); hunger_med=sa["food_insecurity_rate"].median()
    def quadrant(row):
        hw=row["tons_per_capita"]>=waste_med; hh=row["food_insecurity_rate"]>=hunger_med
        if hw and hh:        return "CRITICAL: High Waste, High Hunger"
        elif hw and not hh:  return "High Waste, Low Hunger"
        elif not hw and hh:  return "Low Waste, High Hunger"
        else:                return "Low Waste, Low Hunger"
    sa["quadrant"]=sa.apply(quadrant,axis=1)
    quad_colors={"CRITICAL: High Waste, High Hunger":"#B71C1C","High Waste, Low Hunger":"#E65100","Low Waste, High Hunger":"#F9A825","Low Waste, Low Hunger":"#2E7D32"}

    fig2=go.Figure()
    for quad,color in quad_colors.items():
        sub=sa[sa["quadrant"]==quad]
        if sub.empty: continue
        fig2.add_trace(go.Scatter(x=sub["tons_per_capita"],y=sub["food_insecurity_rate"],mode="markers+text",text=sub["abbrev"],textposition="top center",marker=dict(size=13,color=color),name=quad,hovertemplate="<b>%{text}</b><br>Waste/capita: %{x:.3f}t<br>Insecurity: %{y:.1f}%<extra></extra>"))
    fig2.add_hline(y=hunger_med,line_dash="dot",line_color="gray",annotation_text=f"Median Hunger ({hunger_med:.1f}%)",annotation_position="bottom right")
    fig2.add_vline(x=waste_med,line_dash="dot",line_color="gray",annotation_text="Median Waste/capita",annotation_position="top left")
    fig2.update_layout(xaxis_title="Food Waste Per Capita (tons)",yaxis_title="Food Insecurity Rate (%)",legend=dict(orientation="h",yanchor="bottom",y=-0.3))
    st.plotly_chart(cl(fig2,"State Quadrant Map: Waste vs Hunger Priority Zones"),width='stretch')

    fig3=px.choropleth(sa,locations="abbrev",locationmode="USA-states",color="quadrant",scope="usa",hover_name="state",color_discrete_map=quad_colors,hover_data={"tons_per_capita":True,"food_insecurity_rate":True,"abbrev":False,"quadrant":False})
    fig3.update_layout(geo=dict(bgcolor="white"),paper_bgcolor="white",legend=dict(orientation="h",yanchor="bottom",y=-0.15),margin=dict(l=0,r=0,t=50,b=0))
    st.plotly_chart(fig3,width='stretch')

    critical=sa[sa["quadrant"]=="CRITICAL: High Waste, High Hunger"].copy()
    if not critical.empty:
        sc2=StandardScaler(); cf=sc2.fit_transform(critical[["tons_per_capita","food_insecurity_rate"]].fillna(0))
        critical["priority_score"]=cf.sum(axis=1)
        critical=critical.sort_values("priority_score",ascending=False)
        st.subheader("Critical States: Immediate Intervention Priority")
        disp=critical[["state","tons_surplus","tons_per_capita","food_insecurity_rate","insecure_people","landfill_rate","donation_rate","priority_score"]].copy()
        disp.columns=["State","Surplus (t)","Waste/Capita (t)","Food Insecurity (%)","Insecure People","Landfill Rate","Donation Rate","Priority Score"]
        st.dataframe(disp.reset_index(drop=True).style.format({"Surplus (t)":"{:,.0f}","Waste/Capita (t)":"{:.3f}","Food Insecurity (%)":"{:.1f}%","Insecure People":"{:,.0f}","Landfill Rate":"{:.1%}","Donation Rate":"{:.1%}","Priority Score":"{:.2f}"}),width='stretch')

    st.subheader("Redirection Opportunity Calculator")
    critical2=sa[sa["quadrant"]=="CRITICAL: High Waste, High Hunger"]
    if not critical2.empty:
        total_cs=critical2["tons_surplus"].sum(); total_ip=critical2["insecure_people"].sum()
        total_cm=critical2["meals_wasted"].sum()
        c6,c7,c8=st.columns(3)
        c6.metric("Surplus in Critical States",fmt(total_cs)+" t"); c7.metric("Food Insecure People",f"{int(total_ip):,}"); c8.metric("Meals Wasted (Critical)",fmt(total_cm))
        redir=st.slider("Redirect this % of critical-state surplus to food banks:",1,50,10)
        redirected_t=total_cs*redir/100; meals_rec=redirected_t*mpt; ppl_fed=meals_rec/365/3
        reduction_in_hunger=ppl_fed/total_ip*100 if total_ip else 0
        c9,c10,c11=st.columns(3)
        c9.metric("Food Redirected",fmt(redirected_t)+" t"); c10.metric("Meals Recovered",fmt(meals_rec)); c11.metric("People Fed Year-Round",f"{int(ppl_fed):,}")
        st.markdown(f'<div class="critical"><b>Core Finding:</b> Redirecting just {redir}% of surplus in critical states could feed <b>{int(ppl_fed):,} people year-round</b>, reducing food insecurity by approximately <b>{reduction_in_hunger:.1f}%</b>. The food exists. The hunger exists. The gap is entirely logistics and incentives.</div>', unsafe_allow_html=True)


def pg_innovations(fs, fss):
    st.title("Innovation Solutions Hub")
    ts=fs["tons_surplus"].sum(); tl=fs["tons_landfill"].sum(); td=fs["tons_donations"].sum()
    fs_s=fs[fs["sector"]=="Foodservice"]["tons_surplus"].sum() if "Foodservice" in fs["sector"].values else 0

    solutions=[
        {"num":"01","title":"The Last Hour Economy","tags":["Technology","Foodservice","Zero New Infrastructure"],
         "problem":"Restaurants know which food expires in 2-4 hours daily. That information goes nowhere. The food goes in the trash.",
         "solution":"Mandatory last-hour disclosure embedded in federal food business licensing. Any licensed business must list surplus on a standardized exchange 2 hours before close.",
         "data":f"Foodservice generates {fmt(fs_s)} tons. Even 5% capture = {fmt(fs_s*0.05)} tons redirected at near-zero cost.",
         "precedent":"Too Good To Go operates this in 17 countries. The policy version makes it public infrastructure.","color":"#1B3A5C"},
        {"num":"02","title":"Food Banks as Emergency Infrastructure","tags":["Federal Investment","Cold Chain","Food Security"],
         "problem":"Food banks operate like 1970s warehouses: limited hours, volunteer-dependent, unable to accept perishables at 10pm.",
         "solution":"Federal Food Security Infrastructure Act funding hospital-grade cold chain systems: 24/7, professional staff, refrigerated receiving.",
         "data":f"National donation rate is only {td/ts:.1%}. The constraint is receiving capacity, not donor willingness.",
         "precedent":"Feeding America: cold chain gaps cause rejection of millions of pounds annually.","color":"#2E7D32"},
        {"num":"03","title":"School Kitchens as Community Food Hubs","tags":["Education","Local Government","Zero New Infrastructure"],
         "problem":"Schools have commercial kitchens, refrigeration, paid staff idle from 3pm to 7am while nearby restaurants discard tons of food nightly.",
         "solution":"Fund schools as after-hours food processing hubs. Surplus delivered, processed into frozen meals, distributed via school programs and community pantries.",
         "data":"Zero new construction. The kitchen, cold storage, and staff already exist in every school district, funded by taxpayers.",
         "precedent":"Chicago and San Francisco pilots proved school kitchens can process 2-5 tons/week.","color":"#6A1B9A"},
        {"num":"04","title":"Dynamic Markdown Pricing (Mandated)","tags":["Retail Policy","Low-Income Access","France Model"],
         "problem":"Food retail never adopted dynamic pricing systematically because the cost of waste is externalized to the environment.",
         "solution":"Require retailers receiving USDA subsidies to implement mandatory markdown pricing starting 48 hours before expiration, graduated from 25% to 90% off.",
         "data":f"Retail surplus = {fmt(fs[fs['sector']=='Retail']['tons_surplus'].sum() if 'Retail' in fs['sector'].values else 0)} tons. France's law reduced supermarket waste ~17% in 18 months.",
         "precedent":"France Loi Garot 2016. Same result possible in US for federally-subsidized retailers.","color":"#0277BD"},
        {"num":"05","title":"Grade B Produce + SNAP Multiplier","tags":["Farm Policy","SNAP Reform","Market Creation"],
         "problem":"20-30% of farm produce never leaves the field due to cosmetic standards. Nutritionally identical to Grade A. Farmers get zero revenue for it.",
         "solution":"Federal Grade B Nutrition Program: $1 SNAP on Grade B produce = $1.50 purchasing power. Farmers get revenue; low-income families get more food per dollar.",
         "data":"Data shows buyer rejections and not-harvested as top farm cause groups. Grade B creates a market where none currently exists.",
         "precedent":"Double Up Food Bucks proves SNAP multipliers change behavior at scale.","color":"#558B2F"},
        {"num":"06","title":"The Proximity Tax Credit","tags":["Tax Policy","Local Food Systems","Multi-Goal"],
         "problem":"Current food waste tax credits are flat and disconnected from local outcomes. The same credit applies whether food travels 500 miles or 5 miles.",
         "solution":"A proximity-weighted federal tax credit: the closer food stays to where it was produced before being donated, the larger the credit.",
         "data":"Three goals in one instrument: emission reduction, supply chain resilience, and donation increase.",
         "precedent":"Proximity-based incentives already exist in renewable energy credits and conservation easements.","color":"#AD1457"},
        {"num":"07","title":"Prison Kitchen Processing Program","tags":["Criminal Justice","Vocational Training","Dual-Value"],
         "problem":"Prisons have commercial kitchens and idle labor. Food banks have surplus they cannot process. Recidivism is driven by lack of vocational skills.",
         "solution":"Surplus processed in prison kitchens into frozen meals distributed through food banks. Participants receive culinary certifications and job placement pipelines.",
         "data":"Addresses three crises simultaneously. No new facilities or equipment required.",
         "precedent":"California and Colorado pilots showed measurable reduction in recidivism and increased food bank capacity.","color":"#00838F"},
        {"num":"08","title":"National Food Logistics Corps","tags":["Federal Program","AmeriCorps Model","System Fix"],
         "problem":"Food rescue works at city scale but has never been funded nationally. Every existing program proves the model.",
         "solution":"National Food Logistics Corps modeled on AmeriCorps. Paid professionals with refrigerated vans and route optimization move surplus to where it is needed.",
         "data":f"{fmt(tl)} tons go to landfill. The gap is entirely logistics and incentives, not production.",
         "precedent":"AmeriCorps VISTA deploys 6,000+ professionals on anti-poverty work. Food Logistics track needs only budget appropriation.","color":"#4E342E"},
    ]

    c=st.columns(4)
    c[0].metric("No New Infrastructure Required","4 of 8")
    c[1].metric("Prevention Focus","3 solutions")
    c[2].metric("Recovery Focus","4 solutions")
    c[3].metric("Diversion Focus","2 solutions")
    st.markdown("---")

    all_tags=sorted(set(t for s in solutions for t in s["tags"]))
    sel_tags=st.multiselect("Filter by Tag",all_tags,default=all_tags)
    st.markdown("---")

    for sol in solutions:
        if not any(t in sel_tags for t in sol["tags"]): continue
        col_n,col_b=st.columns([1,10])
        with col_n:
            st.markdown(f'<div style="font-size:52px;font-weight:900;color:{sol["color"]};line-height:1.1;">{sol["num"]}</div>', unsafe_allow_html=True)
        with col_b:
            st.markdown(f"**{sol['title']}**")
            tags_html="".join([f'<span class="sol-tag">{t}</span>' for t in sol["tags"]])
            st.markdown(tags_html,unsafe_allow_html=True)
            tabs=st.tabs(["Problem","Solution","Data Link","Precedent"])
            with tabs[0]: st.write(sol["problem"])
            with tabs[1]: st.write(sol["solution"])
            with tabs[2]: st.markdown(f'<div class="insight">{sol["data"]}</div>',unsafe_allow_html=True)
            with tabs[3]: st.write(sol["precedent"])
        st.markdown("---")

    st.subheader("Implementation Roadmap")
    roadmap=pd.DataFrame({"Solution":["Last Hour Economy","Food Bank Infrastructure","School Kitchen Hubs","Dynamic Pricing Mandate","Grade B + SNAP","Proximity Tax Credit","Prison Kitchen Program","Food Logistics Corps"],"Timeline":["6-12 mo","2-3 yr","1-2 yr","1-3 yr","2-4 yr","1-2 yr","1-2 yr","1-2 yr"],"Cost Level":["Low","High","Low","Low","Medium","Low","Low","Medium"],"Impact Level":["Medium","High","Medium","High","High","Medium","Medium","High"],"New Infrastructure":["No","Yes","No","No","No","No","No","No"],"Federal Legislation":["Yes","Yes","No","Yes","Yes","Yes","No","No"]})
    st.dataframe(roadmap,width='stretch')


def pg_simulator(fs):
    st.title("Policy Impact Simulator")
    st.caption("Quantify intervention effects before enacting them -- all outputs are data-driven")

    c1,c2,c3=st.columns(3)
    with c1:
        ldr=st.slider("Landfill Diversion (%)",0,100,20)
        don=st.slider("Donation Rate Increase (%)",0,100,15)
    with c2:
        cmp=st.slider("Composting Increase (%)",0,100,10)
        dfc=st.slider("Demand Forecasting Adoption (%)",0,100,30)
    with c3:
        pkr=st.slider("Smart Packaging Reform (%)",0,100,20)
        sec_s=st.selectbox("Target Sector",["All"]+sorted(fs["sector"].dropna().unique().tolist()))

    df_s=fs if sec_s=="All" else fs[fs["sector"]==sec_s]
    ts=df_s["tons_surplus"].sum(); tl=df_s["tons_landfill"].sum()
    td=df_s["tons_donations"].sum(); tc2=df_s["tons_composting"].sum()
    tv=df_s["us_dollars_surplus"].sum(); tco=df_s["surplus_total_100_year_mtco2e_footprint"].sum()
    tw_sim=df_s["gallons_water_footprint"].sum()
    mpt=mpt_calc(df_s)

    diverted=tl*(ldr/100); new_l=tl-diverted; new_d=td*(1+don/100); new_c=tc2*(1+cmp/100)
    fs_s2=fs[fs["sector"]=="Foodservice"]["tons_surplus"].sum() if "Foodservice" in fs["sector"].values else 0
    fs_red=fs_s2*(dfc/100)*0.30 if sec_s in ["All","Foodservice"] else 0
    res_s2=fs[fs["sector"]=="Residential"]["tons_surplus"].sum() if "Residential" in fs["sector"].values else 0
    res_red=res_s2*(pkr/100)*0.15 if sec_s in ["All","Residential"] else ts*(pkr/100)*0.10
    new_s=max(ts-fs_red-res_red,0)
    vpt=tv/ts if ts>0 else 0; co2pt=tco/ts if ts>0 else 0; wpt=tw_sim/ts if ts>0 else 0
    co2_saved=diverted*co2pt*0.35; water_saved=diverted*wpt*0.20
    meal_saved=(new_d-td)*1.0; val_rec=(ts-new_s)*vpt

    st.markdown("---")
    st.markdown("### Simulated Policy Outcomes")
    r=st.columns(3)
    r[0].metric("Surplus Reduced",fmt(ts-new_s)+" t",delta=f"-{(ts-new_s)/ts*100:.1f}%" if ts>0 else "N/A")
    r[1].metric("Landfill Avoided",fmt(diverted)+" t",delta=f"-{diverted/tl*100:.1f}%" if tl>0 else "N/A")
    r[2].metric("Extra Meals Donated",fmt(meal_saved))
    r2=st.columns(3)
    r2[0].metric("GHG Saved",f"{co2_saved/1e3:.1f}K MTCO2e"); r2[1].metric("Water Conserved",fmt(water_saved)+" gal"); r2[2].metric("Value Recovered",fmt(val_rec,"$"))

    comp=pd.DataFrame({"Metric":["Surplus","Landfill","Donations","Composting"],"Before":[ts,tl,td,tc2],"After":[new_s,new_l,new_d,new_c]})
    fig=go.Figure([go.Bar(name="Before Policy",x=comp["Metric"],y=comp["Before"],marker_color="#B71C1C"),go.Bar(name="After Policy",x=comp["Metric"],y=comp["After"],marker_color="#2E7D32")])
    fig.update_layout(barmode="group",yaxis_title="Tons")
    st.plotly_chart(cl(fig,"Before vs After: Policy Simulation"),width='stretch')

    meals_per_insecure=(ts-new_s)*mpt
    st.markdown(f'<div class="ok"><b>Food Security Impact:</b> Simulated reduction could generate approximately <b>{fmt(meals_per_insecure)}</b> meals -- enough to feed <b>{int(meals_per_insecure/365/3):,}</b> people year-round. Value recovered: <b>{fmt(val_rec,"$")}</b>.</div>', unsafe_allow_html=True)


def pg_roi(fs, fss):
    st.title("Policy ROI Calculator")
    ts=fs["tons_surplus"].sum(); tl=fs["tons_landfill"].sum(); td=fs["tons_donations"].sum()
    tv=fs["us_dollars_surplus"].sum(); tco2=fs["surplus_total_100_year_mtco2e_footprint"].sum()
    twat=fs["gallons_water_footprint"].sum(); mpt=mpt_calc(fs)
    vpt=tv/ts if ts>0 else 0; co2pt=tco2/ts if ts>0 else 0; watpt=twat/ts if ts>0 else 0

    c_price=st.sidebar.number_input("Carbon Credit ($/MTCO2e)",10,200,65)
    w_price=st.sidebar.number_input("Water Value ($/1K gal)",1,20,4)

    st.markdown("### Adjust Intervention Parameters")
    r1c1,r1c2=st.columns(2)
    with r1c1:
        roi1=st.slider("Date Labeling: Surplus Reduction (%)",1,20,8,key="roi1"); rc1=st.slider("Federal Cost ($M)",1,500,50,key="rc1")
        roi2=st.slider("Foodservice Diversion: Landfill Reduction (%)",5,50,20,key="roi2"); rc2=st.slider("Compliance Cost ($M)",10,2000,400,key="rc2")
    with r1c2:
        roi3=st.slider("Cold Chain: Donation Rate Increase (%)",1,30,10,key="roi3"); rc3=st.slider("Investment ($M)",50,5000,800,key="rc3")
        roi4=st.slider("Grade B: Not-Harvested Recovery (%)",5,40,15,key="roi4"); rc4=st.slider("SNAP Cost Increase ($M/yr)",100,3000,600,key="rc4")
    roi5=st.slider("Logistics Corps: Landfill Diversion (%)",1,20,7,key="roi5"); rc5=st.slider("Annual Budget ($M)",100,5000,1200,key="rc5")

    fs_s=fs[fs["sector"]=="Foodservice"]["tons_landfill"].sum() if "Foodservice" in fs["sector"].values else tl*0.3
    res_s=fs[fs["sector"]=="Residential"]["tons_surplus"].sum() if "Residential" in fs["sector"].values else ts*0.3
    farm_nh=fs[fs["sector"]=="Farm"]["tons_not_harvested"].sum() if "Farm" in fs["sector"].values and "tons_not_harvested" in fs.columns else ts*0.03

    interventions=[
        {"name":"Standardized Date Labeling","tons":res_s*(roi1/100),"cost_m":rc1},
        {"name":"Foodservice Diversion Mandate","tons":fs_s*(roi2/100),"cost_m":rc2},
        {"name":"Cold Chain Food Bank Act","tons":td*(roi3/100),"cost_m":rc3},
        {"name":"Grade B + SNAP Multiplier","tons":farm_nh*(roi4/100),"cost_m":rc4},
        {"name":"National Food Logistics Corps","tons":tl*(roi5/100),"cost_m":rc5},
    ]
    results=[]
    for iv in interventions:
        usd_saved=iv["tons"]*vpt; co2_saved_mt=iv["tons"]*co2pt
        water_saved_g=iv["tons"]*watpt; meals_rec=iv["tons"]*mpt*0.4
        carbon_rev=co2_saved_mt*c_price; water_rev=(water_saved_g/1000)*w_price
        total_return=usd_saved+carbon_rev+water_rev; cost=iv["cost_m"]*1e6
        roi_ratio=total_return/cost if cost>0 else 0
        results.append({"Intervention":iv["name"],"Tons Reduced":iv["tons"],"USD Value Saved":usd_saved,"Meals":meals_rec,"Carbon Revenue ($)":carbon_rev,"Total Return ($)":total_return,"Federal Cost ($)":cost,"ROI Ratio":roi_ratio})

    rdf=pd.DataFrame(results)
    fig=px.bar(rdf.sort_values("ROI Ratio",ascending=True),x="ROI Ratio",y="Intervention",orientation="h",color="ROI Ratio",color_continuous_scale="RdYlGn",text=rdf.sort_values("ROI Ratio")["ROI Ratio"].map(lambda x:f"{x:.2f}x"),labels={"ROI Ratio":"Return per $ Invested","Intervention":""})
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(cl(fig,"Policy ROI: For Every $1 Invested, This Many $ Returned"),width='stretch')

    c1,c2=st.columns(2)
    with c1:
        fig2=px.bar(rdf,x="Intervention",y=["USD Value Saved","Carbon Revenue ($)"],barmode="stack",color_discrete_sequence=["#1B3A5C","#2E7D32"])
        fig2.update_xaxes(tickangle=25); st.plotly_chart(cl(fig2,"Economic Return by Intervention"),width='stretch')
    with c2:
        fig3=px.scatter(rdf,x="Federal Cost ($)",y="Total Return ($)",size="Tons Reduced",color="Intervention",color_discrete_sequence=COLORS,hover_name="Intervention",text="Intervention")
        fig3.update_traces(textposition="top center",textfont_size=9)
        st.plotly_chart(cl(fig3,"Cost vs Return"),width='stretch')

    best=rdf.nlargest(1,"ROI Ratio")
    st.markdown(f'<div class="ok"><b>Highest-ROI Intervention:</b> <b>{best["Intervention"].values[0]}</b> returns <b>{best["ROI Ratio"].values[0]:.2f}x</b> for every dollar invested.</div>', unsafe_allow_html=True)


def pg_policy(fs, fc_df, fss):
    st.title("Policy Recommendations")
    st.markdown('<div style="background:#1B3A5C;color:#ECEFF1;border-radius:10px;padding:14px 20px;margin-bottom:14px;font-size:14px;line-height:1.7;"><b>This platform is the bridge between data and action.</b> A senator can see that their state generates a disproportionate share of national surplus -- and know exactly where to write the bill. An NGO can run the Policy Simulator and show a donor precisely what a targeted investment recovers in meals.</div>', unsafe_allow_html=True)

    ts=fs["tons_surplus"].sum(); tv=fs["us_dollars_surplus"].sum(); vpt=tv/ts if ts>0 else 0
    sa=fs.groupby("sector")[["tons_surplus","tons_landfill","tons_donations"]].sum()
    sa["lr"]=sdiv(sa["tons_landfill"].values,sa["tons_surplus"].values); sa["dr"]=sdiv(sa["tons_donations"].values,sa["tons_surplus"].values)
    top_cause=fc_df.groupby("cause_group")["tons_surplus_due_to_cause"].sum().idxmax()
    top_cause_tons=fc_df.groupby("cause_group")["tons_surplus_due_to_cause"].sum().max()
    top_state=fss.groupby("state")["tons_surplus"].sum().idxmax()
    top_state_pct=fss.groupby("state")["tons_surplus"].sum().max()/fss["tons_surplus"].sum()*100 if fss["tons_surplus"].sum() else 0
    res_s=sa.loc["Residential","tons_surplus"] if "Residential" in sa.index else ts*0.3
    fs_lr=sa.loc["Foodservice","lr"] if "Foodservice" in sa.index else 0.7
    ret_dr=sa.loc["Retail","dr"] if "Retail" in sa.index else 0.2
    fs_ts=sa.loc["Foodservice","tons_surplus"] if "Foodservice" in sa.index else ts*0.2
    top_ft=fs.groupby("food_type").apply(lambda x: x["tons_waste"].sum()/x["tons_surplus"].sum() if x["tons_surplus"].sum()>0 else 0).idxmax()
    top_ft_rate=fs.groupby("food_type").apply(lambda x: x["tons_waste"].sum()/x["tons_surplus"].sum() if x["tons_surplus"].sum()>0 else 0).max()

    policies=[
        {"title":"Policy 1: Residential Waste Reduction","cls":"ok","insight":f"Residential generates {fmt(res_s)} tons -- more than any other sector. Most policy targets restaurants. The data proves households are the primary source.","actions":["Mandate a single unified federal date label standard.","Fund municipal smart-bin programs with household waste tracking.","Subsidize composting via utility bill credits.","National meal planning and food storage education campaign."],"est_reduction":int(res_s*0.12),"timeline":"1-2 years","feasibility":"High"},
        {"title":"Policy 2: Foodservice Mandatory Landfill Diversion","cls":"warn","insight":f"Foodservice sends {fs_lr:.1%} to landfill -- worst of any sector. Retail donates {ret_dr:.1%}. This gap is addressable with a clear legal mandate.","actions":["Foodservice Organic Waste Diversion Act for operators above 1 ton/week.","Mandatory certified diversion contracts.","Tiered tax credits for 20%+ year-over-year waste reduction.","AI demand forecasting grants for small operators."],"est_reduction":int(fs_ts*0.15),"timeline":"1-3 years","feasibility":"High"},
        {"title":f"Policy 3: {top_ft} Supply Optimization","cls":"insight","insight":f"{top_ft} has a {top_ft_rate:.1%} waste rate -- highest of any food type. Structural overproduction, not consumer behavior.","actions":["Annual waste rate reporting mandate for large manufacturers.","Federal tax incentive for predictive demand planning.","Dynamic markdown regulations for near-expiry items.","Extended shelf-life packaging R&D grants."],"est_reduction":int(ts*0.05),"timeline":"3-5 years","feasibility":"Medium"},
        {"title":"Policy 4: Processing Byproduct Valorization Act","cls":"warn","insight":f"Largest cause group: '{top_cause}' at {fmt(top_cause_tons)} tons -- structural byproducts invisible in public reporting.","actions":["Tax credits for converting byproducts to animal feed or biofuel.","Federal Byproduct Exchange marketplace.","EPA reporting mandates for streams above 1,000 tons/year.","Biorefinery feasibility studies."],"est_reduction":int(ts*0.08),"timeline":"3-7 years","feasibility":"Medium"},
        {"title":f"Policy 5: {top_state} Federal Innovation Pilot","cls":"ok","insight":f"{top_state} generates {top_state_pct:.1f}% of national surplus. A pilot here produces measurable national-scale results within 3 years.","actions":[f"Designate {top_state} a Federal Food Waste Innovation Zone.",f"Statewide real-time surplus exchange connecting donors with food banks.","County-level composting mandates for municipalities over 50,000 residents.","State procurement commitments for Grade B produce."],"est_reduction":int(fss[fss["state"]==top_state]["tons_surplus"].sum()*0.10),"timeline":"1-2 years","feasibility":"High"},
    ]

    for pol in policies:
        st.markdown(f"### {pol['title']}")
        st.markdown(f'<div class="{pol["cls"]}">{pol["insight"]}</div>', unsafe_allow_html=True)
        st.markdown("**Proposed Actions:**")
        for a in pol["actions"]: st.markdown(f"- {a}")
        c_pol=st.columns(3)
        c_pol[0].metric("Est. Reduction",fmt(pol["est_reduction"])+" t"); c_pol[1].metric("Value Recovered",fmt(pol["est_reduction"]*vpt,"$")); c_pol[2].metric("Timeline / Feasibility",f"{pol['timeline']} | {pol['feasibility']}")
        st.markdown("---")

    st.subheader("Projected Combined Impact")
    total_r=sum(p["est_reduction"] for p in policies); mpt=mpt_calc(fs)
    total_meals=total_r*mpt; total_ppl=total_meals/365/3; pct_r=total_r/ts*100 if ts>0 else 0
    c=st.columns(4)
    c[0].metric("Total Surplus Reduction",fmt(total_r)+" t"); c[1].metric("Value Recovered",fmt(total_r*vpt,"$")); c[2].metric("Meals Recovered",fmt(total_meals)); c[3].metric("People Fed Year-Round",f"{int(total_ppl):,}")
    st.markdown(f'<div class="ok"><b>Combined Impact:</b> Full implementation reduces U.S. food surplus by approximately <b>{pct_r:.1f}%</b>, feeds <b>{int(total_ppl):,} people year-round</b>, without producing a single additional pound of food.</div>', unsafe_allow_html=True)

    pdf=pd.DataFrame({"Policy":[p["title"].split(":")[0] for p in policies],"Reduction (tons)":[p["est_reduction"] for p in policies],"Feasibility":[p["feasibility"] for p in policies]})
    fig=px.bar(pdf,x="Policy",y="Reduction (tons)",color="Feasibility",color_discrete_map={"High":"#2E7D32","Medium":"#E65100","Low":"#B71C1C"})
    st.plotly_chart(cl(fig,"Policy Impact Comparison"),width='stretch')


def pg_forecast(s_df):
    st.title("Trend Forecasting (ML)")
    c1,c2,c3=st.columns(3)
    with c1: sec=st.selectbox("Sector",sorted(s_df["sector"].dropna().unique()))
    with c2: met=st.selectbox("Metric",["tons_surplus","tons_landfill","tons_donations","tons_waste","tons_composting","us_dollars_surplus","meals_wasted","surplus_total_100_year_mtco2e_footprint"])
    with c3: hor=st.slider("Horizon (years)",1,10,5)

    sub=s_df[s_df["sector"]==sec].groupby("year")[met].sum().reset_index().sort_values("year")
    if len(sub)<2: st.warning("Not enough data points."); return
    X=sub[["year"]].values; y=sub[met].values
    lr=LinearRegression().fit(X,y)
    fy=np.arange(sub["year"].max()+1,sub["year"].max()+hor+1).reshape(-1,1); fp=lr.predict(fy); r2=lr.score(X,y)

    fig=go.Figure([go.Scatter(x=sub["year"],y=y,mode="lines+markers",name="Historical",line=dict(color="#1B3A5C",width=2.5)),go.Scatter(x=fy.flatten(),y=fp,mode="lines+markers",name="Forecast",line=dict(color="#2E7D32",width=2,dash="dash"))])
    fig.add_vrect(x0=sub["year"].max(),x1=fy.max(),fillcolor="#E8F5E9",opacity=0.4,line_width=0,annotation_text="Forecast Zone",annotation_position="top left")
    fig.update_layout(yaxis_title=met.replace("_"," ").title())
    st.plotly_chart(cl(fig,f"{sec} -- {met.replace('_',' ').title()} Forecast"),width='stretch')

    c=st.columns(4)
    c[0].metric("Current",fmt(y[-1])); c[1].metric(f"Projected ({int(fy[-1][0])})",fmt(fp[-1]))
    delta=(fp[-1]-y[-1])/y[-1]*100 if y[-1]!=0 else 0
    c[2].metric("Projected Change",f"{delta:+.1f}%"); c[3].metric("Model R2",f"{r2:.3f}")

    nat=s_df.groupby("year")[met].sum().reset_index().sort_values("year")
    Xn=nat[["year"]].values; yn=nat[met].values
    lrn=LinearRegression().fit(Xn,yn); fpn=lrn.predict(fy)
    fig2=go.Figure([go.Scatter(x=nat["year"],y=yn,mode="lines+markers",name="Historical",line=dict(color="#1B3A5C",width=2.5)),go.Scatter(x=fy.flatten(),y=fpn,mode="lines+markers",name="Forecast",line=dict(color="#E65100",width=2,dash="dash"))])
    st.plotly_chart(cl(fig2,f"National {met.replace('_',' ').title()} Forecast"),width='stretch')


def pg_one_law(fs, fd, fc_df, fss):
    st.title("The One Law")
    st.markdown('<div style="background:#1B3A5C;color:#ECEFF1;border-radius:10px;padding:18px 24px;margin-bottom:20px;font-size:15px;line-height:1.75;"><b>The hardest question in policy is not "what should we do?"</b> The hard question is: <b>"If Congress could pass exactly ONE law tomorrow, which one would move the national metric the most?"</b> This page answers that question using the data.</div>', unsafe_allow_html=True)

    ts=fs["tons_surplus"].sum(); tl=fs["tons_landfill"].sum(); td=fs["tons_donations"].sum()
    tv=fs["us_dollars_surplus"].sum(); tco2=fs["surplus_total_100_year_mtco2e_footprint"].sum()
    tw=fs["gallons_water_footprint"].sum(); tmeal=fs["meals_wasted"].sum()
    vpt=tv/ts if ts>0 else 0; co2pt=tco2/ts if ts>0 else 0; watpt=tw/ts if ts>0 else 0
    mpt=mpt_calc(fs,fd)
    res_surplus=fs[fs["sector"]=="Residential"]["tons_surplus"].sum() if "Residential" in fs["sector"].values else ts*0.3
    fs_landfill=fs[fs["sector"]=="Foodservice"]["tons_landfill"].sum() if "Foodservice" in fs["sector"].values else tl*0.3
    ret_surplus=fs[fs["sector"]=="Retail"]["tons_surplus"].sum() if "Retail" in fs["sector"].values else ts*0.2
    farm_nh=fs[fs["sector"]=="Farm"]["tons_not_harvested"].sum() if "Farm" in fs["sector"].values and "tons_not_harvested" in fs.columns else ts*0.03
    top_state=fss.groupby("state")["tons_surplus"].sum().idxmax()
    top_state_t=fss.groupby("state")["tons_surplus"].sum().max()

    interventions=[
        {"name":"Unified Federal Date Label Standard","layer":"Prevention","tons":res_surplus*0.09,"feasibility":0.90,"speed":0.85,"cost_m":50,"mechanism":"Single USDA/FDA executive order. NRDC estimates 8-10% residential waste reduction. No new infrastructure.","one_liner":"One rule change that makes billions of pounds of safe food safe to eat again."},
        {"name":"Foodservice Organic Waste Diversion Mandate","layer":"Prevention + Diversion","tons":fs_landfill*0.40,"feasibility":0.78,"speed":0.65,"cost_m":200,"mechanism":"Mirrors Massachusetts 2014 law. Operators above 1 ton/week must divert. Proven at state scale.","one_liner":"The law Massachusetts already passed. Apply it to every state."},
        {"name":"Grade B Produce + SNAP Multiplier Act","layer":"Prevention + Recovery","tons":farm_nh*0.25,"feasibility":0.70,"speed":0.55,"cost_m":600,"mechanism":"Creates a market that does not exist. Farmers earn revenue. $1 SNAP = $1.50 on Grade B. No new infrastructure.","one_liner":"Simultaneously reduces farm waste and hunger with one price signal."},
        {"name":"Federal Cold Chain Food Bank Infrastructure Act","layer":"Recovery","tons":td*0.30,"feasibility":0.65,"speed":0.45,"cost_m":800,"mechanism":"Funds food banks to operate 24/7 with refrigerated receiving. Perishable donations at 10pm currently go in the trash.","one_liner":"The food exists. The hunger exists. Build the bridge between them."},
        {"name":"Dynamic Markdown Pricing Mandate (Retail)","layer":"Prevention","tons":ret_surplus*0.17,"feasibility":0.62,"speed":0.60,"cost_m":30,"mechanism":"France Loi Garot 2016 reduced supermarket waste 17% in 18 months. USDA-subsidy retailers mark down 48hr before expiry.","one_liner":"Airlines solved this for seats. France solved it for food. We have the template."},
        {"name":f"{top_state} Federal Innovation Pilot Zone","layer":"All Three Layers","tons":top_state_t*0.12,"feasibility":0.80,"speed":0.75,"cost_m":400,"mechanism":f"{top_state} generates {top_state_t/ts*100:.1f}% of national surplus. Pilot combines all layers, demonstrates national replicability.","one_liner":f"One state. {top_state_t/ts*100:.1f}% of the national problem. Prove it here, replicate everywhere."},
        {"name":"Mandatory Commercial Composting (Vermont/MA Model)","layer":"Diversion","tons":tl*0.35,"feasibility":0.68,"speed":0.55,"cost_m":300,"mechanism":"Vermont Act 148 and Massachusetts 2022 ban prove mandatory composting works. Federal mandate with 18-month phase-in.","one_liner":"Food that cannot be eaten should not create methane. It should create soil."},
        {"name":"National Food Logistics Corps","layer":"Recovery","tons":tl*0.08,"feasibility":0.55,"speed":0.50,"cost_m":1200,"mechanism":"Paid professionals, not volunteers. Refrigerated vans, route optimization. City Harvest proves the model at city scale.","one_liner":"Hunger is a logistics problem. Hire people to solve it."},
    ]
    max_tons=max(iv["tons"] for iv in interventions)
    for iv in interventions:
        iv["tons_norm"]=iv["tons"]/max_tons if max_tons>0 else 0
        iv["usd"]=iv["tons"]*vpt; iv["co2_saved"]=iv["tons"]*co2pt; iv["meals"]=iv["tons"]*mpt*0.4
        iv["roi"]=(iv["usd"]+iv["co2_saved"]*65+(iv["tons"]*watpt/1000)*4)/(iv["cost_m"]*1e6) if iv["cost_m"]>0 else 0
        iv["score"]=iv["tons_norm"]*0.35+iv["feasibility"]*0.30+iv["speed"]*0.20+min(iv["roi"],5)/5*0.15
    interventions.sort(key=lambda x:x["score"],reverse=True)
    winner=interventions[0]

    st.markdown(f'<div style="background:linear-gradient(135deg,#1B3A5C,#2E7D32);color:#ECEFF1;border-radius:14px;padding:26px 30px;text-align:center;margin:10px 0 22px 0;"><div style="font-size:12px;letter-spacing:.15em;text-transform:uppercase;color:#A5D6A7;margin-bottom:8px;">The Data Says: Pass This Law First</div><div style="font-size:26px;font-weight:900;margin-bottom:10px;">{winner["name"]}</div><div style="font-size:15px;color:#C8E6C9;font-style:italic;margin-bottom:14px;">"{winner["one_liner"]}"</div><div style="display:flex;justify-content:center;gap:28px;flex-wrap:wrap;margin-top:10px;"><span><b style="font-size:20px;">{fmt(winner["tons"])} t</b><br><span style="font-size:10px;color:#A5D6A7;">SURPLUS REDUCED</span></span><span><b style="font-size:20px;">{fmt(winner["usd"],"$")}</b><br><span style="font-size:10px;color:#A5D6A7;">VALUE RECOVERED</span></span><span><b style="font-size:20px;">{fmt(winner["meals"])}</b><br><span style="font-size:10px;color:#A5D6A7;">MEALS RECOVERED</span></span><span><b style="font-size:20px;">{winner["score"]:.2f}/1.00</b><br><span style="font-size:10px;color:#A5D6A7;">COMPOSITE SCORE</span></span></div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ok"><b>Why this law wins:</b> {winner["mechanism"]}<br><b>Layer:</b> {winner["layer"]} | <b>Feasibility:</b> {winner["feasibility"]:.0%} | <b>Speed:</b> {winner["speed"]:.0%} | <b>Fed Cost:</b> ${winner["cost_m"]}M</div>', unsafe_allow_html=True)

    st.markdown("---")
    df_iv=pd.DataFrame([{"Intervention":iv["name"],"Layer":iv["layer"],"Feasibility":iv["feasibility"],"Tons Reduced":iv["tons"],"Score":iv["score"],"Cost ($M)":iv["cost_m"]} for iv in interventions])
    fig=px.scatter(df_iv,x="Feasibility",y="Tons Reduced",size="Score",color="Layer",hover_name="Intervention",text="Intervention",color_discrete_sequence=COLORS,size_max=50)
    fig.update_traces(textposition="top center",textfont_size=9)
    fig.add_vline(x=0.70,line_dash="dot",line_color="gray",opacity=0.4,annotation_text="Feasibility threshold")
    fig.add_hline(y=df_iv["Tons Reduced"].median(),line_dash="dot",line_color="gray",opacity=0.4,annotation_text="Median impact")
    fig.update_xaxes(tickformat=".0%",range=[0.4,1.0])
    fig.update_layout(annotations=[dict(x=0.88,y=df_iv["Tons Reduced"].max()*0.92,text="SWEET SPOT",showarrow=False,font=dict(size=11,color="#2E7D32"),bgcolor="#E8F5E9",bordercolor="#2E7D32",borderwidth=1)],legend=dict(orientation="h",yanchor="bottom",y=-0.3))
    st.plotly_chart(cl(fig,"Impact vs Feasibility Matrix -- All Interventions"),width='stretch')

    rank_df=pd.DataFrame([{"Rank":i+1,"Intervention":iv["name"],"Layer":iv["layer"],"Tons Reduced":int(iv["tons"]),"Feasibility":iv["feasibility"],"Speed":iv["speed"],"Fed Cost $M":iv["cost_m"],"Score":iv["score"]} for i,iv in enumerate(interventions)])
    st.dataframe(rank_df.style.format({"Tons Reduced":"{:,.0f}","Feasibility":"{:.0%}","Speed":"{:.0%}","Score":"{:.3f}"}),width='stretch',hide_index=True)


def pg_compare(fss):
    st.title("State vs State Comparison Engine")
    available=sorted(fss["state"].dropna().unique().tolist())
    default_a="California" if "California" in available else available[0]
    default_b="Texas" if "Texas" in available else available[1]
    c1,c2=st.columns(2)
    with c1: state_a=st.selectbox("State A",available,index=available.index(default_a),key="cmp_a")
    with c2: state_b=st.selectbox("State B",available,index=available.index(default_b),key="cmp_b")
    if state_a==state_b: st.warning("Select two different states."); return

    def agg_state(state_name):
        sub=fss[fss["state"]==state_name]
        r=sub.agg({c:"sum" for c in ["tons_surplus","tons_landfill","tons_donations","tons_composting","tons_anaerobic_digestion","tons_animal_feed","tons_industrial_uses","tons_waste","meals_wasted","us_dollars_surplus","surplus_total_100_year_mtco2e_footprint","gallons_water_footprint"] if c in sub.columns})
        ts=r.get("tons_surplus",1)
        r["landfill_rate"]=r.get("tons_landfill",0)/ts if ts>0 else 0
        r["donation_rate"]=r.get("tons_donations",0)/ts if ts>0 else 0
        r["diversion_rate"]=(r.get("tons_donations",0)+r.get("tons_composting",0)+r.get("tons_animal_feed",0)+r.get("tons_anaerobic_digestion",0)+r.get("tons_industrial_uses",0))/ts if ts>0 else 0
        r["fi_rate"]=FOOD_INSECURITY.get(state_name,0)
        return r

    a=agg_state(state_a); b=agg_state(state_b)

    metrics=[("Total Surplus","tons_surplus",lambda x:fmt(x)+" t",False),("Landfill Rate","landfill_rate",lambda x:f"{x:.1%}",True),("Donation Rate","donation_rate",lambda x:f"{x:.1%}",False),("Diversion Rate","diversion_rate",lambda x:f"{x:.1%}",False),("Food Insecurity","fi_rate",lambda x:f"{x:.1f}%",True),("Meals Wasted","meals_wasted",lambda x:fmt(x),True)]
    cols=st.columns(len(metrics))
    for col,(label,key,fmtfn,lower_better) in zip(cols,metrics):
        va=float(a.get(key,0)); vb=float(b.get(key,0))
        winner_a=va<vb if lower_better else va>vb
        col.markdown(f'<div style="background:{"#E8F5E9" if winner_a else "#FFEBEE"};border-radius:8px;padding:10px;text-align:center;margin-bottom:6px;"><div style="font-size:10px;color:#555;font-weight:600;">{label}</div><div style="font-size:15px;font-weight:800;color:{"#1B5E20" if winner_a else "#B71C1C"};">{fmtfn(va)}</div><div style="font-size:10px;color:#888;">{state_a}</div></div>', unsafe_allow_html=True)
        col.markdown(f'<div style="background:{"#E8F5E9" if not winner_a else "#FFEBEE"};border-radius:8px;padding:10px;text-align:center;margin-bottom:6px;"><div style="font-size:10px;color:#555;font-weight:600;">{label}</div><div style="font-size:15px;font-weight:800;color:{"#1B5E20" if not winner_a else "#B71C1C"};">{fmtfn(vb)}</div><div style="font-size:10px;color:#888;">{state_b}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    c1,c2=st.columns(2)
    with c1:
        dc=[c for c in DEST_BASE if c in fss.columns]
        a_dest=fss[fss["state"]==state_a][dc].sum(); b_dest=fss[fss["state"]==state_b][dc].sum()
        dest_df=pd.DataFrame({"Destination":[dlabel(c) for c in dc],state_a:a_dest.values,state_b:b_dest.values})
        dest_df=dest_df[dest_df[[state_a,state_b]].sum(axis=1)>0]
        fig=go.Figure([go.Bar(name=state_a,x=dest_df["Destination"],y=dest_df[state_a],marker_color="#1B3A5C"),go.Bar(name=state_b,x=dest_df["Destination"],y=dest_df[state_b],marker_color="#2E7D32")])
        fig.update_layout(barmode="group",yaxis_title="Tons",legend=dict(orientation="h"))
        st.plotly_chart(cl(fig,"Destination Mix Comparison"),width='stretch')
    with c2:
        trend_a=fss[fss["state"]==state_a].groupby("year")["tons_landfill"].sum()/fss[fss["state"]==state_a].groupby("year")["tons_surplus"].sum()
        trend_b=fss[fss["state"]==state_b].groupby("year")["tons_landfill"].sum()/fss[fss["state"]==state_b].groupby("year")["tons_surplus"].sum()
        trend_df=pd.DataFrame({state_a:trend_a,state_b:trend_b}).reset_index()
        fig2=px.line(trend_df,x="year",y=[state_a,state_b],markers=True,color_discrete_sequence=["#1B3A5C","#2E7D32"])
        fig2.update_yaxes(tickformat=".0%"); st.plotly_chart(cl(fig2,"Landfill Rate Trend: Head to Head"),width='stretch')

    lf_a=float(a.get("landfill_rate",0)); lf_b=float(b.get("landfill_rate",0))
    worse=state_a if lf_a>lf_b else state_b; better=state_b if lf_a>lf_b else state_a
    worse_lf=max(lf_a,lf_b); better_lf=min(lf_a,lf_b)
    worse_ts=float(a.get("tons_surplus",0)) if lf_a>lf_b else float(b.get("tons_surplus",0))
    gap_tons=(worse_lf-better_lf)*worse_ts
    st.markdown(f'<div class="insight"><b>{worse} vs {better}:</b> If {worse} matched {better}\'s landfill rate of {better_lf:.1%}, it would divert {fmt(gap_tons)} additional tons per year.</div>', unsafe_allow_html=True)


def pg_inaction(fs, fss):
    """FIX: fss is now correctly passed as a parameter."""
    st.title("Cost of Inaction")
    ts=fs["tons_surplus"].sum(); tl=fs["tons_landfill"].sum(); tv=fs["us_dollars_surplus"].sum()
    tco2=fs["surplus_total_100_year_mtco2e_footprint"].sum(); tmeal=fs["meals_wasted"].sum()
    tw=fs["gallons_water_footprint"].sum()
    yrs=max(len(fs["year"].unique()),1)
    per_year={"surplus_t":ts/yrs,"landfill_t":tl/yrs,"usd":tv/yrs,"co2e":tco2/yrs,"meals":tmeal/yrs,"water":tw/yrs}
    per_day={k:v/365 for k,v in per_year.items()}; per_hour={k:v/8760 for k,v in per_year.items()}
    per_minute={k:v/525600 for k,v in per_year.items()}; per_second={k:v/31536000 for k,v in per_year.items()}

    rows=[]
    for label,d in [("Per Year",per_year),("Per Day",per_day),("Per Hour",per_hour),("Per Minute",per_minute),("Per Second",per_second)]:
        rows.append({"Time Unit":label,"Surplus":fmt(d["surplus_t"])+" t","Landfill":fmt(d["landfill_t"])+" t","Economic Value Lost":fmt(d["usd"],"$"),"Meals Wasted":fmt(d["meals"]),"CO2e (MT)":f"{d['co2e']:.4f}","Water":fmt(d["water"])+" gal"})
    st.dataframe(pd.DataFrame(rows),width='stretch',hide_index=True)

    st.markdown("---")
    st.subheader("Live Waste Counter")
    st.caption("Counter starts from page load. Rates derived from your filtered dataset.")
    mps=per_second["meals"]; ups=per_second["usd"]; tps=per_second["surplus_t"]
    lfps=per_second["landfill_t"]; cps=per_second["co2e"]

    st.components.v1.html(f"""
    <style>
    .cg{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;font-family:'Segoe UI',Arial,sans-serif;}}
    .cc{{background:#fff;border:1px solid #DDE3EC;border-radius:10px;padding:16px 18px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.06);}}
    .cl{{font-size:10px;font-weight:700;color:#566573;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;}}
    .cv{{font-size:26px;font-weight:900;color:#B71C1C;font-variant-numeric:tabular-nums;}}
    .cu{{font-size:11px;color:#888;margin-top:4px;}}
    </style>
    <div class="cg">
      <div class="cc"><div class="cl">Meals Wasted Since Page Load</div><div class="cv" id="c-meals">0</div><div class="cu">meals</div></div>
      <div class="cc"><div class="cl">Economic Value Lost</div><div class="cv" id="c-usd">$0</div><div class="cu">USD</div></div>
      <div class="cc"><div class="cl">Surplus Generated</div><div class="cv" id="c-tons">0</div><div class="cu">tons</div></div>
      <div class="cc"><div class="cl">Sent to Landfill</div><div class="cv" id="c-lf">0</div><div class="cu">tons</div></div>
      <div class="cc"><div class="cl">CO2e Emitted</div><div class="cv" id="c-co2">0.0000</div><div class="cu">metric tons</div></div>
      <div class="cc"><div class="cl">Seconds of Inaction</div><div class="cv" id="c-secs">0</div><div class="cu">seconds since page load</div></div>
    </div>
    <script>
    const mps={mps:.4f},ups={ups:.2f},tps={tps:.6f},lfps={lfps:.6f},cps={cps:.8f};
    const start=Date.now();
    function fmt(n){{if(n>=1e9)return(n/1e9).toFixed(2)+'B';if(n>=1e6)return(n/1e6).toFixed(2)+'M';if(n>=1e3)return(n/1e3).toFixed(1)+'K';return Math.round(n).toLocaleString();}}
    function tick(){{const s=(Date.now()-start)/1000;document.getElementById('c-meals').textContent=fmt(s*mps);document.getElementById('c-usd').textContent='$'+fmt(s*ups);document.getElementById('c-tons').textContent=fmt(s*tps);document.getElementById('c-lf').textContent=fmt(s*lfps);document.getElementById('c-co2').textContent=(s*cps).toFixed(4);document.getElementById('c-secs').textContent=Math.floor(s).toLocaleString();}}
    setInterval(tick,250);tick();
    </script>
    """, height=260)

    st.markdown("---")
    st.subheader("Annual Cost of Inaction by State")
    # FIX: fss is now a proper parameter, not undefined
    state_cost=fss.groupby("state").agg(surplus=("tons_surplus","sum"),landfill=("tons_landfill","sum"),meals=("meals_wasted","sum"),usd=("us_dollars_surplus","sum")).reset_index()
    state_cost["abbrev"]=state_cost["state"].map(STATE_ABBREV)
    state_cost["annual_usd"]=state_cost["usd"]/yrs
    fig=px.choropleth(state_cost,locations="abbrev",locationmode="USA-states",color="annual_usd",scope="usa",hover_name="state",color_continuous_scale="Reds",title="Annual Economic Value Lost Per State (Cost of Inaction)",hover_data={"annual_usd":True,"abbrev":False})
    fig.update_layout(geo=dict(bgcolor="white"),paper_bgcolor="white")
    st.plotly_chart(fig,width='stretch')

    st.markdown(f'<div class="critical"><b>Cost of doing nothing for one more year:</b> {fmt(per_year["usd"],"$")} in food value lost. {fmt(per_year["meals"])} meals wasted. {per_year["co2e"]/1e6:.2f}M metric tons of CO2e emitted. Every year without a federal policy is a choice -- and this is exactly what that choice costs.</div>', unsafe_allow_html=True)


def pg_subsector(fs, fd):
    st.title("Sub-Sector Intelligence")
    dc=dcols(fs)
    ss_agg=fs.groupby(["sector","sub_sector"]).agg(tons_surplus=("tons_surplus","sum"),tons_landfill=("tons_landfill","sum"),tons_donations=("tons_donations","sum"),us_dollars_surplus=("us_dollars_surplus","sum"),meals_wasted=("meals_wasted","sum")).reset_index()
    ss_agg["landfill_rate"]=sdiv(ss_agg["tons_landfill"].values,ss_agg["tons_surplus"].values)
    ss_agg["donation_rate"]=sdiv(ss_agg["tons_donations"].values,ss_agg["tons_surplus"].values)
    ss_agg=ss_agg[ss_agg["sub_sector"]!="Not Applicable"].sort_values("tons_surplus",ascending=False)

    c1,c2=st.columns(2)
    with c1:
        fig=px.bar(ss_agg.head(15).sort_values("tons_surplus"),x="tons_surplus",y="sub_sector",orientation="h",color="sector",color_discrete_sequence=COLORS,labels={"tons_surplus":"Tons","sub_sector":""})
        st.plotly_chart(cl(fig,"Top 15 Sub-Sectors by Surplus"),width='stretch')
    with c2:
        fig2=px.bar(ss_agg.sort_values("landfill_rate",ascending=False).head(15),x="landfill_rate",y="sub_sector",orientation="h",color="sector",color_discrete_sequence=COLORS,labels={"landfill_rate":"Landfill Rate","sub_sector":""})
        fig2.update_xaxes(tickformat=".0%"); fig2.update_layout(showlegend=False)
        st.plotly_chart(cl(fig2,"Landfill Rate by Sub-Sector"),width='stretch')

    dest_sub=fs[fs["sub_sector"]!="Not Applicable"].groupby("sub_sector")[dc].sum().reset_index()
    dm=dest_sub.melt(id_vars="sub_sector",value_vars=dc,var_name="Dest",value_name="Tons")
    dm["Dest"]=dm["Dest"].apply(dlabel)
    fig3=px.bar(dm,x="sub_sector",y="Tons",color="Dest",barmode="stack",color_discrete_sequence=COLORS)
    fig3.update_xaxes(tickangle=35)
    st.plotly_chart(cl(fig3,"Destination Mix by Sub-Sector",h=480),width='stretch')

    if "sub_sector_category" in fs.columns:
        st.subheader("Sub-Sector Category Treemap (31 Categories)")
        cat_agg=fs[fs["sub_sector_category"]!="Not Applicable"].groupby(["sector","sub_sector","sub_sector_category"]).agg(tons_surplus=("tons_surplus","sum"),tons_landfill=("tons_landfill","sum")).reset_index()
        cat_agg["landfill_rate"]=sdiv(cat_agg["tons_landfill"].values,cat_agg["tons_surplus"].values)
        fig4=px.treemap(cat_agg[cat_agg["tons_surplus"]>0],path=["sector","sub_sector","sub_sector_category"],values="tons_surplus",color="landfill_rate",color_continuous_scale="RdYlGn_r")
        st.plotly_chart(cl(fig4,"Sub-Sector Category Treemap (color = landfill rate)",h=500),width='stretch')


def pg_action_plan(ss_df, fc_df, fs):
    st.title("State Action Plan Generator")
    available=sorted(ss_df["state"].dropna().unique().tolist())
    default_idx=available.index("California") if "California" in available else 0
    c1,c2=st.columns([2,4])
    with c1: sel_state=st.selectbox("Select State",available,index=default_idx)

    st.markdown(f"## Action Plan: {sel_state}")
    sd=ss_df[ss_df["state"]==sel_state].groupby("state").agg(tons_surplus=("tons_surplus","sum"),tons_landfill=("tons_landfill","sum"),tons_donations=("tons_donations","sum"),tons_composting=("tons_composting","sum"),tons_waste=("tons_waste","sum"),meals_wasted=("meals_wasted","sum"),us_dollars_surplus=("us_dollars_surplus","sum"),co2e=("surplus_total_100_year_mtco2e_footprint","sum"),water=("gallons_water_footprint","sum")).reset_index().iloc[0]
    nat=ss_df.groupby("state").agg(tons_surplus=("tons_surplus","sum"),tons_landfill=("tons_landfill","sum"),tons_donations=("tons_donations","sum"),tons_composting=("tons_composting","sum")).reset_index()
    nat["landfill_rate"]=sdiv(nat["tons_landfill"].values,nat["tons_surplus"].values)
    nat["donation_rate"]=sdiv(nat["tons_donations"].values,nat["tons_surplus"].values)
    nat["composting_rate"]=sdiv(nat["tons_composting"].values,nat["tons_surplus"].values)
    nat_lf=nat["landfill_rate"].mean(); nat_dr=nat["donation_rate"].mean(); nat_cr=nat["composting_rate"].mean()
    ts_state=sd["tons_surplus"]; state_lf=sdiv([sd["tons_landfill"]],[ts_state])[0]
    state_dr=sdiv([sd["tons_donations"]],[ts_state])[0]; state_cr=sdiv([sd["tons_composting"]],[ts_state])[0]
    state_pct=ts_state/ss_df["tons_surplus"].sum()*100 if ss_df["tons_surplus"].sum()>0 else 0
    vpt=sd["us_dollars_surplus"]/ts_state if ts_state>0 else 0; mpt=mpt_calc(fs)

    c=st.columns(4)
    c[0].metric("Total Surplus",fmt(ts_state)+" t",delta=f"{state_pct:.1f}% of national")
    c[1].metric("Landfill Rate",f"{state_lf:.1%}",delta=f"{(state_lf-nat_lf)*100:+.1f}pp vs avg",delta_color="inverse")
    c[2].metric("Donation Rate",f"{state_dr:.1%}",delta=f"{(state_dr-nat_dr)*100:+.1f}pp vs avg")
    c[3].metric("Composting Rate",f"{state_cr:.1%}",delta=f"{(state_cr-nat_cr)*100:+.1f}pp vs avg")

    c3,c4=st.columns(2)
    with c3:
        sec_data=ss_df[ss_df["state"]==sel_state].groupby("sector")[["tons_surplus","tons_landfill"]].sum().reset_index()
        fig1=px.bar(sec_data,x="sector",y="tons_surplus",color="sector",color_discrete_sequence=COLORS)
        fig1.update_layout(showlegend=False); st.plotly_chart(cl(fig1,f"{sel_state}: Surplus by Sector"),width='stretch')
    with c4:
        dc=dcols(ss_df); dest_st=ss_df[ss_df["state"]==sel_state][dc].sum()
        dest_st.index=[dlabel(i) for i in dest_st.index]; dest_st=dest_st[dest_st>0]
        fig2=px.pie(values=dest_st.values,names=dest_st.index,color_discrete_sequence=COLORS)
        fig2.update_traces(textposition="inside",textinfo="percent+label"); st.plotly_chart(cl(fig2,f"{sel_state}: Destination Mix"),width='stretch')

    st.markdown("---")
    st.subheader("Prioritized Action Plan")
    actions=[]
    if state_lf>nat_lf*1.15:
        gap=( state_lf-nat_lf)*ts_state
        actions.append({"p":1,"title":"Urgent: Reduce Landfill Rate to National Average","cls":"critical","detail":f"{sel_state}'s landfill rate of {state_lf:.1%} is {(state_lf-nat_lf)*100:.1f}pp above the national average of {nat_lf:.1%}. Closing this gap would divert {fmt(gap)} tons annually. Recommended: mandatory commercial organic waste diversion for generators above 1 ton/week.","impact":fmt(gap*vpt,"$"),"timeline":"12-24 months"})
    if state_dr<nat_dr*0.85:
        gap_dr=(nat_dr-state_dr)*ts_state; meals_gap=gap_dr*mpt
        actions.append({"p":2,"title":"Expand Food Donation Infrastructure","cls":"warn","detail":f"{sel_state}'s donation rate of {state_dr:.1%} is below the national average of {nat_dr:.1%}. Raising to the national average would redirect {fmt(gap_dr)} tons, recovering {fmt(meals_gap)} additional meals.","impact":fmt(meals_gap)+" meals","timeline":"6-18 months"})
    if state_cr<nat_cr*0.80:
        actions.append({"p":3,"title":"Expand Municipal Composting","cls":"warn","detail":f"{sel_state}'s composting rate of {state_cr:.1%} lags the national average of {nat_cr:.1%}. Mandate organic waste separation for municipalities above 50,000 residents.","impact":fmt((nat_cr-state_cr)*ts_state)+" t diverted","timeline":"18-36 months"})
    if state_pct>5.0:
        actions.append({"p":4,"title":"Leverage Scale: State-Level Innovation Pilot","cls":"insight","detail":f"{sel_state} generates {state_pct:.1f}% of national food surplus. Apply for Federal Food Waste Innovation Zone designation.","impact":f"{state_pct:.1f}% national impact","timeline":"6-12 months"})

    if not actions:
        st.markdown(f'<div class="ok"><b>{sel_state} is a national leader.</b> Performance is at or above national benchmarks. Recommended role: serve as model state for national replication.</div>', unsafe_allow_html=True)
    else:
        for action in sorted(actions,key=lambda x:x["p"]):
            st.markdown(f'<div class="{action["cls"]}"><b>Priority {action["p"]}: {action["title"]}</b><br>{action["detail"]}<br><b>Est. Impact:</b> {action["impact"]} | <b>Timeline:</b> {action["timeline"]}</div>', unsafe_allow_html=True)


def main():
    with st.spinner("Loading datasets..."):
        s,d,c,ss,sd=load_data()
        s,d,c,ss,sd=engineer(s,d,c,ss,sd)

    yr,sec,ft,state=build_sidebar(s,ss)

    # Apply filters
    fs  = F(s,  yr, sec, ft)
    fd  = F(d,  yr, sec, ft)
    fc  = F(c,  yr, sec, ft)
    fss = FS(ss, yr, state)

    if fs.empty:
        st.warning("No data matches the selected filters. Adjust the sidebar controls.")
        return

    # Sticky top navigation (renders at top of main content)
    page = render_top_nav()

    # Route to page
    routing = {
        "01 Executive Dashboard":   lambda: pg_executive(fs, fss),
        "02 Supply Chain Funnel":   lambda: pg_supply_funnel(fs, fd),
        "03 Temporal Analysis":     lambda: pg_temporal(s),
        "04 Sector Deep Dive":      lambda: pg_sector(fs),
        "05 Sub-Sector Intel":      lambda: pg_subsector(fs, fd),
        "06 Food Type & Category":  lambda: pg_food(fd),
        "07 Root Cause Engine":     lambda: pg_causes(fc),
        "08 Hidden Waste":          lambda: pg_hidden(fc, fd),
        "09 Waste Flow (Sankey)":   lambda: pg_sankey(fs),
        "10 State Intelligence":    lambda: pg_states(fss, state),
        "11 State Clustering":      lambda: pg_clustering(ss),
        "12 Zero Waste Index":      lambda: pg_zero_waste_index(ss),
        "13 Environmental":         lambda: pg_environment(fs, fss),
        "14 Waste x Hunger":        lambda: pg_nexus(ss),
        "15 Innovation Hub":        lambda: pg_innovations(fs, fss),
        "16 Policy Simulator":      lambda: pg_simulator(fs),
        "17 ROI Calculator":        lambda: pg_roi(fs, fss),
        "18 Policy Recs":           lambda: pg_policy(fs, fc, fss),
        "19 State Action Plan":     lambda: pg_action_plan(ss, fc, fs),
        "20 Trend Forecast":        lambda: pg_forecast(s),
        "22 The One Law":           lambda: pg_one_law(fs, fd, fc, fss),
        "23 State vs State":        lambda: pg_compare(fss),
        "24 Cost of Inaction":      lambda: pg_inaction(fs, fss),  # FIX: fss passed correctly
    }

    if page in routing:
        routing[page]()
    else:
        st.warning(f"Page '{page}' not found.")


if __name__ == "__main__":
    main()