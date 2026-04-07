"""
ImmoPred AI — France Edition
Données réelles : DVF (Demandes de Valeurs Foncières) 2023
Source officielle : data.gouv.fr
"""
from __future__ import annotations
import os, sys, json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ImmoPred AI France — DVF Réel",
    page_icon="🏡", layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#060614 0%,#0d1b30 60%,#060614 100%);color:#e2e8f0;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0a0a28,#141438);border-right:1px solid rgba(0,183,120,.2);}
[data-testid="stSidebar"] *{color:#e2e8f0!important;}
h1{background:linear-gradient(90deg,#00b778,#0ea5e9);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:800!important;font-size:2.4rem!important;}
h2{color:#34d399!important;font-weight:700!important;}
h3{color:#60a5fa!important;font-weight:600!important;}
[data-testid="stMetric"]{background:rgba(0,183,120,.08);border:1px solid rgba(0,183,120,.25);border-radius:16px;padding:16px 20px;transition:transform .2s,box-shadow .2s;}
[data-testid="stMetric"]:hover{transform:translateY(-3px);box-shadow:0 8px 30px rgba(0,183,120,.25);}
[data-testid="stMetricLabel"]{color:#94a3b8!important;font-size:.8rem!important;}
[data-testid="stMetricValue"]{color:#e2e8f0!important;font-weight:700!important;}
[data-testid="stTabs"] button{font-weight:600;color:#94a3b8;border-radius:10px 10px 0 0;transition:all .2s;}
[data-testid="stTabs"] button[aria-selected="true"]{color:#00b778!important;border-bottom:3px solid #00b778!important;background:rgba(0,183,120,.1)!important;}
.stButton>button{background:linear-gradient(135deg,#00b778,#0ea5e9);color:#fff!important;border:none;border-radius:12px;padding:12px 32px;font-weight:700;font-size:1rem;transition:all .3s;width:100%;}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 8px 25px rgba(0,183,120,.4);}
.hero{background:linear-gradient(135deg,rgba(0,183,120,.15),rgba(14,165,233,.1));border:1px solid rgba(0,183,120,.3);border-radius:20px;padding:36px 40px;text-align:center;margin-bottom:28px;}
.card{background:rgba(255,255,255,.04);border:1px solid rgba(0,183,120,.2);border-radius:16px;padding:22px;backdrop-filter:blur(8px);margin-bottom:14px;}
.result-box{background:linear-gradient(135deg,rgba(0,183,120,.2),rgba(14,165,233,.15));border:2px solid #00b778;border-radius:20px;padding:32px;text-align:center;margin-top:20px;}
.result-price{font-size:2.8rem;font-weight:800;background:linear-gradient(90deg,#00b778,#0ea5e9);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.badge{display:inline-block;padding:4px 12px;border-radius:20px;font-size:.78rem;font-weight:600;margin:2px;}
.badge-green{background:rgba(0,183,120,.2);color:#34d399;border:1px solid #059669;}
.badge-blue{background:rgba(14,165,233,.15);color:#60a5fa;border:1px solid #0284c7;}
.badge-gold{background:rgba(251,191,36,.15);color:#fbbf24;border:1px solid #d97706;}
.dvf-note{background:rgba(0,183,120,.08);border-left:4px solid #00b778;border-radius:0 8px 8px 0;padding:12px 18px;font-size:.85rem;color:#94a3b8;margin-bottom:16px;}
hr{border-color:rgba(0,183,120,.2)!important;margin:22px 0!important;}
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ──────────────────────────────────────────────────────────────
PT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,10,30,.6)',
    font=dict(family='Inter', color='#e2e8f0', size=13),
    title_font=dict(size=15, color='#34d399'),
    xaxis=dict(gridcolor='rgba(0,183,120,.12)', linecolor='rgba(0,183,120,.2)'),
    yaxis=dict(gridcolor='rgba(0,183,120,.12)', linecolor='rgba(0,183,120,.2)'),
    colorway=['#00b778','#0ea5e9','#f59e0b','#f87171','#a78bfa','#34d399','#fb923c','#e879f9'],
)

REGION_COLORS = {
    'Île-de-France': '#f87171', 'PACA': '#fbbf24',
    'Auvergne-Rhône-Alpes': '#34d399', 'Nouvelle-Aquitaine': '#60a5fa',
    'Occitanie': '#a78bfa', 'Pays de la Loire': '#00b778',
    'Bretagne': '#fb923c', 'Hauts-de-France': '#e879f9',
    'Grand Est': '#38bdf8', 'Normandie': '#4ade80',
    'Centre-Val de Loire': '#f472b6', 'Bourgogne-Franche-Comté': '#facc15',
    'Corse': '#fb7185', 'Autre': '#94a3b8',
}

MODEL_COLORS = {
    'Linear Regression':'#60a5fa','Ridge Regression':'#34d399',
    'Gradient Boosting':'#f59e0b','Random Forest':'#00b778','XGBoost':'#a78bfa'
}
MODELS_ORDER = ['Linear Regression','Ridge Regression','Gradient Boosting','Random Forest','XGBoost']


# ── Cache helpers ─────────────────────────────────────────────────────────────
def _mtime(path):
    return os.path.getmtime(path) if os.path.exists(path) else 0

@st.cache_data(show_spinner=False)
def load_data(_mtime=0):
    p_gz = os.path.join(ROOT,'data','processed','housing_engineered.csv.gz')
    p_csv = os.path.join(ROOT,'data','processed','housing_engineered.csv')
    if os.path.exists(p_gz):
        return pd.read_csv(p_gz, compression='gzip')
    elif os.path.exists(p_csv):
        return pd.read_csv(p_csv)
    return None

@st.cache_data(show_spinner=False)
def load_results(_mtime=0):
    p = os.path.join(ROOT,'models','results.json')
    if not os.path.exists(p): return None, None
    d = json.load(open(p))
    return d.get('results',{}), d.get('best_model','')

@st.cache_data(show_spinner=False)
def load_predictions(_mtime=0):
    p = os.path.join(ROOT,'models','test_predictions.csv')
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_resource(show_spinner=False)
def load_model(name, _mtime=0):
    p = os.path.join(ROOT,'models', name.lower().replace(' ','_')+'.pkl')
    return joblib.load(p) if os.path.exists(p) else None

@st.cache_data(show_spinner=False)
def load_fi(_mtime=0):
    p = os.path.join(ROOT,'models','feature_importance.csv')
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data(show_spinner=False)
def get_features(_mtime=0):
    p = os.path.join(ROOT,'models','features.json')
    return json.load(open(p)) if os.path.exists(p) else []

def run_pipeline():
    with st.spinner("⚙️ Téléchargement DVF + Pipeline ML… (~5-15 min selon connexion)"):
        import subprocess
        r = subprocess.run([sys.executable, os.path.join(ROOT,'main.py')],
                          capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        st.error(f"Erreur :\n```\n{r.stderr[-3000:]}\n```")
        st.stop()
    st.cache_data.clear(); st.cache_resource.clear(); st.rerun()

def fmt_eur(v): return f"{v:,.0f} €"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0 10px'>
      <div style='font-size:2.8rem'>🏡</div>
      <div style='font-size:1.3rem;font-weight:800;
           background:linear-gradient(90deg,#00b778,#0ea5e9);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
        ImmoPred AI
      </div>
      <div style='color:#64748b;font-size:.8rem;margin-top:4px'>
        France · Données DVF Réelles
      </div>
    </div><hr>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='dvf-note'>
      📋 <b>Source officielle</b><br>
      Demandes de Valeurs Foncières 2023<br>
      data.gouv.fr — DGFIP<br>
      <a href='https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres-geolocalisees/'
         target='_blank' style='color:#00b778'>Voir la source ↗</a>
    </div>""", unsafe_allow_html=True)

    st.markdown("**🤖 Modèles ML**")
    for m in MODELS_ORDER:
        st.markdown(f"<div style='padding:3px 0;font-size:.82rem'>"
                    f"<span style='color:{MODEL_COLORS[m]}'>●</span> {m}</div>",
                    unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("🔄 Relancer le pipeline"):
        st.cache_data.clear(); st.cache_resource.clear(); run_pipeline()
    st.markdown("<div style='color:#475569;font-size:.72rem;text-align:center;margin-top:16px'>"
                "© 2024 ImmoPred AI · France DVF Edition</div>", unsafe_allow_html=True)


# ── Check données ─────────────────────────────────────────────────────────────
models_ok = os.path.exists(os.path.join(ROOT,'models','results.json'))
data_csv = os.path.join(ROOT,'data','processed','housing_engineered.csv')
data_gz = os.path.join(ROOT,'data','processed','housing_engineered.csv.gz')
data_ok = os.path.exists(data_csv) or os.path.exists(data_gz)

if not models_ok or not data_ok:
    st.markdown("""
    <div class='hero'>
      <h1>🏡 ImmoPred AI France</h1>
      <p>Données DVF réelles — Premier lancement</p>
    </div>""", unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        st.markdown("""
        <div class='dvf-note'>
          📥 Le pipeline va télécharger les données DVF officielles depuis data.gouv.fr
          (~35 départements, ~200-400 Mo) puis entraîner les modèles ML.<br>
          <b>Durée estimée : 5-15 minutes.</b>
        </div>""", unsafe_allow_html=True)
        if st.button("🚀 Télécharger DVF + Entraîner les modèles"): run_pipeline()
    st.stop()

_mt_csv = max(_mtime(data_gz), _mtime(data_csv))
_mt_model = _mtime(os.path.join(ROOT,'models','results.json'))

df       = load_data(_mtime=_mt_csv)
results, best_name = load_results(_mtime=_mt_model)
preds    = load_predictions(_mtime=_mt_model)
fi_df    = load_fi(_mtime=_mt_model)
features = get_features(_mtime=_mt_model)

if df is None or results is None:
    st.error("Données introuvables. Relancez le pipeline."); st.stop()


# ════════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(
    ["🏡  Accueil", "📊  Exploration", "🤖  Modèles", "🔮  Prédiction"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — ACCUEIL
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    br2 = results[best_name]
    st.markdown(f"""
    <div class='hero'>
      <h1>🏡 ImmoPred AI — France Édition</h1>
      <p>Prédiction de prix immobilier · <b>Vraies données DVF 2023</b></p>
      <div style='margin-top:14px'>
        <span class='badge badge-green'>✅ Données Réelles</span>
        <span class='badge badge-green'>DVF 2023</span>
        <span class='badge badge-blue'>XGBoost</span>
        <span class='badge badge-blue'>Random Forest</span>
        <span class='badge badge-gold'>data.gouv.fr</span>
      </div>
    </div>""", unsafe_allow_html=True)

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("🏠 Transactions réelles", f"{len(df):,}")
    k2.metric("🗺️ Départements", f"{df['department'].nunique() if 'department' in df.columns else '—'}")
    k3.metric("🏘️ Communes", f"{df['city'].nunique() if 'city' in df.columns else '—'}")
    k4.metric("🏆 R² (meilleur modèle)", f"{br2['R2']:.4f}")
    k5.metric("📉 MAE", fmt_eur(br2['MAE']))
    k6.metric("📐 Features", f"{len(features)}")

    st.markdown("""
    <div class='dvf-note' style='margin-top:16px'>
      ✅ <b>Ces données sont 100% réelles</b> — issues des actes notariés enregistrés par la Direction Générale des
      Finances Publiques (DGFIP) et publiées en open data sur data.gouv.fr.
      Chaque ligne correspond à une vraie transaction immobilière en France en 2023.
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Carte France
    st.markdown("### 🗺️ Carte des prix immobiliers réels (France 2023)")
    sample_map = df.sample(min(15_000, len(df)), random_state=42)
    fig_map = px.scatter_mapbox(
        sample_map,
        lat='latitude', lon='longitude',
        color='price',
        color_continuous_scale='Plasma',
        size_max=6,
        opacity=0.6,
        zoom=4.8,
        center={'lat': 46.8, 'lon': 2.3},
        mapbox_style='carto-darkmatter',
        hover_data={'city': True, 'price': ':,.0f',
                    'surface_sqm': ':.0f', 'rooms': True,
                    'latitude': False, 'longitude': False}
                    if 'city' in df.columns else {},
        labels={'price': 'Prix (€)'},
        template='plotly_dark'
    )
    fig_map.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0,r=0,t=0,b=0), height=480,
        coloraxis_colorbar=dict(
            tickfont=dict(color='#e2e8f0'),
            title=dict(text='Prix (€)', font=dict(color='#e2e8f0'))
        )
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    c_a, c_b = st.columns(2)
    with c_a:
        st.markdown("### 💰 Prix médian par région")
        if 'region' in df.columns:
            reg_stats = (df.groupby('region')['price']
                         .median().sort_values(ascending=False).reset_index())
            reg_stats.columns = ['Région', 'Prix médian (€)']
            fig_reg = px.bar(
                reg_stats, x='Région', y='Prix médian (€)',
                color='Région',
                color_discrete_map=REGION_COLORS,
                template='plotly_dark',
                text='Prix médian (€)'
            )
            fig_reg.update_traces(texttemplate='%{text:,.0f} €', textposition='outside',
                                   textfont=dict(color='#e2e8f0'))
            fig_reg.update_layout(**PT, height=400, showlegend=False,
                                   xaxis_tickangle=-30)
            st.plotly_chart(fig_reg, use_container_width=True)

    with c_b:
        st.markdown("### 📊 Répartition par type de bien")
        if 'property_type' in df.columns:
            prop_counts = df['property_type'].value_counts().reset_index()
            prop_counts.columns = ['Type', 'count']
            fig_pie = px.pie(
                prop_counts, values='count', names='Type',
                color_discrete_sequence=['#00b778','#0ea5e9'],
                template='plotly_dark', hole=0.52
            )
            fig_pie.update_traces(textfont=dict(color='#e2e8f0', size=13))
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                    font=dict(family='Inter',color='#e2e8f0'),
                                    legend=dict(font=dict(color='#e2e8f0')),
                                    height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Évolution temporelle (si données de mois disponibles)
    if 'month' in df.columns and 'year' in df.columns:
        st.markdown("### 📅 Évolution mensuelle des prix (2023)")
        monthly = df.groupby(['year','month'])['price'].agg(['median','count']).reset_index()
        monthly.columns = ['year','month','prix_median','volume']
        monthly['mois'] = monthly['year'].astype(str) + '-' + monthly['month'].astype(str).str.zfill(2)
        monthly = monthly.sort_values('mois')
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=monthly['mois'], y=monthly['prix_median'],
            mode='lines+markers', name='Prix médian (€)',
            line=dict(color='#00b778', width=3),
            marker=dict(size=7, color='#00b778'),
        ))
        fig_time.update_layout(**PT, height=300, xaxis_title='Mois', yaxis_title='Prix médian (€)')
        st.plotly_chart(fig_time, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — EXPLORATION
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("## 📊 Exploration des données DVF réelles")

    # Filtres
    f1, f2, f3 = st.columns(3)
    regions_list = sorted(df['region'].dropna().unique().tolist()) if 'region' in df.columns else []
    sel_reg = f1.multiselect("Région", regions_list, default=regions_list[:6] if len(regions_list) > 6 else regions_list)
    prop_list = sorted(df['property_type'].dropna().unique().tolist()) if 'property_type' in df.columns else []
    sel_prop = f2.multiselect("Type de bien", prop_list, default=prop_list)
    price_max = int(df['price'].quantile(0.97))
    sel_price = f3.slider("Prix max (€)", 10_000, price_max, price_max, 10_000)

    dff = df.copy()
    if sel_reg and 'region' in df.columns:
        dff = dff[dff['region'].isin(sel_reg)]
    if sel_prop and 'property_type' in df.columns:
        dff = dff[dff['property_type'].isin(sel_prop)]
    dff = dff[dff['price'] <= sel_price]
    st.caption(f"**{len(dff):,}** transactions sélectionnées")

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Surface ↔ Prix")
        sample = dff.sample(min(5000, len(dff)), random_state=42)
        col_c = 'region' if 'region' in sample.columns else None
        fig = px.scatter(
            sample, x='surface_sqm', y='price',
            color=col_c,
            color_discrete_map=REGION_COLORS if col_c else None,
            opacity=0.4,
            labels={'surface_sqm':'Surface (m²)','price':'Prix (€)','region':'Région'},
            template='plotly_dark'
        )
        fig.update_layout(**PT, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Pièces ↔ Prix")
        rooms_stats = (dff.groupby('rooms')['price']
                       .agg(['median','count']).reset_index()
                       .rename(columns={'median':'Prix médian','count':'Volume'}))
        rooms_stats = rooms_stats[rooms_stats['rooms'] <= 10]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=rooms_stats['rooms'], y=rooms_stats['Prix médian'],
            name='Prix médian (€)',
            marker_color='#00b778',
            text=[f"{v:,.0f} €" for v in rooms_stats['Prix médian']],
            textposition='outside', textfont=dict(color='#e2e8f0')
        ))
        fig2.update_layout(**PT, height=380, xaxis_title='Nombre de pièces',
                            yaxis_title='Prix médian (€)', showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### Distribution des prix par région")
        if 'region' in dff.columns:
            fig3 = px.box(
                dff, x='region', y='price',
                color='region', color_discrete_map=REGION_COLORS,
                points=False,
                labels={'region':'Région','price':'Prix (€)'},
                template='plotly_dark'
            )
            fig3.update_layout(**PT, height=400, showlegend=False, xaxis_tickangle=-30)
            st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("### Prix/m² par département (Top 20)")
        if 'department' in dff.columns and 'price_per_sqm' in dff.columns:
            dept_ppsqm = (dff.groupby('department')['price_per_sqm']
                          .median().sort_values(ascending=False).head(20).reset_index())
            _th = {k: v for k, v in PT.items() if k != 'yaxis'}
            fig4 = px.bar(
                dept_ppsqm, x='price_per_sqm', y='department',
                orientation='h', color='price_per_sqm',
                color_continuous_scale='Viridis',
                labels={'price_per_sqm':'Prix/m² (€)','department':'Département'},
                template='plotly_dark'
            )
            fig4.update_layout(**_th, height=420, showlegend=False,
                                coloraxis_showscale=False)
            fig4.update_yaxes(categoryorder='total ascending',
                               gridcolor='rgba(0,183,120,.12)')
            st.plotly_chart(fig4, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 🔗 Matrice de corrélation")
    corr_cols = ['price','surface_sqm','rooms','dist_paris_km','dist_coast_km',
                 'is_apartment','is_house','region_enc','dept_enc','is_idf']
    corr_cols = [c for c in corr_cols if c in dff.columns]
    labels_fr = {
        'price':'Prix','surface_sqm':'Surface','rooms':'Pièces',
        'dist_paris_km':'Dist. Paris','dist_coast_km':'Dist. côte',
        'is_apartment':'Appartement','is_house':'Maison',
        'region_enc':'Région','dept_enc':'Département','is_idf':'IDF'
    }
    corr = dff[corr_cols].corr().round(3)
    corr.rename(index=labels_fr, columns=labels_fr, inplace=True)
    fig_corr = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                          zmin=-1, zmax=1, template='plotly_dark', aspect='auto')
    fig_corr.update_layout(**PT, height=400,
                            coloraxis_colorbar=dict(tickfont=dict(color='#e2e8f0')))
    st.plotly_chart(fig_corr, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — MODÈLES
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("## 🤖 Comparaison des modèles ML")

    mdata = []
    for name, m in results.items():
        mdata.append({'Modèle':name,'R²':round(m['R2'],4),
                      'MAE (€)':round(m['MAE']),'RMSE (€)':round(m['RMSE']),
                      'MAPE (%)':round(m.get('MAPE',0),1),
                      'Temps (s)':round(m.get('training_time',0),1)})
    mdf = pd.DataFrame(mdata).sort_values('R²', ascending=False).reset_index(drop=True)
    mdf.insert(0, '🏆', ['🥇']+['']*(len(mdf)-1))

    st.dataframe(
        mdf.style
           .background_gradient(subset=['R²'], cmap='Greens')
           .background_gradient(subset=['MAE (€)','RMSE (€)'], cmap='Reds_r')
           .format({'MAE (€)':'{:,.0f}','RMSE (€)':'{:,.0f}'}),
        use_container_width=True, hide_index=True
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    cr1, cr2 = st.columns(2)
    with cr1:
        st.markdown("### R² par modèle")
        names  = mdf['Modèle'].tolist()
        r2vals = mdf['R²'].tolist()
        fig_r2 = go.Figure(go.Bar(
            x=r2vals, y=names, orientation='h',
            marker_color=[MODEL_COLORS.get(n,'#00b778') for n in names],
            text=[f"{v:.4f}" for v in r2vals], textposition='outside',
            textfont=dict(color='#e2e8f0')
        ))
        fig_r2.update_layout(**PT, height=300, xaxis_range=[0,1.05],
                              xaxis_title='R²', showlegend=False)
        st.plotly_chart(fig_r2, use_container_width=True)

    with cr2:
        st.markdown("### Radar — Performances comparées")
        cats = ['R²×100','Vitesse','Précision MAE','Précision RMSE']
        mx_mae  = max(r['MAE']  for r in results.values())
        mx_rmse = max(r['RMSE'] for r in results.values())
        mx_t    = max(r.get('training_time',1) for r in results.values()) or 1
        fig_rad = go.Figure()
        for nm, m in results.items():
            vals = [m['R2']*100,
                    max(0,(1-m.get('training_time',1)/mx_t))*100,
                    max(0,(1-m['MAE']/mx_mae))*100,
                    max(0,(1-m['RMSE']/mx_rmse))*100]
            vals += [vals[0]]
            fig_rad.add_trace(go.Scatterpolar(
                r=vals, theta=cats+[cats[0]], name=nm,
                line=dict(color=MODEL_COLORS.get(nm,'#00b778'), width=2),
                fill='toself', fillcolor=MODEL_COLORS.get(nm,'#00b778'), opacity=0.12
            ))
        fig_rad.update_layout(
            polar=dict(bgcolor='rgba(10,10,30,.6)',
                       radialaxis=dict(range=[0,100],gridcolor='rgba(0,183,120,.2)',
                                       tickfont=dict(color='#94a3b8',size=9)),
                       angularaxis=dict(gridcolor='rgba(0,183,120,.2)',
                                        tickfont=dict(color='#e2e8f0',size=11))),
            paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter',color='#e2e8f0'),
            legend=dict(font=dict(size=10),bgcolor='rgba(0,0,0,0)'),
            height=340, margin=dict(l=40,r=40,t=20,b=20)
        )
        st.plotly_chart(fig_rad, use_container_width=True)

    # Prédit vs Réel
    st.markdown("<hr>", unsafe_allow_html=True)
    sel_m = st.selectbox("Modèle à analyser", list(results.keys()),
                         index=list(results.keys()).index(best_name)
                         if best_name in results else 0, key='sel_m')

    if preds is not None and f'y_pred_{sel_m}' in preds.columns:
        cs1, cs2 = st.columns(2)
        with cs1:
            st.markdown(f"### 🎯 Prédit vs Réel — {sel_m}")
            sp = preds.sample(min(2000,len(preds)), random_state=42)
            mn = min(sp['y_true'].min(), sp[f'y_pred_{sel_m}'].min())
            mx = max(sp['y_true'].max(), sp[f'y_pred_{sel_m}'].max())
            fpv = go.Figure()
            fpv.add_trace(go.Scatter(x=sp['y_true'],y=sp[f'y_pred_{sel_m}'],
                                      mode='markers',
                                      marker=dict(color='#00b778',size=3,opacity=0.4)))
            fpv.add_trace(go.Scatter(x=[mn,mx],y=[mn,mx],mode='lines',
                                      line=dict(color='#0ea5e9',dash='dash',width=2),name='Parfait'))
            fpv.update_layout(**PT, height=360,
                               xaxis_title='Prix réel (€)', yaxis_title='Prix prédit (€)')
            st.plotly_chart(fpv, use_container_width=True)

        with cs2:
            st.markdown(f"### 📊 Résidus — {sel_m}")
            res = preds[f'y_pred_{sel_m}'].values - preds['y_true'].values
            fres = px.histogram(x=res, nbins=60, color_discrete_sequence=['#00b778'],
                                 labels={'x':'Résidu (€)'}, template='plotly_dark')
            fres.add_vline(x=0,line_dash='dash',line_color='#0ea5e9')
            fres.update_layout(**PT, height=360, showlegend=False, bargap=0.05)
            st.plotly_chart(fres, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 🔑 Importance des variables")
    if fi_df is not None and not fi_df.empty:
        top = fi_df.head(15)
        FEAT_LABELS = {
            'surface_sqm': 'Surface (m²)', 'rooms': 'Nb. pièces',
            'latitude': 'Latitude', 'longitude': 'Longitude',
            'dept_enc': 'Département', 'region_enc': 'Région',
            'dist_paris_km': 'Distance Paris', 'dist_coast_km': 'Distance côte',
            'is_idf': 'Île-de-France', 'log_surface': 'Log surface',
            'is_apartment': 'Appartement', 'is_house': 'Maison',
            'month_sin': 'Saisonnalité (sin)', 'month_cos': 'Saisonnalité (cos)',
            'year': 'Année',
        }
        top = top.copy()
        top['feature_label'] = top['feature'].map(FEAT_LABELS).fillna(top['feature'])
        _th3 = {k: v for k, v in PT.items() if k != 'yaxis'}
        ffi = go.Figure(go.Bar(
            x=top['importance_pct'], y=top['feature_label'], orientation='h',
            marker=dict(color=top['importance_pct'], colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title='%',tickfont=dict(color='#e2e8f0'))),
            text=[f"{v:.1f}%" for v in top['importance_pct']],
            textposition='outside', textfont=dict(color='#e2e8f0')
        ))
        ffi.update_layout(**_th3, height=450, xaxis_title='Importance (%)')
        ffi.update_yaxes(categoryorder='total ascending', gridcolor='rgba(0,183,120,.12)')
        st.plotly_chart(ffi, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — PRÉDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("## 🔮 Estimateur de prix immobilier (France)")
    st.markdown("""
    <div class='dvf-note'>
      🤖 L'estimation utilise un modèle entraîné sur de <b>vraies transactions DVF 2023</b>.
      Les prédictions reflètent les prix du marché immobilier français réel.
    </div>""", unsafe_allow_html=True)

    best_idx = MODELS_ORDER.index(best_name) if best_name in MODELS_ORDER else 0
    sel_model_name = st.selectbox("🤖 Modèle", MODELS_ORDER, index=best_idx)
    model_obj = load_model(sel_model_name, _mtime=_mt_model)
    scaler = None
    if sel_model_name in ('Linear Regression','Ridge Regression'):
        sp = os.path.join(ROOT,'models','scaler.pkl')
        if os.path.exists(sp): scaler = joblib.load(sp)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 📍 Localisation")

    loc1, loc2 = st.columns(2)
    dept_list = sorted(df['department'].dropna().unique().tolist()) if 'department' in df.columns else []
    sel_dept = loc1.selectbox("Département", dept_list)

    # Valeurs par défaut selon département
    dept_mask = df['department'] == sel_dept if 'department' in df.columns else pd.Series([True]*len(df))
    def_lat = float(df.loc[dept_mask,'latitude'].median()) if dept_mask.any() else 46.8
    def_lon = float(df.loc[dept_mask,'longitude'].median()) if dept_mask.any() else 2.3
    sel_ptype = loc2.selectbox("Type de bien", ['Appartement','Maison'])

    lat_bounds = (max(41.0, def_lat-1.5), min(51.5, def_lat+1.5))
    lon_bounds = (max(-5.5, def_lon-1.5), min(10.0, def_lon+1.5))
    latitude  = st.slider("Latitude",  float(lat_bounds[0]), float(lat_bounds[1]), def_lat, 0.01)
    longitude = st.slider("Longitude", float(lon_bounds[0]), float(lon_bounds[1]), def_lon, 0.01)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 🏗️ Caractéristiques du bien")
    p1, p2 = st.columns(2)
    with p1:
        surface_sqm = st.slider("Surface (m²)", 10, 500, 75, 5)
        rooms       = st.slider("Nombre de pièces", 1, 10, 3, 1)
    with p2:
        year_sale  = st.slider("Année de vente", 2023, 2025, 2024, 1)
        month_sale = st.slider("Mois de vente", 1, 12, 6, 1)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Estimer le prix", key='predict_btn')

    if predict_btn and model_obj is not None:
        # Encodage département/région
        if 'department' in df.columns and 'price' in df.columns:
            dept_med = df.groupby('department')['price'].median()
            dept_rank = dept_med.rank(method='dense').astype(int).to_dict()
            dept_enc = int(dept_rank.get(sel_dept, 0))
        else:
            dept_enc = 0

        if 'region' in df.columns and 'price' in df.columns:
            region_of_dept = df.loc[df['department']==sel_dept,'region'].mode()
            sel_region = region_of_dept.iloc[0] if len(region_of_dept) > 0 else 'Autre'
            reg_med = df.groupby('region')['price'].median()
            reg_rank = reg_med.rank(method='dense').astype(int).to_dict()
            region_enc = int(reg_rank.get(sel_region, 0))
        else:
            sel_region = 'Autre'
            region_enc = 0

        IDF_DEPTS = {'75','77','78','91','92','93','94','95'}
        is_idf = 1 if sel_dept in IDF_DEPTS else 0

        # Distance à Paris
        R = 6371.0
        dlat = np.radians(latitude - 48.8566)
        dlon = np.radians(longitude - 2.3522)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(latitude))*np.cos(np.radians(48.8566))*np.sin(dlon/2)**2
        dist_paris = R * 2 * np.arcsin(np.sqrt(a))

        # Distance à la côte (approx simple)
        COAST_FR = np.array([
            (51.0,2.5),(50.7,1.6),(49.5,-1.5),(48.7,-2.0),
            (47.5,-2.5),(47.1,-2.2),(46.6,-1.4),(46.2,-1.1),
            (45.7,-1.1),(44.7,-1.2),(43.7,-1.5),(43.5,-1.8),
            (43.3,-1.5),(43.3,3.5),(43.1,5.9),(43.4,6.8),(43.8,7.4),
        ])
        dists_coast = []
        for clat, clon in COAST_FR:
            d_ = np.radians(latitude - clat)
            d2_ = np.radians(longitude - clon)
            a_ = np.sin(d_/2)**2 + np.cos(np.radians(latitude))*np.cos(np.radians(clat))*np.sin(d2_/2)**2
            dists_coast.append(R * 2 * np.arcsin(np.sqrt(a_)))
        dist_coast = min(dists_coast)

        month_sin = float(np.sin(2*np.pi*month_sale/12))
        month_cos = float(np.cos(2*np.pi*month_sale/12))
        log_surface = float(np.log1p(surface_sqm))
        is_apt = 1 if sel_ptype == 'Appartement' else 0
        is_house = 1 - is_apt

        feat = {
            'surface_sqm': surface_sqm,
            'rooms': rooms,
            'latitude': latitude,
            'longitude': longitude,
            'is_apartment': is_apt,
            'is_house': is_house,
            'dept_enc': dept_enc,
            'region_enc': region_enc,
            'dist_paris_km': round(dist_paris, 1),
            'dist_coast_km': round(dist_coast, 1),
            'is_idf': is_idf,
            'log_surface': round(log_surface, 4),
            'month_sin': round(month_sin, 4),
            'month_cos': round(month_cos, 4),
            'year': year_sale,
        }

        X_in = pd.DataFrame([feat])[features]
        X_arr = scaler.transform(X_in) if scaler else X_in.values
        price_pred = float(model_obj.predict(X_arr)[0])
        ppsm       = price_pred / surface_sqm
        mae_model  = results[sel_model_name]['MAE']

        r1, r2, r3 = st.columns([1,2,1])
        with r2:
            st.markdown(f"""
            <div class='result-box'>
              <div style='color:#94a3b8;font-size:.9rem;margin-bottom:8px'>
                💡 Estimation — <b>{sel_model_name}</b> · Dépt {sel_dept}
                ({sel_region}) · {sel_ptype}
              </div>
              <div class='result-price'>{fmt_eur(price_pred)}</div>
              <div style='color:#94a3b8;font-size:.95rem;margin-top:6px'>Prix estimé du bien</div>
              <hr style='border-color:rgba(0,183,120,.3);margin:18px 0'>
              <div style='display:flex;justify-content:space-around'>
                <div><div style='color:#34d399;font-size:1.3rem;font-weight:700'>{fmt_eur(ppsm)}</div>
                     <div style='color:#64748b;font-size:.82rem'>Prix au m²</div></div>
                <div><div style='color:#60a5fa;font-size:1.3rem;font-weight:700'>{surface_sqm} m²</div>
                     <div style='color:#64748b;font-size:.82rem'>Surface</div></div>
                <div><div style='color:#fbbf24;font-size:1.3rem;font-weight:700'>{rooms} pièces</div>
                     <div style='color:#64748b;font-size:.82rem'>Pièces</div></div>
                <div><div style='color:#f87171;font-size:1rem;font-weight:700'>≈{dist_paris:.0f} km</div>
                     <div style='color:#64748b;font-size:.82rem'>de Paris</div></div>
              </div>
              <div style='margin-top:14px;color:#64748b;font-size:.82rem'>
                Fourchette (±MAE) :
                <b style='color:#34d399'>{fmt_eur(max(0,price_pred-mae_model))}</b> —
                <b style='color:#34d399'>{fmt_eur(price_pred+mae_model)}</b>
              </div>
            </div>""", unsafe_allow_html=True)

        # Comparaison avec médiane du département
        if dept_mask.any():
            dept_median = float(df.loc[dept_mask,'price'].median())
            diff_pct = (price_pred - dept_median) / dept_median * 100
            g1, g2, g3 = st.columns([1,2,1])
            with g2:
                st.markdown("<br>", unsafe_allow_html=True)
                fig_g = go.Figure(go.Indicator(
                    mode='gauge+number+delta',
                    value=price_pred/1000,
                    delta={'reference':dept_median/1000,'suffix':'k€',
                           'relative':True, 'valueformat':'.1%'},
                    title={'text':f"Prix estimé vs médiane dépt {sel_dept}",
                           'font':{'color':'#34d399','size':13}},
                    number={'suffix':'k€','font':{'color':'#e2e8f0','size':26}},
                    gauge={
                        'axis':{'range':[0, df['price'].quantile(0.97)/1000],
                                'tickcolor':'#94a3b8'},
                        'bar':{'color':'#00b778'},
                        'steps':[
                            {'range':[0,max(0,(price_pred-mae_model)/1000)],'color':'rgba(0,183,120,.05)'},
                            {'range':[max(0,(price_pred-mae_model)/1000),(price_pred+mae_model)/1000],'color':'rgba(0,183,120,.18)'},
                        ],
                        'threshold':{'line':{'color':'#0ea5e9','width':2},
                                     'thickness':.8,'value':dept_median/1000},
                        'bgcolor':'rgba(10,10,30,.6)','bordercolor':'rgba(0,183,120,.3)'
                    }
                ))
                fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                     font=dict(family='Inter',color='#e2e8f0'),
                                     height=280, margin=dict(l=20,r=20,t=40,b=10))
                st.plotly_chart(fig_g, use_container_width=True)

    elif predict_btn:
        st.error("Modèle introuvable. Relancez le pipeline.")
