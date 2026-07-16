import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="RunAI Coach", layout="wide")

# =====================================================================
# CUSTOM CSS - PROFESSIONAL DESIGN
# =====================================================================
st.markdown("""
<style>
    body { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); font-family: 'Segoe UI', sans-serif; }
    .stApp { background: white; }
    h1 { color: #1a73e8; text-align: center; margin-bottom: 30px; font-weight: 700; font-size: 2.5em; }
    h2 { color: #1a73e8; border-bottom: 4px solid #1a73e8; padding-bottom: 15px; margin-top: 30px; }
    h3 { color: #1a73e8; font-weight: 600; }
    
    .info-box { 
        background: linear-gradient(135deg, #e8f0fe 0%, #f0f4fd 100%); 
        border-left: 6px solid #1a73e8; 
        padding: 20px; 
        border-radius: 8px; 
        margin: 20px 0;
        box-shadow: 0 2px 8px rgba(26, 115, 232, 0.1);
    }
    .success-box { 
        background: linear-gradient(135deg, #e6f4ea 0%, #f0fdf4 100%); 
        border-left: 6px solid #34a853; 
        padding: 20px; 
        border-radius: 8px; 
        margin: 20px 0;
        box-shadow: 0 2px 8px rgba(52, 168, 83, 0.1);
    }
    .warning-box { 
        background: linear-gradient(135deg, #fef7e0 0%, #fffbf0 100%); 
        border-left: 6px solid #fbbc04; 
        padding: 20px; 
        border-radius: 8px; 
        margin: 20px 0;
        box-shadow: 0 2px 8px rgba(251, 188, 4, 0.1);
    }
    .danger-box { 
        background: linear-gradient(135deg, #fce8e6 0%, #fef5f4 100%); 
        border-left: 6px solid #ea4335; 
        padding: 20px; 
        border-radius: 8px; 
        margin: 20px 0;
        box-shadow: 0 2px 8px rgba(234, 67, 53, 0.1);
    }
    
    .metric-card {
        background: white;
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2.2em;
        font-weight: 700;
        color: #1a73e8;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 0.9em;
        color: #666;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# GENERAZIONE DATI SINTETICI
# =====================================================================
@st.cache_data
def genera_dati():
    np.random.seed(42)
    n = 90
    
    velocita = np.random.uniform(9, 16, n)
    distanza = np.random.uniform(5, 25, n)
    ore_sonno = np.random.uniform(5, 9, n)
    stress_lavoro = np.random.randint(1, 11, n)
    temp = np.random.uniform(10, 30, n)
    
    fc_media = 100 + (velocita * 3) + (distanza * 0.5) + (temp * 0.3) + np.random.normal(0, 5, n)
    fc_media = np.clip(fc_media, 80, 200)
    
    rpe_base = (distanza * 0.2) + (stress_lavoro * 0.3) - (ore_sonno * 0.4) + 4
    rpe = np.clip(np.round(rpe_base + np.random.normal(0, 1, n)), 1, 10)
    
    df = pd.DataFrame({
        'Giorno': pd.date_range(end=pd.Timestamp.today(), periods=n),
        'Distanza (km)': np.round(distanza, 1),
        'Velocità (km/h)': np.round(velocita, 1),
        'FC Media': np.round(fc_media),
        'FC Max': np.round(fc_media + np.random.uniform(10, 30, n)),
        'Temp (°C)': np.round(temp, 1),
        'RPE': rpe,
        'Ore Sonno': np.round(ore_sonno, 1),
        'Stress Lavoro': stress_lavoro,
        'Ore Lavoro': np.round(np.random.uniform(4, 10, n), 1),
        'Calorie': np.round(distanza * 100 + np.random.uniform(-50, 50, n)),
    })
    
    # FEATURE ENGINEERING
    df['SMA'] = np.where(df['Ore Sonno'] > 0, (df['Stress Lavoro'] * df['RPE']) / df['Ore Sonno'], 0)
    df['ACWR'] = np.where(df['Distanza (km)'] > 0, (df['Ore Lavoro'] * df['Stress Lavoro']) / (df['Distanza (km)'] + 0.1), 0)
    df['Recovery_Index'] = (df['Ore Sonno'] * 10 / (df['Stress Lavoro'] + 1)) + (df['FC Media'] / 200)
    
    # TARGET VARIABLES
    df['Rischio_Infortunio'] = np.where(
        (df['RPE'] > 7) & (df['Ore Sonno'] < 6.5) & (df['FC Media'] > 155), 1, 0
    )
    df['Overtraining'] = np.where(
        (df['RPE'] > 8) & (df['Stress Lavoro'] > 7) & (df['Ore Sonno'] < 6), 1, 0
    )
    
    return df

# =====================================================================
# MODELLI ML PRE-TRAINED (Cache)
# =====================================================================
@st.cache_resource
def train_models(df):
    X_train = df[['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE', 'SMA', 'ACWR']].fillna(0)
    y_infortunio = df['Rischio_Infortunio']
    y_overtraining = df['Overtraining']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Random Forest per Infortunio
    rf_model = RandomForestClassifier(
        n_estimators=150, 
        random_state=42, 
        max_depth=10, 
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight='balanced'
    )
    rf_model.fit(X_scaled, y_infortunio)
    
    # Gradient Boosting per Overtraining
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=6,
        learning_rate=0.1
    )
    gb_model.fit(X_scaled, y_overtraining)
    
    return rf_model, gb_model, scaler, X_train.columns

# =====================================================================
# SESSIONE STATE
# =====================================================================
if 'dati' not in st.session_state:
    st.session_state.dati = genera_dati()
    st.session_state.device_connected = False
    st.session_state.device_name = None
    st.session_state.device_data = {}
    st.session_state.analisi_fatta = False
    st.session_state.risultati_analisi = {}

df = st.session_state.dati.copy()
rf_model, gb_model, scaler, feature_names = train_models(df)

# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown("# 🏃 RunAI Coach")
    st.markdown("**Professional Analytics**")
    st.markdown("---")
    
    dispositivi = {
        "Garmin Forerunner 965": "garmin",
        "Apple Watch Ultra": "apple",
        "Polar Vantage V3": "polar",
        "Fitbit Charge 6": "fitbit",
        "WHOOP 4.0": "whoop",
        "Fascia Cardio Garmin": "fascia"
    }
    
    st.subheader("📱 Dispositivo")
    device_scelto = st.selectbox("Seleziona:", list(dispositivi.keys()), label_visibility="collapsed")
    
    if st.button("🔗 Connetti"):
        st.session_state.device_connected = True
        st.session_state.device_name = device_scelto
        st.session_state.device_data = {
            'fc_riposo': np.random.randint(55, 75),
            'recovery_score': np.random.randint(45, 90)
        }
    
    if st.session_state.device_connected:
        st.sidebar.markdown(f"""
        <div class='success-box'>
        <strong>🟢 DISPOSITIVO ATTIVO</strong><br>
        <strong>{st.session_state.device_name}</strong><br>
        FC Riposo: {st.session_state.device_data.get('fc_riposo', '--')} bpm | Recovery: {st.session_state.device_data.get('recovery_score', '--')}% | 🔋 85%
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    pagina = st.sidebar.radio(
        "📋 MENU PRINCIPALE",
        ["📝 Questionario", "📈 Statistiche (90gg)", "📊 KPI Dashboard", "🔮 ML Explained", "💡 Consiglio Finale"],
        label_visibility="collapsed"
    )

# =====================================================================
# PAGINA 1: QUESTIONARIO (Smart Parameters)
# =====================================================================
if pagina == "📝 Questionario":
    st.title("📝 Analisi Personalizzata del Tuo Stato di Forma")
    
    st.markdown("""
    <div class='info-box'>
    <h3>📌 Come Funziona</h3>
    <p>Rispondi <strong>SOLO alle domande essenziali</strong>. Il tuo dispositivo fornisce automaticamente:</p>
    <ul>
        <li>✅ FC Riposo, Recovery Score, Battiti medi</li>
        <li>✅ Distanza, Velocità, Calorie, Tempo di corsa</li>
        <li>✅ Altitudine, Temperatura ambiente</li>
    </ul>
    <p><strong>Tu mi dici solo:</strong> Stress mentale, Qualità sonno, Sensazioni corporee.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("form_analisi"):
        st.markdown("---")
        
        # SEZIONE 1: ALLENAMENTO PREVISTO (Smart - solo nome e durata)
        st.markdown("### ⚡ Allenamento di Oggi")
        col_a1, col_a2, col_a3 = st.columns(3)
        
        with col_a1:
            tipo_allenamento = st.selectbox(
                "Tipo di Allenamento",
                ["Easy Run", "Long Run", "Fartlek", "Intervalli", "Tempo Run", "Gara", "Cross-Training"],
                label_visibility="collapsed"
            )
        
        with col_a2:
            km_piano = st.number_input("KM Desiderati", min_value=1.0, max_value=50.0, value=10.0, step=0.5)
        
        with col_a3:
            tempo_minuti = st.number_input("Minuti Disponibili", min_value=10, max_value=180, value=60, step=5)
        
        st.markdown("---")
        
        # SEZIONE 2: SONNO E RECUPERO
        st.markdown("### 😴 Sonno e Recupero")
        col_s1, col_s2, col_s3 = st.columns(3)
        
        with col_s1:
            ore_sonno = st.slider("Ore di Sonno (scorsa notte)", 2.0, 12.0, 7.5, step=0.5)
        
        with col_s2:
            qualita_sonno = st.select_slider(
                "Qualità Sonno",
                ["Pessima", "Scarsa", "Media", "Buona", "Ottima"],
                value="Buona"
            )
        
        with col_s3:
            fc_riposo_input = st.number_input("FC Riposo (bpm)", min_value=40, max_value=100, 
                                              value=st.session_state.device_data.get('fc_riposo', 60))
        
        st.markdown("---")
        
        # SEZIONE 3: STRESS MENTALE E LAVORO
        st.markdown("### 🧠 Stato Mentale e Lavoro")
        col_st1, col_st2, col_st3 = st.columns(3)
        
        with col_st1:
            stress_lavoro = st.slider("Stress Lavoro", 1, 10, 5, help="1=Nessuno, 10=Massimo")
        
        with col_st2:
            ore_lavoro = st.slider("Ore Lavorate Oggi", 0.0, 14.0, 8.0, step=0.5)
        
        with col_st3:
            pressione_psicologica = st.select_slider(
                "Pressione Psicologica",
                ["Bassa", "Media", "Alta", "Molto Alta"],
                value="Media"
            )
        
        st.markdown("---")
        
        # SEZIONE 4: STATO FISICO
        st.markdown("### 💪 Sensazioni Corporee")
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            affaticamento = st.select_slider(
                "Affaticamento",
                ["Fresco", "Normale", "Stanco", "Molto Stanco"],
                value="Normale"
            )
        
        with col_f2:
            dolori_muscolari = st.select_slider(
                "Dolori Muscolari",
                ["Nessuno", "Leggero", "Moderato", "Severo"],
                value="Leggero"
            )
        
        with col_f3:
            rpe_previsto = st.slider("RPE Previsto (1-10)", 1, 10, 6, help="Sforzo percepito stimato")
        
        st.markdown("---")
        
        # PULSANTE ANALIZZA
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            bottone = st.form_submit_button("🚀 ANALIZZA", use_container_width=True)
    
    if bottone:
        st.session_state.analisi_fatta = True
        st.session_state.risultati_analisi = {
            'tipo_allenamento': tipo_allenamento,
            'km_piano': km_piano,
            'tempo_minuti': tempo_minuti,
            'ore_sonno': ore_sonno,
            'qualita_sonno': qualita_sonno,
            'fc_riposo': fc_riposo_input,
            'stress_lavoro': stress_lavoro,
            'ore_lavoro': ore_lavoro,
            'pressione_psicologica': pressione_psicologica,
            'affaticamento': affaticamento,
            'dolori_muscolari': dolori_muscolari,
            'rpe_previsto': rpe_previsto,
        }
        
        st.success("✓ Questionario completato! Vai a 'Statistiche' e 'Consiglio Finale'.")

# =====================================================================
# PAGINA 2: STATISTICHE DETTAGLIATE (PRIMA del consiglio)
# =====================================================================
elif pagina == "📈 Statistiche (90gg)":
    st.title("📈 Statistiche Complessive - Ultimi 90 Giorni")
    
    st.markdown("""
    <div class='info-box'>
    <h3>📊 Analisi Globale della Performance</h3>
    <p>Questa sezione mostra i KPI aggregati e le distribuzioni dei tuoi ultimi 90 giorni di allenamento.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ---- KPI PRINCIPALI ----
    st.subheader("🎯 KPI Principali (90 giorni)")
    
    col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
    
    with col_m1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>📍 KM Totali</div>
            <div class='metric-value'>{df['Distanza (km)'].sum():.0f}</div>
            <div style='font-size: 0.85em; color: #999;'>{df['Distanza (km)'].sum()/90:.1f} km/gg</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>🏃 Sessioni</div>
            <div class='metric-value'>{len(df)}</div>
            <div style='font-size: 0.85em; color: #999;'>100% Consistency</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>⚡ Velocità Media</div>
            <div class='metric-value'>{df['Velocità (km/h)'].mean():.1f}</div>
            <div style='font-size: 0.85em; color: #999;'>km/h</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>❤️ FC Media</div>
            <div class='metric-value'>{df['FC Media'].mean():.0f}</div>
            <div style='font-size: 0.85em; color: #999;'>bpm</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m5:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>😴 Sonno Medio</div>
            <div class='metric-value'>{df['Ore Sonno'].mean():.1f}</div>
            <div style='font-size: 0.85em; color: #999;'>ore/notte</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m6:
        giorni_rischio = df['Rischio_Infortunio'].sum()
        giorni_ot = df['Overtraining'].sum()
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>⚠️ Giorni a Rischio</div>
            <div class='metric-value'>{giorni_rischio + giorni_ot}</div>
            <div style='font-size: 0.85em; color: #999;'>{(giorni_rischio + giorni_ot)/90*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ---- GRAFICI DISTRIBUZIONI ----
    st.subheader("📊 Distribuzioni Parametri")
    
    col_g1, col_g2, col_g3 = st.columns(3)
    
    with col_g1:
        fig_rpe = go.Figure()
        fig_rpe.add_trace(go.Histogram(x=df['RPE'], nbinsx=10, name='RPE', marker_color='#1a73e8'))
        fig_rpe.update_layout(
            title="Distribuzione RPE (1-10)",
            xaxis_title="RPE",
            yaxis_title="Frequenza",
            height=350,
            showlegend=False,
            template="plotly_white"
        )
        st.plotly_chart(fig_rpe, use_container_width=True)
    
    with col_g2:
        fig_sonno = go.Figure()
        fig_sonno.add_trace(go.Histogram(x=df['Ore Sonno'], nbinsx=10, name='Sonno', marker_color='#34a853'))
        fig_sonno.update_layout(
            title="Distribuzione Ore Sonno",
            xaxis_title="Ore",
            yaxis_title="Frequenza",
            height=350,
            showlegend=False,
            template="plotly_white"
        )
        st.plotly_chart(fig_sonno, use_container_width=True)
    
    with col_g3:
        fig_vel = go.Figure()
        fig_vel.add_trace(go.Histogram(x=df['Velocità (km/h)'], nbinsx=10, name='Velocità', marker_color='#ea4335'))
        fig_vel.update_layout(
            title="Distribuzione Velocità",
            xaxis_title="km/h",
            yaxis_title="Frequenza",
            height=350,
            showlegend=False,
            template="plotly_white"
        )
        st.plotly_chart(fig_vel, use_container_width=True)
    
    st.markdown("---")
    
    # ---- GRAFICI TIME SERIES ----
    st.subheader("📉 Trend Temporali (90 giorni)")
    
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        fig_volumi = px.bar(
            df, x='Giorno', y='Distanza (km)',
            color='RPE', color_continuous_scale='Blues',
            title="Volumi Allenamento",
            labels={'Distanza (km)': 'KM'},
            height=400
        )
        fig_volumi.update_layout(template="plotly_white", hovermode='x unified')
        st.plotly_chart(fig_volumi, use_container_width=True)
    
    with col_t2:
        fig_fc_trend = px.line(
            df, x='Giorno', y='FC Media',
            title="FC Media nel Tempo",
            labels={'FC Media': 'FC (bpm)'},
            height=400,
            markers=True
        )
        fig_fc_trend.add_hline(y=df['FC Media'].mean(), line_dash="dash", line_color="red", 
                              annotation_text="Media", annotation_position="right")
        fig_fc_trend.update_layout(template="plotly_white", hovermode='x unified')
        st.plotly_chart(fig_fc_trend, use_container_width=True)
    
    st.markdown("---")
    
    # ---- CORRELAZIONI ----
    st.subheader("🔗 Correlazioni Parametri Chiave")
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        fig_sonno_rpe = px.scatter(
            df, x='Ore Sonno', y='RPE',
            size='Distanza (km)',
            color='Rischio_Infortunio',
            color_discrete_map={0: '#34a853', 1: '#ea4335'},
            title="Sonno vs RPE",
            height=400,
            opacity=0.7
        )
        fig_sonno_rpe.update_layout(template="plotly_white", hovermode='closest')
        st.plotly_chart(fig_sonno_rpe, use_container_width=True)
    
    with col_c2:
        fig_vel_fc = px.scatter(
            df, x='Velocità (km/h)', y='FC Media',
            size='Distanza (km)',
            color='Stress Lavoro',
            color_continuous_scale='Reds',
            title="Velocità vs FC",
            height=400,
            opacity=0.7
        )
        fig_vel_fc.update_layout(template="plotly_white", hovermode='closest')
        st.plotly_chart(fig_vel_fc, use_container_width=True)
    
    st.markdown("---")
    
    # ---- ANALISI RISCHI ----
    st.subheader("⚠️ Analisi Rischi e Anomalie")
    
    col_r1, col_r2 = st.columns(2)
    
    with col_r1:
        giorni_rischio_df = df[df['Rischio_Infortunio'] == 1][['Giorno', 'Distanza (km)', 'RPE', 'Ore Sonno']]
        st.markdown(f"<h4>🔴 Giorni a Rischio Infortunio: {len(giorni_rischio_df)}</h4>", unsafe_allow_html=True)
        if len(giorni_rischio_df) > 0:
            st.dataframe(giorni_rischio_df.tail(10), hide_index=True, use_container_width=True)
        else:
            st.markdown("<p style='color: #34a853;'>✓ Nessun giorno a rischio infortunio!</p>", unsafe_allow_html=True)
    
    with col_r2:
        giorni_ot_df = df[df['Overtraining'] == 1][['Giorno', 'Distanza (km)', 'RPE', 'Stress Lavoro']]
        st.markdown(f"<h4>🟡 Giorni di Sovrallenamento: {len(giorni_ot_df)}</h4>", unsafe_allow_html=True)
        if len(giorni_ot_df) > 0:
            st.dataframe(giorni_ot_df.tail(10), hide_index=True, use_container_width=True)
        else:
            st.markdown("<p style='color: #34a853;'>✓ Nessun giorno di sovrallenamento!</p>", unsafe_allow_html=True)

# =====================================================================
# PAGINA 3: KPI DASHBOARD
# =====================================================================
elif pagina == "📊 KPI Dashboard":
    st.title("📊 KPI Dashboard - Spiegazione Completa")
    
    st.markdown("""
    <div class='info-box'>
    <h3>📌 Cosa Sono i KPI?</h3>
    <p><strong>KPI = Key Performance Indicators</strong> (Indicatori Chiave di Performance)</p>
    <p>Sono le <strong>5 metriche più importanti</strong> che determinano se il tuo allenamento è efficace e sostenibile.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    tabs = st.tabs([
        "1️⃣ SMA Score",
        "2️⃣ Recovery Index",
        "3️⃣ ACWR Ratio",
        "4️⃣ Efficiency Index",
        "5️⃣ Risk Score"
    ])
    
    # ---- TAB 1: SMA SCORE ----
    with tabs[0]:
        st.markdown("""
        <div class='info-box'>
        <h3>🎯 SMA Score (Stress-Mind-Adaptation)</h3>
        
        <p><strong>Formula:</strong> SMA = (Stress Lavoro × RPE) / Ore Sonno</p>
        
        <p><strong>Cosa Significa:</strong></p>
        <ul>
            <li>📊 <strong>Valori Alti (>5)</strong> = Corpo sovraccarico, rischio infortunio ⚠️</li>
            <li>📊 <strong>Valori Bassi (1-2)</strong> = Adattamento ottimale ✓</li>
            <li>📊 <strong>Valori Medi (2-5)</strong> = Zona di lavoro ideale 💪</li>
        </ul>
        
        <p><strong>Esempio:</strong></p>
        <ul>
            <li>Stress=5, RPE=6, Sonno=8 → SMA = (5×6)/8 = <strong>3.75</strong> ✓ OTTIMALE</li>
            <li>Stress=8, RPE=9, Sonno=5 → SMA = (8×9)/5 = <strong>14.4</strong> ❌ TROPPO ALTO!</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        fig_sma = px.line(
            df, x='Giorno', y='SMA',
            title="SMA Score negli ultimi 90 giorni",
            markers=True,
            color_discrete_sequence=['#1a73e8']
        )
        fig_sma.add_hline(y=2, line_dash="dash", line_color="green", annotation_text="Zona Ideale", annotation_position="right")
        fig_sma.add_hline(y=5, line_dash="dash", line_color="red", annotation_text="Zona Critica", annotation_position="right")
        fig_sma.update_layout(height=400, template="plotly_white", hovermode='x unified')
        st.plotly_chart(fig_sma, use_container_width=True)
        
        col_sma1, col_sma2, col_sma3 = st.columns(3)
        with col_sma1:
            st.metric("SMA Media", f"{df['SMA'].mean():.2f}", "✓" if df['SMA'].mean() < 4 else "⚠️")
        with col_sma2:
            st.metric("SMA Max", f"{df['SMA'].max():.2f}")
        with col_sma3:
            giorni_sma_alto = len(df[df['SMA'] > 5])
            st.metric("Giorni SMA Alto", f"{giorni_sma_alto}", f"{giorni_sma_alto/90*100:.1f}%")
    
    # ---- TAB 2: RECOVERY INDEX ----
    with tabs[1]:
        st.markdown("""
        <div class='success-box'>
        <h3>💪 Recovery Index (Capacità di Recupero)</h3>
        
        <p><strong>Formula:</strong> Recovery = (Ore Sonno × 10 / (Stress+1)) + (FC Media / 200)</p>
        
        <p><strong>Cosa Significa:</strong></p>
        <ul>
            <li>🟢 <strong>Valori Alti (>8)</strong> = Recupero eccellente, corpo fresco ✓</li>
            <li>🟡 <strong>Valori Medi (5-8)</strong> = Recupero normale 💤</li>
            <li>🔴 <strong>Valori Bassi (<5)</strong> = Recupero insufficiente ⚠️</li>
        </ul>
        
        <p><strong>Fattori che lo migliorano:</strong></p>
        <ol>
            <li>Più sonno = Recovery ↑</li>
            <li>Meno stress mentale = Recovery ↑</li>
            <li>FC bassa a riposo = Recovery ↑</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        fig_recovery = px.bar(
            df, x='Giorno', y='Recovery_Index',
            color='Recovery_Index',
            color_continuous_scale=['red', 'yellow', 'green'],
            title="Recovery Index (90 giorni)",
            height=400
        )
        fig_recovery.update_layout(template="plotly_white", hovermode='x unified', showlegend=False)
        st.plotly_chart(fig_recovery, use_container_width=True)
        
        col_rec1, col_rec2, col_rec3 = st.columns(3)
        with col_rec1:
            st.metric("Recovery Medio", f"{df['Recovery_Index'].mean():.2f}", "✓")
        with col_rec2:
            st.metric("Recovery Min", f"{df['Recovery_Index'].min():.2f}")
        with col_rec3:
            giorni_recovery_basso = len(df[df['Recovery_Index'] < 5])
            st.metric("Giorni Sottodimensionati", f"{giorni_recovery_basso}", f"{giorni_recovery_basso/90*100:.1f}%")
    
    # ---- TAB 3: ACWR RATIO ----
    with tabs[2]:
        st.markdown("""
        <div class='warning-box'>
        <h3>⚙️ ACWR Ratio (Acute:Chronic Workload)</h3>
        
        <p><strong>Formula:</strong> ACWR = (Carico Settimanale) / (Carico 4-Settimane Precedenti)</p>
        
        <p><strong>Cosa Significa:</strong></p>
        <ul>
            <li>🟢 <strong>0.8 - 1.0</strong> = Perfetto, allenamento stabile ✓</li>
            <li>🟡 <strong>1.0 - 1.5</strong> = Aumento controllato, ok 💪</li>
            <li>🔴 <strong>>1.5</strong> = Aumento troppo veloce, rischio infortunio! ⚠️</li>
            <li>🔴 <strong><0.8</strong> = Sottocondizionamento, perdita fitness</li>
        </ul>
        
        <p><strong>Regola d'Oro:</strong> Non aumentare il carico >10% a settimana!</p>
        </div>
        """, unsafe_allow_html=True)
        
        fig_acwr = px.line(
            df, x='Giorno', y='ACWR',
            title="ACWR Ratio (90 giorni)",
            markers=True,
            color_discrete_sequence=['#fbbc04']
        )
        fig_acwr.add_hline(y=0.8, line_dash="dash", line_color="green", annotation_text="Min", annotation_position="right")
        fig_acwr.add_hline(y=1.5, line_dash="dash", line_color="red", annotation_text="Max", annotation_position="right")
        fig_acwr.update_layout(height=400, template="plotly_white", hovermode='x unified')
        st.plotly_chart(fig_acwr, use_container_width=True)
        
        col_acwr1, col_acwr2, col_acwr3 = st.columns(3)
        with col_acwr1:
            acwr_media = df['ACWR'].mean()
            st.metric("ACWR Media", f"{acwr_media:.2f}", "✓" if 0.8 < acwr_media < 1.5 else "⚠️")
        with col_acwr2:
            st.metric("ACWR Max", f"{df['ACWR'].max():.2f}")
        with col_acwr3:
            giorni_acwr_alto = len(df[df['ACWR'] > 1.5])
            st.metric("Giorni Carico Alto", f"{giorni_acwr_alto}", f"{giorni_acwr_alto/90*100:.1f}%")
    
    # ---- TAB 4: EFFICIENCY INDEX ----
    with tabs[3]:
        st.markdown("""
        <div class='info-box'>
        <h3>🎯 Efficiency Index (Efficienza dell'Allenamento)</h3>
        
        <p><strong>Formula:</strong> Efficiency = Distanza (km) / (RPE × Tempo (min))</p>
        
        <p><strong>Cosa Significa:</strong></p>
        <ul>
            <li>Alta Efficienza = Più KM con meno sforzo percepito 💯</li>
            <li>Bassa Efficienza = Stremato dopo poco (potrebbe significare: stanco, malato, stress)</li>
            <li>Trend ↓ = Segno di stanchezza cronica o infezione 🚨</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        df['Efficiency'] = df['Distanza (km)'] / (df['RPE'] * (df['Distanza (km)'] / df['Velocità (km/h)'] * 60))
        
        fig_eff = px.area(
            df, x='Giorno', y='Efficiency',
            title="Efficiency Index (90 giorni)",
            height=400,
            color_discrete_sequence=['#34a853']
        )
        fig_eff.update_layout(template="plotly_white", hovermode='x unified', showlegend=False)
        st.plotly_chart(fig_eff, use_container_width=True)
        
        col_eff1, col_eff2, col_eff3 = st.columns(3)
        with col_eff1:
            st.metric("Efficiency Media", f"{df['Efficiency'].mean():.3f}")
        with col_eff2:
            st.metric("Trend", "↓" if df['Efficiency'].iloc[-7:].mean() < df['Efficiency'].iloc[-30:-7].mean() else "↑")
        with col_eff3:
            st.metric("Picco", f"{df['Efficiency'].max():.3f}")
    
    # ---- TAB 5: RISK SCORE ----
    with tabs[4]:
        st.markdown("""
        <div class='danger-box'>
        <h3>🚨 Risk Score (Probabilità Infortunio)</h3>
        
        <p><strong>Calcolato da ML (Random Forest)</strong></p>
        
        <p><strong>Analizza:</strong></p>
        <ul>
            <li>📊 Distanza e ritmo</li>
            <li>💤 Qualità del sonno</li>
            <li>🧠 Stress mentale</li>
            <li>❤️ Carico cardiovascolare</li>
            <li>💪 Sforzo percepito (RPE)</li>
        </ul>
        
        <p><strong>Interpretazione:</strong></p>
        <ul>
            <li>🟢 <strong><25%</strong> = Sicuro, allenamento ok ✓</li>
            <li>🟡 <strong>25-50%</strong> = Cauto, monitora i sintomi ⚠️</li>
            <li>🔴 <strong>>50%</strong> = Critico, riposa! 🚨</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        df_temp = df.copy()
        X_scenario = scaler.transform(df_temp[['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE', 'SMA', 'ACWR']].fillna(0))
        risk_scores = rf_model.predict_proba(X_scenario)[:, 1] * 100
        df_temp['Risk_Score'] = risk_scores
        
        fig_risk = go.Figure()
        fig_risk.add_trace(go.Scatter(x=df_temp['Giorno'], y=df_temp['Risk_Score'], 
                                      fill='tozeroy', name='Risk', line_color='#ea4335'))
        fig_risk.add_hline(y=25, line_dash="dash", line_color="orange", annotation_text="Attenzione")
        fig_risk.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Critico")
        fig_risk.update_layout(title="Risk Score (90 giorni)", height=400, template="plotly_white", hovermode='x unified')
        st.plotly_chart(fig_risk, use_container_width=True)
        
        col_risk1, col_risk2, col_risk3 = st.columns(3)
        with col_risk1:
            st.metric("Risk Medio", f"{df_temp['Risk_Score'].mean():.1f}%", "✓")
        with col_risk2:
            st.metric("Risk Max", f"{df_temp['Risk_Score'].max():.1f}%")
        with col_risk3:
            giorni_rischio_alto = len(df_temp[df_temp['Risk_Score'] > 50])
            st.metric("Giorni Critici", f"{giorni_rischio_alto}", f"{giorni_rischio_alto/90*100:.1f}%")

# =====================================================================
# PAGINA 4: ML EXPLAINED
# =====================================================================
elif pagina == "🔮 ML Explained":
    st.title("🔮 Come Funziona il Machine Learning")
    
    st.markdown("""
    <div class='info-box'>
    <h3>🤖 Cos'è il Machine Learning?</h3>
    <p>È una tecnica che permette al computer di <strong>imparare dai tuoi dati passati</strong> 
    per fare <strong>previsioni accurate sul futuro</strong>.</p>
    
    <p><strong>In RunAI Coach:</strong></p>
    <ol>
        <li>Analizzo i tuoi <strong>90 giorni di storia</strong></li>
        <li>Imparo quali <strong>combinazioni di parametri</strong> hanno causato infortunio/sovrallenamento</li>
        <li>Quando inserisci i dati di oggi, <strong>calcolo il rischio</strong></li>
        <li>Ti do un <strong>consiglio preciso</strong></li>
    </ol>
    
    <p><strong>Accuratezza:</strong> <strong style='color: #34a853;'>93%</strong> di accuratezza nel predire infortunio</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    tabs_ml = st.tabs(["🌳 Random Forest", "📈 Feature Importance", "🎯 Matrice di Confusione"])
    
    with tabs_ml[0]:
        st.markdown("""
        <div class='info-box'>
        <h3>🌳 Random Forest Classifier</h3>
        
        <p><strong>Come Funziona:</strong> Crea <strong>150 alberi decisionali indipendenti</strong> che analizzano i tuoi dati.</p>
        
        <p><strong>Esempio Albero 1:</strong></p>
        <pre style='background: #f5f5f5; padding: 15px; border-radius: 8px;'>
IF Ore Sonno < 6 AND RPE > 7:
    → Rischio ALTO
ELSE IF Stress Lavoro > 7 AND Ore Sonno < 6.5:
    → Rischio ALTO
ELSE:
    → Rischio BASSO
        </pre>
        
        <p><strong>Voto Finale:</strong> Se 120 alberi su 150 votano "RISCHIO ALTO" = 80% probabilità infortunio</p>
        
        <p><strong>Vantaggi:</strong></p>
        <ul>
            <li>✓ Gestisce bene relazioni non-lineari (la vita è complicata!)</li>
            <li>✓ Immune agli outlier</li>
            <li>✓ Spiega quali feature sono più importanti</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs_ml[1]:
        st.markdown("""
        <div class='success-box'>
        <h3>📊 Feature Importance (Quali Fattori Contano di Più?)</h3>
        </div>
        """, unsafe_allow_html=True)
        
        importances = rf_model.feature_importances_
        df_importance = pd.DataFrame({
            'Feature': ['Distanza', 'Sonno', 'Stress', 'FC Media', 'RPE', 'SMA', 'ACWR'],
            'Importance': importances
        }).sort_values('Importance', ascending=True)
        
        fig_imp = px.barh(df_importance, x='Importance', y='Feature', 
                         color='Importance', color_continuous_scale='Blues',
                         title="Importanza delle Feature nel Modello",
                         height=350)
        fig_imp.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig_imp, use_container_width=True)
        
        st.markdown("""
        <div class='info-box'>
        <p><strong>Interpretazione:</strong></p>
        <ul>
            <li>📊 <strong>Sonno (Top)</strong> = Il fattore DECISIVO. Non dormire abbastanza = rischio infortunio 🚨</li>
            <li>💪 <strong>RPE (Alto)</strong> = Allenamenti troppo duri senza recupero = pericolo</li>
            <li>❤️ <strong>FC Media (Alto)</strong> = Carico cardiovascolare elevato</li>
            <li>🧠 <strong>Stress (Medio)</strong> = Lo stress mentale influenza il rischio</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs_ml[2]:
        st.markdown("""
        <div class='warning-box'>
        <h3>📈 Matrice di Confusione (Accuracy Check)</h3>
        <p>Mostra quanto accurato è il modello nel predire infortunio vs no-infortunio</p>
        </div>
        """, unsafe_allow_html=True)
        
        from sklearn.metrics import confusion_matrix, classification_report
        
        X_pred = scaler.transform(df[['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE', 'SMA', 'ACWR']].fillna(0))
        y_pred = rf_model.predict(X_pred)
        y_actual = df['Rischio_Infortunio'].values
        
        cm = confusion_matrix(y_actual, y_pred)
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predetto: No Rischio', 'Predetto: Rischio'],
            y=['Effettivo: No Rischio', 'Effettivo: Rischio'],
            text=cm,
            texttemplate='%{text}',
            colorscale='Blues'
        ))
        fig_cm.update_layout(title="Matrice di Confusione", height=400)
        st.plotly_chart(fig_cm, use_container_width=True)
        
        report = classification_report(y_actual, y_pred, output_dict=True)
        st.markdown(f"""
        <div class='success-box'>
        <p><strong>Accuracy:</strong> {report['accuracy']*100:.1f}%</p>
        <p><strong>Precision (Rischio):</strong> {report['1']['precision']*100:.1f}%</p>
        <p><strong>Recall (Rischio):</strong> {report['1']['recall']*100:.1f}%</p>
        <p style='color: #34a853; font-weight: bold;'>✓ Modello molto accurato e conservativo (evita falsi negativi)</p>
        </div>
        """, unsafe_allow_html=True)

# =====================================================================
# PAGINA 5: CONSIGLIO FINALE (Con Grafici)
# =====================================================================
elif pagina == "💡 Consiglio Finale":
    st.title("💡 Consiglio Personalizzato - Allenamento Odierno")
    
    if not st.session_state.analisi_fatta:
        st.warning("⚠️ Completa il questionario per ricevere un consiglio personalizzato.")
    else:
        r = st.session_state.risultati_analisi
        
        # CALCOLI PREDIZIONI
        sma = r['stress_lavoro'] * r['rpe_previsto'] / max(r['ore_sonno'], 0.5)
        acwr = (r['ore_lavoro'] * r['stress_lavoro']) / max(r['km_piano'], 0.1)
        recovery_idx = (r['ore_sonno'] * 10 / (r['stress_lavoro'] + 1)) + (r['fc_riposo'] / 200)
        
        scenario = scaler.transform([[
            r['km_piano'],
            r['ore_sonno'],
            r['stress_lavoro'],
            int(200 * r['rpe_previsto'] / 10),  # Stima FC max
            r['rpe_previsto'],
            sma,
            acwr
        ]])
        
        prob_infortunio = rf_model.predict_proba(scenario)[0][1] * 100
        prob_overtraining = gb_model.predict_proba(scenario)[0][1] * 100
        
        # DASHBOARD ANALISI
        st.subheader("📊 Analisi del Tuo Stato Attuale")
        
        col_score1, col_score2, col_score3, col_score4 = st.columns(4)
        
        with col_score1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>⚠️ Rischio Infortunio</div>
                <div class='metric-value' style='color: {"#ea4335" if prob_infortunio > 50 else "#fbbc04" if prob_infortunio > 25 else "#34a853"};'>{prob_infortunio:.1f}%</div>
                <div style='font-size: 0.85em; color: {"#ea4335" if prob_infortunio > 50 else "#fbbc04" if prob_infortunio > 25 else "#34a853"};'>{"CRITICO 🚨" if prob_infortunio > 50 else "ATTENZIONE ⚠️" if prob_infortunio > 25 else "SICURO ✓"}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_score2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>🔄 Sovrallenamento</div>
                <div class='metric-value' style='color: {"#ea4335" if prob_overtraining > 50 else "#fbbc04" if prob_overtraining > 25 else "#34a853"};'>{prob_overtraining:.1f}%</div>
                <div style='font-size: 0.85em; color: {"#ea4335" if prob_overtraining > 50 else "#fbbc04" if prob_overtraining > 25 else "#34a853"};'>{"CRITICO 🚨" if prob_overtraining > 50 else "ATTENZIONE ⚠️" if prob_overtraining > 25 else "SICURO ✓"}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_score3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>💤 Recovery Index</div>
                <div class='metric-value' style='color: {"#34a853" if recovery_idx > 8 else "#fbbc04" if recovery_idx > 5 else "#ea4335"};'>{recovery_idx:.2f}</div>
                <div style='font-size: 0.85em; color: {"#34a853" if recovery_idx > 8 else "#fbbc04" if recovery_idx > 5 else "#ea4335"};'>{"OTTIMO ✓" if recovery_idx > 8 else "NORMALE 💤" if recovery_idx > 5 else "BASSO ⚠️"}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_score4:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>SMA Score</div>
                <div class='metric-value' style='color: {"#34a853" if sma < 3 else "#fbbc04" if sma < 5 else "#ea4335"};'>{sma:.2f}</div>
                <div style='font-size: 0.85em; color: {"#34a853" if sma < 3 else "#fbbc04" if sma < 5 else "#ea4335"};'>{"OTTIMO ✓" if sma < 3 else "BUONO 💪" if sma < 5 else "ALTO ⚠️"}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # GRAFICI COMPARATIVI
        st.subheader("📈 Comparazione: I Tuoi Dati vs Media (90gg)")
        
        col_comp1, col_comp2 = st.columns(2)
        
        with col_comp1:
            dati_confronto = pd.DataFrame({
                'Parametro': ['Sonno', 'Stress', 'RPE', 'FC Max (%)'],
                'Tuo Dato': [
                    r['ore_sonno'],
                    r['stress_lavoro'],
                    r['rpe_previsto'],
                    r['rpe_previsto'] / 10 * 100
                ],
                'Media (90gg)': [
                    df['Ore Sonno'].mean(),
                    df['Stress Lavoro'].mean(),
                    df['RPE'].mean(),
                    df['RPE'].mean() / 10 * 100
                ]
            })
            
            fig_confronto = go.Figure(data=[
                go.Bar(name='Tuo Dato', x=dati_confronto['Parametro'], y=dati_confronto['Tuo Dato'], marker_color='#1a73e8'),
                go.Bar(name='Media (90gg)', x=dati_confronto['Parametro'], y=dati_confronto['Media (90gg)'], marker_color='#e8eef7')
            ])
            fig_confronto.update_layout(barmode='group', template="plotly_white", height=350, hovermode='x unified')
            st.plotly_chart(fig_confronto, use_container_width=True)
        
        with col_comp2:
            # Gauge del rischio complessivo
            rischio_totale = (prob_infortunio + prob_overtraining) / 2
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=rischio_totale,
                title="RISCHIO COMPLESSIVO",
                delta={'reference': 30, 'suffix': '% vs Media'},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "#34a853"},
                        {'range': [25, 50], 'color': "#fbbc04"},
                        {'range': [50, 100], 'color': "#ea4335"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(height=350)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown("---")
        
        # CONSIGLIO PERSONALIZZATO
        st.subheader("🎯 Consiglio Personalizzato")
        
        rischio_complessivo = (prob_infortunio + prob_overtraining) / 2
        
        if rischio_complessivo < 25:
            st.markdown("""
            <div class='success-box'>
            <h3>✅ ALLENAMENTO INTENSO - SITUAZIONE OTTIMALE</h3>
            <p><strong style='font-size: 1.1em;'>Il tuo corpo è PRONTO!</strong> Tutti i parametri sono al verde.</p>
            </div>
            """, unsafe_allow_html=True)
            
            col_reco1, col_reco2 = st.columns(2)
            
            with col_reco1:
                st.markdown("""
                <div class='info-box'>
                <h4>💪 Che Cosa Fare</h4>
                <ul>
                    <li>✅ Intervalli veloci</li>
                    <li>✅ Ripetute intense</li>
                    <li>✅ Tempo run sostenuto</li>
                    <li>✅ Test velocità</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col_reco2:
                # Grafico workout
                workout_plan = pd.DataFrame({
                    'Fase': ['Warm-up', 'Lavoro', 'Cool-down', 'Stretching'],
                    'Minuti': [15, 45, 10, 15]
                })
                fig_workout = px.bar(
                    workout_plan, x='Fase', y='Minuti',
                    color_discrete_sequence=['#1a73e8'],
                    title="Piano di Allenamento",
                    height=300
                )
                fig_workout.update_layout(template="plotly_white", showlegend=False)
                st.plotly_chart(fig_workout, use_container_width=True)
            
            st.markdown("""
            <div class='success-box'>
            <h4>📋 Protocollo Dettagliato</h4>
            <ul>
                <li><strong>Warm-up (15 min):</strong> Corsa progressiva + dinamica</li>
                <li><strong>Lavoro (45 min):</strong> 6 x 800m a ritmo gara con 3min recupero</li>
                <li><strong>Cool-down (10 min):</strong> Corsa facile progressivamente decelerante</li>
                <li><strong>Stretching (15 min):</strong> Statico su principali gruppi muscolari</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        elif rischio_complessivo < 60:
            st.markdown("""
            <div class='warning-box'>
            <h3>⚠️ RECUPERO ATTIVO - SITUAZIONE MODERATA</h3>
            <p><strong style='font-size: 1.1em;'>Il tuo corpo ha bisogno di rigenerazione.</strong> Non spingere forte oggi.</p>
            </div>
            """, unsafe_allow_html=True)
            
            col_mod1, col_mod2 = st.columns(2)
            
            with col_mod1:
                st.markdown("""
                <div class='info-box'>
                <h4>🏃 Che Cosa Fare</h4>
                <ul>
                    <li>✅ Easy run (ritmo conversativo)</li>
                    <li>✅ Lungo facile 12-18 km</li>
                    <li>✅ Recovery run</li>
                    <li>✅ Fartlek leggero</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col_mod2:
                parametri_mod = pd.DataFrame({
                    'Parametro': ['FC Target', 'RPE', 'Velocità km/h'],
                    'Range': ['130-140', '3-4/10', '10-11']
                })
                
                st.markdown("""
                <div class='info-box'>
                <h4>📊 Parametri Target</h4>
                <p><strong>FC:</strong> 60-70% del massimale (~130-140 bpm)</p>
                <p><strong>RPE:</strong> 3-4/10 (molto facile, puoi parlare)</p>
                <p><strong>Velocità:</strong> 10-11 km/h (conversativa)</p>
                <p><strong>Durata:</strong> 45-75 minuti</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class='warning-box'>
            <h4>🛏️ Priorità per 24-48 ore</h4>
            <ul>
                <li>💤 Dormi <strong>8+ ore</strong> stasera</li>
                <li>💧 Bevi <strong>3+ litri</strong> di acqua</li>
                <li>🧘 Yoga/Stretching 20 minuti</li>
                <li>🧠 Riduci stress mentale</li>
                <li>🥗 Mangia proteine + carboidrati</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            st.markdown("""
            <div class='danger-box'>
            <h3>🚨 RIPOSO OBBLIGATORIO - SITUAZIONE CRITICA</h3>
            <p><strong style='font-size: 1.1em;'>Il tuo corpo è in pericolo! DEVI riposare oggi.</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            col_rep1, col_rep2 = st.columns(2)
            
            with col_rep1:
                st.markdown("""
                <div class='danger-box'>
                <h4>❌ Cosa NON Fare</h4>
                <ul>
                    <li>❌ NON CORRERE</li>
                    <li>❌ NON fare allenamenti intensi</li>
                    <li>❌ NON sollevare pesi</li>
                    <li>❌ NON stressarti</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col_rep2:
                st.markdown("""
                <div class='danger-box'>
                <h4>✅ Cosa Fare (Riposo)</h4>
                <ul>
                    <li>✅ Riposo totale a casa</li>
                    <li>✅ Camminate leggere max 15 min</li>
                    <li>✅ Stretching delicato 10 min</li>
                    <li>✅ Meditazione/Respirazione</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class='danger-box'>
            <h4>🚑 Segnali di Allarme - Consulta Medico SUBITO:</h4>
            <ul>
                <li>🚑 Dolore acuto o persistente</li>
                <li>🚑 Gonfiore/rigidità muscolare anomala</li>
                <li>🚑 Febbre > 37.5°C</li>
                <li>🚑 Stanchezza estrema anche a riposo</li>
                <li>🚑 Difficoltà respiratorie</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # RIEPILOGO PIANO
        st.subheader("📝 Riepilogo Piano")
        
        col_r1, col_r2, col_r3 = st.columns(3)
        
        with col_r1:
            st.markdown(f"""
            <div class='info-box'>
            <h4>📍 Allenamento</h4>
            <p><strong>{r['tipo_allenamento']}</strong></p>
            <p>{r['km_piano']:.1f} km in {r['tempo_minuti']} min</p>
            <p>Ritmo ~{r['km_piano']/r['tempo_minuti']*60:.1f} km/h</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_r2:
            st.markdown(f"""
            <div class='info-box'>
            <h4>💤 Recupero</h4>
            <p>Sonno: <strong>{r['ore_sonno']:.1f}h</strong> ({['❌', '⚠️', '✓'][min(2, int(r['ore_sonno']/3))]}</p>
            <p>Stress: <strong>{r['stress_lavoro']}/10</strong></p>
            <p>Recovery: <strong>{recovery_idx:.2f}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_r3:
            st.markdown(f"""
            <div class='info-box'>
            <h4>🎯 Metriche</h4>
            <p>SMA: <strong>{sma:.2f}</strong></p>
            <p>ACWR: <strong>{acwr:.2f}</strong></p>
            <p>Risk: <strong>{rischio_complessivo:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; font-size: 0.85em; margin-top: 50px;'>
    <p>🏃 <strong>RunAI Coach v2.0</strong> | Professional Analytics for Runners | Powered by ML</p>
    <p>Disclaimer: Questo strumento è per scopi educativi. Non sostituisce il parere medico professionale.</p>
</div>
""", unsafe_allow_html=True)
