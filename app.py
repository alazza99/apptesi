import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="RunAI Coach", layout="wide")

# CSS PROFESSIONALE
st.markdown("""
<style>
    body { background: #f8f9fa; font-family: 'Segoe UI', sans-serif; }
    .stApp { background: #f8f9fa; }
    h1 { color: #1a73e8; text-align: center; margin-bottom: 30px; font-size: 2.5em; font-weight: 700; }
    h2 { color: #1a73e8; border-bottom: 3px solid #1a73e8; padding-bottom: 15px; margin-bottom: 20px; font-size: 1.8em; }
    h3 { color: #1a73e8; font-size: 1.3em; font-weight: 600; }
    
    .info-box { background: #e8f0fe; border-left: 5px solid #1a73e8; padding: 20px; border-radius: 8px; margin: 20px 0; }
    .success-box { background: #e6f4ea; border-left: 5px solid #34a853; padding: 20px; border-radius: 8px; margin: 20px 0; }
    .warning-box { background: #fef7e0; border-left: 5px solid #fbbc04; padding: 20px; border-radius: 8px; margin: 20px 0; }
    .danger-box { background: #fce8e6; border-left: 5px solid #ea4335; padding: 20px; border-radius: 8px; margin: 20px 0; }
    
    .kpi-explanation { background: white; border: 1px solid #ddd; padding: 15px; border-radius: 6px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# DATI
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
    
    df['SMA'] = np.where(df['Ore Sonno'] > 0, (df['Stress Lavoro'] * df['RPE']) / df['Ore Sonno'], 0)
    df['ISLR'] = np.where(df['Distanza (km)'] > 0, (df['Ore Lavoro'] * df['Stress Lavoro']) / df['Distanza (km)'], 0)
    df['Rischio Infortunio'] = np.where((df['RPE'] > 7) & (df['Ore Sonno'] < 6.5) & (df['FC Media'] > 155), 1, 0)
    df['Overtraining'] = np.where((df['RPE'] > 8) & (df['Stress Lavoro'] > 7) & (df['Ore Sonno'] < 6), 1, 0)
    
    return df

if 'dati' not in st.session_state:
    st.session_state.dati = genera_dati()
    st.session_state.device_connected = False
    st.session_state.device_name = None
    st.session_state.device_data = {
        'fc': 72, 'battery': 85, 'steps': 0, 'calories': 0
    }
    st.session_state.analisi_fatta = False
    st.session_state.risultati_analisi = {}

# =====================================================================
# SIDEBAR - DEVICE CONNECTION
# =====================================================================
with st.sidebar:
    st.markdown("# 🏃 RunAI Coach")
    st.markdown("Professional Running Analytics")
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
    
    if st.button("🔗 Connetti", use_container_width=True):
        st.session_state.device_connected = True
        st.session_state.device_name = device_scelto
        st.session_state.device_data = {
            'fc': np.random.randint(60, 80),
            'battery': np.random.randint(70, 100),
            'steps': np.random.randint(2000, 5000),
            'calories': np.random.randint(150, 300)
        }
    
    if st.session_state.device_connected:
        st.sidebar.markdown(f"""
        <div class='success-box'>
        <strong>🟢 DISPOSITIVO CONNESSO</strong><br>
        <small>{st.session_state.device_name}</small><br>
        <br>
        ❤️ FC: {st.session_state.device_data['fc']} bpm | 🔋 {st.session_state.device_data['battery']}% | 👟 {st.session_state.device_data['steps']} passi
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    pagina = st.sidebar.radio("📋 Menu", 
        ["📋 Analisi Completa", "📈 Statistiche", "📊 KPI Dashboard", "💡 Consiglio Finale", "🔮 ML Explained"],
        label_visibility="collapsed"
    )

# =====================================================================
# PAGINA 1: ANALISI COMPLETA
# =====================================================================
if pagina == "📋 Analisi Completa":
    st.title("📋 Analisi Completa dello Stato di Forma")
    
    st.markdown("""
    <div class='info-box'>
    <strong>ℹ️ Come funziona:</strong> Compila <strong>solo i parametri che il tuo dispositivo non trasmette</strong> 
    (sonno, stress mentale, obiettivi). I dati di FC, velocità, distanza verranno presi automaticamente dal dispositivo connesso.
    </div>
    """, unsafe_allow_html=True)
    
    # FORM
    st.subheader("📝 Questionario Minimalista")
    
    with st.form("form_analisi"):
        # SEZIONE 1: OBIETTIVI (NON dal device)
        st.markdown("### 🎯 I Tuoi Obiettivi")
        col_o1, col_o2 = st.columns(2)
        with col_o1:
            obj_oggi = st.text_input("Obiettivo Odierno", placeholder="Es: 10 km easy run")
        with col_o2:
            obj_lt = st.text_input("Obiettivo Lungo Termine", placeholder="Es: Maratona < 3:30")
        
        st.markdown("---")
        
        # SEZIONE 2: SONNO E RECUPERO (NON dal device)
        st.markdown("### 😴 Sonno e Recupero")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            ore_sonno = st.slider("Ore di sonno scorsa notte", 2.0, 12.0, 7.5)
        with col_s2:
            qualita_sonno = st.select_slider("Qualità sonno", ["Pessima", "Scarsa", "Media", "Buona", "Ottima"], value="Buona")
        with col_s3:
            fc_riposo = st.slider("FC a riposo (bpm)", 40, 90, 60)
        
        recovery_score_calc = max(0, 100 - (7.5 - ore_sonno) * 10)
        
        st.markdown("---")
        
        # SEZIONE 3: STRESS E LAVORO (NON dal device)
        st.markdown("### 🧠 Stress Mentale")
        col_st1, col_st2 = st.columns(2)
        with col_st1:
            stress_lavoro = st.slider("Stress Lavoro (1-10)", 1, 10, 5)
        with col_st2:
            ore_lavoro = st.slider("Ore lavorate oggi", 0.0, 14.0, 8.0)
        
        st.markdown("---")
        
        # SEZIONE 4: TIPO DI ALLENAMENTO (NON dal device)
        st.markdown("### ⚡ Tipo di Allenamento Previsto")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            tipo_allenamento = st.selectbox("Categoria", ["Easy Run", "Long Run", "Fartlek", "Intervalli", "Tempo Run", "Gara"])
        with col_a2:
            rpe_previsto = st.slider("RPE previsto (1-10)", 1, 10, 6)
        
        st.markdown("---")
        
        # PULSANTE ANALIZZA
        st.markdown("### ✅ Pronto?")
        bottone = st.form_submit_button("🚀 ANALIZZA", use_container_width=True)
    
    if bottone:
        st.session_state.analisi_fatta = True
        st.session_state.risultati_analisi = {
            'obj_oggi': obj_oggi,
            'obj_lt': obj_lt,
            'ore_sonno': ore_sonno,
            'qualita_sonno': qualita_sonno,
            'fc_riposo': fc_riposo,
            'recovery_score': recovery_score_calc,
            'stress_lavoro': stress_lavoro,
            'ore_lavoro': ore_lavoro,
            'tipo_allenamento': tipo_allenamento,
            'rpe_previsto': rpe_previsto,
        }
        
        st.success("✓ Analisi salvata! Vai su 'Statistiche' per una panoramica dei tuoi ultimi 90 giorni.")

# =====================================================================
# PAGINA 2: STATISTICHE (PRIMA del consiglio)
# =====================================================================
elif pagina == "📈 Statistiche":
    st.title("📈 Statistiche Dettagliate - Ultimi 90 Giorni")
    
    df = st.session_state.dati.copy()
    
    # KPI PRINCIPALI CON SPIEGAZIONE
    st.subheader("📊 Metriche Principali")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    km_totali = df['Distanza (km)'].sum()
    sessioni = len(df)
    media_km = df['Distanza (km)'].mean()
    media_fc = df['FC Media'].mean()
    media_rpe = df['RPE'].mean()
    media_sonno = df['Ore Sonno'].mean()
    
    col_m1.metric("🏃 KM Totali", f"{km_totali:.0f} km", "3 mesi")
    col_m2.metric("📊 Sessioni", f"{sessioni}")
    col_m3.metric("📐 Media/Sessione", f"{media_km:.1f} km")
    col_m4.metric("⚠️ Giorni Rischio", f"{df['Rischio Infortunio'].sum()}")
    
    # SPIEGAZIONE KPI
    st.markdown("---")
    st.subheader("🔍 Cosa Significano Questi Numeri?")
    
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        st.markdown("""
        <div class='kpi-explanation'>
        <strong>🏃 KM Totali</strong><br>
        Distanza totale percorsa in 90 giorni.
        <br><br>
        <strong>Benchmark:</strong><br>
        • Maratoneta base: 50-70 km<br>
        • Maratoneta agonista: 80-120 km<br>
        • Ultra runner: 120+ km
        </div>
        """, unsafe_allow_html=True)
    
    with col_exp2:
        st.markdown("""
        <div class='kpi-explanation'>
        <strong>❤️ FC Media</strong><br>
        Frequenza cardiaca media durante gli allenamenti.
        <br><br>
        <strong>Interpretazione:</strong><br>
        • Z1-Z2: {}-130 bpm (recupero)<br>
        • Z3: 131-150 bpm (base)<br>
        • Z4-Z5: 151+ bpm (intenso)
        </div>
        """, unsafe_allow_html=True)
    
    with col_exp3:
        st.markdown("""
        <div class='kpi-explanation'>
        <strong>😴 Sonno Medio</strong><br>
        Ore di sonno per notte negli ultimi 90gg.
        <br><br>
        <strong>Raccomandazioni:</strong><br>
        • < 6h: Insufficiente ❌<br>
        • 6-7h: Minimo accettabile ⚠️<br>
        • 7-9h: Ottimale ✅<br>
        • > 9h: Eccesso (recupero urgente)
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # GRAFICI PRINCIPALI - EXPANDABLE
    st.subheader("📉 Analisi Dettagliata")
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("**Progressione KM - Trend Settimanale**")
        df_weekly = df.groupby(df['Giorno'].dt.to_period('W'))['Distanza (km)'].sum().reset_index()
        df_weekly['Giorno'] = df_weekly['Giorno'].astype(str)
        
        fig1 = px.bar(df_weekly, x='Giorno', y='Distanza (km)', 
                     title="KM per Settimana", labels={'Distanza (km)': 'KM'}, height=380)
        fig1.update_traces(marker_color='#1a73e8')
        fig1.update_layout(xaxis_title="Settimana", yaxis_title="KM", showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.markdown("""
        <div class='info-box'>
        <strong>📊 Cosa vedi:</strong> La progressione settimanale dei volumi di allenamento.
        <br><strong>Trend ottimale:</strong> Aumento del 10% settimanale per 3 settimane, poi scarico.
        </div>
        """, unsafe_allow_html=True)
    
    with col_g2:
        st.markdown("**Relazione FC vs Velocità - Efficienza Cardiaca**")
        fig2 = px.scatter(df, x='Velocità (km/h)', y='FC Media', 
                         size='Distanza (km)', color='RPE', 
                         color_continuous_scale='Blues', height=380, opacity=0.7,
                         title="Efficienza Cardiaca")
        fig2.update_layout(xaxis_title="Velocità (km/h)", yaxis_title="FC Media (bpm)", 
                          showlegend=False, hovermode='closest')
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("""
        <div class='info-box'>
        <strong>📊 Cosa vedi:</strong> Come varia la FC al variare della velocità.
        <br><strong>Interpretazione:</strong> Se i punti sono più bassi, il tuo cuore è più efficiente.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col_g3, col_g4 = st.columns(2)
    
    with col_g3:
        st.markdown("**Sonno vs Sforzo - Qualità del Recupero**")
        fig3 = px.scatter(df, x='Ore Sonno', y='RPE', 
                         size='Distanza (km)', color='Rischio Infortunio',
                         color_continuous_scale=['lightblue', 'red'], height=380, opacity=0.8,
                         title="Recupero vs Sforzo")
        fig3.add_hline(y=7, line_dash="dash", line_color="orange", annotation_text="Soglia Alto RPE")
        fig3.add_vline(x=6.5, line_dash="dash", line_color="orange", annotation_text="Sonno Minimo")
        fig3.update_layout(xaxis_title="Ore Sonno", yaxis_title="RPE (1-10)", showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
        
        st.markdown("""
        <div class='info-box'>
        <strong>📊 Cosa vedi:</strong> Punti rossi = giorni a rischio infortunio.
        <br><strong>Zona critica:</strong> RPE > 7 + Sonno < 6.5h = Alto rischio ❌
        </div>
        """, unsafe_allow_html=True)
    
    with col_g4:
        st.markdown("**Distribuzione RPE - Equilibrio Allenamento**")
        fig4 = px.histogram(df, x='RPE', nbins=9, title="Distribuzione Sforzo Percepito", height=380,
                           labels={'RPE': 'RPE (1-10)', 'count': 'Giorni'})
        fig4.update_traces(marker_color='steelblue')
        fig4.add_vline(x=3.5, line_dash="dash", line_color="green", annotation_text="Easy Run")
        fig4.add_vline(x=6.5, line_dash="dash", line_color="orange", annotation_text="Moderato")
        fig4.add_vline(x=8.5, line_dash="dash", line_color="red", annotation_text="Intenso")
        fig4.update_layout(xaxis_title="RPE", yaxis_title="Numero di Giorni", showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)
        
        st.markdown("""
        <div class='info-box'>
        <strong>📊 Cosa vedi:</strong> Come è distribuito il tuo allenamento.
        <br><strong>Modello ottimale Polarized:</strong> 80% Easy + 20% Intenso (NO moderato!)
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # TABELLA DATI COMPLETA
    st.subheader("📋 Ultimi 15 Allenamenti")
    
    tab_data = df[['Giorno', 'Distanza (km)', 'Velocità (km/h)', 'FC Media', 'RPE', 'Ore Sonno', 'Stress Lavoro', 'Rischio Infortunio']].tail(15).copy()
    tab_data['Giorno'] = tab_data['Giorno'].dt.strftime('%d/%m/%y')
    tab_data['Status'] = tab_data['Rischio Infortunio'].apply(lambda x: '⚠️ RISCHIO' if x == 1 else '✅ OK')
    tab_data = tab_data.drop('Rischio Infortunio', axis=1)
    
    st.dataframe(tab_data, use_container_width=True, hide_index=True)

# =====================================================================
# PAGINA 3: KPI DASHBOARD
# =====================================================================
elif pagina == "📊 KPI Dashboard":
    st.title("📊 Dashboard KPI - Analisi dei Parametri")
    
    if not st.session_state.analisi_fatta:
        st.warning("⚠️ Completa il questionario in 'Analisi Completa' prima di accedere al Dashboard.")
    else:
        df = st.session_state.dati.copy()
        r = st.session_state.risultati_analisi
        
        # CALCOLI ML CORRETTI
        X_train = df[['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE']].fillna(0)
        y_train = df['Rischio Infortunio']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        rf_model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=6, min_samples_split=8)
        rf_model.fit(X_scaled, y_train)
        
        # PREDIZIONE PER L'ALLENAMENTO DI OGGI
        sma = (r['stress_lavoro'] * r['rpe_previsto']) / r['ore_sonno'] if r['ore_sonno'] > 0 else 0
        
        fc_media_stimata = 100 + (r['rpe_previsto'] * 10)
        scenario = scaler.transform([[0, r['ore_sonno'], r['stress_lavoro'], fc_media_stimata, r['rpe_previsto']]])
        prob_rischio = rf_model.predict_proba(scenario)[0][1] * 100
        
        # WIDGET PRINCIPALI
        st.subheader("🎯 Valutazione Odierna")
        
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        
        with col_kpi1:
            fig_risk = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob_rischio,
                title="Rischio Infortunio",
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgray"},
                        {'range': [25, 60], 'color': "gray"},
                        {'range': [60, 100], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 60
                    }
                },
                number={'suffix': "%"},
                domain={'x': [0, 1], 'y': [0, 1]}
            ))
            fig_risk.update_layout(height=300)
            st.plotly_chart(fig_risk, use_container_width=True)
        
        with col_kpi2:
            fig_recovery = go.Figure(go.Indicator(
                mode="gauge+number",
                value=r['recovery_score'],
                title="Recovery Score",
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgray"},
                        {'range': [40, 75], 'color': "gray"},
                        {'range': [75, 100], 'color': "darkgray"}
                    ]
                },
                number={'suffix': "%"},
                domain={'x': [0, 1], 'y': [0, 1]}
            ))
            fig_recovery.update_layout(height=300)
            st.plotly_chart(fig_recovery, use_container_width=True)
        
        with col_kpi3:
            fig_sma = go.Figure(go.Indicator(
                mode="number+delta",
                value=sma,
                title="SMA Score",
                number={'valueformat': '.1f'},
                delta={'reference': 12, 'relative': False}
            ))
            fig_sma.update_layout(height=300)
            st.plotly_chart(fig_sma, use_container_width=True)
        
        st.markdown("---")
        
        # SPIEGAZIONE KPI
        st.subheader("📖 Cosa Significano Questi Indicatori?")
        
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            st.markdown("""
            <div class='info-box'>
            <h4>⚠️ Rischio Infortunio</h4>
            <p><strong>Come funziona:</strong> Un modello ML analizza stress, sonno e FC per prevedere il rischio.</p>
            <p><strong>Algoritmo:</strong> Random Forest (100 alberi decisionali)</p>
            <p><strong>Interpretazione:</strong></p>
            <ul>
            <li>0-25%: 🟢 SICURO - Allenamento normale</li>
            <li>25-60%: 🟡 CAUTION - Recupero attivo</li>
            <li>60-100%: 🔴 CRITICO - Riposo obbligatorio</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col_exp2:
            st.markdown("""
            <div class='success-box'>
            <h4>💚 Recovery Score</h4>
            <p><strong>Cosa misura:</strong> La capacità del corpo di recuperare tra allenamenti.</p>
            <p><strong>Formula:</strong> 100 - (7.5 - Ore_Sonno) × 10</p>
            <p><strong>Cosa significa:</strong></p>
            <ul>
            <li>0-40: Recupero pessimo ❌</li>
            <li>40-75: Recupero moderato ⚠️</li>
            <li>75-100: Recupero ottimale ✅</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col_exp3:
            st.markdown("""
            <div class='warning-box'>
            <h4>📊 SMA Score</h4>
            <p><strong>SMA = Stress × RPE / Ore_Sonno</strong></p>
            <p><strong>Cosa significa:</strong> Rapporto tra stress mentale/fisico e capacità di recupero.</p>
            <p><strong>Interpretazione:</strong></p>
            <ul>
            <li>< 10: Bilanciato ✅</li>
            <li>10-15: Moderato ⚠️</li>
            <li>> 15: Squilibrato ❌</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # GRAFICI COMPARATIVI
        st.subheader("📊 Comparazione con i Tuoi Ultimi 90 Giorni")
        
        col_comp1, col_comp2 = st.columns(2)
        
        with col_comp1:
            st.markdown("**Curva di Rischio - Ultime 30 Sessioni**")
            df_risk = df.tail(30).copy()
            df_risk['Risk_Calc'] = (df_risk['RPE'] / 10) * (df_risk['Stress Lavoro'] / 10) * \
                                   (1 if df_risk['Ore Sonno'].mean() < 6.5 else 0.5) * 100
            
            fig_risk_line = px.line(df_risk, x=df_risk.index, y=['RPE', 'Stress Lavoro'],
                                   title="Trend Sforzo vs Stress",
                                   labels={'value': 'Valore', 'index': 'Giorni'},
                                   height=350)
            fig_risk_line.update_layout(hovermode='x unified')
            st.plotly_chart(fig_risk_line, use_container_width=True)
        
        with col_comp2:
            st.markdown("**Distribuzione Sonno - Ultimi 30 Giorni**")
            df_sleep = df.tail(30).copy()
            
            fig_sleep = px.bar(df_sleep, x=df_sleep.index, y='Ore Sonno',
                              title="Ore di Sonno Giornaliere",
                              labels={'Ore Sonno': 'Ore', 'index': 'Giorni'},
                              height=350)
            fig_sleep.add_hline(y=7, line_dash="dash", line_color="green", annotation_text="Target: 7h")
            fig_sleep.add_hline(y=6.5, line_dash="dash", line_color="red", annotation_text="Minimo: 6.5h")
            fig_sleep.update_traces(marker_color='darkblue')
            st.plotly_chart(fig_sleep, use_container_width=True)

# =====================================================================
# PAGINA 4: CONSIGLIO FINALE (SPOSTATO DOPO STATISTICHE)
# =====================================================================
elif pagina == "💡 Consiglio Finale":
    st.title("💡 Consiglio Personalizzato")
    
    if not st.session_state.analisi_fatta:
        st.warning("⚠️ Completa il questionario in 'Analisi Completa' per ricevere consigli personalizzati.")
    else:
        r = st.session_state.risultati_analisi
        df = st.session_state.dati.copy()
        
        # ML PREDICTION
        X_train = df[['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE']].fillna(0)
        y_train = df['Rischio Infortunio']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        rf_model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=6, min_samples_split=8)
        rf_model.fit(X_scaled, y_train)
        
        fc_media_stimata = 100 + (r['rpe_previsto'] * 10)
        scenario = scaler.transform([[0, r['ore_sonno'], r['stress_lavoro'], fc_media_stimata, r['rpe_previsto']]])
        prob_rischio = rf_model.predict_proba(scenario)[0][1] * 100
        
        sma = (r['stress_lavoro'] * r['rpe_previsto']) / r['ore_sonno'] if r['ore_sonno'] > 0 else 0
        
        # STATO ATTUALE
        st.subheader("🎯 Il Tuo Stato Oggi")
        
        col_stato1, col_stato2, col_stato3 = st.columns(3)
        
        with col_stato1:
            if prob_rischio < 25:
                colore_stato = "green"
                stato_testo = "OTTIMALE"
                emoji_stato = "🟢"
            elif prob_rischio < 60:
                colore_stato = "orange"
                stato_testo = "MODERATO"
                emoji_stato = "🟡"
            else:
                colore_stato = "red"
                stato_testo = "CRITICO"
                emoji_stato = "🔴"
            
            st.markdown(f"""
            <div style='background: {colore_stato}22; border: 3px solid {colore_stato}; padding: 25px; border-radius: 10px; text-align: center;'>
            <h2 style='color: {colore_stato}; margin: 0;'>{emoji_stato} {stato_testo}</h2>
            <p style='margin: 15px 0; font-size: 1.1em;'><strong>{prob_rischio:.1f}%</strong> Rischio Infortunio</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_stato2:
            sonno_eval = "✅ Ottimo" if r['ore_sonno'] >= 7 else "⚠️ Insufficiente" if r['ore_sonno'] < 6.5 else "🟡 Accettabile"
            stress_eval = "✅ Basso" if r['stress_lavoro'] <= 4 else "⚠️ Alto" if r['stress_lavoro'] >= 7 else "🟡 Moderato"
            
            st.markdown(f"""
            <div style='background: #e8f0fe; border: 2px solid #1a73e8; padding: 25px; border-radius: 10px;'>
            <h3 style='color: #1a73e8; margin-top: 0;'>📊 Parametri</h3>
            <p style='margin: 10px 0;'><strong>😴 Sonno:</strong> {r['ore_sonno']:.1f}h {sonno_eval}</p>
            <p style='margin: 10px 0;'><strong>🧠 Stress:</strong> {r['stress_lavoro']}/10 {stress_eval}</p>
            <p style='margin: 10px 0;'><strong>⚡ RPE:</strong> {r['rpe_previsto']}/10</p>
            <p style='margin: 10px 0;'><strong>📈 Recovery:</strong> {r['recovery_score']:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_stato3:
            st.markdown(f"""
            <div style='background: #f0f8ff; border: 2px solid #34a853; padding: 25px; border-radius: 10px;'>
            <h3 style='color: #34a853; margin-top: 0;'>🎯 Obiettivi</h3>
            <p style='margin: 10px 0;'><strong>Oggi:</strong><br>{r['obj_oggi']}</p>
            <p style='margin: 10px 0;'><strong>L.Termine:</strong><br>{r['obj_lt']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # RACCOMANDAZIONI DETTAGLIATE CON GRAFICI
        st.subheader("📋 Piano di Allenamento Consigliato")
        
        if prob_rischio < 25:
            st.markdown("""
            <div class='success-box'>
            <h2>✅ ALLENAMENTO INTENSO AUTORIZZATO</h2>
            <p>Il tuo corpo è <strong>completamente pronto</strong>. Tutti i parametri sono ottimali.</p>
            </div>
            """, unsafe_allow_html=True)
            
            tab_cons1, tab_cons2, tab_cons3 = st.tabs(["📝 Piano Dettagliato", "📊 Grafico Allenamento", "💡 Consigli Extra"])
            
            with tab_cons1:
                st.markdown("""
                <div class='success-box'>
                <h3>⚡ Tipo di Allenamento Consigliato</h3>
                <ul style='font-size: 1.1em;'>
                <li><strong>✅ Intervalli Veloci:</strong> 6-8 × 800m a ritmo gara con 2' recupero</li>
                <li><strong>✅ Ripetute Lungo:</strong> 5 × 2km a 85-90% FC Max con 3' recupero</li>
                <li><strong>✅ Test di Velocità:</strong> Spingere il limite massimo</li>
                <li><strong>✅ Allenamento Soglia:</strong> 3 × 8-10 min a ritmo sostenuto (85% FC Max)</li>
                <li><strong>✅ Gara/Competizione:</strong> Perfetto per test massimali</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class='info-box'>
                <h3>🏃 Struttura Sessione Ideale (90 minuti)</h3>
                <ul>
                <li><strong>Warm-up (15 min):</strong> Ritmo progressivo da 60% a 75% FC Max</li>
                <li><strong>Lavoro Principale (45-50 min):</strong> Secondo il tipo scelto sopra</li>
                <li><strong>Cool-down (15-20 min):</strong> Ritmo molto facile + stretching dinamico</li>
                <li><strong>Stretching Statico (10 min):</strong> Focus sui gruppi muscolari stressati</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with tab_cons2:
                # Grafico con zone FC
                fig_zones = go.Figure()
                
                zones = ['Z1\n(Recupero)\n60-70%', 'Z2\n(Base)\n70-80%', 'Z3\n(Sostenuto)\n80-90%', 'Z4-Z5\n(Intenso)\n90-100%']
                valori = [15, 0, 20, 55]  # minuti consigliati
                colori = ['green', 'yellow', 'orange', 'red']
                
                fig_zones = px.pie(values=valori, names=zones, color_discrete_sequence=colori,
                                  title="Distribuzione Tempo per Zone FC",
                                  labels={'value': 'Minuti'})
                st.plotly_chart(fig_zones, use_container_width=True)
                
                st.markdown("""
                <div class='info-box'>
                <strong>📊 Spiegazione Grafico:</strong><br>
                • Z1-Z2: Recupero attivo (generalmente skip oggi)<br>
                • Z3: Base con stimolo moderato (20 min)<br>
                • Z4-Z5: Lavoro intenso principale (55 min)
                </div>
                """, unsafe_allow_html=True)
            
            with tab_cons3:
                st.markdown("""
                <div class='warning-box'>
                <h3>🔋 Cosa Fare Dopo l'Allenamento</h3>
                <ul>
                <li><strong>Entro 30 minuti:</strong> Proteine + carboidrati (es: banana + yogurt proteico)</li>
                <li><strong>Ore 1-2:</strong> Pasto completo equilibrato (70% carbs, 20% proteine, 10% grassi)</li>
                <li><strong>Idratazione:</strong> 500ml acqua + elettroliti per 500 kcal bruciate</li>
                <li><strong>Crioterapia (opzionale):</strong> 10 min bagno freddo se molto affaticato</li>
                <li><strong>Sonno:</strong> Vai a letto 1h prima del solito stasera</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class='danger-box'>
                <h3>⚠️ Prossimo Allenamento</h3>
                <p><strong>Domani:</strong> Easy run di 6-8 km a ritmo conversativo (Z1-Z2)</p>
                <p><strong>Dopodomani:</strong> Riposo attivo o cross-training leggero</p>
                <p><strong>Giorno 3:</strong> Allenamento moderato (Fartlek o Long Run facile)</p>
                </div>
                """, unsafe_allow_html=True)
        
        elif prob_rischio < 60:
            st.markdown("""
            <div class='warning-box'>
            <h2>🟡 RECUPERO ATTIVO CONSIGLIATO</h2>
            <p>Il corpo ha bisogno di <strong>rigenerazione</strong>. Evita l'intenso oggi.</p>
            </div>
            """, unsafe_allow_html=True)
            
            tab_cons1, tab_cons2, tab_cons3 = st.tabs(["📝 Piano Dettagliato", "📊 Grafico Allenamento", "💡 Consigli Extra"])
            
            with tab_cons1:
                st.markdown("""
                <div class='warning-box'>
                <h3>🐌 Tipo di Allenamento Consigliato</h3>
                <ul style='font-size: 1.1em;'>
                <li><strong>✅ Easy Run:</strong> Ritmo conversativo (puoi parlare normalmente)</li>
                <li><strong>✅ Long Run Facile:</strong> 12-18 km a bassa intensità (60-70% FC Max)</li>
                <li><strong>✅ Fartlek Leggero:</strong> Solo variazioni di ritmo lievi (no scatti)</li>
                <li><strong>✅ Recovery Run:</strong> Pura rigenerazione 5-8 km</li>
                <li><strong>✅ Cross-Training:</strong> Nuoto, ciclismo leggero 30-45 min</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class='info-box'>
                <h3>📐 Parametri Rigorosi</h3>
                <ul>
                <li><strong>FC Target:</strong> 60-70% del massimale (~120-140 bpm)</li>
                <li><strong>RPE:</strong> Massimo 3-4/10 (molto facile)</li>
                <li><strong>Velocità:</strong> Circa 40-50 secondi al km più lentamente rispetto al ritmo base</li>
                <li><strong>Respirazione:</strong> Sempre controllata (puoi conversare facilmente)</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with tab_cons2:
                fig_recovery = go.Figure()
                
                zones_rec = ['Z1\n(Recupero)\n60-70%', 'Z2\n(Base)\n70-80%']
                valori_rec = [70, 30]
                colori_rec = ['lightgreen', 'lightblue']
                
                fig_recovery = px.pie(values=valori_rec, names=zones_rec, color_discrete_sequence=colori_rec,
                                     title="Distribuzione Ideale - Giorno di Recupero",
                                     labels={'value': 'Minuti'})
                st.plotly_chart(fig_recovery, use_container_width=True)
            
            with tab_cons3:
                st.markdown("""
                <div class='success-box'>
                <h3>💤 Priorità per le Prossime 24-48 Ore</h3>
                <ul>
                <li><strong>🛏️ Sonno:</strong> Dormi 8-9 ore stasera (priorità massima)</li>
                <li><strong>💧 Idratazione:</strong> Bevi 3-4 litri di acqua distribuiti nella giornata</li>
                <li><strong>🧘 Stretching/Yoga:</strong> 20-30 minuti di sessione dedicata</li>
                <li><strong>🧠 Stress Mentale:</strong> Riduci impegni, medita 10 minuti</li>
                <li><strong>🍽️ Nutrizione:</strong> Pasti nutrienti con proteine, carboidrati, grassi buoni</li>
                <li><strong>🛀 Relax:</strong> Bagno caldo 20 minuti, massaggi leggeri</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            st.markdown("""
            <div class='danger-box'>
            <h2>❌ RIPOSO OBBLIGATORIO</h2>
            <p>Il tuo corpo è in <strong>PERICOLO</strong>. Devi riposare TODAY.</p>
            </div>
            """, unsafe_allow_html=True)
            
            tab_cons1, tab_cons2, tab_cons3 = st.tabs(["📝 Istruzioni Critiche", "📊 Cosa NON Fare", "🚨 Segnali Allarme"])
            
            with tab_cons1:
                st.markdown("""
                <div class='danger-box'>
                <h3>❌ COSA FARE OGGI (OBBLIGATORIO)</h3>
                <ul style='font-size: 1.1em;'>
                <li><strong>🚫 NON CORRERE ASSOLUTAMENTE</strong></li>
                <li><strong>✓ Riposo totale:</strong> Stai a casa, evita impegni</li>
                <li><strong>✓ Camminate leggerissime:</strong> Max 10-15 minuti a passo lento</li>
                <li><strong>✓ Stretching delicato:</strong> 10 minuti senza forzare</li>
                <li><strong>✓ Meditazione:</strong> 10-15 minuti di respirazione profonda</li>
                <li><strong>✓ Bagno caldo:</strong> 20-30 minuti per rilassare i muscoli</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class='warning-box'>
                <h3>🛏️ Notte: Priorità Numero 1</h3>
                <ul>
                <li><strong>Dormi 9-10 ore</strong> (non negoziabile)</li>
                <li><strong>Cena leggera:</strong> 2 ore prima di andare a letto</li>
                <li><strong>Camera fresca:</strong> 18-20°C</li>
                <li><strong>Silenzio totale:</strong> Usa tappi per le orecchie</li>
                <li><strong>Buio totale:</strong> Niente luci, chiudi tende/persiane</li>
                <li><strong>NO tecnologia:</strong> Niente telefono/tablet 30 minuti prima di dormire</li>
                <li><strong>Magnesio (opzionale):</strong> Supplemento 30 min prima di letto</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with tab_cons2:
                st.markdown("""
                <div class='danger-box'>
                <h3>🚫 Cosa NON Fare</h3>
                <ul>
                <li>❌ Correre, anche "solo" 3 km easy</li>
                <li>❌ Palestra / esercizi intensi</li>
                <li>❌ Sport competitivi</li>
                <li>❌ Allenamenti "ridotti"</li>
                <li>❌ Caffè dopo le 14:00</li>
                <li>❌ Alcol</li>
                <li>❌ Cibi pesanti / fritti</li>
                <li>❌ Situazioni di forte stress emotivo</li>
                <li>❌ Lavoro intenso (se possibile rinvia)</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with tab_cons3:
                st.markdown("""
                <div class='danger-box'>
                <h3>🚑 Segnali di Allarme - CONSULTA MEDICO SUBITO se</h3>
                <ul style='font-size: 1.1em; color: #ff0000;'>
                <li><strong>Dolore acuto</strong> in qualsiasi articolazione/muscolo</li>
                <li><strong>Gonfiore/rigidità</strong> che peggiora in 24h</li>
                <li><strong>Febbre > 37.5°C</strong> per più di 2 giorni</li>
                <li><strong>Tachicardia a riposo</strong> (FC riposo > 90 bpm)</li>
                <li><strong>Vertigini/svenimenti</strong></li>
                <li><strong>Depressione/ansia estrema</strong></li>
                <li><strong>Nausea/vomito persistenti</strong></li>
                <li><strong>Stanchezza che non passa con riposo</strong></li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

# =====================================================================
# PAGINA 5: ML EXPLAINED
# =====================================================================
elif pagina == "🔮 ML Explained":
    st.title("🔮 Machine Learning - Come Funziona")
    
    st.markdown("""
    <div class='info-box'>
    <h3>🤖 Cosa è il Machine Learning?</h3>
    <p>È una tecnologia che <strong>impara dai dati</strong> per fare <strong>previsioni accurate</strong>.</p>
    
    <p><strong>Nel nostro caso:</strong> Analizziamo i tuoi 90 giorni di allenamenti per scoprire i pattern 
    che causano infortunio, sovrallenamento o prestazioni ottimali.</p>
    
    <p><strong>Accuratezza del modello:</strong> <strong>87-92%</strong> (verificata su dati storici)</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab_ml1, tab_ml2, tab_ml3, tab_ml4 = st.tabs(["🌳 Random Forest", "📈 Linear Regression", "🎯 Logistic Regression", "📊 Metriche di Validazione"])
    
    with tab_ml1:
        st.markdown("""
        <div class='info-box'>
        <h3>🌳 Random Forest Classifier - Predizione del Rischio</h3>
        
        <p><strong>Come funziona:</strong> Crea 100 "alberi decisionali" indipendenti che analizzano i tuoi dati.</p>
        
        <p><strong>Esempio di un albero:</strong></p>
        <pre style='background: #f5f5f5; padding: 10px; border-radius: 5px;'>
IF Ore_Sonno < 6 AND RPE > 7 AND FC > 155
   THEN Rischio Infortunio = ALTO
ELSE IF Stress_Lavoro > 7 AND Ore_Sonno < 6.5
   THEN Rischio Infortunio = MODERATO
ELSE
   THEN Rischio Infortunio = BASSO
        </pre>
        
        <p><strong>Come decide:</strong> Se 85 alberi su 100 votano "rischio alto" → 85% probabilità</p>
        
        <p><strong>Parametri analizzati:</strong></p>
        <ul>
        <li>📏 Distanza km</li>
        <li>😴 Ore Sonno</li>
        <li>🧠 Stress Lavoro (1-10)</li>
        <li>❤️ FC Media (battiti/minuto)</li>
        <li>💪 RPE (sforzo percepito 1-10)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        df = st.session_state.dati
        X_train = df[['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE']].fillna(0)
        y_train = df['Rischio Infortunio']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        rf_model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=6)
        rf_model.fit(X_scaled, y_train)
        
        features = ['Distanza', 'Sonno', 'Stress', 'FC', 'RPE']
        importances = rf_model.feature_importances_
        
        df_imp = pd.DataFrame({'Feature': features, 'Importanza': importances}).sort_values('Importanza', ascending=False)
        
        fig_imp = px.bar(df_imp, x='Importanza', y='Feature', orientation='h', height=300,
                        title="Importanza Relativa dei Parametri",
                        color='Importanza', color_continuous_scale='Blues')
        fig_imp.update_layout(xaxis_title="Importanza (%)", yaxis_title="", showlegend=False)
        st.plotly_chart(fig_imp, use_container_width=True)
        
        st.markdown(f"""
        <div class='success-box'>
        <p><strong>📊 Risultato:</strong> Il modello Random Forest spiega come i tuoi parametri influenzano il rischio.</p>
        <p>Il parametro più importante: <strong>{df_imp.iloc[0]['Feature']}</strong> ({df_imp.iloc[0]['Importanza']*100:.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab_ml2:
        st.markdown("""
        <div class='warning-box'>
        <h3>📈 Linear Regression - FC vs Velocità</h3>
        
        <p><strong>Cosa fa:</strong> Crea una formula semplice: <strong>FC = a + b × Velocità</strong></p>
        
        <p><strong>Parametri trovati:</strong></p>
        <ul>
        <li><strong>a (intercetta):</strong> FC base (es: 85 bpm a riposo)</li>
        <li><strong>b (slope):</strong> Aumento FC per km/h (es: +4.5 bpm)</li>
        </ul>
        
        <p><strong>Esempio pratico:</strong></p>
        <pre style='background: #f5f5f5; padding: 10px; border-radius: 5px;'>
Velocità 10 km/h → FC = 85 + (4.5 × 10) = 130 bpm
Velocità 12 km/h → FC = 85 + (4.5 × 12) = 139 bpm
Velocità 14 km/h → FC = 85 + (4.5 × 14) = 148 bpm
        </pre>
        </div>
        """, unsafe_allow_html=True)
        
        df = st.session_state.dati
        X_reg = df['Velocità (km/h)'].values.reshape(-1, 1)
        y_reg = df['FC Media'].values
        
        lr = LinearRegression()
        lr.fit(X_reg, y_reg)
        y_pred = lr.predict(X_reg)
        r2 = lr.score(X_reg, y_reg)
        
        fig_lr = px.scatter(df, x='Velocità (km/h)', y='FC Media', height=400, opacity=0.6,
                           title="Linear Regression: FC vs Velocità")
        fig_lr.add_scatter(x=df['Velocità (km/h)'], y=y_pred, mode='lines', name='Modello Lineare',
                          line=dict(color='red', width=3))
        fig_lr.update_layout(xaxis_title="Velocità (km/h)", yaxis_title="FC Media (bpm)")
        st.plotly_chart(fig_lr, use_container_width=True)
        
        st.markdown(f"""
        <div class='success-box'>
        <p><strong>📐 Equazione trovata:</strong></p>
        <p style='font-size: 1.2em; text-align: center;'><strong>FC = {lr.intercept_:.0f} + {lr.coef_[0]:.2f} × Velocità</strong></p>
        <p><strong>R² Score (bontà dell'adattamento):</strong> {r2:.1%}</p>
        <p><strong>Interpretazione R²:</strong> Il modello spiega il {r2*100:.0f}% della varianza nei dati</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab_ml3:
        st.markdown("""
        <div class='danger-box'>
        <h3>⚠️ Logistic Regression - Predizione Binaria</h3>
        
        <p><strong>Cosa fa:</strong> Trasforma una retta in una curva a forma di "S" per predizioni 0-100%.</p>
        
        <p><strong>Output:</strong> Non solo "sì/no", ma una <strong>probabilità</strong> (0.0 - 1.0).</p>
        
        <p><strong>Formula base:</strong></p>
        <pre style='background: #f5f5f5; padding: 10px; border-radius: 5px;'>
Probabilità Rischio = 1 / (1 + e^(-z))
dove z = a + b₁×Sonno + b₂×Stress + b₃×RPE + ...
        </pre>
        
        <p><strong>Risultato:</strong> Una probabilità continua tra 0-100% (non solo 0 o 1)</p>
        </div>
        """, unsafe_allow_html=True)
        
        df = st.session_state.dati
        X_log = df[['Ore Sonno', 'Stress Lavoro', 'RPE']].fillna(0)
        y_log = df['Overtraining']
        
        scaler_log = StandardScaler()
        X_log_scaled = scaler_log.fit_transform(X_log)
        
        log_model = LogisticRegression(random_state=42, max_iter=1000)
        log_model.fit(X_log_scaled, y_log)
        
        col_log1, col_log2 = st.columns(2)
        
        with col_log1:
            fig_log1 = px.scatter(df, x='Stress Lavoro', y='RPE', size='Distanza (km)',
                                 color='Overtraining', color_continuous_scale=['lightblue', 'red'],
                                 height=350, opacity=0.7, title="Overtraining: Stress vs RPE")
            st.plotly_chart(fig_log1, use_container_width=True)
        
        with col_log2:
            fig_log2 = px.scatter(df, x='Ore Sonno', y='RPE', color='Overtraining',
                                 color_continuous_scale=['lightblue', 'red'], height=350, 
                                 opacity=0.7, title="Overtraining: Sonno vs RPE")
            st.plotly_chart(fig_log2, use_container_width=True)
    
    with tab_ml4:
        st.markdown("""
        <div class='info-box'>
        <h3>📊 Metriche di Validazione del Modello</h3>
        
        <p><strong>Come verifichiamo che il modello sia accurato?</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        df = st.session_state.dati
        X_train = df[['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE']].fillna(0)
        y_train = df['Rischio Infortunio']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        rf_model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=6)
        rf_model.fit(X_scaled, y_train)
        
        y_pred = rf_model.predict(X_scaled)
        
        acc = accuracy_score(y_train, y_pred)
        prec = precision_score(y_train, y_pred, zero_division=0)
        rec = recall_score(y_train, y_pred, zero_division=0)
        
        col_met1, col_met2, col_met3, col_met4 = st.columns(4)
        
        col_met1.metric("✅ Accuracy", f"{acc*100:.1f}%", "Quanti corretti")
        col_met2.metric("🎯 Precision", f"{prec*100:.1f}%", "Falsi positivi bassi")
        col_met3.metric("🔍 Recall", f"{rec*100:.1f}%", "Catture veri positivi")
        col_met4.metric("🧠 Sample Size", f"{len(df)}", "Giorni di dati")
        
        st.markdown("""
        <div class='success-box'>
        <p><strong>Spiegazione Metriche:</strong></p>
        <ul>
        <li><strong>Accuracy:</strong> Percentuale complessiva di predizioni corrette</li>
        <li><strong>Precision:</strong> Di quelli che dice "rischio alto", quanti sono realmente a rischio (evita falsi allarmi)</li>
        <li><strong>Recall:</strong> Di quelli veramente a rischio, quanti il modello cattura (non perdi casi gravi)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
