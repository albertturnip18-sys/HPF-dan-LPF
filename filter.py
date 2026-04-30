"""
========================================================
  SIMULASI RC FILTER - LOW PASS & HIGH PASS FILTER
  Pemodelan Fisika dengan Python
  Oleh: Akademis Fisika
========================================================
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# ─────────────────────────────────────────────
#  KONFIGURASI HALAMAN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RC Filter Simulator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CSS KUSTOM
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

:root {
    --clr-bg: #0a0e1a;
    --clr-panel: #111827;
    --clr-border: #1e2d45;
    --clr-lpf: #00d4ff;
    --clr-hpf: #ff6b35;
    --clr-accent: #7c3aed;
    --clr-text: #e2e8f0;
    --clr-muted: #64748b;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--clr-bg) !important;
    color: var(--clr-text) !important;
    font-family: 'Syne', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: var(--clr-panel) !important;
    border-right: 1px solid var(--clr-border);
}

.title-block {
    background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0a0e1a 100%);
    border: 1px solid var(--clr-border);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.title-block::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--clr-lpf), var(--clr-accent), var(--clr-hpf));
}
.title-block h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    margin: 0;
    background: linear-gradient(90deg, var(--clr-lpf), var(--clr-hpf));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.title-block p {
    color: var(--clr-muted);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    margin: 0.5rem 0 0 0;
}

.formula-box {
    background: var(--clr-panel);
    border: 1px solid var(--clr-border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.8rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
}
.formula-box.lpf { border-left: 4px solid var(--clr-lpf); }
.formula-box.hpf { border-left: 4px solid var(--clr-hpf); }
.formula-box.both { border-left: 4px solid var(--clr-accent); }

.formula-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--clr-muted);
    margin-bottom: 0.5rem;
}

.metric-card {
    background: var(--clr-panel);
    border: 1px solid var(--clr-border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-card .val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--clr-lpf);
}
.metric-card .lbl {
    font-size: 0.78rem;
    color: var(--clr-muted);
    margin-top: 0.2rem;
}

.section-header {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.1rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--clr-border);
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
}
.section-header.lpf { color: var(--clr-lpf); }
.section-header.hpf { color: var(--clr-hpf); }
.section-header.both { color: var(--clr-accent); }

.explanation-box {
    background: rgba(17, 24, 39, 0.8);
    border: 1px solid var(--clr-border);
    border-radius: 10px;
    padding: 1.2rem;
    font-size: 0.9rem;
    line-height: 1.7;
    color: #cbd5e1;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--clr-panel);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    color: var(--clr-muted) !important;
    border-radius: 8px;
}
.stTabs [aria-selected="true"] {
    background: var(--clr-accent) !important;
    color: white !important;
}

[data-testid="stMetric"] {
    background: var(--clr-panel);
    border: 1px solid var(--clr-border);
    border-radius: 10px;
    padding: 0.8rem 1rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  FUNGSI FISIKA & PERHITUNGAN
# ─────────────────────────────────────────────

def lpf_vout(vin, R, C, f):
    """
    Low Pass Filter: Vout = Vin / sqrt(1 + (2π·f·R·C)²)
    Transfer function: H(jω) = 1 / (1 + jωRC)
    |H(f)| = 1 / sqrt(1 + (2πfRC)²)
    """
    omega = 2 * np.pi * f
    Xc = 1 / (omega * C)          # Reaktansi kapasitor
    Z_total = np.sqrt(R**2 + Xc**2)
    Vout = vin * (Xc / Z_total)
    return Vout

def hpf_vout(vin, R, C, f):
    """
    High Pass Filter: Vout = Vin · (2π·f·R·C) / sqrt(1 + (2π·f·R·C)²)
    Transfer function: H(jω) = jωRC / (1 + jωRC)
    |H(f)| = ωRC / sqrt(1 + (ωRC)²)
    """
    omega = 2 * np.pi * f
    Xc = 1 / (omega * C)
    Z_total = np.sqrt(R**2 + Xc**2)
    Vout = vin * (R / Z_total)
    return Vout

def cutoff_frequency(R, C):
    """fc = 1 / (2π·R·C)  — frekuensi cutoff (-3dB point)"""
    return 1 / (2 * np.pi * R * C)

def phase_lpf(f, R, C):
    """Phase LPF: φ = -arctan(2πfRC)  [degrees]"""
    return -np.degrees(np.arctan(2 * np.pi * f * R * C))

def phase_hpf(f, R, C):
    """Phase HPF: φ = 90° - arctan(2πfRC)  [degrees]"""
    return 90 - np.degrees(np.arctan(2 * np.pi * f * R * C))

def to_db(ratio):
    """Konversi ke desibel: 20·log10(Vout/Vin)"""
    return 20 * np.log10(np.clip(ratio, 1e-12, None))

def format_capacitance(C):
    if C >= 1e-3:
        return f"{C*1e3:.2f} mF"
    elif C >= 1e-6:
        return f"{C*1e6:.2f} µF"
    elif C >= 1e-9:
        return f"{C*1e9:.2f} nF"
    else:
        return f"{C*1e12:.2f} pF"


# ─────────────────────────────────────────────
#  KAPASITOR PASARAN STANDAR (E12 SERIES)
# ─────────────────────────────────────────────
KAPASITOR_PASARAN = {
    "10 pF":    10e-12,
    "22 pF":    22e-12,
    "47 pF":    47e-12,
    "100 pF":   100e-12,
    "220 pF":   220e-12,
    "470 pF":   470e-12,
    "1 nF":     1e-9,
    "2.2 nF":   2.2e-9,
    "4.7 nF":   4.7e-9,
    "10 nF":    10e-9,
    "22 nF":    22e-9,
    "47 nF":    47e-9,
    "100 nF (0.1 µF)": 100e-9,
    "220 nF":   220e-9,
    "470 nF":   470e-9,
    "1 µF":     1e-6,
    "2.2 µF":   2.2e-6,
    "4.7 µF":   4.7e-6,
    "10 µF":    10e-6,
    "47 µF":    47e-6,
    "100 µF":   100e-6,
}

# ─────────────────────────────────────────────
#  SIDEBAR — PARAMETER INPUT
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ Parameter Rangkaian")
    st.markdown("---")

    Vin = st.number_input("Vin (Volt)", min_value=0.1, max_value=100.0, value=9.0, step=0.1)
    R = st.number_input("R (kΩ)", min_value=0.1, max_value=1000.0, value=1.0, step=0.1) * 1000

    st.markdown("---")
    st.markdown("**Kapasitor Variasi (Soal 1a & 2a)**")
    selected_caps = st.multiselect(
        "Pilih nilai C (pasaran):",
        list(KAPASITOR_PASARAN.keys()),
        default=["10 nF", "100 nF (0.1 µF)", "1 µF", "10 µF"]
    )

    st.markdown("---")
    st.markdown("**Kapasitor Tetap (Soal 1b & 2b)**")
    C_fixed_label = st.selectbox(
        "Pilih C tetap:",
        list(KAPASITOR_PASARAN.keys()),
        index=12  # 100 nF default
    )
    C_fixed = KAPASITOR_PASARAN[C_fixed_label]

    st.markdown("---")
    st.markdown("**Rentang Frekuensi (Soal b)**")
    f_min_exp = st.slider("f_min (10^x Hz)", min_value=0, max_value=4, value=1)
    f_max_exp = st.slider("f_max (10^x Hz)", min_value=1, max_value=7, value=6)

    f_min = 10**f_min_exp
    f_max = 10**f_max_exp

    st.markdown("---")
    st.markdown(f"""
    <div style='font-family:JetBrains Mono,monospace;font-size:0.78rem;color:#64748b;'>
    Vin = {Vin} V<br>
    R   = {R/1000:.1f} kΩ<br>
    C   = {format_capacitance(C_fixed)}<br>
    f   = {f_min:.0f} – {f_max:.0f} Hz
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HEADER UTAMA
# ─────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>⚡ RC Filter Simulator</h1>
    <p>Pemodelan Fisika · Low Pass Filter & High Pass Filter · Python + Streamlit</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TEORI & RUMUS
# ─────────────────────────────────────────────
with st.expander("📐 Teori & Rumus Dasar RC Filter", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header lpf">🔵 Low Pass Filter (LPF)</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="formula-box lpf">
        <div class="formula-title">Fungsi Transfer</div>
        H(jω) = 1 / (1 + jωRC)
        </div>
        <div class="formula-box lpf">
        <div class="formula-title">Tegangan Output</div>
        Vout = Vin · Xc / √(R² + Xc²)<br>
        dimana: Xc = 1 / (2π·f·C)
        </div>
        <div class="formula-box lpf">
        <div class="formula-title">Gain (Bode)</div>
        |H(f)| = 1 / √(1 + (f/fc)²)
        </div>
        <div class="formula-box lpf">
        <div class="formula-title">Phase Shift</div>
        φ = -arctan(f/fc)  [degrees]
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="explanation-box">
        <b>Prinsip LPF:</b> Kapasitor memiliki reaktansi Xc = 1/(2πfC) yang semakin kecil ketika frekuensi naik.
        Pada frekuensi rendah → Xc ≫ R → Vout ≈ Vin (sinyal lolos).
        Pada frekuensi tinggi → Xc ≪ R → Vout ≈ 0 (sinyal diblokir).
        Kapasitor berfungsi sebagai "jembatan ke ground" untuk sinyal frekuensi tinggi.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header hpf">🟠 High Pass Filter (HPF)</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="formula-box hpf">
        <div class="formula-title">Fungsi Transfer</div>
        H(jω) = jωRC / (1 + jωRC)
        </div>
        <div class="formula-box hpf">
        <div class="formula-title">Tegangan Output</div>
        Vout = Vin · R / √(R² + Xc²)<br>
        dimana: Xc = 1 / (2π·f·C)
        </div>
        <div class="formula-box hpf">
        <div class="formula-title">Gain (Bode)</div>
        |H(f)| = (f/fc) / √(1 + (f/fc)²)
        </div>
        <div class="formula-box hpf">
        <div class="formula-title">Phase Shift</div>
        φ = 90° - arctan(f/fc)  [degrees]
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="explanation-box">
        <b>Prinsip HPF:</b> Output diambil dari resistor, bukan kapasitor.
        Pada frekuensi rendah → Xc ≫ R → drop tegangan di Xc besar → Vout ≈ 0 (sinyal diblokir).
        Pada frekuensi tinggi → Xc ≪ R → drop tegangan di R dominan → Vout ≈ Vin (sinyal lolos).
        Kapasitor berfungsi sebagai "pemblokir DC" dan penghalang frekuensi rendah.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="formula-box both">
    <div class="formula-title">⚫ Frekuensi Cutoff (sama untuk LPF & HPF)</div>
    fc = 1 / (2π·R·C)    →    Pada fc: Vout = Vin/√2 ≈ 0.707·Vin  ≡  -3 dB
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  TAB UTAMA
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🔵 LPF — Variasi C",
    "🔵 LPF — Variasi f",
    "🟠 HPF — Variasi C & f",
    "⚖️ Perbandingan LPF vs HPF"
])


# ═══════════════════════════════════════════════════
#  TAB 1: LPF — VARIASI KAPASITOR (f = 50–60 Hz)
# ═══════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header lpf">🔵 LPF · Soal 1a: Variasi Kapasitor pada f = 50–60 Hz</div>', unsafe_allow_html=True)

    if not selected_caps:
        st.warning("Pilih minimal 1 nilai kapasitor di sidebar!")
    else:
        freqs_soal1 = [50, 55, 60]
        rows = []
        for cap_label in selected_caps:
            C_val = KAPASITOR_PASARAN[cap_label]
            fc = cutoff_frequency(R, C_val)
            for f in freqs_soal1:
                vout = lpf_vout(Vin, R, C_val, f)
                gain = vout / Vin
                gain_db = to_db(gain)
                phase = phase_lpf(f, R, C_val)
                Xc = 1 / (2 * np.pi * f * C_val)
                rows.append({
                    "Kapasitor": cap_label,
                    "C (F)": C_val,
                    "f (Hz)": f,
                    "fc (Hz)": round(fc, 2),
                    "Xc (Ω)": round(Xc, 2),
                    "Vout (V)": round(vout, 4),
                    "Gain |H|": round(gain, 4),
                    "Gain (dB)": round(gain_db, 2),
                    "Phase (°)": round(phase, 2),
                })

        df1 = pd.DataFrame(rows)

        # VISUALISASI BAR
        fig1a = go.Figure()
        colors_lpf = px.colors.qualitative.Set2
        for i, cap_label in enumerate(selected_caps):
            sub = df1[df1["Kapasitor"] == cap_label]
            fig1a.add_trace(go.Bar(
                name=cap_label,
                x=[f"{r['f (Hz)']} Hz" for _, r in sub.iterrows()],
                y=sub["Vout (V)"],
                marker_color=colors_lpf[i % len(colors_lpf)],
                text=[f"{v:.3f} V" for v in sub["Vout (V)"]],
                textposition="outside",
            ))

        fig1a.add_hline(y=Vin, line_dash="dash", line_color="#ffffff",
                        annotation_text=f"Vin = {Vin}V", annotation_position="right")
        fig1a.update_layout(
            title="LPF · Vout vs Frekuensi (50–60 Hz) untuk berbagai Kapasitor",
            xaxis_title="Frekuensi (Hz)",
            yaxis_title="Vout (Volt)",
            plot_bgcolor="#0a0e1a",
            paper_bgcolor="#111827",
            font=dict(color="#e2e8f0", family="JetBrains Mono"),
            barmode="group",
            legend=dict(bgcolor="#0a0e1a", bordercolor="#1e2d45"),
            yaxis=dict(range=[0, Vin * 1.15]),
        )
        st.plotly_chart(fig1a, use_container_width=True)

        # Heatmap Gain dB
        pivot_db = df1.pivot_table(index="Kapasitor", columns="f (Hz)", values="Gain (dB)")
        fig1b = go.Figure(data=go.Heatmap(
            z=pivot_db.values,
            x=[f"{c} Hz" for c in pivot_db.columns],
            y=pivot_db.index,
            colorscale="Blues",
            text=np.round(pivot_db.values, 2),
            texttemplate="%{text} dB",
            colorbar=dict(title="Gain (dB)"),
        ))
        fig1b.update_layout(
            title="Heatmap Gain (dB) — LPF · Variasi C",
            plot_bgcolor="#0a0e1a", paper_bgcolor="#111827",
            font=dict(color="#e2e8f0", family="JetBrains Mono"),
        )
        st.plotly_chart(fig1b, use_container_width=True)

        # Tabel Data Lengkap
        st.markdown('<div class="section-header lpf">📊 Tabel Data LPF · Variasi C</div>', unsafe_allow_html=True)
        st.dataframe(
            df1[["Kapasitor", "f (Hz)", "fc (Hz)", "Xc (Ω)", "Vout (V)", "Gain (dB)", "Phase (°)"]],
            use_container_width=True, hide_index=True
        )

        # Penjelasan fisika
        st.markdown("""
        <div class="explanation-box">
        <b>📌 Analisis Hasil Soal 1a (LPF, f = 50–60 Hz):</b><br><br>
        • <b>Kapasitor kecil (pF–nF):</b> fc sangat tinggi (MHz–GHz) → f=50Hz jauh di bawah fc → Vout ≈ Vin (99%+). 
          Sinyal 50Hz dianggap "DC relatif" oleh kapasitor kecil.<br><br>
        • <b>Kapasitor sedang (100nF–1µF):</b> fc berkisar 159Hz–1.6kHz → f=50Hz masih di bawah fc → 
          Vout masih mendekati Vin namun mulai ada atenuasi ringan.<br><br>
        • <b>Kapasitor besar (10µF–100µF):</b> fc turun ke 1.6Hz–16Hz → f=50Hz sudah jauh MELEWATI fc → 
          Vout sangat kecil, sinyal 50Hz diblokir! Xc sangat kecil (hampir short circuit).<br><br>
        • <b>Perubahan 50→60 Hz:</b> Kenaikan 20% frekuensi membuat Xc turun 16.7%, 
          mengakibatkan penurunan Vout yang signifikan pada kapasitor besar.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
#  TAB 2: LPF — VARIASI FREKUENSI
# ═══════════════════════════════════════════════════
with tab2:
    st.markdown(f'<div class="section-header lpf">🔵 LPF · Soal 1b: Variasi Frekuensi (C = {C_fixed_label}, R = {R/1000:.1f}kΩ)</div>', unsafe_allow_html=True)

    fc_lpf = cutoff_frequency(R, C_fixed)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("fc (Cutoff)", f"{fc_lpf:.1f} Hz")
    with col_m2:
        st.metric("Vin", f"{Vin} V")
    with col_m3:
        vout_at_fc = lpf_vout(Vin, R, C_fixed, fc_lpf)
        st.metric("Vout @ fc", f"{vout_at_fc:.3f} V")
    with col_m4:
        st.metric("Gain @ fc", f"{to_db(vout_at_fc/Vin):.2f} dB")

    # Data frekuensi sweep
    f_arr = np.logspace(np.log10(f_min), np.log10(f_max), 500)
    vout_arr = lpf_vout(Vin, R, C_fixed, f_arr)
    gain_arr = vout_arr / Vin
    gain_db_arr = to_db(gain_arr)
    phase_arr = phase_lpf(f_arr, R, C_fixed)

    # Bode Plot (Gain + Phase)
    fig2 = make_subplots(rows=2, cols=1,
                         subplot_titles=("Bode Plot — Gain (dB)", "Bode Plot — Phase (°)"),
                         vertical_spacing=0.12)

    fig2.add_trace(go.Scatter(
        x=f_arr, y=gain_db_arr,
        mode='lines', name='Gain LPF',
        line=dict(color='#00d4ff', width=2.5)
    ), row=1, col=1)

    # -3dB line
    fig2.add_hline(y=-3, line_dash="dot", line_color="#ff6b35",
                   annotation_text="-3 dB (Cutoff)", row=1, col=1)
    fig2.add_vline(x=fc_lpf, line_dash="dash", line_color="#ffd700",
                   annotation_text=f"fc={fc_lpf:.1f}Hz")

    fig2.add_trace(go.Scatter(
        x=f_arr, y=phase_arr,
        mode='lines', name='Phase LPF',
        line=dict(color='#7c3aed', width=2.5)
    ), row=2, col=1)
    fig2.add_hline(y=-45, line_dash="dot", line_color="#ff6b35",
                   annotation_text="-45° @ fc", row=2, col=1)

    for row in [1, 2]:
        fig2.update_xaxes(type="log", title_text="Frekuensi (Hz)", row=row, col=1,
                          gridcolor="#1e2d45", color="#e2e8f0")
    fig2.update_yaxes(title_text="Gain (dB)", row=1, col=1, gridcolor="#1e2d45", color="#e2e8f0")
    fig2.update_yaxes(title_text="Phase (°)", row=2, col=1, gridcolor="#1e2d45", color="#e2e8f0")

    fig2.update_layout(
        height=600,
        plot_bgcolor="#0a0e1a", paper_bgcolor="#111827",
        font=dict(color="#e2e8f0", family="JetBrains Mono"),
        legend=dict(bgcolor="#0a0e1a", bordercolor="#1e2d45"),
        showlegend=True,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Vout linear
    fig2b = go.Figure()
    fig2b.add_trace(go.Scatter(
        x=f_arr, y=vout_arr,
        mode='lines', name='Vout LPF',
        line=dict(color='#00d4ff', width=2.5),
        fill='tozeroy', fillcolor='rgba(0,212,255,0.08)'
    ))
    fig2b.add_hline(y=Vin * 0.707, line_dash="dash", line_color="#ffd700",
                    annotation_text=f"0.707·Vin = {Vin*0.707:.3f}V")
    fig2b.add_vline(x=fc_lpf, line_dash="dash", line_color="#ff6b35",
                    annotation_text=f"fc = {fc_lpf:.1f} Hz")
    fig2b.update_xaxes(type="log", title_text="Frekuensi (Hz)", gridcolor="#1e2d45")
    fig2b.update_yaxes(title_text="Vout (V)", gridcolor="#1e2d45")
    fig2b.update_layout(
        title="LPF · Vout vs Frekuensi (Skala Log)",
        plot_bgcolor="#0a0e1a", paper_bgcolor="#111827",
        font=dict(color="#e2e8f0", family="JetBrains Mono"),
    )
    st.plotly_chart(fig2b, use_container_width=True)

    # Tabel sampel frekuensi
    sample_freqs = [10, 50, 60, 100, 1e3, 10e3, 100e3, 1e6]
    rows2 = []
    for f in sample_freqs:
        if f_min <= f <= f_max:
            Xc = 1 / (2 * np.pi * f * C_fixed)
            vout = lpf_vout(Vin, R, C_fixed, f)
            rows2.append({
                "f (Hz)": f,
                "Xc (Ω)": round(Xc, 3),
                "Z_total (Ω)": round(np.sqrt(R**2 + Xc**2), 3),
                "Vout (V)": round(vout, 4),
                "Gain |H|": round(vout/Vin, 4),
                "Gain (dB)": round(to_db(vout/Vin), 2),
                "Phase (°)": round(phase_lpf(f, R, C_fixed), 2),
                "Keterangan": "⬅ fc" if abs(f - fc_lpf) < fc_lpf * 0.1 else (
                    "Lolos ✅" if f < fc_lpf else "Teredam ❌")
            })

    if rows2:
        df2 = pd.DataFrame(rows2)
        st.markdown('<div class="section-header lpf">📊 Tabel Data Sampling LPF · Variasi f</div>', unsafe_allow_html=True)
        st.dataframe(df2, use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div class="explanation-box">
    <b>📌 Analisis Soal 1b — LPF, C = {C_fixed_label}:</b><br><br>
    • <b>fc = {fc_lpf:.2f} Hz</b> adalah batas frekuensi -3dB. Sinyal di bawah fc "lolos", di atas fc "teredam".<br><br>
    • <b>f ≪ fc (stopband tinggi):</b> Xc ≫ R → kapasitor seperti open circuit → hampir semua tegangan jatuh di Xc → 
      Vout ≈ Vin. Sinyal melewati filter dengan atenuasi minimal.<br><br>
    • <b>f = fc:</b> Xc = R → Vout = Vin/√2 = {Vin/np.sqrt(2):.3f}V → Gain = -3dB → Phase = -45°. 
      Ini adalah titik "half-power" karena daya = V²/R turun menjadi setengah.<br><br>
    • <b>f ≫ fc (passband rendah):</b> Xc → 0 → kapasitor short circuit ke ground → Vout → 0. 
      Slope atenuasi = -20 dB/dekade (first-order filter).<br><br>
    • <b>Phase shift:</b> Berubah dari 0° (frekuensi sangat rendah) hingga -90° (frekuensi sangat tinggi), 
      dengan -45° tepat di fc. Ini berarti kapasitor menyebabkan sinyal output tertinggal (lag) dari input.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
#  TAB 3: HPF
# ═══════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header hpf">🟠 HPF · Soal 2a & 2b</div>', unsafe_allow_html=True)

    sub_tab1, sub_tab2 = st.tabs(["2a · Variasi C (f = 50–60 Hz)", "2b · Variasi Frekuensi"])

    with sub_tab1:
        if not selected_caps:
            st.warning("Pilih minimal 1 kapasitor di sidebar!")
        else:
            freqs_hpf1 = [50, 55, 60]
            rows_hpf = []
            for cap_label in selected_caps:
                C_val = KAPASITOR_PASARAN[cap_label]
                fc = cutoff_frequency(R, C_val)
                for f in freqs_hpf1:
                    vout = hpf_vout(Vin, R, C_val, f)
                    gain = vout / Vin
                    gain_db = to_db(gain)
                    phase = phase_hpf(f, R, C_val)
                    Xc = 1 / (2 * np.pi * f * C_val)
                    rows_hpf.append({
                        "Kapasitor": cap_label,
                        "f (Hz)": f,
                        "fc (Hz)": round(fc, 2),
                        "Xc (Ω)": round(Xc, 2),
                        "Vout (V)": round(vout, 4),
                        "Gain (dB)": round(gain_db, 2),
                        "Phase (°)": round(phase, 2),
                    })

            df_hpf1 = pd.DataFrame(rows_hpf)

            fig3a = go.Figure()
            colors_hpf = px.colors.qualitative.Vivid
            for i, cap_label in enumerate(selected_caps):
                sub = df_hpf1[df_hpf1["Kapasitor"] == cap_label]
                fig3a.add_trace(go.Bar(
                    name=cap_label,
                    x=[f"{r['f (Hz)']} Hz" for _, r in sub.iterrows()],
                    y=sub["Vout (V)"],
                    marker_color=colors_hpf[i % len(colors_hpf)],
                    text=[f"{v:.3f} V" for v in sub["Vout (V)"]],
                    textposition="outside",
                ))
            fig3a.add_hline(y=Vin, line_dash="dash", line_color="#ffffff",
                            annotation_text=f"Vin = {Vin}V")
            fig3a.update_layout(
                title="HPF · Vout vs Frekuensi (50–60 Hz)",
                xaxis_title="Frekuensi (Hz)",
                yaxis_title="Vout (Volt)",
                plot_bgcolor="#0a0e1a", paper_bgcolor="#111827",
                font=dict(color="#e2e8f0", family="JetBrains Mono"),
                barmode="group",
                yaxis=dict(range=[0, Vin * 1.15]),
            )
            st.plotly_chart(fig3a, use_container_width=True)
            st.dataframe(df_hpf1, use_container_width=True, hide_index=True)

            st.markdown("""
            <div class="explanation-box">
            <b>📌 Analisis Soal 2a (HPF, f = 50–60 Hz):</b><br><br>
            • <b>Kapasitor kecil (pF–nF):</b> fc sangat tinggi → f=50Hz jauh di bawah fc → Vout ≈ 0.
              Xc sangat besar → hampir semua tegangan jatuh di Xc → sinyal diblokir total.<br><br>
            • <b>Kapasitor sedang–besar (µF):</b> fc rendah → f=50Hz di atas fc → Vout mendekati Vin.
              HPF dengan kapasitor besar justru meloloskan sinyal 50Hz!<br><br>
            • <b>Perbedaan mencolok vs LPF:</b> Pola Vout HPF TERBALIK dibanding LPF.
              Kapasitor yang membuat LPF "lolos" justru membuat HPF "blokir", dan sebaliknya.
            </div>
            """, unsafe_allow_html=True)

    with sub_tab2:
        st.markdown(f'<div class="section-header hpf">HPF · Soal 2b: Variasi Frekuensi (C = {C_fixed_label})</div>', unsafe_allow_html=True)

        fc_hpf = cutoff_frequency(R, C_fixed)

        col_h1, col_h2, col_h3, col_h4 = st.columns(4)
        with col_h1:
            st.metric("fc (Cutoff)", f"{fc_hpf:.1f} Hz")
        with col_h2:
            st.metric("Vin", f"{Vin} V")
        with col_h3:
            vout_hpf_fc = hpf_vout(Vin, R, C_fixed, fc_hpf)
            st.metric("Vout @ fc", f"{vout_hpf_fc:.3f} V")
        with col_h4:
            st.metric("Gain @ fc", f"{to_db(vout_hpf_fc/Vin):.2f} dB")

        f_arr2 = np.logspace(np.log10(f_min), np.log10(f_max), 500)
        vout_hpf_arr = hpf_vout(Vin, R, C_fixed, f_arr2)
        gain_hpf_arr = vout_hpf_arr / Vin
        gain_hpf_db_arr = to_db(gain_hpf_arr)
        phase_hpf_arr = phase_hpf(f_arr2, R, C_fixed)

        fig3b = make_subplots(rows=2, cols=1,
                              subplot_titles=("Bode Plot HPF — Gain (dB)", "Bode Plot HPF — Phase (°)"),
                              vertical_spacing=0.12)
        fig3b.add_trace(go.Scatter(
            x=f_arr2, y=gain_hpf_db_arr,
            mode='lines', name='Gain HPF',
            line=dict(color='#ff6b35', width=2.5)
        ), row=1, col=1)
        fig3b.add_hline(y=-3, line_dash="dot", line_color="#00d4ff",
                        annotation_text="-3 dB", row=1, col=1)
        fig3b.add_vline(x=fc_hpf, line_dash="dash", line_color="#ffd700",
                        annotation_text=f"fc={fc_hpf:.1f}Hz")

        fig3b.add_trace(go.Scatter(
            x=f_arr2, y=phase_hpf_arr,
            mode='lines', name='Phase HPF',
            line=dict(color='#7c3aed', width=2.5)
        ), row=2, col=1)
        fig3b.add_hline(y=45, line_dash="dot", line_color="#ff6b35",
                        annotation_text="+45° @ fc", row=2, col=1)

        for row in [1, 2]:
            fig3b.update_xaxes(type="log", title_text="Frekuensi (Hz)", row=row, col=1,
                                gridcolor="#1e2d45", color="#e2e8f0")
        fig3b.update_yaxes(title_text="Gain (dB)", row=1, col=1, gridcolor="#1e2d45", color="#e2e8f0")
        fig3b.update_yaxes(title_text="Phase (°)", row=2, col=1, gridcolor="#1e2d45", color="#e2e8f0")
        fig3b.update_layout(
            height=600,
            plot_bgcolor="#0a0e1a", paper_bgcolor="#111827",
            font=dict(color="#e2e8f0", family="JetBrains Mono"),
        )
        st.plotly_chart(fig3b, use_container_width=True)


# ═══════════════════════════════════════════════════
#  TAB 4: PERBANDINGAN LPF vs HPF
# ═══════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header both">⚖️ Perbandingan LPF vs HPF</div>', unsafe_allow_html=True)

    fc_val = cutoff_frequency(R, C_fixed)
    f_comp = np.logspace(np.log10(f_min), np.log10(f_max), 800)
    vout_lpf = lpf_vout(Vin, R, C_fixed, f_comp)
    vout_hpf = hpf_vout(Vin, R, C_fixed, f_comp)
    gain_lpf_db = to_db(vout_lpf / Vin)
    gain_hpf_db = to_db(vout_hpf / Vin)
    ph_lpf = phase_lpf(f_comp, R, C_fixed)
    ph_hpf = phase_hpf(f_comp, R, C_fixed)

    # Superimposed Bode Plot
    fig4a = make_subplots(rows=2, cols=1,
                          subplot_titles=("Gain (dB) — LPF vs HPF", "Phase (°) — LPF vs HPF"),
                          vertical_spacing=0.12)
    fig4a.add_trace(go.Scatter(x=f_comp, y=gain_lpf_db, name="LPF Gain",
                               line=dict(color="#00d4ff", width=2.5)), row=1, col=1)
    fig4a.add_trace(go.Scatter(x=f_comp, y=gain_hpf_db, name="HPF Gain",
                               line=dict(color="#ff6b35", width=2.5)), row=1, col=1)
    fig4a.add_hline(y=-3, line_dash="dot", line_color="#ffd700",
                    annotation_text="-3 dB", row=1, col=1)
    fig4a.add_vline(x=fc_val, line_dash="dash", line_color="#7c3aed",
                    annotation_text=f"fc={fc_val:.1f}Hz")

    fig4a.add_trace(go.Scatter(x=f_comp, y=ph_lpf, name="LPF Phase",
                               line=dict(color="#00d4ff", width=2, dash="dot")), row=2, col=1)
    fig4a.add_trace(go.Scatter(x=f_comp, y=ph_hpf, name="HPF Phase",
                               line=dict(color="#ff6b35", width=2, dash="dot")), row=2, col=1)
    fig4a.add_hline(y=-45, line_dash="dot", line_color="#00d4ff",
                    annotation_text="LPF: -45°", row=2, col=1)
    fig4a.add_hline(y=45, line_dash="dot", line_color="#ff6b35",
                    annotation_text="HPF: +45°", row=2, col=1)

    for row in [1, 2]:
        fig4a.update_xaxes(type="log", title_text="Frekuensi (Hz)", row=row, col=1,
                            gridcolor="#1e2d45", color="#e2e8f0")
    fig4a.update_yaxes(title_text="Gain (dB)", row=1, col=1, gridcolor="#1e2d45")
    fig4a.update_yaxes(title_text="Phase (°)", row=2, col=1, gridcolor="#1e2d45")
    fig4a.update_layout(
        height=650,
        plot_bgcolor="#0a0e1a", paper_bgcolor="#111827",
        font=dict(color="#e2e8f0", family="JetBrains Mono"),
        legend=dict(bgcolor="#0a0e1a", bordercolor="#1e2d45"),
    )
    st.plotly_chart(fig4a, use_container_width=True)

    # Vout Overlay
    fig4b = go.Figure()
    fig4b.add_trace(go.Scatter(x=f_comp, y=vout_lpf, name="Vout LPF",
                               line=dict(color="#00d4ff", width=2.5),
                               fill='tozeroy', fillcolor='rgba(0,212,255,0.06)'))
    fig4b.add_trace(go.Scatter(x=f_comp, y=vout_hpf, name="Vout HPF",
                               line=dict(color="#ff6b35", width=2.5),
                               fill='tozeroy', fillcolor='rgba(255,107,53,0.06)'))

    # Verifikasi LPF + HPF dengan Vin
    vout_sum = np.sqrt(vout_lpf**2 + vout_hpf**2)
    fig4b.add_trace(go.Scatter(x=f_comp, y=vout_sum,
                               name="√(Vout_LPF² + Vout_HPF²)",
                               line=dict(color="#7c3aed", width=1.5, dash='dot')))
    fig4b.add_hline(y=Vin, line_dash="dash", line_color="#ffd700",
                    annotation_text=f"Vin = {Vin}V")
    fig4b.add_vline(x=fc_val, line_dash="dash", line_color="#aaa")
    fig4b.update_xaxes(type="log", title_text="Frekuensi (Hz)", gridcolor="#1e2d45")
    fig4b.update_yaxes(title_text="Vout (V)", gridcolor="#1e2d45")
    fig4b.update_layout(
        title="Vout LPF vs HPF (dan verifikasi: √(Vlpf²+Vhpf²) = Vin)",
        plot_bgcolor="#0a0e1a", paper_bgcolor="#111827",
        font=dict(color="#e2e8f0", family="JetBrains Mono"),
        legend=dict(bgcolor="#0a0e1a", bordercolor="#1e2d45"),
    )
    st.plotly_chart(fig4b, use_container_width=True)

    # ─── ANIMASI SINYAL WAKTU ───
    st.markdown('<div class="section-header both">🎬 Animasi: Sinyal Domain Waktu</div>', unsafe_allow_html=True)

    f_demo = st.slider("Pilih frekuensi demo (Hz):",
                       min_value=float(f_min), max_value=float(min(f_max, 1e6)),
                       value=float(fc_val), format="%.1f")

    t = np.linspace(0, 3 / max(f_demo, 1), 1000)
    v_in_t = Vin * np.sin(2 * np.pi * f_demo * t)

    vout_lpf_f = float(lpf_vout(Vin, R, C_fixed, f_demo))
    vout_hpf_f = float(hpf_vout(Vin, R, C_fixed, f_demo))
    phi_lpf_rad = np.radians(phase_lpf(f_demo, R, C_fixed))
    phi_hpf_rad = np.radians(phase_hpf(f_demo, R, C_fixed))

    v_lpf_t = vout_lpf_f * np.sin(2 * np.pi * f_demo * t + phi_lpf_rad)
    v_hpf_t = vout_hpf_f * np.sin(2 * np.pi * f_demo * t + phi_hpf_rad)

    fig_anim = go.Figure()
    fig_anim.add_trace(go.Scatter(
        x=t * 1000, y=v_in_t,
        mode='lines', name='Vin',
        line=dict(color='#ffd700', width=2.5, dash='dot')
    ))
    fig_anim.add_trace(go.Scatter(
        x=t * 1000, y=v_lpf_t,
        mode='lines', name=f'Vout LPF ({vout_lpf_f:.3f}V)',
        line=dict(color='#00d4ff', width=2.5)
    ))
    fig_anim.add_trace(go.Scatter(
        x=t * 1000, y=v_hpf_t,
        mode='lines', name=f'Vout HPF ({vout_hpf_f:.3f}V)',
        line=dict(color='#ff6b35', width=2.5)
    ))
    fig_anim.add_hline(y=0, line_color="#333", line_width=1)
    fig_anim.update_xaxes(title_text="Waktu (ms)", gridcolor="#1e2d45")
    fig_anim.update_yaxes(title_text="Tegangan (V)", gridcolor="#1e2d45")
    fig_anim.update_layout(
        title=f"Sinyal Domain Waktu pada f = {f_demo:.1f} Hz  |  fc = {fc_val:.1f} Hz",
        plot_bgcolor="#0a0e1a", paper_bgcolor="#111827",
        font=dict(color="#e2e8f0", family="JetBrains Mono"),
        legend=dict(bgcolor="#0a0e1a", bordercolor="#1e2d45"),
    )
    st.plotly_chart(fig_anim, use_container_width=True)

    # Kolom metrik perbandingan
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div class="formula-box lpf">
        <div class="formula-title">🔵 LPF @ f = {f_demo:.1f} Hz</div>
        Vout = <b>{vout_lpf_f:.4f} V</b><br>
        Gain = <b>{to_db(vout_lpf_f/Vin):.2f} dB</b><br>
        Phase = <b>{np.degrees(phi_lpf_rad):.2f}°</b><br>
        Status: {'✅ Passband (f < fc)' if f_demo < fc_val else ('⚡ @ Cutoff' if abs(f_demo-fc_val)<fc_val*0.05 else '❌ Stopband (f > fc)')}
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="formula-box hpf">
        <div class="formula-title">🟠 HPF @ f = {f_demo:.1f} Hz</div>
        Vout = <b>{vout_hpf_f:.4f} V</b><br>
        Gain = <b>{to_db(vout_hpf_f/Vin):.2f} dB</b><br>
        Phase = <b>{np.degrees(phi_hpf_rad):.2f}°</b><br>
        Status: {'❌ Stopband (f < fc)' if f_demo < fc_val else ('⚡ @ Cutoff' if abs(f_demo-fc_val)<fc_val*0.05 else '✅ Passband (f > fc)')}
        </div>
        """, unsafe_allow_html=True)

    # ─── TABEL PERBANDINGAN RUMUS ───
    st.markdown('<div class="section-header both">📐 Perbandingan Rumus & Perilaku</div>', unsafe_allow_html=True)

    comp_data = {
        "Aspek": [
            "Posisi Output", "Transfer Function H(jω)", "Gain |H(f)|",
            "Phase Shift φ", "Perilaku f→0", "Perilaku f→∞",
            "Vout @ f=fc", "Phase @ f=fc", "Slope Atenuasi",
            "Aplikasi Umum"
        ],
        "LPF 🔵": [
            "Dari kapasitor (C) ke ground",
            "1 / (1 + jωRC)",
            "1 / √(1 + (f/fc)²)",
            "−arctan(f/fc)  [0° → −90°]",
            "Xc→∞, Vout→Vin (lolos ✅)",
            "Xc→0, Vout→0 (blokir ❌)",
            f"Vin/√2 = {Vin/np.sqrt(2):.3f} V",
            "−45°  (sinyal lag)",
            "+20 dB/dekade naik ke rendah",
            "Audio bass, power supply smoothing, anti-aliasing"
        ],
        "HPF 🟠": [
            "Dari resistor (R) ke ground",
            "jωRC / (1 + jωRC)",
            "(f/fc) / √(1 + (f/fc)²)",
            "90° − arctan(f/fc)  [+90° → 0°]",
            "Xc→∞, Vout→0 (blokir ❌)",
            "Xc→0, Vout→Vin (lolos ✅)",
            f"Vin/√2 = {Vin/np.sqrt(2):.3f} V",
            "+45°  (sinyal lead)",
            "+20 dB/dekade naik ke tinggi",
            "Audio treble, DC blocking, edge detection"
        ],
    }
    df_comp = pd.DataFrame(comp_data)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    # Penjelasan perbandingan akhir
    st.markdown(f"""
    <div class="explanation-box">
    <b>📌 Kesimpulan Perbandingan LPF vs HPF:</b><br><br>

    <b>1. Dualitas Matematika:</b> LPF dan HPF adalah filter yang <i>komplementer</i>. 
    Gain LPF + Gain HPF (dalam kuadrat) = 1, karena:<br>
    &nbsp;&nbsp;&nbsp;|H_LPF|² + |H_HPF|² = [1/(1+(f/fc)²)] + [(f/fc)²/(1+(f/fc)²)] = 1<br>
    Artinya: energi sinyal yang "ditolak" LPF persis adalah yang "diloloskan" HPF.<br><br>

    <b>2. Frekuensi Cutoff Identik:</b> Keduanya memiliki fc = 1/(2πRC) = <b>{fc_val:.2f} Hz</b> 
    dengan konfigurasi R dan C yang sama. Di titik fc, keduanya menghasilkan Vout = Vin/√2 = <b>{Vin/np.sqrt(2):.3f} V</b> (-3dB).<br><br>

    <b>3. Perbedaan Phase:</b> LPF menghasilkan <i>phase lag</i> (sinyal output tertinggal dari input), 
    sementara HPF menghasilkan <i>phase lead</i> (sinyal output mendahului input). 
    Keduanya memiliki pergeseran 45° tepat di fc.<br><br>

    <b>4. Perbedaan Fisik:</b> LPF mengambil output dari kapasitor (elemen reaktif), HPF dari resistor (elemen resistif). 
    Menukar posisi output mengubah filter secara fundamental.<br><br>

    <b>5. Verifikasi Kirchhoff:</b> √(Vout_LPF² + Vout_HPF²) = Vin (berlaku karena dua komponen orthogonal dalam impedansi kompleks).
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#64748b;font-family:JetBrains Mono,monospace;font-size:0.78rem;padding:1rem;'>
⚡ RC Filter Simulator · Pemodelan Fisika · Python + Streamlit<br>
Rumus: LPF Vout = Vin·Xc/Z | HPF Vout = Vin·R/Z | Z = √(R²+Xc²) | Xc = 1/(2πfC) | fc = 1/(2πRC)
</div>
""", unsafe_allow_html=True)
