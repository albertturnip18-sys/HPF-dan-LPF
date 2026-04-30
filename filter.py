"""
========================================================
  SIMULASI RC FILTER - LOW PASS & HIGH PASS FILTER
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
    <p>Low Pass Filter & High Pass Filter · Analisis Vout & Frekuensi</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  GAMBAR RANGKAIAN LPF & HPF
# ─────────────────────────────────────────────
st.markdown('<div class="section-header both">🔌 Gambar Rangkaian RC Filter</div>', unsafe_allow_html=True)

col_circ1, col_circ2 = st.columns(2)

with col_circ1:
    st.markdown("""
    <div class="formula-box lpf" style="padding:1rem;">
    <div class="formula-title">🔵 LOW PASS FILTER (LPF) — Output dari Kapasitor</div>
    <svg viewBox="0 0 420 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;background:#0a0e1a;border-radius:8px;padding:4px;">
      <!-- Vin label & source -->
      <text x="14" y="85" fill="#ffd700" font-size="13" font-family="monospace">Vin</text>
      <line x1="40" y1="80" x2="80" y2="80" stroke="#ffd700" stroke-width="2"/>
      <!-- Resistor R (zig-zag) -->
      <polyline points="80,80 92,68 104,92 116,68 128,92 140,68 152,92 164,80" fill="none" stroke="#00d4ff" stroke-width="2.5"/>
      <text x="110" y="58" fill="#00d4ff" font-size="13" font-family="monospace" text-anchor="middle">R</text>
      <!-- Wire after R to node -->
      <line x1="164" y1="80" x2="240" y2="80" stroke="#e2e8f0" stroke-width="2"/>
      <!-- Node dot -->
      <circle cx="240" cy="80" r="4" fill="#e2e8f0"/>
      <!-- Capacitor C (vertical plates) -->
      <line x1="240" y1="80" x2="240" y2="115" stroke="#e2e8f0" stroke-width="2"/>
      <line x1="215" y1="115" x2="265" y2="115" stroke="#7c3aed" stroke-width="3"/>
      <line x1="215" y1="125" x2="265" y2="125" stroke="#7c3aed" stroke-width="3"/>
      <line x1="240" y1="125" x2="240" y2="160" stroke="#e2e8f0" stroke-width="2"/>
      <text x="275" y="123" fill="#7c3aed" font-size="13" font-family="monospace">C</text>
      <!-- Ground -->
      <line x1="220" y1="160" x2="260" y2="160" stroke="#e2e8f0" stroke-width="2"/>
      <line x1="228" y1="166" x2="252" y2="166" stroke="#e2e8f0" stroke-width="1.5"/>
      <line x1="235" y1="172" x2="245" y2="172" stroke="#e2e8f0" stroke-width="1"/>
      <!-- Bottom wire back to Vin- -->
      <line x1="40" y1="160" x2="220" y2="160" stroke="#e2e8f0" stroke-width="2"/>
      <line x1="40" y1="80" x2="40" y2="160" stroke="#ffd700" stroke-width="2"/>
      <!-- Vout arrow & label -->
      <line x1="240" y1="80" x2="360" y2="80" stroke="#e2e8f0" stroke-width="2"/>
      <line x1="360" y1="80" x2="360" y2="160" stroke="#00d4ff" stroke-width="2" stroke-dasharray="5,3"/>
      <text x="370" y="85" fill="#00d4ff" font-size="13" font-family="monospace">Vout</text>
      <text x="370" y="100" fill="#7c3aed" font-size="11" font-family="monospace">(dari C)</text>
      <line x1="300" y1="160" x2="360" y2="160" stroke="#e2e8f0" stroke-width="2"/>
      <!-- Arrow head for Vout -->
      <polygon points="362,78 356,73 356,83" fill="#00d4ff"/>
    </svg>
    <div style="font-size:0.82rem;color:#94a3b8;margin-top:0.5rem;line-height:1.6;">
    📌 <b>Vout diukur pada kapasitor (C)</b>.<br>
    Frekuensi rendah → Xc besar → Vout ≈ Vin ✅<br>
    Frekuensi tinggi → Xc kecil → Vout → 0 ❌
    </div>
    </div>
    """, unsafe_allow_html=True)

with col_circ2:
    st.markdown("""
    <div class="formula-box hpf" style="padding:1rem;">
    <div class="formula-title">🟠 HIGH PASS FILTER (HPF) — Output dari Resistor</div>
    <svg viewBox="0 0 420 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;background:#0a0e1a;border-radius:8px;padding:4px;">
      <!-- Vin label & source -->
      <text x="14" y="85" fill="#ffd700" font-size="13" font-family="monospace">Vin</text>
      <line x1="40" y1="80" x2="80" y2="80" stroke="#ffd700" stroke-width="2"/>
      <!-- Capacitor C (horizontal plates) -->
      <line x1="80" y1="80" x2="110" y2="80" stroke="#e2e8f0" stroke-width="2"/>
      <line x1="110" y1="65" x2="110" y2="95" stroke="#7c3aed" stroke-width="3"/>
      <line x1="120" y1="65" x2="120" y2="95" stroke="#7c3aed" stroke-width="3"/>
      <line x1="120" y1="80" x2="160" y2="80" stroke="#e2e8f0" stroke-width="2"/>
      <text x="95" y="58" fill="#7c3aed" font-size="13" font-family="monospace" text-anchor="middle">C</text>
      <!-- Wire after C to node -->
      <line x1="160" y1="80" x2="240" y2="80" stroke="#e2e8f0" stroke-width="2"/>
      <!-- Node dot -->
      <circle cx="240" cy="80" r="4" fill="#e2e8f0"/>
      <!-- Resistor R (vertical zig-zag) -->
      <line x1="240" y1="80" x2="240" y2="100" stroke="#e2e8f0" stroke-width="2"/>
      <polyline points="240,100 228,110 252,120 228,130 252,140 240,148" fill="none" stroke="#ff6b35" stroke-width="2.5"/>
      <line x1="240" y1="148" x2="240" y2="160" stroke="#e2e8f0" stroke-width="2"/>
      <text x="260" y="130" fill="#ff6b35" font-size="13" font-family="monospace">R</text>
      <!-- Ground -->
      <line x1="220" y1="160" x2="260" y2="160" stroke="#e2e8f0" stroke-width="2"/>
      <line x1="228" y1="166" x2="252" y2="166" stroke="#e2e8f0" stroke-width="1.5"/>
      <line x1="235" y1="172" x2="245" y2="172" stroke="#e2e8f0" stroke-width="1"/>
      <!-- Bottom wire back to Vin- -->
      <line x1="40" y1="160" x2="220" y2="160" stroke="#e2e8f0" stroke-width="2"/>
      <line x1="40" y1="80" x2="40" y2="160" stroke="#ffd700" stroke-width="2"/>
      <!-- Vout arrow & label (from R node) -->
      <line x1="240" y1="80" x2="360" y2="80" stroke="#e2e8f0" stroke-width="2"/>
      <line x1="360" y1="80" x2="360" y2="160" stroke="#ff6b35" stroke-width="2" stroke-dasharray="5,3"/>
      <text x="370" y="85" fill="#ff6b35" font-size="13" font-family="monospace">Vout</text>
      <text x="370" y="100" fill="#ff6b35" font-size="11" font-family="monospace">(dari R)</text>
      <line x1="300" y1="160" x2="360" y2="160" stroke="#e2e8f0" stroke-width="2"/>
      <!-- Arrow head for Vout -->
      <polygon points="362,78 356,73 356,83" fill="#ff6b35"/>
    </svg>
    <div style="font-size:0.82rem;color:#94a3b8;margin-top:0.5rem;line-height:1.6;">
    📌 <b>Vout diukur pada resistor (R)</b>.<br>
    Frekuensi rendah → Xc besar → Vout → 0 ❌<br>
    Frekuensi tinggi → Xc kecil → Vout ≈ Vin ✅
    </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="explanation-box" style="margin-bottom:1.2rem;">
<b>🔑 Kunci Perbedaan Rangkaian LPF vs HPF:</b><br><br>
Kedua rangkaian menggunakan komponen <b>R dan C yang sama</b> — yang berbeda hanyalah <b>di mana Vout diukur</b>.<br>
▸ <b>LPF</b>: Vout diambil dari <span style="color:#7c3aed"><b>kapasitor</b></span> → tegangan output tergantung reaktansi Xc yang berubah terhadap frekuensi.<br>
▸ <b>HPF</b>: Vout diambil dari <span style="color:#ff6b35"><b>resistor</b></span> → tegangan output tergantung drop tegangan di R yang juga berubah seiring frekuensi.<br>
Perubahan posisi output inilah yang menghasilkan perilaku filter yang <b>berlawanan secara fundamental</b>.
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
        <b>Apa yang terjadi pada Vout LPF ketika frekuensi berubah?</b><br><br>
        Vout pada LPF diambil dari kapasitor. Rumus Vout = Vin · Xc / √(R² + Xc²) menunjukkan bahwa
        <b>Vout berbanding lurus dengan Xc</b>. Karena Xc = 1/(2πfC), ketika frekuensi naik, Xc turun,
        sehingga Vout pun turun.<br><br>
        ▸ <b>f rendah (f ≪ fc):</b> Xc ≫ R → penyebut ≈ Xc → Vout ≈ Vin (sinyal lolos penuh)<br>
        ▸ <b>f = fc:</b> Xc = R → Vout = Vin/√2 = 0.707·Vin (titik -3 dB)<br>
        ▸ <b>f tinggi (f ≫ fc):</b> Xc → 0 → Vout → 0 (sinyal teredam habis)
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
        <b>Apa yang terjadi pada Vout HPF ketika frekuensi berubah?</b><br><br>
        Vout pada HPF diambil dari resistor. Rumus Vout = Vin · R / √(R² + Xc²) menunjukkan bahwa
        <b>Vout berbanding terbalik dengan Xc</b>. Ketika frekuensi naik, Xc turun, penyebut mengecil,
        sehingga Vout naik mendekati Vin.<br><br>
        ▸ <b>f rendah (f ≪ fc):</b> Xc ≫ R → penyebut ≈ Xc → Vout ≈ 0 (sinyal diblokir)<br>
        ▸ <b>f = fc:</b> Xc = R → Vout = Vin/√2 = 0.707·Vin (titik -3 dB)<br>
        ▸ <b>f tinggi (f ≫ fc):</b> Xc → 0 → penyebut ≈ R → Vout ≈ Vin (sinyal lolos penuh)
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
                phase = phase_
