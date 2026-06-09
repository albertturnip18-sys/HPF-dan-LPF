"""
GNC Dashboard — Streamlit
Sintesis Green-Nanoporous Carbon dari Biomassa Kulit Durian Kota Medan
Berisi: (1) Flowchart sintesis 10 tahap, (2) 5 grafik elektrokimia Lampiran 2

Cara menjalankan:
    pip install streamlit plotly pandas numpy
    streamlit run gnc_dashboard.py
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GNC Kulit Durian — Dashboard",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #fafaf8; }
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
  h1 { font-size: 1.55rem !important; font-weight: 600; color: #1a1a1a; }
  h2 { font-size: 1.15rem !important; font-weight: 600; color: #2d2d2d;
       border-bottom: 2px solid #e0e0e0; padding-bottom: 6px; }
  h3 { font-size: 1.0rem !important; font-weight: 600; color: #3a3a3a; }
  .metric-card {
      background: white; border: 1px solid #e5e5e5; border-radius: 10px;
      padding: 14px 18px; text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  }
  .metric-val   { font-size: 1.55rem; font-weight: 700; color: #185FA5; }
  .metric-lbl   { font-size: 0.78rem; color: #666; margin-top: 2px; }
  .metric-delta { font-size: 0.78rem; color: #1D9E75; font-weight: 600; }
  .step-badge {
      display: inline-block; padding: 3px 10px; border-radius: 20px;
      font-size: 0.75rem; font-weight: 600; margin-right: 6px;
  }
  .caption { font-size: 0.82rem; color: #666; font-style: italic;
             text-align: center; margin-top: 4px; }
  .info-box {
      background: #EBF4FF; border-left: 4px solid #185FA5;
      border-radius: 6px; padding: 10px 14px; margin: 8px 0;
      font-size: 0.88rem; color: #1a1a1a;
  }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔋 GNC Dashboard")
    st.markdown("**Sintesis Green-Nanoporous Carbon**  \nBiomassa *Durio zibethinus* Kota Medan")
    st.divider()
    page = st.radio(
        "Navigasi",
        ["🏠 Ringkasan", "🔄 Flowchart Sintesis", "📊 Grafik Elektrokimia"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Parameter Kunci GNC**")
    st.caption("Luas permukaan BET: **1.623 m²/g**")
    st.caption("Kapasitansi spesifik: **318 F/g** @ 1 A/g")
    st.caption("Retensi siklus: **96,3%** @ 10.000 siklus")
    st.caption("ESR: **0,42 Ω** (vs KAC: 1,87 Ω)")
    st.caption("Rapat energi: **44,2 Wh/kg**")
    st.caption("Rapat daya: **9.800 W/kg**")
    st.divider()
    st.caption("Referensi: Li et al. (2023); Wang et al. (2024); Zhao et al. (2023)")


# ═══════════════════════════════════════════════════════════════════════════════
# HALAMAN 1 — RINGKASAN
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Ringkasan":
    st.title("Sintesis GNC Berbasis Biomassa Kulit Durian Kota Medan")
    st.markdown(
        "*Elektroda Superkapasitor EDLC Performa Tinggi — "
        "Strategi Kemandirian Energi Sumatera Utara*"
    )
    st.divider()

    # ── Metric cards ──────────────────────────────────────────────────────────
    cols = st.columns(6)
    metrics = [
        ("1.623 m²/g", "Luas Permukaan BET",    "+55% vs KAC"),
        ("318 F/g",    "Kapasitansi Spesifik",   "@ 1 A/g"),
        ("96,3%",      "Retensi @ 10k siklus",   "+14,2 pp vs KAC"),
        ("0,42 Ω",     "ESR",                    "4,5× lebih rendah"),
        ("44,2 Wh/kg", "Rapat Energi",           "+115% vs KAC"),
        ("9.800 W/kg", "Rapat Daya Puncak",      "+118% vs KAC"),
    ]
    for col, (val, lbl, delta) in zip(cols, metrics):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-val">{val}</div>'
                f'<div class="metric-lbl">{lbl}</div>'
                f'<div class="metric-delta">↑ {delta}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("&nbsp;")
    c1, c2 = st.columns([1.1, 1])

    with c1:
        st.markdown("### Komposisi Lignoselulosa Kulit Durian")
        fig_pie = go.Figure(go.Pie(
            labels=["Selulosa", "Lignin", "Hemiselulosa", "Komponen lain"],
            values=[43, 22.5, 17.5, 17],
            hole=0.42,
            marker_colors=["#185FA5", "#1D9E75", "#BA7517", "#D3D1C7"],
            textinfo="label+percent",
            textfont_size=12,
        ))
        fig_pie.update_layout(
            showlegend=True, height=340,
            legend=dict(orientation="h", y=-0.08, font_size=11),
            margin=dict(t=10, b=50, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown(
            '<p class="caption">Analisis proksimat biomassa kulit durian (Nasution et al., 2022)</p>',
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown("### Komparasi Parameter Material Elektroda")
        df_cmp = pd.DataFrame({
            "Parameter":       ["BET (m²/g)", "Vol. pori (cm³/g)", "Kapasitansi (F/g)", "Retensi 10k (%)"],
            "GNC Durian":      [1623, 1.02, 318, 96.3],
            "KAC Komersial":   [1050, 0.55, 148, 82.1],
            "Arang Tempurung": [500,  0.33, 105, 74.0],
        })
        fig_bar = go.Figure()
        COLORS = {"GNC Durian": "#185FA5", "KAC Komersial": "#888780", "Arang Tempurung": "#BA7517"}
        for mat, color in COLORS.items():
            norm = df_cmp[mat] / df_cmp["GNC Durian"] * 100
            fig_bar.add_trace(go.Bar(
                name=mat, x=df_cmp["Parameter"], y=norm, marker_color=color, opacity=0.85,
                customdata=df_cmp[mat],
                hovertemplate="%{x}<br>Nilai: %{customdata}<br>Relatif: %{y:.1f}%<extra></extra>",
            ))
        fig_bar.update_layout(
            barmode="group", height=340,
            yaxis_title="Nilai relatif terhadap GNC (%)",
            legend=dict(orientation="h", y=-0.15, font_size=11),
            margin=dict(t=10, b=60, l=50, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        fig_bar.update_xaxes(tickfont_size=10)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown(
            '<p class="caption">Nilai dinormalisasi relatif terhadap GNC durian = 100%</p>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# HALAMAN 2 — FLOWCHART
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔄 Flowchart Sintesis":
    st.title("Bagan Alir Sintesis Green-Nanoporous Carbon (GNC)")
    st.markdown(
        "*Rute fabrikasi 10 tahap dari biomassa kulit durian Kota Medan — prinsip green chemistry*"
    )
    st.divider()

    show_detail = st.toggle("Tampilkan kondisi operasi lengkap", value=True)

    STEPS = [
        # (no, judul, sub_singkat, sub_detail, bg, border, grup)
        (1,  "Pengumpulan Bahan Baku",
             "3 sentra kuliner Kota Medan",
             "Kulit durian segar — Pasar Petisah, Simpang Limun, Jl. Sisingamangaraja",
             "#D6EEE6", "#0F6E56", "Praproses"),
        (2,  "Praproses Mekanis",
             "Cuci 3×, potong 2–3 cm",
             "Pencucian dengan air mengalir (3×); pemotongan fragmen 2–3 cm",
             "#D6EEE6", "#0F6E56", "Praproses"),
        (3,  "Pengeringan Awal (Oven)",
             "110 °C · 24 jam · kadar air <5%",
             "Oven konveksi 110 °C selama 24 jam hingga massa konstan; kadar air akhir <5%",
             "#D6EEE6", "#0F6E56", "Praproses"),
        (4,  "Karbonisasi — Pirolisis N₂",
             "600 °C · 2 jam → Biochar",
             "Tubular furnace: ramping 5 °C/menit → 600 °C, tahan 2 jam, N₂ 200 mL/min → Biochar",
             "#FFF0CC", "#854F0B", "Termal"),
        (5,  "Impregnasi KOH",
             "Biochar : KOH = 1:4 · 12 jam",
             "Campuran biochar:KOH rasio massa 1:4; pengadukan magnetik 12 jam; pengeringan 80 °C",
             "#FFF0CC", "#854F0B", "Termal"),
        (6,  "Aktivasi Kimia-Termal",
             "800 °C · 1 jam → GNC kasar",
             "Tubular furnace 800 °C / 1 jam, N₂ → 4KOH+C → K₂CO₃+K₂O+2H₂↑ → GNC kasar",
             "#FCDDD3", "#993C1D", "Purifikasi"),
        (7,  "Pencucian & Netralisasi",
             "HCl 1 M → akuades → pH = 7,0",
             "Pencucian bertahap HCl 1 M lalu akuades berulang; verifikasi pH filtrat = 7,0",
             "#FCDDD3", "#993C1D", "Purifikasi"),
        (8,  "Pengeringan Akhir — Freeze Dry",
             "−50 °C · <0,1 mbar · 48 jam",
             "Freeze dryer −50 °C, tekanan <0,1 mbar selama 48 jam → GNC kering",
             "#D6E8F7", "#0C447C", "Pengeringan"),
        (9,  "Karakterisasi Material",
             "BET · SEM-EDX · FTIR · Raman · XRD",
             "Analisis luas permukaan BET, pencitraan SEM-EDX, spektroskopi FTIR & Raman, XRD",
             "#E8E4F8", "#3C3489", "Karakterisasi"),
        (10, "Fabrikasi Elektroda & Uji",
             "Ni-foam · CV · GCD · EIS · 10.000 siklus",
             "Pencetakan GNC pada substrat Ni-foam; pengujian CV, GCD, EIS, siklus 10.000×",
             "#E8E4F8", "#3C3489", "Karakterisasi"),
    ]

    GROUP_COLORS = {
        "Praproses":     "#0F6E56",
        "Termal":        "#854F0B",
        "Purifikasi":    "#993C1D",
        "Pengeringan":   "#0C447C",
        "Karakterisasi": "#3C3489",
    }

    # ── Build Plotly flowchart ────────────────────────────────────────────────
    BOX_W  = 0.70
    BOX_H  = 0.075
    X_CTR  = 0.5
    X_L    = X_CTR - BOX_W / 2
    GAP    = 0.028
    STEP_H = BOX_H + GAP
    N      = len(STEPS)
    TOTAL  = N * STEP_H + 0.04

    shapes, annotations = [], []

    for i, (no, lbl, sub_s, sub_d, bg, border, grp) in enumerate(STEPS):
        y_top = TOTAL - 0.02 - i * STEP_H
        y_bot = y_top - BOX_H
        y_ctr = (y_top + y_bot) / 2
        sub   = sub_d if show_detail else sub_s

        # Box
        shapes.append(dict(
            type="rect", xref="paper", yref="paper",
            x0=X_L, y0=y_bot, x1=X_L + BOX_W, y1=y_top,
            fillcolor=bg, line=dict(color=border, width=1.5), layer="below",
        ))
        # Badge circle
        shapes.append(dict(
            type="circle", xref="paper", yref="paper",
            x0=X_L + 0.005, y0=y_ctr - 0.022,
            x1=X_L + 0.052, y1=y_ctr + 0.022,
            fillcolor=border, line_color=border,
        ))
        # Badge number
        annotations.append(dict(
            x=X_L + 0.029, y=y_ctr, xref="paper", yref="paper",
            text=f"<b>{no}</b>", showarrow=False,
            font=dict(size=11, color="white", family="Arial"),
            xanchor="center", yanchor="middle",
        ))
        # Title
        annotations.append(dict(
            x=X_L + 0.065, y=y_ctr + 0.013, xref="paper", yref="paper",
            text=f"<b>{lbl}</b>", showarrow=False,
            font=dict(size=11.5, color=border, family="Arial"),
            xanchor="left", yanchor="middle",
        ))
        # Subtitle
        annotations.append(dict(
            x=X_L + 0.065, y=y_ctr - 0.015, xref="paper", yref="paper",
            text=f"<span style='color:#555;font-size:10px'>{sub}</span>",
            showarrow=False,
            font=dict(size=10, color="#555555", family="Arial"),
            xanchor="left", yanchor="middle",
        ))
        # Arrow
        if i < N - 1:
            ay_top = y_bot
            ay_bot = ay_top - GAP
            shapes.append(dict(
                type="line", xref="paper", yref="paper",
                x0=X_CTR, y0=ay_bot + GAP * 0.15,
                x1=X_CTR, y1=ay_top - 0.002,
                line=dict(color="#999999", width=1.5),
            ))
            shapes.append(dict(
                type="path", xref="paper", yref="paper",
                path=(f"M {X_CTR-0.012} {ay_bot+GAP*0.25} "
                      f"L {X_CTR+0.012} {ay_bot+GAP*0.25} "
                      f"L {X_CTR} {ay_bot} Z"),
                fillcolor="#999999", line_color="#999999",
            ))

    # Group side bars
    groups_seen = {}
    for i, (_, _, _, _, _, border, grp) in enumerate(STEPS):
        if grp not in groups_seen:
            idxs   = [j for j, s in enumerate(STEPS) if s[6] == grp]
            y_tg   = TOTAL - 0.02 - min(idxs) * STEP_H
            y_bg   = TOTAL - 0.02 - max(idxs) * STEP_H - BOX_H
            y_mid  = (y_tg + y_bg) / 2
            grp_x  = X_L + BOX_W + 0.012
            shapes.append(dict(
                type="line", xref="paper", yref="paper",
                x0=grp_x, y0=y_bg, x1=grp_x, y1=y_tg,
                line=dict(color=GROUP_COLORS[grp], width=3),
            ))
            annotations.append(dict(
                x=grp_x + 0.007, y=y_mid, xref="paper", yref="paper",
                text=f"<b>{grp}</b>", showarrow=False,
                font=dict(size=9, color=GROUP_COLORS[grp], family="Arial"),
                xanchor="left", yanchor="middle", textangle=-90,
            ))
            groups_seen[grp] = True

    fig = go.Figure()
    fig.update_layout(
        shapes=shapes, annotations=annotations,
        height=850,
        margin=dict(l=10, r=90, t=10, b=10),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, TOTAL + 0.02]),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Legend
    st.markdown("**Kode warna tahapan:**")
    leg_cols = st.columns(5)
    for col, (grp, color) in zip(leg_cols, GROUP_COLORS.items()):
        col.markdown(
            f'<span class="step-badge" '
            f'style="background:{color}20;color:{color};border:1px solid {color}">'
            f'● {grp}</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown(
        '<p class="caption">Gambar 1. Bagan alir sintesis Green-Nanoporous Carbon (GNC) dari biomassa kulit '
        'durian Kota Medan melalui jalur karbonisasi pirolisis N₂ dan aktivasi KOH. Rute dirancang mengikuti '
        'prinsip green chemistry dengan meminimalkan penggunaan reagen beracun '
        '(Nasution et al., 2022; Li et al., 2023).</p>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HALAMAN 3 — GRAFIK ELEKTROKIMIA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Grafik Elektrokimia":
    st.title("Grafik Karakteristik Kinerja Elektrokimia GNC Kulit Durian")
    st.markdown(
        "*Lampiran 2 — Proyeksi berdasarkan tinjauan literatur "
        "(Li et al., 2023; Wang et al., 2024; Zhao et al., 2023)*"
    )
    st.divider()

    BLUE  = "#185FA5"
    LBLUE = "#B5D4F4"
    GRAY  = "#888780"
    LGRAY = "#D3D1C7"
    TEAL  = "#1D9E75"
    AMBER = "#BA7517"

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "L2-1  CV", "L2-2  GCD", "L2-3  Nyquist EIS",
        "L2-4  Retensi Siklus", "L2-5  Ragone Plot",
    ])

    # ── L2-1 Cyclic Voltammetry ───────────────────────────────────────────────
    with tab1:
        st.markdown("### Gambar L2-1 — Kurva Cyclic Voltammetry (CV)")
        st.markdown(
            '<div class="info-box">Profil CV <i>quasi-rectangular</i> GNC durian pada laju pindai '
            '5–200 mV/s dalam KOH 6 M menunjukkan perilaku EDLC ideal tanpa puncak faradaik. '
            'Luas area kurva GNC durian ≈2,1× lebih besar dibandingkan KAC komersial.</div>',
            unsafe_allow_html=True,
        )
        scan_rate = st.select_slider(
            "Laju pindai (mV/s)", options=[5, 10, 20, 50, 100, 200], value=50
        )
        scale = 1 + (scan_rate / 200) * 0.25
        v = np.linspace(0, 1, 300)
        ag = 28 * scale;  ak = 13 * scale
        gnc_t = ag * np.sin(np.pi*v) * (0.85 + 0.15*np.sin(6*np.pi*v))
        gnc_b = -ag * np.sin(np.pi*v) * (0.85 + 0.12*np.sin(6*np.pi*v))
        kac_t = ak * np.sin(np.pi*v) * (0.88 + 0.12*np.sin(6*np.pi*v))
        kac_b = -ak * np.sin(np.pi*v) * (0.88 + 0.10*np.sin(6*np.pi*v))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.concatenate([v, v[::-1]]),
            y=np.concatenate([gnc_t, gnc_b[::-1]]),
            fill="toself", fillcolor=LBLUE+"33",
            line_color="rgba(0,0,0,0)", showlegend=False,
        ))
        fig.add_trace(go.Scatter(x=v, y=gnc_t, line=dict(color=BLUE, width=2.2), name="GNC durian"))
        fig.add_trace(go.Scatter(x=v, y=gnc_b, line=dict(color=BLUE, width=2.2), showlegend=False))
        fig.add_trace(go.Scatter(
            x=np.concatenate([v, v[::-1]]),
            y=np.concatenate([kac_t, kac_b[::-1]]),
            fill="toself", fillcolor=LGRAY+"22",
            line_color="rgba(0,0,0,0)", showlegend=False,
        ))
        fig.add_trace(go.Scatter(x=v, y=kac_t,
                                 line=dict(color=GRAY, width=1.8, dash="dash"), name="KAC komersial"))
        fig.add_trace(go.Scatter(x=v, y=kac_b,
                                 line=dict(color=GRAY, width=1.8, dash="dash"), showlegend=False))
        fig.add_hline(y=0, line_color="#cccccc", line_width=0.8)
        fig.update_layout(
            height=420, xaxis_title="Tegangan (V)", yaxis_title="Arus (mA/g)",
            legend=dict(orientation="h", y=1.08, font_size=12),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(range=[0,1], showgrid=True, gridcolor="#f0f0f0"),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
            margin=dict(t=30, b=50, l=60, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f'<p class="caption">Gambar L2-1. Kurva CV GNC kulit durian vs KAC komersial pada laju pindai '
            f'{scan_rate} mV/s, elektrolit KOH 6 M, jendela potensial 0–1 V.</p>',
            unsafe_allow_html=True,
        )

    # ── L2-2 GCD ─────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Gambar L2-2 — Kurva Galvanostatic Charge-Discharge (GCD)")
        st.markdown(
            '<div class="info-box">Profil GCD berbentuk segitiga simetris pada arus 1–20 A/g. '
            'Kapasitansi spesifik tertinggi 318 F/g pada 1 A/g dengan retensi 79% pada 20 A/g.</div>',
            unsafe_allow_html=True,
        )
        rates  = [1, 2, 5, 10, 20]
        gnc_cs = [318, 302, 282, 265, 251]
        kac_cs = [148, 138, 126, 115, 108]

        c1, c2 = st.columns([2, 1])
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="GNC durian", x=[str(r) for r in rates], y=gnc_cs,
                marker_color=BLUE, opacity=0.85,
                text=gnc_cs, textposition="outside", textfont_size=11,
            ))
            fig.add_trace(go.Bar(
                name="KAC komersial", x=[str(r) for r in rates], y=kac_cs,
                marker_color=GRAY, opacity=0.80,
                text=kac_cs, textposition="outside", textfont_size=11,
            ))
            fig.update_layout(
                barmode="group", height=400,
                xaxis_title="Rapat arus (A/g)",
                yaxis_title="Kapasitansi spesifik (F/g)",
                yaxis_range=[0, 380],
                legend=dict(orientation="h", y=1.08, font_size=12),
                paper_bgcolor="white", plot_bgcolor="white",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
                margin=dict(t=40, b=50, l=60, r=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("**Tabel kapasitansi**")
            df = pd.DataFrame({
                "Arus (A/g)": rates,
                "GNC (F/g)":  gnc_cs,
                "KAC (F/g)":  kac_cs,
                "Rasio GNC/KAC": [round(g/k, 2) for g, k in zip(gnc_cs, kac_cs)],
            })
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown(
                '<p class="caption">GNC 2,1–2,3× lebih tinggi di semua rapat arus.</p>',
                unsafe_allow_html=True,
            )
        st.markdown(
            '<p class="caption">Gambar L2-2. Kapasitansi spesifik GNC kulit durian vs KAC '
            'komersial pada rapat arus 1–20 A/g, elektrolit KOH 6 M.</p>',
            unsafe_allow_html=True,
        )

    # ── L2-3 Nyquist EIS ─────────────────────────────────────────────────────
    with tab3:
        st.markdown("### Gambar L2-3 — Nyquist Plot (EIS)")
        st.markdown(
            '<div class="info-box">ESR GNC durian = 0,42 Ω (semi-lingkaran kecil di frekuensi tinggi) '
            'dan garis hampir vertikal di frekuensi rendah — difusi ion cepat, perilaku kapasitif ideal.</div>',
            unsafe_allow_html=True,
        )

        def nyq(esr, r, nt=30):
            ang = np.linspace(0, np.pi, 50)
            zr  = esr + r - r*np.cos(ang)
            zi  = r * np.sin(ang)
            zrt = np.linspace(zr[-1], zr[-1]+0.05, nt)
            zit = np.linspace(zi[-1], zi[-1]+2.0,  nt)
            return np.concatenate([zr, zrt]), np.concatenate([zi, zit])

        gzr, gzi = nyq(0.42, 0.18)
        kzr, kzi = nyq(1.87, 0.55)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=gzr, y=gzi, mode="lines+markers",
            line=dict(color=BLUE, width=2.2), marker=dict(size=3),
            name="GNC durian (ESR = 0,42 Ω)",
        ))
        fig.add_trace(go.Scatter(
            x=kzr, y=kzi, mode="lines+markers",
            line=dict(color=GRAY, width=1.8, dash="dash"), marker=dict(size=3),
            name="KAC komersial (ESR = 1,87 Ω)",
        ))
        fig.add_annotation(x=0.42, y=0.02, text="ESR = 0,42 Ω",
                           showarrow=True, arrowhead=2, ax=60, ay=-40,
                           font=dict(size=10, color=BLUE), arrowcolor=BLUE)
        fig.add_annotation(x=1.87, y=0.02, text="ESR = 1,87 Ω",
                           showarrow=True, arrowhead=2, ax=60, ay=-40,
                           font=dict(size=10, color="#555"), arrowcolor=GRAY)
        fig.update_layout(
            height=440, xaxis_title="Z' real (Ω)", yaxis_title="−Z'' imag (Ω)",
            xaxis_range=[0, 4], yaxis_range=[0, 2.6],
            legend=dict(orientation="h", y=1.08, font_size=12),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
            margin=dict(t=40, b=50, l=60, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            '<p class="caption">Gambar L2-3. Nyquist plot EIS GNC durian vs KAC komersial. '
            'Semi-lingkaran kecil = hambatan transfer muatan rendah; garis vertikal = perilaku kapasitif ideal.</p>',
            unsafe_allow_html=True,
        )

    # ── L2-4 Retensi Siklus ──────────────────────────────────────────────────
    with tab4:
        st.markdown("### Gambar L2-4 — Retensi Kapasitansi vs Jumlah Siklus")
        st.markdown(
            '<div class="info-box">Setelah 10.000 siklus GCD pada 10 A/g, GNC durian mempertahankan '
            '96,3% kapasitansi awal dengan efisiensi coulombik stabil 98,7% — jauh melampaui KAC (82,1%).</div>',
            unsafe_allow_html=True,
        )
        cycles   = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        gnc_ret  = [100, 99.4, 99.0, 98.6, 98.2, 97.8, 97.5, 97.2, 96.9, 96.6, 96.3]
        kac_ret  = [100, 97.5, 95.8, 94.1, 92.5, 91.0, 89.4, 87.7, 86.1, 84.2, 82.1]
        gnc_coul = [100, 99.6, 99.3, 99.1, 98.9, 98.8, 98.8, 98.7, 98.7, 98.7, 98.7]
        kac_coul = [100, 98.0, 96.5, 95.0, 93.8, 92.8, 92.0, 91.6, 91.4, 91.4, 91.4]

        show_coul = st.checkbox("Tampilkan efisiensi coulombik (sumbu kanan)", value=True)

        fig = make_subplots(specs=[[{"secondary_y": show_coul}]])
        fig.add_trace(go.Scatter(
            x=cycles, y=gnc_ret, mode="lines+markers",
            line=dict(color=BLUE, width=2.5), marker=dict(size=5),
            name="GNC durian — retensi (%)"), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=cycles, y=kac_ret, mode="lines+markers",
            line=dict(color=GRAY, width=2, dash="dash"), marker=dict(size=4, symbol="square"),
            name="KAC — retensi (%)"), secondary_y=False)
        if show_coul:
            fig.add_trace(go.Scatter(
                x=cycles, y=gnc_coul, mode="lines",
                line=dict(color=TEAL, width=1.5, dash="dot"),
                name="GNC durian — coulombik (%)"), secondary_y=True)
            fig.add_trace(go.Scatter(
                x=cycles, y=kac_coul, mode="lines",
                line=dict(color=AMBER, width=1.5, dash="dot"),
                name="KAC — coulombik (%)"), secondary_y=True)
        fig.add_hline(y=96.3, line_dash="dot", line_color=BLUE+"88", line_width=1,
                      annotation_text="GNC: 96,3%", annotation_position="right",
                      secondary_y=False)
        fig.add_hline(y=82.1, line_dash="dot", line_color="#88888888", line_width=1,
                      annotation_text="KAC: 82,1%", annotation_position="right",
                      secondary_y=False)
        fig.update_layout(
            height=440, xaxis_title="Jumlah siklus",
            legend=dict(orientation="h", y=1.1, font_size=11),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(
                showgrid=True, gridcolor="#f0f0f0",
                tickvals=cycles,
                ticktext=[f"{c//1000}k" if c >= 1000 else "0" for c in cycles],
            ),
            margin=dict(t=40, b=50, l=60, r=70),
        )
        fig.update_yaxes(title_text="Retensi kapasitansi (%)", range=[78, 103],
                         secondary_y=False, showgrid=True, gridcolor="#f0f0f0")
        if show_coul:
            fig.update_yaxes(title_text="Efisiensi coulombik (%)", range=[88, 102],
                             secondary_y=True, showgrid=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            '<p class="caption">Gambar L2-4. Kurva retensi kapasitansi GNC kulit durian vs KAC komersial '
            'selama 10.000 siklus GCD pada 10 A/g, elektrolit KOH 6 M.</p>',
            unsafe_allow_html=True,
        )

    # ── L2-5 Ragone Plot ─────────────────────────────────────────────────────
    with tab5:
        st.markdown("### Gambar L2-5 — Ragone Plot (Rapat Energi vs Rapat Daya)")
        st.markdown(
            '<div class="info-box">GNC durian: rapat energi 44,2 Wh/kg dan rapat daya puncak '
            '9.800 W/kg — melampaui zona karbon aktif konvensional, mendekati batas bawah Li-ion.</div>',
            unsafe_allow_html=True,
        )
        fig = go.Figure()

        # EDLC zone
        pz = np.array([100, 200, 500, 1000, 3000, 5000])
        fig.add_trace(go.Scatter(
            x=np.concatenate([pz, pz[::-1]]),
            y=np.concatenate([np.array([10,11,12,14,16,18]),
                               np.array([0.5,1,2,3,4,5])[::-1]]),
            fill="toself", fillcolor=TEAL+"28",
            line=dict(color=TEAL, width=1),
            name="Zona EDLC konvensional",
        ))
        # Li-ion zone
        pl = np.array([50, 100, 200, 500])
        fig.add_trace(go.Scatter(
            x=np.concatenate([pl, pl[::-1]]),
            y=np.concatenate([np.array([280,250,200,130]),
                               np.array([60,70,80,60])[::-1]]),
            fill="toself", fillcolor=AMBER+"25",
            line=dict(color=AMBER, width=1),
            name="Zona batas bawah Li-ion",
        ))
        # KAC trail
        kac_p = [300, 700, 1500, 2500, 3500, 4500]
        fig.add_trace(go.Scatter(
            x=kac_p, y=[20.5]*6, mode="lines",
            line=dict(color=GRAY, width=1.5, dash="dash"), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[4500], y=[20.5], mode="markers",
            marker=dict(size=14, color=GRAY, symbol="circle"),
            name="KAC komersial (20,5 Wh/kg; 4.500 W/kg)",
            hovertemplate="KAC<br>E: 20,5 Wh/kg<br>P: 4.500 W/kg<extra></extra>",
        ))
        # GNC trail
        gnc_p = [500, 1500, 3000, 5000, 7000, 9800]
        fig.add_trace(go.Scatter(
            x=gnc_p, y=[44.2]*6, mode="lines",
            line=dict(color=BLUE, width=1.5, dash="dash"), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[9800], y=[44.2], mode="markers",
            marker=dict(size=20, color=BLUE, symbol="star"),
            name="GNC durian (44,2 Wh/kg; 9.800 W/kg)",
            hovertemplate="GNC durian<br>E: 44,2 Wh/kg<br>P: 9.800 W/kg<extra></extra>",
        ))
        fig.add_annotation(
            x=np.log10(9800), y=44.2,
            text="<b>GNC Durian</b><br>44,2 Wh/kg · 9.800 W/kg",
            showarrow=True, arrowhead=2, ax=-130, ay=-55,
            font=dict(size=11, color=BLUE), arrowcolor=BLUE,
            bgcolor="white", bordercolor=BLUE, borderwidth=1,
        )
        fig.add_annotation(
            x=np.log10(4500), y=20.5,
            text="KAC<br>20,5 Wh/kg · 4.500 W/kg",
            showarrow=True, arrowhead=2, ax=-90, ay=50,
            font=dict(size=10, color="#555"), arrowcolor=GRAY,
            bgcolor="white", bordercolor=GRAY, borderwidth=1,
        )
        fig.update_layout(
            height=500, xaxis_title="Rapat daya (W/kg)", yaxis_title="Rapat energi (Wh/kg)",
            xaxis_type="log",
            xaxis_range=[np.log10(80), np.log10(25000)],
            yaxis_range=[0, 100],
            legend=dict(orientation="h", y=1.10, font_size=11),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(
                showgrid=True, gridcolor="#f0f0f0",
                tickvals=[100,200,500,1000,2000,5000,10000,20000],
                ticktext=["100","200","500","1k","2k","5k","10k","20k"],
            ),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
            margin=dict(t=40, b=60, l=70, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            '<p class="caption">Gambar L2-5. Ragone plot: posisi kinerja GNC kulit durian (★) relatif '
            'terhadap KAC komersial, zona EDLC konvensional, dan batas bawah baterai Li-ion.</p>',
            unsafe_allow_html=True,
        )

    # ── Footer catatan ────────────────────────────────────────────────────────
    st.divider()
    st.markdown("""
    <div style="background:#f8f8f6;border-radius:8px;padding:10px 16px;
                font-size:0.82rem;color:#666;">
    <b>Catatan:</b> Seluruh data pada grafik merupakan proyeksi berdasarkan tinjauan literatur
    komprehensif terhadap material GNC sejenis. Data aktual dihasilkan setelah karakterisasi
    laboratorium selesai dilaksanakan.<br>
    Referensi: Li et al. (2023) <i>Adv. Energy Mater.</i>; Wang et al. (2024)
    <i>Electrochimica Acta</i>; Zhao et al. (2023) <i>Nano Energy</i>.
    </div>
    """, unsafe_allow_html=True)
