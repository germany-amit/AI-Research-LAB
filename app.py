# AI 2100 Simulator — Research-Lab Showcase (Free GitHub + Free Streamlit)
# Author: You
# Focus: Truthful-by-design, transparent assumptions, future forecasting, resilience scenarios

import streamlit as st
import pandas as pd
import numpy as np
import io
import math
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

# -----------------------------
# Global constants & helpers
# -----------------------------
C_KM_PER_S = 299_792.458  # speed of light (km/s)
EARTH_MOON_AVG_KM = 384_400
EARTH_MARS_MIN_KM = 54_600_000
EARTH_MARS_MAX_KM = 401_000_000

WORKLOAD_POWER_KW = {
    "LLM Inference (small)": 1.5,
    "LLM Training (1 GPU)": 5.0,
    "Vision Inference": 1.0,
    "Tabular Training": 0.6,
}

# 2025 baseline demos (illustrative, not vendor-official)
CLOUD_2025 = {
    "AWS":  {"lat_ms": 120.0, "usd_per_kwh": 0.12, "gco2_per_kwh": 350.0},
    "GCP":  {"lat_ms": 180.0, "usd_per_kwh": 0.11, "gco2_per_kwh": 450.0},
    "Azure":{"lat_ms": 150.0, "usd_per_kwh": 0.13, "gco2_per_kwh": 200.0},
}

REGION_RTT_TERR = {
    "us-east": 20.0,
    "us-west": 25.0,
    "eu-west": 25.0,
    "ap-south": 35.0,
}

def clamp(v, lo, hi):
    return max(lo, min(v, hi))

# Forecasting curves (simple, explainable)
def cost_factor(year):
    # ~1.5% improvement per year
    return (1.0 - 0.015) ** max(0, year - 2025)

def carbon_factor(year):
    # ~3% decarbonization per year
    return (1.0 - 0.03) ** max(0, year - 2025)

def latency_factor(year):
    # slight improvement, floor at 0.6x
    return max(0.60, (1.0 - 0.005) ** max(0, year - 2025))

# Physics-based RTT (ms) for interplanetary links
def rtt_ms_from_distance_km(distance_km: float) -> float:
    if distance_km <= 0:
        return 0.0
    one_way_s = distance_km / C_KM_PER_S
    return 2 * one_way_s * 1000.0

def cost_usd(power_kw: float, hours: float, usd_per_kwh: float) -> float:
    return max(power_kw, 0) * max(hours, 0) * max(usd_per_kwh, 0)

def carbon_kg(power_kw: float, hours: float, g_per_kwh: float) -> float:
    kwh = max(power_kw, 0) * max(hours, 0)
    return (kwh * max(g_per_kwh, 0)) / 1000.0

def score_strategy(lat_ms: float, cost_val: float, co2_kg: float, weights: dict) -> float:
    # Normalize to [0,1] bands; lower is better
    lat_norm = clamp(lat_ms / 1000.0, 0.0, 1.0)   # cap at 1s RTT
    cost_norm = clamp(cost_val / 100.0, 0.0, 1.0) # cap at $100
    co2_norm = clamp(co2_kg / 50.0, 0.0, 1.0)     # cap at 50 kg
    return (weights["latency"] * lat_norm +
            weights["cost"]    * cost_norm +
            weights["carbon"]  * co2_norm)

def agent_explanations(agent_name: str, row: dict, scenario: str):
    lat = row["lat_ms"]; costv = row["cost_usd"]; co2 = row["co2_kg"]
    loc = row["location"]
    if agent_name == "Performance Agent 🏎️":
        msg = f"{loc}: targets lowest latency (~{lat:.0f} ms)."
        if scenario in ("Outage", "Network congestion", "Solar flare comms disruption"):
            msg += " Penalizes links affected by the incident."
        return msg
    if agent_name == "Finance Agent 💰":
        return f"{loc}: minimizes cost (~${costv:.2f})."
    if agent_name == "Sustainability Agent 🌱":
        return f"{loc}: minimizes emissions (~{co2:.2f} kg CO₂e)."
    if agent_name == "Risk/Resilience Agent 🛡️":
        msg = f"{loc}: prefers stable paths; avoids long/fragile links."
        if scenario in ("Outage", "Cyberattack (DDoS)"):
            msg += " Adds risk premium under active incidents."
        return msg
    return ""

def export_pdf(title: str, summary: dict, table_rows: list[dict]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter
    y = h - 50

    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, title)
    y -= 18
    c.setFont("Helvetica", 9)
    c.drawString(40, y, f"Generated: {datetime.utcnow().isoformat()}Z")
    y -= 14

    # Assumptions
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Assumptions")
    y -= 14
    c.setFont("Helvetica", 9)
    for k, v in summary["assumptions"].items():
        c.drawString(50, y, f"- {k}: {v}")
        y -= 12
        if y < 80: c.showPage(); y = h - 50

    # Weights
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Decision weights (lower score is better)")
    y -= 14
    c.setFont("Helvetica", 9)
    for k, v in summary["weights"].items():
        c.drawString(50, y, f"- {k}: {v:.2f}")
        y -= 12
        if y < 80: c.showPage(); y = h - 50

    # Table
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Strategies compared")
    y -= 14
    c.setFont("Helvetica-Bold", 9)
    c.drawString(50, y, "Location")
    c.drawString(180, y, "Latency(ms)")
    c.drawString(270, y, "Cost($)")
    c.drawString(330, y, "CO2(kg)")
    c.drawString(390, y, "Score↓")
    y -= 12
    c.setFont("Helvetica", 9)
    for r in table_rows:
        c.drawString(50, y, r["location"])
        c.drawRightString(255, y, f"{r['lat_ms']:.0f}")
        c.drawRightString(325, y, f"{r['cost_usd']:.2f}")
        c.drawRightString(385, y, f"{r['co2_kg']:.2f}")
        c.drawRightString(445, y, f"{r['score']:.3f}")
        y -= 12
        if y < 80: c.showPage(); y = h - 50

    if table_rows:
        best = min(table_rows, key=lambda x: x["score"])
        y -= 8
        c.setFont("Helvetica-Bold", 11)
        c.drawString(40, y, f"Recommended: {best['location']} (score {best['score']:.3f})")
    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI 2100 Simulator — Research-Lab Showcase", layout="wide")
st.title("🌌 AI 2100 Simulator — Research-Lab Showcase")
st.caption("Truthful simulation • Future forecasting • Resilience • Multi-agent rationale")

with st.sidebar:
    st.subheader("⚙️ Workload & Forecast")
    workload = st.selectbox("Workload", list(WORKLOAD_POWER_KW.keys()))
    power_kw = st.number_input("Estimated power (kW)", 0.1, 500.0, float(WORKLOAD_POWER_KW[workload]), 0.1)
    hours = st.number_input("Runtime (hours)", 0.1, 240.0, 1.0, 0.1)
    year = st.slider("Forecast year", 2025, 2100, 2035, 1)

    st.markdown("---")
    st.subheader("📉 Decision Weights (sum ≈ 1)")
    w_latency = st.slider("Latency weight", 0.0, 1.0, 0.50, 0.01)
    w_cost    = st.slider("Cost weight",    0.0, 1.0, 0.25, 0.01)
    w_carbon  = st.slider("Carbon weight",  0.0, 1.0, 0.25, 0.01)

    st.markdown("---")
    st.subheader("🛡️ Failure/Risk scenario")
    scenario_multi = st.selectbox("Earth multi-cloud scenario", ["Normal", "Outage", "Network congestion", "Cyberattack (DDoS)"])
    scenario_space = st.selectbox("Interplanetary scenario", ["Normal", "Solar flare comms disruption", "Mars relay outage", "Lunar base power cap"])

tabs = st.tabs(["🌐 Earth Multi-Cloud (AWS/GCP/Azure)", "🌌 Interplanetary (Earth/Moon/Mars)"])

# -----------------------------
# Tab 1: Earth Multi-Cloud
# -----------------------------
with tabs[0]:
    st.subheader("🌐 Earth Multi-Cloud (AWS / GCP / Azure)")

    # Allow user to tweak provider baselines (remain free & transparent)
    st.markdown("**2025 baselines (editable):**")
    baselines = {}
    cols = st.columns(3)
    for i, prov in enumerate(["AWS", "GCP", "Azure"]):
        with cols[i]:
            st.caption(f"**{prov}** baseline (2025)")
            lat = st.number_input(f"{prov} latency (ms)", 20.0, 400.0, float(CLOUD_2025[prov]["lat_ms"]), 5.0, key=f"{prov}_lat")
            price = st.number_input(f"{prov} $/kWh", 0.0, 2.0, float(CLOUD_2025[prov]["usd_per_kwh"]), 0.01, key=f"{prov}_price")
            gco2 = st.number_input(f"{prov} gCO₂/kWh", 0.0, 2000.0, float(CLOUD_2025[prov]["gco2_per_kwh"]), 10.0, key=f"{prov}_gco2")
            baselines[prov] = {"lat_ms": lat, "usd_per_kwh": price, "gco2_per_kwh": gco2}

    # Apply forecast factors
    rows = []
    for prov, base in baselines.items():
        lat_ms = base["lat_ms"] * latency_factor(year)
        usd_kwh = base["usd_per_kwh"] * cost_factor(year)
        gco2_kwh = base["gco2_per_kwh"] * carbon_factor(year)

        # Scenario effects
        if scenario_multi == "Outage":
            # Represent severe impact: huge latency / effectively unusable
            lat_ms *= 5.0
        elif scenario_multi == "Network congestion":
            lat_ms *= 1.5
        elif scenario_multi == "Cyberattack (DDoS)":
            lat_ms += 80.0
            usd_kwh *= 1.10  # incident handling overhead

        c_usd = cost_usd(power_kw, hours, usd_kwh)
        co2 = carbon_kg(power_kw, hours, gco2_kwh)
        score = score_strategy(lat_ms, c_usd, co2, {"latency": w_latency, "cost": w_cost, "carbon": w_carbon})
        rows.append({"location": prov, "lat_ms": lat_ms, "cost_usd": c_usd, "co2_kg": co2, "score": score})

    df_mc = pd.DataFrame(rows).sort_values("score", ascending=True)
    st.dataframe(df_mc.reset_index(drop=True), use_container_width=True)

    # Charts
    ch1, ch2, ch3 = st.columns(3)
    with ch1:
        fig, ax = plt.subplots()
        ax.bar(df_mc["location"], df_mc["lat_ms"])
        ax.set_title("Latency (ms) ↓")
        st.pyplot(fig)
    with ch2:
        fig, ax = plt.subplots()
        ax.bar(df_mc["location"], df_mc["cost_usd"])
        ax.set_title("Cost (USD) ↓")
        st.pyplot(fig)
    with ch3:
        fig, ax = plt.subplots()
        ax.bar(df_mc["location"], df_mc["co2_kg"])
        ax.set_title("Carbon (kg CO₂e) ↓")
        st.pyplot(fig)

    # Recommendation + agent debate
    best_mc = df_mc.iloc[0]
    st.markdown(
        f"**🏆 Multi-Cloud Recommendation:** {best_mc['location']} — "
        f"Score {best_mc['score']:.3f} (≈ {best_mc['lat_ms']:.0f} ms, "
        f"${best_mc['cost_usd']:.2f}, {best_mc['co2_kg']:.2f} kg CO₂e)"
    )

    st.subheader("🤝 Agent Debate (number-aware)")
    for agent in ["Performance Agent 🏎️", "Finance Agent 💰", "Sustainability Agent 🌱", "Risk/Resilience Agent 🛡️"]:
        st.write(f"- **{agent}**: {agent_explanations(agent, dict(best_mc), scenario_multi)}")

    # Exports
    st.subheader("📑 Export (Multi-Cloud)")
    csv_bytes = df_mc.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ CSV — Multi-Cloud", data=csv_bytes, file_name="multicloud_results.csv", mime="text/csv")

    summary_mc = {
        "assumptions": {
            "Year": year,
            "Forecast factors": f"cost x{cost_factor(year):.3f}, carbon x{carbon_factor(year):.3f}, latency x{latency_factor(year):.3f}",
            "Workload": workload,
            "Power/Hours": f"{power_kw:.2f} kW × {hours:.2f} h",
            "Scenario": scenario_multi,
        },
        "weights": {"latency": w_latency, "cost": w_cost, "carbon": w_carbon},
    }
    pdf_bytes_mc = export_pdf("AI 2100 — Multi-Cloud Summary", summary_mc, rows)
    st.download_button("⬇️ PDF — Multi-Cloud Summary", data=pdf_bytes_mc, file_name="multicloud_summary.pdf", mime="application/pdf")

# -----------------------------
# Tab 2: Interplanetary
# -----------------------------
with tabs[1]:
    st.subheader("🌌 Interplanetary (Earth / Moon / Mars)")

    cold, cole = st.columns(2)
    with cold:
        region = st.selectbox("Earth region baseline RTT", list(REGION_RTT_TERR.keys()))
        extra_overhead_ms = st.number_input("Extra overhead (switch/queue) ms", 0.0, 500.0, 30.0, 5.0)
        usd_per_kwh = st.number_input("Electricity price (USD/kWh)", 0.0, 2.0, 0.12, 0.01)
    with cole:
        gco2_earth = st.number_input("Earth grid intensity (gCO₂/kWh)", 0.0, 2000.0, 400.0, 10.0)
        gco2_moon  = st.number_input("Moon intensity (gCO₂/kWh)", 0.0, 2000.0, 50.0, 10.0)
        gco2_mars  = st.number_input("Mars intensity (gCO₂/kWh)", 0.0, 2000.0, 80.0, 10.0)

    moon_km = st.slider("Earth–Moon distance (km)", 360_000, 405_000, EARTH_MOON_AVG_KM, 500)
    mars_km = st.slider("Earth–Mars distance (km)", EARTH_MARS_MIN_KM, EARTH_MARS_MAX_KM, 225_000_000, 1_000_000)

    # Compute rows
    rows_ip = []
    # Earth
    lat_e = REGION_RTT_TERR[region] * latency_factor(year) + extra_overhead_ms
    # Moon / Mars physics
    lat_moon = rtt_ms_from_distance_km(moon_km) * latency_factor(year) + extra_overhead_ms
    lat_mars = rtt_ms_from_distance_km(mars_km) * latency_factor(year) + extra_overhead_ms

    # Scenario effects (space)
    if scenario_space == "Solar flare comms disruption":
        lat_moon *= 2.5; lat_mars *= 2.5
    elif scenario_space == "Mars relay outage":
        lat_mars *= 3.0
    elif scenario_space == "Lunar base power cap":
        # model as cost bump later
        pass

    # Forecast decarbonization & cost
    usd_kwh = usd_per_kwh * cost_factor(year)
    g_e = gco2_earth * carbon_factor(year)
    g_moon = gco2_moon * carbon_factor(year)
    g_mars = gco2_mars * carbon_factor(year)

    # Cost per location (same price unless lunar cap)
    cost_e = cost_usd(power_kw, hours, usd_kwh)
    cost_moon = cost_usd(power_kw, hours, usd_kwh) * (1.25 if scenario_space == "Lunar base power cap" else 1.0)
    cost_mars = cost_usd(power_kw, hours, usd_kwh)

    co2_e = carbon_kg(power_kw, hours, g_e)
    co2_moon = carbon_kg(power_kw, hours, g_moon)
    co2_mars = carbon_kg(power_kw, hours, g_mars)

    for label, lat, cst, co2 in [
        (f"Earth/{region}", lat_e, cost_e, co2_e),
        ("Moon Base", lat_moon, cost_moon, co2_moon),
        ("Mars Colony", lat_mars, cost_mars, co2_mars),
    ]:
        rows_ip.append({
            "location": label,
            "lat_ms": lat,
            "cost_usd": cst,
            "co2_kg": co2,
            "score": score_strategy(lat, cst, co2, {"latency": w_latency, "cost": w_cost, "carbon": w_carbon})
        })

    df_ip = pd.DataFrame(rows_ip).sort_values("score", ascending=True)
    st.dataframe(df_ip.reset_index(drop=True), use_container_width=True)

    # Charts
    ch1, ch2, ch3 = st.columns(3)
    with ch1:
        fig, ax = plt.subplots()
        ax.bar(df_ip["location"], df_ip["lat_ms"])
        ax.set_title("Latency (ms) ↓")
        st.pyplot(fig)
    with ch2:
        fig, ax = plt.subplots()
        ax.bar(df_ip["location"], df_ip["cost_usd"])
        ax.set_title("Cost (USD) ↓")
        st.pyplot(fig)
    with ch3:
        fig, ax = plt.subplots()
        ax.bar(df_ip["location"], df_ip["co2_kg"])
        ax.set_title("Carbon (kg CO₂e) ↓")
        st.pyplot(fig)

    best_ip = df_ip.iloc[0]
    st.markdown(
        f"**🏆 Interplanetary Recommendation:** {best_ip['location']} — "
        f"Score {best_ip['score']:.3f} (≈ {best_ip['lat_ms']:.0f} ms, "
        f"${best_ip['cost_usd']:.2f}, {best_ip['co2_kg']:.2f} kg CO₂e)"
    )

    st.subheader("🤝 Agent Debate (number-aware)")
    for agent in ["Performance Agent 🏎️", "Finance Agent 💰", "Sustainability Agent 🌱", "Risk/Resilience Agent 🛡️"]:
        st.write(f"- **{agent}**: {agent_explanations(agent, dict(best_ip), scenario_space)}")

    # Exports
    st.subheader("📑 Export (Interplanetary)")
    csv_bytes_ip = df_ip.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ CSV — Interplanetary", data=csv_bytes_ip, file_name="interplanetary_results.csv", mime="text/csv")

    summary_ip = {
        "assumptions": {
            "Year": year,
            "Forecast factors": f"cost x{cost_factor(year):.3f}, carbon x{carbon_factor(year):.3f}, latency x{latency_factor(year):.3f}",
            "Workload": workload,
            "Power/Hours": f"{power_kw:.2f} kW × {hours:.2f} h",
            "Scenario": scenario_space,
            "Earth region RTT": f"{REGION_RTT_TERR[region]} ms base",
            "Extra overhead": f"{extra_overhead_ms:.0f} ms",
            "Moon distance": f"{EARTH_MOON_AVG_KM:,} km (slider set: {moon_km:,} km)",
            "Mars distance": f"{EARTH_MARS_MIN_KM:,}–{EARTH_MARS_MAX_KM:,} km (slider set: {mars_km:,} km)",
        },
        "weights": {"latency": w_latency, "cost": w_cost, "carbon": w_carbon},
    }
    pdf_bytes_ip = export_pdf("AI 2100 — Interplanetary Summary", summary_ip, rows_ip)
    st.download_button("⬇️ PDF — Interplanetary Summary", data=pdf_bytes_ip, file_name="interplanetary_summary.pdf", mime="application/pdf")

# Footer transparency
with st.expander("🔎 Transparency & Formulas"):
    st.markdown("""
**Latency**
- Earth multi-cloud: starts from editable 2025 baselines; applies year improvement (`latency_factor(year)`) and scenario impact.
- Interplanetary: physics RTT `2 × distance / c` + overhead, then year and scenario impacts.

**Cost**
- USD = `kW × hours × USD/kWh` with year factor (`cost_factor(year)`).

**Carbon**
- CO₂e (kg) = `kW × hours × gCO₂/kWh ÷ 1000` with decarbonization factor (`carbon_factor(year)`).

**Decision score (lower is better)**
- Normalizes latency, cost, carbon to [0,1] bands and applies your weights. All assumptions are editable for auditability.
""")
