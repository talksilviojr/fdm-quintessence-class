"""
FDM + Quintessence Cosmology Simulator
Based on the CLASS modification by Silvio A. Corrêa Junior

Interactive Streamlit interface for cosmological background evolution
with Fuzzy Dark Matter and Quintessence (axion-like dark energy).
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from cosmology import CosmologyParams, BackgroundEvolution

st.set_page_config(
    page_title="FDM + Quintessence Cosmology",
    page_icon="🌌",
    layout="wide"
)

st.title("Cosmological Background Evolution")
st.markdown("""
**FDM + Quintessence Dynamic Dark Sector Model**

Based on the CLASS modification by Silvio A. Corrêa Junior  
Paper: [Dynamical Dark Sector: FDM + Quintessence](https://doi.org/10.5281/zenodo.17566560)

This simulator solves the background cosmological equations including:
- **Fuzzy Dark Matter (FDM)**: Ultra-light scalar field as dark matter
- **Quintessence (PNGB)**: Axion-like scalar field as dynamic dark energy
""")

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Standard Cosmological Parameters")
    
    h = st.slider("Hubble parameter (h)", 0.5, 0.8, 0.67, 0.01,
                  help="H₀ = 100h km/s/Mpc")
    
    Omega_b = st.slider("Ω_b (Baryons)", 0.01, 0.10, 0.05, 0.005,
                        help="Baryon density parameter")
    
    Omega_cdm = st.slider("Ω_cdm (Cold Dark Matter)", 0.0, 0.35, 0.25, 0.01,
                          help="CDM density parameter (reduce if using FDM)")

with col2:
    st.subheader("Dark Sector Configuration")
    
    model_type = st.radio(
        "Dark Sector Model",
        ["ΛCDM (Standard)", "Quintessence Only", "FDM Only", "FDM + Quintessence"],
        index=0,
        help="Choose the dark sector model to simulate"
    )
    
    has_quintessence = model_type in ["Quintessence Only", "FDM + Quintessence"]
    has_fdm = model_type in ["FDM Only", "FDM + Quintessence"]

st.divider()

if has_quintessence:
    st.subheader("Quintessence Parameters")
    st.markdown("**Potential:** V(φ) = M⁴[1 + cos(φ/f)]")
    
    col_q1, col_q2 = st.columns(2)
    
    with col_q1:
        M_quint_exp = st.slider(
            "log₁₀(M_quint/M_pl)", 
            -6, 0, -3,
            help="Mass scale of quintessence potential"
        )
        M_quint = 10**M_quint_exp
        st.markdown(f"**M_quint = {M_quint:.2e} M_pl**")
        
        phi_ini = st.slider(
            "φ_ini (initial field value)", 
            0.01, 3.14, 0.5, 0.01,
            help="Initial value of quintessence field"
        )
    
    with col_q2:
        f_quint_exp = st.slider(
            "log₁₀(f_quint/M_pl)", 
            -2, 2, 0,
            help="Decay constant of quintessence potential"
        )
        f_quint = 10**f_quint_exp
        st.markdown(f"**f_quint = {f_quint:.2e} M_pl**")
        
        phi_prime_ini = st.number_input(
            "φ'_ini (initial derivative)", 
            value=0.0, 
            format="%.2e",
            help="Initial derivative dφ/dt"
        )
else:
    M_quint, f_quint, phi_ini, phi_prime_ini = 1e-3, 1.0, 0.5, 0.0

if has_fdm:
    st.subheader("Fuzzy Dark Matter Parameters")
    st.markdown("**FDM**: Ultra-light scalar field dark matter with mass m ~ 10⁻²² eV")
    
    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        m_Psi_exp = st.slider(
            "log₁₀(m_Psi/eV)", 
            -26, -18, -22,
            help="FDM particle mass (typical: 10⁻²² eV)"
        )
        m_Psi = 10**m_Psi_exp
        st.markdown(f"**m_Psi = {m_Psi:.2e} eV**")
    
    with col_f2:
        Omega_fdm = st.slider(
            "Ω_FDM", 
            0.01, 0.35, 0.25, 0.01,
            help="FDM density parameter (typically replaces CDM)"
        )
        
        st.info("FDM behaves like dark energy at early times (frozen field) and like CDM at late times (oscillating field).")
else:
    m_Psi, Omega_fdm = 1e-22, 0.25

st.divider()

run_button = st.button("Run Simulation", type="primary", use_container_width=True)

if run_button:
    with st.spinner("Computing cosmological evolution..."):
        try:
            params = CosmologyParams(
                h=h,
                Omega_b=Omega_b,
                Omega_cdm=Omega_cdm,
                has_quintessence=has_quintessence,
                M_quint=M_quint,
                f_quint=f_quint,
                phi_ini=phi_ini,
                phi_prime_ini=phi_prime_ini,
                has_fdm=has_fdm,
                m_Psi=m_Psi,
                Omega_fdm=Omega_fdm,
            )
            
            bg = BackgroundEvolution(params)
            results = bg.solve()
            
            st.session_state['results'] = results
            st.session_state['params'] = params
            st.success("Simulation completed successfully!")
            
        except Exception as e:
            st.error(f"Simulation failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()

if 'results' in st.session_state:
    results = st.session_state['results']
    params = st.session_state['params']
    
    st.header("Results")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Density Evolution", 
        "Hubble Parameter", 
        "Dark Sector Fields",
        "Data Export"
    ])
    
    with tab1:
        st.subheader("Density Parameters vs Redshift")
        
        fig = make_subplots(rows=1, cols=1)
        
        z = results['z']
        mask = (z < 1e6) & (z > 0)
        
        fig.add_trace(go.Scatter(
            x=z[mask], y=results['Omega_r'][mask],
            name='Ω_radiation', mode='lines',
            line=dict(color='orange', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=z[mask], y=results['Omega_m'][mask],
            name='Ω_matter', mode='lines',
            line=dict(color='blue', width=2)
        ))
        
        if params.has_quintessence:
            fig.add_trace(go.Scatter(
                x=z[mask], y=results['Omega_quint'][mask],
                name='Ω_quintessence', mode='lines',
                line=dict(color='purple', width=2, dash='dash')
            ))
        
        if params.has_fdm:
            fig.add_trace(go.Scatter(
                x=z[mask], y=results['Omega_fdm'][mask],
                name='Ω_FDM', mode='lines',
                line=dict(color='green', width=2, dash='dot')
            ))
        
        if not params.has_quintessence:
            fig.add_trace(go.Scatter(
                x=z[mask], y=results['Omega_lambda'][mask],
                name='Ω_Λ', mode='lines',
                line=dict(color='red', width=2)
            ))
        
        fig.update_xaxes(type="log", title="Redshift z", range=[np.log10(max(z[mask].min(), 0.001)), 6])
        fig.update_yaxes(type="log", title="Density Parameter Ω", range=[-6, 0.5])
        fig.update_layout(
            height=500,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Ω_m(z=0)", f"{results['Omega_m'][-1]:.4f}")
        with col_info2:
            st.metric("Ω_DE(z=0)", f"{results['Omega_de'][-1]:.4f}")
        with col_info3:
            st.metric("Ω_r(z=0)", f"{results['Omega_r'][-1]:.2e}")
    
    with tab2:
        st.subheader("Hubble Parameter Evolution")
        
        fig2 = make_subplots(rows=1, cols=1)
        
        z = results['z']
        mask = (z < 1e4) & (z > 0)
        
        H = results['H'][mask]
        z_plot = z[mask]
        
        fig2.add_trace(go.Scatter(
            x=z_plot, y=H,
            name='H(z)', mode='lines',
            line=dict(color='darkblue', width=2)
        ))
        
        fig2.update_xaxes(type="log", title="Redshift z")
        fig2.update_yaxes(type="log", title="H(z) [km/s/Mpc]")
        fig2.update_layout(height=400)
        
        st.plotly_chart(fig2, use_container_width=True)
        
        st.metric("H₀ (today)", f"{results['H'][-1]:.2f} km/s/Mpc")
    
    with tab3:
        st.subheader("Scalar Field Evolution")
        
        if params.has_quintessence or params.has_fdm:
            
            if params.has_quintessence:
                fig3 = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(
                        'Quintessence Field φ(a)',
                        'Equation of State w_φ(z)',
                        'Field Derivative dφ/dt',
                        'Potential V(φ)'
                    )
                )
                
                a = results['a']
                z = results['z']
                mask = (z < 100) & (z >= 0)
                
                fig3.add_trace(go.Scatter(
                    x=a[mask], y=results['phi'][mask],
                    name='φ', mode='lines',
                    line=dict(color='purple', width=2)
                ), row=1, col=1)
                
                fig3.add_trace(go.Scatter(
                    x=z[mask], y=results['w_quint'][mask],
                    name='w_φ', mode='lines',
                    line=dict(color='purple', width=2)
                ), row=1, col=2)
                
                fig3.add_trace(go.Scatter(
                    x=a[mask], y=results['phi_prime'][mask],
                    name="dφ/dt", mode='lines',
                    line=dict(color='purple', width=2)
                ), row=2, col=1)
                
                fig3.add_trace(go.Scatter(
                    x=a[mask], y=results['V_quint'][mask],
                    name='V(φ)', mode='lines',
                    line=dict(color='purple', width=2)
                ), row=2, col=2)
                
                fig3.add_hline(y=-1, line_dash="dash", line_color="gray", row=1, col=2)
                
                fig3.update_xaxes(type="log", title="Scale factor a", row=1, col=1)
                fig3.update_xaxes(type="log", title="Redshift z", row=1, col=2)
                fig3.update_xaxes(type="log", title="Scale factor a", row=2, col=1)
                fig3.update_xaxes(type="log", title="Scale factor a", row=2, col=2)
                
                fig3.update_layout(height=600, showlegend=True)
                
                st.plotly_chart(fig3, use_container_width=True)
                
                w_today = results['w_quint'][-1]
                st.metric("w_φ (today)", f"{w_today:.3f}")
            
            if params.has_fdm:
                st.subheader("FDM Behavior")
                
                a_osc = results.get('a_osc_fdm', 1e-6)
                z_osc = 1/a_osc - 1 if a_osc > 0 else float('inf')
                
                col_fdm1, col_fdm2 = st.columns(2)
                with col_fdm1:
                    st.metric("Transition scale factor", f"{a_osc:.2e}")
                with col_fdm2:
                    st.metric("Transition redshift", f"{z_osc:.2e}")
                
                st.info(f"""
                **FDM Transition:**
                - For a < {a_osc:.2e} (z > {z_osc:.1e}): FDM is frozen, acts like dark energy (w ≈ -1)
                - For a > {a_osc:.2e} (z < {z_osc:.1e}): FDM oscillates, acts like CDM (w ≈ 0)
                """)
                
                fig_fdm = go.Figure()
                z = results['z']
                mask = (z < 1e6) & (z > 0)
                
                fig_fdm.add_trace(go.Scatter(
                    x=z[mask], y=results['w_fdm'][mask],
                    name='w_FDM', mode='lines',
                    line=dict(color='green', width=2)
                ))
                
                fig_fdm.update_xaxes(type="log", title="Redshift z")
                fig_fdm.update_yaxes(title="Equation of State w", range=[-1.2, 0.2])
                fig_fdm.update_layout(height=300, title="FDM Equation of State")
                
                st.plotly_chart(fig_fdm, use_container_width=True)
        else:
            st.info("Enable Quintessence or FDM to see scalar field evolution.")
    
    with tab4:
        st.subheader("Export Data")
        
        export_data = {
            'z': results['z'],
            'a': results['a'],
            'H_km_s_Mpc': results['H'],
            'E': results['E'],
            'Omega_r': results['Omega_r'],
            'Omega_m': results['Omega_m'],
            'Omega_b': results['Omega_b'],
            'Omega_cdm': results['Omega_cdm'],
            'Omega_lambda': results['Omega_lambda'],
        }
        
        if params.has_quintessence:
            export_data['phi'] = results['phi']
            export_data['phi_prime'] = results['phi_prime']
            export_data['Omega_quint'] = results['Omega_quint']
            export_data['w_quint'] = results['w_quint']
        
        if params.has_fdm:
            export_data['Omega_fdm'] = results['Omega_fdm']
            export_data['w_fdm'] = results['w_fdm']
        
        df = pd.DataFrame(export_data)
        
        st.dataframe(df.head(100), use_container_width=True)
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Full Data (CSV)",
            data=csv,
            file_name="fdm_quintessence_background.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.subheader("Simulation Parameters")
        params_dict = {
            'h': params.h,
            'Omega_b': params.Omega_b,
            'Omega_cdm': params.Omega_cdm,
            'Omega_lambda': params.Omega_lambda,
            'has_quintessence': params.has_quintessence,
            'has_fdm': params.has_fdm,
        }
        
        if params.has_quintessence:
            params_dict.update({
                'M_quint': params.M_quint,
                'f_quint': params.f_quint,
                'phi_ini': params.phi_ini,
            })
        
        if params.has_fdm:
            params_dict.update({
                'm_Psi_eV': params.m_Psi,
                'Omega_fdm': params.Omega_fdm,
            })
        
        st.json(params_dict)

st.divider()
st.markdown("""
---
**About this simulator:**

This tool implements the cosmological background evolution equations from the CLASS 
modification for Fuzzy Dark Matter + Quintessence. It solves the coupled differential 
equations for the Hubble parameter and scalar fields.

**Key Features:**
- **FDM**: Treated using fluid approximation - frozen at early times (w=-1), oscillating at late times (w=0)
- **Quintessence**: PNGB potential V(φ) = M⁴[1 + cos(φ/f)]
- **Accurate Friedmann equation** evolution from a=10⁻¹⁰ to today

**References:**
- [CLASS Code](http://class-code.net/)
- [FDM + Quintessence GitHub](https://github.com/talksilviojr/fdm-quintessence-class)
""")
