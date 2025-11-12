# FDM + Quintessence in CLASS

**Paper**: [Dynamical Dark Sector: FDM + Quintessence](https://doi.org/10.5281/zenodo.17566560)  
**Author**: Silvio A. Corrêa Junior

## Overview

This is a modified version of the [Cosmological Code CLASS](http://class-code.net/) that incorporates a dynamic dark sector model consisting of:

- **Fuzzy Dark Matter (FDM):** An ultra-light scalar field (`ψ`) with a quadratic potential $V(\psi) = \frac{1}{2} m_\psi^2 \psi^2$.
- **Quintessence:** An axion-like scalar field (`ϕ`) with a cosine potential $V(\phi) = M^4 [1 + \cos(\phi/f)]$ acting as dynamic dark energy.

This implementation allows for the study of cosmological models where the dark sector has a rich, dynamic nature.

## Key Scientific Results

The model was used to explore cosmological tensions, yielding the following key results:

| Observable      | Best-fit Value |
| :-------------- | :------------- |
| `S8`            | 0.79           |
| `w_φ,0` (hoje)  | -0.92          |

## Installation and Compilation

This repository contains a full, modified version of CLASS. The patching process described in the original README is not required when using this repository directly.

1.  **Clone this repository:**
    ```bash
    git clone https://github.com/talksilviojr/fdm-quintessence-class.git
    cd fdm-quintessence-class
    ```

2.  **Compile the code:**
    ```bash
    make clean
    make
    ```
    This will create the `class` executable in the main directory.

## How to Use: New Parameters

To run a simulation with the new components, you must set the following parameters in your `.ini` or `.param` file.

### Quintessence Parameters

-   `has_quintessence` (`yes`/`no`): Enables or disables the quintessence field.
-   `M_quint` (double): The mass scale `M` in the potential, in reduced Planck units (`M_pl`).
-   `f_quint` (double): The decay constant `f` in the potential, in reduced Planck units (`M_pl`).
-   `phi_ini` (double): The initial value of the field `ϕ`.
-   `phi_prime_ini` (double): The initial derivative of the field, `dϕ/dτ`.

### Fuzzy Dark Matter (FDM) Parameters

-   `has_fdm` (`yes`/`no`): Enables or disables the FDM field.
-   `m_Psi` (double): The mass of the FDM particle `m_ψ` in **electron-volts (eV)**.
-   `psi_ini` (double): The initial value of the field `ψ`.
-   `psi_prime_ini` (double): The initial derivative of the field, `dψ/dτ`.

## Example

1.  Create a parameter file (e.g., `my_run.ini`) with the desired configuration:

    ```ini
    # Standard cosmological parameters
    h = 0.674
    omega_b = 0.0224
    omega_cdm = 0.12  # Remainder CDM density
    
    # --- Quintessence Field ---
    has_quintessence = yes
    M_quint = 2.3e-3
    f_quint = 1.0
    phi_ini = 0.1
    phi_prime_ini = 0.0
    
    # --- Fuzzy Dark Matter Field ---
    has_fdm = yes
    m_Psi = 1.0e-22      # Mass in eV
    psi_ini = 1.0e-5
    psi_prime_ini = 0.0
    
    # Required output
    output = tCl, mPk
    lensing = yes
    ```

2.  Run CLASS from the command line:

    ```bash
    ./class my_run.ini
    ```

3.  To quickly verify the background evolution, you can use the provided Python script:

    ```bash
    python test_model.py
    ```
    This will generate two plots (`test_Ha_fdm_quint.png` and `test_rho_fdm_quint.png`) showing the evolution of the Hubble parameter and the CDM density.
