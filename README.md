# FDM + Quintessence in CLASS

**Paper**: [Dynamical Dark Sector: FDM + Quintessence](https://doi.org/10.5281/zenodo.17566560)  
**Author**: Silvio A. Corrêa Junior

## Overview

This repository provides the necessary files to patch the official [Cosmological Code CLASS](http://class-code.net/) and introduce a dynamic dark sector model, which includes:

-   **Fuzzy Dark Matter (FDM):** An ultra-light scalar field (`ψ`).
-   **Quintessence:** An axion-like scalar field (`ϕ`) acting as dynamic dark energy.

## Key Scientific Results

The model was used to explore cosmological tensions, yielding the following key results:

| Observable      | Best-fit Value |
| :-------------- | :------------- |
| `S8`            | 0.79           |
| `w_φ,0` (hoje)  | -0.92          |

## Installation and Usage

To use this modification, you need to apply the patch to a fresh clone of the official CLASS repository.

1.  **Clone the official CLASS repository:**
    ```bash
    git clone https://github.com/lesgourg/class_public.git class_with_fdmq
    cd class_with_fdmq
    ```

2.  **Apply the patch from this repository:**
    *(Assumindo que este repositório foi clonado ao lado de `class_with_fdmq`)*
    ```bash
    patch -p1 < ../fdm-quintessence-class/modifications/class_fdm_quintessence.patch
    ```

3.  **Compile the patched code:**
    ```bash
    make clean
    make
    ```

4.  **(Opcional, mas recomendado) Atualize a documentação dos parâmetros:**
    Anexe o conteúdo do arquivo de documentação fornecido ao `explanatory.ini` principal do CLASS:
    ```bash
    cat ../fdm-quintessence-class/modifications/explanatory >> explanatory.ini
    ```

## New Parameters

After applying the patch, the following parameters become available in your `.ini` file:

#### Quintessence
-   `has_quintessence` (`yes`/`no`)
-   `M_quint` (double, in `M_pl` units)
-   `f_quint` (double, in `M_pl` units)
-   `phi_ini` (double)
-   `phi_prime_ini` (double)

#### Fuzzy Dark Matter (FDM)
-   `has_fdm` (`yes`/`no`)
-   `m_Psi` (double, in **eV**)
-   `psi_ini` (double)
-   `psi_prime_ini` (double)

A documentação detalhada para esses parâmetros pode ser encontrada no arquivo `modifications/explanatory`.
