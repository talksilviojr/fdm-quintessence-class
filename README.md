# FDM + Quintessence in CLASS

@misc{zenodo_fdm_quint_2025,
  author = {Corrêa Junior, Silvio A.},
  title = {Dynamical Dark Sector: FDM + Quintessence},
  doi = {10.5281/zenodo.17566560},
  url = {https://zenodo.org/records/17566560},
  year = {2025},
  publisher = {Zenodo}
}

---

## Overview

Modified **CLASS** code implementing a **unified dark sector** model with:

- **Fuzzy Dark Matter (FDM)**: ultralight scalar field  
  $$
  m_\Psi = 10^{-22}~\text{eV}
  $$

- **Axion-like Quintessence**: thawing dynamics  
  $$
  V(\phi) = M^4 \left[1 + \cos\left(\frac{\phi}{f}\right)\right]
  $$

### Key Results
| Observable | Value | Impact |
|----------|-------|--------|
| $ S_8 $ | $ \simeq 0.79 $ | Alivia Hubble + $\sigma_8$ tensions |
| $ w_{\phi,0} $ | $ \simeq -0.92 $ | Thawing behavior (consistent with DESI hints) |

---

## Installation

```bash
# 1. Clone CLASS public
git clone https://github.com/lesgourgues/class_public.git class_fdmq
cd class_fdmq

# 2. Clone modifications
git clone https://github.com/talksilviojr/fdm-quintessence-class.git modifications

# 3. Apply patch
patch -p1 < modifications/class_fdm_quintessence.patch

# 4. Compile
make clean && make -j$(nproc)
