# FDM + Quintessence in CLASS

**Paper**: [Dynamical Dark Sector: FDM + Quintessence](https://doi.org/10.5281/zenodo.17566560)  
**Author**: Silvio A. Corrêa Junior

## Model
- **FDM**: ultra-light scalar, `m_Ψ = 10⁻²² eV`
- **Quintessence**: axion-like, `V(ϕ) = M⁴ [1 + cos(ϕ/f)]`

## Key Results
| Observable | Value | 
|-----------|--------|
| `S8` | 0.79 |
| `w_φ,0` | -0.92 |

## Installation
```bash
git clone https://github.com/lesgourg/class_public.git class_fdmq
cd class_fdmq
git clone https://github.com/talksilviojr/fdm-quintessence-class.git mods
patch -p1 < mods/class_fdm_quintessence.patch
make -j
