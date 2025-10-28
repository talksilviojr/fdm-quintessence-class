# FDM + Quintessence in CLASS  
**Author**: Silvio A. Corrêa Junior  
**Paper**: [arXiv:XXXX.XXXXX](https://arxiv.org) *(to be updated)*  
**JCAP Submission**: October 2025  

## Overview
Modified CLASS code for a unified dark sector:  
- **Fuzzy Dark Matter (FDM)**: \( m_\Psi = 10^{-22} \) eV  
- **Axion-like Quintessence**: \( V(\phi) = M^4 [1 + \cos(\phi/f)] \)  

**Results**:  
- \( S_8 \approx 0.79 \) (alivia tensão)  
- \( w_{\phi,0} \approx -0.92 \) (thawing)  

## Installation
```bash
git clone https://github.com/lesgourgues/class_public.git class_fdmq
cd class_fdmq
git clone https://github.com/silviocorrea/fdm-quintessence-class.git modifications
patch -p1 < modifications/class_fdm_quintessence.patch
make
