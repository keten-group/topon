# Bundled CHARMM files

`top_all35_ethers.rtf` and `par_all35_ethers.prm` are the CHARMM C35r ether
force field, copied unchanged from `toppar/` in `toppar_c36_feb26.tgz` of the
MacKerell lab's official CHARMM36 repository
(https://github.com/mackerell-lab/charmm36-force-field), which distributes the
files under the MIT License reproduced in `LICENSE.toppar`. MD5 sums are
`ca6719ec668d203d664f970d62466ed1` (RTF) and
`61aebfa67c6f0e5b8b8fad1d5f34986d` (PRM). The RTF includes `PEGM`, the PEG
repeat unit (-CH2-O-CH2-, joined C1 to the previous unit's C2).

A config refers to them as `bundled:top_all35_ethers.rtf` and
`bundled:par_all35_ethers.prm`. For current parameters, download the latest
toppar release from the same repository or from
https://mackerell.umaryland.edu/charmm_ff.shtml.

Please cite, when using them,

- I. Vorobyov, V. M. Anisimov, S. Greene, R. M. Venable, A. Moser,
  R. W. Pastor and A. D. MacKerell Jr., Additive and classical Drude
  polarizable force fields for linear and cyclic ethers, J. Chem. Theory
  Comput. 3, 1120-1133 (2007).
- H. Lee, R. M. Venable, A. D. MacKerell Jr. and R. W. Pastor, Molecular
  dynamics studies of polyethylene oxide and polyethylene glycol:
  hydrodynamic radius and shape anisotropy, Biophys. J. 95, 1590-1599
  (2008) (the revised O-C-C-O torsion, C35r).
