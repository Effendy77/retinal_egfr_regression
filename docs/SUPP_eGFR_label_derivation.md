# Supplementary Methods: Derivation of eGFR (UK Biobank)

## Overview
We derived baseline estimated glomerular filtration rate (eGFR; mL/min/1.73 m²) from serum creatinine using the CKD-EPI 2009 creatinine equation (non-Black; no race multiplier). The implementation used for this study is provided in `utils/egfr_ckdepi2009.py`.

## UK Biobank variables
eGFR was computed using the following baseline variables (instance 0):

- **Serum creatinine**: UK Biobank Field **30700-0.0** (µmol/L)
- **Age**: UK Biobank Field **21022-0.0** (years)  
  *(If your pipeline uses 21003-0.0 instead, state that explicitly and remain consistent.)*
- **Sex**: UK Biobank Field **31-0.0**

### Sex encoding
In our exported dataset, sex was encoded as:
- **0 = Female**
- **1 = Male**

(If your export uses "Female/Male" strings, the code supports this; document the exact encoding used.)

## Unit conversion
Creatinine was converted from µmol/L to mg/dL:
- **Scr (mg/dL) = Scr (µmol/L) / 88.4**

## CKD-EPI 2009 creatinine equation (non-Black)
eGFR was calculated as:

**eGFR = 141 × min(Scr/κ, 1)^α × max(Scr/κ, 1)^(-1.209) × 0.993^Age × (1.018 if female)**

where:
- **Scr** is serum creatinine in **mg/dL**
- **κ = 0.7 (female)**, **0.9 (male)**
- **α = −0.329 (female)**, **−0.411 (male)**
- **Age** is in years

This corresponds to the CKD-EPI 2009 creatinine equation with the **race coefficient omitted** (i.e., assuming non-Black ethnicity).

## Race coefficient (not used)
For completeness, the repository also includes a function implementing the CKD-EPI 2009 Black race coefficient (multiplier 1.159). This function was **not used** in any analyses reported in the manuscript.

## Quality control / exclusions
We excluded records with:
- missing creatinine, age, or sex
- non-positive creatinine values (≤ 0)
- non-positive age values (≤ 0)

(If you applied additional QC thresholds, add them here explicitly.)

## Code reference
- Implementation: `utils/egfr_ckdepi2009.py`
