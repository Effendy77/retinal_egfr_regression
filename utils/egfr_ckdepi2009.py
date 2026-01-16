import numpy as np

__all__ = [
    "egfr_ckdepi_2009_creatinine_nonblack",
    "egfr_ckdepi_2009_creatinine_black",
]


def _is_female_array(sex: np.ndarray) -> np.ndarray:
    """
    Convert common UKBB sex encodings to a boolean array (True = female).

    Accepts:
      - strings: "Female"/"Male", "F"/"M" (case-insensitive supported indirectly via exact matches below)
      - numeric encodings: 0/1 where 0=female, 1=male (common in UKBB exports)

    IMPORTANT: If your export uses a different coding, update this function and document it in docs.
    """
    sex = np.asarray(sex)

    # Handle strings and numeric (vectorized)
    return (
        (sex == "Female")
        | (sex == "F")
        | (sex == 0)
        | (sex == "0")
    )


def egfr_ckdepi_2009_creatinine_nonblack(scr_umolL, age_years, sex):
    """
    CKD-EPI 2009 creatinine eGFR, assuming non-Black ethnicity (no race multiplier).
    Output: eGFR in mL/min/1.73 m^2.

    Typical UK Biobank baseline fields:
      - Creatinine: 30700-0.0 (µmol/L)
      - Age: 21022-0.0 (years) or 21003-0.0 depending on your dataset
      - Sex: 31-0.0

    Equation (CKD-EPI 2009, creatinine; non-Black):
      eGFR = 141 * min(Scr/k, 1)^a * max(Scr/k, 1)^(-1.209) * 0.993^Age * (1.018 if female)

    where Scr is in mg/dL, k=0.7 (female) or 0.9 (male),
    a=-0.329 (female) or -0.411 (male).
    """
    scr_umolL = np.asarray(scr_umolL, dtype=float)
    age_years = np.asarray(age_years, dtype=float)
    is_female = _is_female_array(sex)

    # Unit conversion: µmol/L -> mg/dL
    scr_mgdl = scr_umolL / 88.4

    kappa = np.where(is_female, 0.7, 0.9)
    alpha = np.where(is_female, -0.329, -0.411)
    female_factor = np.where(is_female, 1.018, 1.0)

    ratio = scr_mgdl / kappa

    egfr = (
        141.0
        * (np.minimum(ratio, 1.0) ** alpha)
        * (np.maximum(ratio, 1.0) ** -1.209)
        * (0.993 ** age_years)
        * female_factor
    )

    return egfr


def egfr_ckdepi_2009_creatinine_black(scr_umolL, age_years, sex):
    """
    CKD-EPI 2009 creatinine eGFR, assuming Black ethnicity (includes race multiplier 1.159).
    Output: eGFR in mL/min/1.73 m^2.

    NOTE: Provided for completeness only. Not recommended for contemporary analyses and
    should NOT be used unless you have a pre-specified rationale and consistent ethnicity definition.
    """
    egfr_nonblack = egfr_ckdepi_2009_creatinine_nonblack(scr_umolL, age_years, sex)
    return egfr_nonblack * 1.159
