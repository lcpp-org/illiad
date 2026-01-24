## IMPORTS
import numpy as np
from numpy.polynomial import Polynomial
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
import classes.class_outputHandler as out
from classes.mesh import *

def rlp_data_loader(root=None):
    files = sorted({*root.rglob("*.CSV"), *root.rglob("*.csv")})
    by_shot: dict[str, list[tuple[int, Path]]] = {}
    SHOT_INFO: dict[str, dict[str, object]] = {
        "6143": {"I_Hel": 900,  "I_Tor": 486, "FWD_REV": "REV", "StartPos_cm": 16.0, "COND": "iota3_rev"},
        "6142": {"I_Hel": 900,  "I_Tor": 486, "FWD_REV": "REV", "StartPos_cm": 16.0, "COND": "iota3_rev"},
        "6141": {"I_Hel": 900,  "I_Tor": 486, "FWD_REV": "REV", "StartPos_cm": 14.0, "COND": "iota3_rev"},
        "6140": {"I_Hel": 900,  "I_Tor": 486, "FWD_REV": "REV", "StartPos_cm": 12.0, "COND": "iota3_rev"},
        "6139": {"I_Hel": 900,  "I_Tor": 486, "FWD_REV": "REV", "StartPos_cm": None, "COND": "n/a"},
        "6138": {"I_Hel": 900,  "I_Tor": 486, "FWD_REV": "REV", "StartPos_cm": None, "COND": "n/a"},
        "6137": {"I_Hel": 790,  "I_Tor": 486, "FWD_REV": "REV", "StartPos_cm": None, "COND": "n/a"},
        "6136": {"I_Hel": 790,  "I_Tor": 486, "FWD_REV": "REV", "StartPos_cm": 12.0, "COND": "iota4_rev"},
        "6135": {"I_Hel": 790,  "I_Tor": 486, "FWD_REV": "REV", "StartPos_cm": 14.0, "COND": "iota4_rev"},
        "6134": {"I_Hel": 790,  "I_Tor": 486, "FWD_REV": "REV", "StartPos_cm": 16.0, "COND": "iota4_rev"},
        "6133": {"I_Hel": 790,  "I_Tor": 486, "FWD_REV": "REV", "StartPos_cm": 16.0, "COND": "iota4_rev"},
        "6132": {"I_Hel": 790,  "I_Tor": 486, "FWD_REV": "REV", "StartPos_cm": None, "COND": "n/a"},
        "6131": {"I_Hel": 1580, "I_Tor": 972, "FWD_REV": "FWD", "StartPos_cm": 16.0, "COND": "n/a"},
        "6130": {"I_Hel": 790,  "I_Tor": 486, "FWD_REV": "FWD", "StartPos_cm": 16.0, "COND": "iota4_dflt"},
        "6129": {"I_Hel": 790,  "I_Tor": 486, "FWD_REV": "FWD", "StartPos_cm": 14.0, "COND": "iota4_dflt"},
        "6128": {"I_Hel": 790,  "I_Tor": 486, "FWD_REV": "FWD", "StartPos_cm": 12.0, "COND": "iota4_dflt"},
        "6127": {"I_Hel": 900,  "I_Tor": 486, "FWD_REV": "FWD", "StartPos_cm": None, "COND": "n/a"},
        "6126": {"I_Hel": 900,  "I_Tor": 486, "FWD_REV": "FWD", "StartPos_cm": 12.0, "COND": "iota3_dflt"},
        "6125": {"I_Hel": 900,  "I_Tor": 486, "FWD_REV": "FWD", "StartPos_cm": 12.0, "COND": "iota3_dflt"},
        "6124": {"I_Hel": 900,  "I_Tor": 486, "FWD_REV": "FWD", "StartPos_cm": 18.0, "COND": "iota3_dflt"},
        "6123": {"I_Hel": 900,  "I_Tor": 486, "FWD_REV": "FWD", "StartPos_cm": 16.0, "COND": "iota3_dflt"},
    }
    SHOT_RE = re.compile(r"(\d{4})")
    REP_RE = re.compile(r"_(\d+)\.[Cc][Ss][Vv]$")

    def shot_and_rep(path: Path) -> tuple[str, int]:
        m = SHOT_RE.search(str(path))
        if not m: raise ValueError("no 4-digit shot id found")
        shot = m.group(1)

        m2 = REP_RE.search(path.name)
        rep = int(m2.group(1)) if m2 else 1
        return shot, rep

    def load_profile(csv_path: Path, shot: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]

        df[POS_COL] = pd.to_numeric(df[POS_COL], errors="coerce") / 10.0  # convert mm to cm (CSV mislabeled)
        df[NE_COL] = pd.to_numeric(df[NE_COL], errors="coerce")
        df = df.dropna(subset=[POS_COL, NE_COL]).sort_values(POS_COL)

        # Attach shot metadata + compute total distance
        info = SHOT_INFO.get(shot, {})
        start = info.get("StartPos_cm", None)

        df["Shot"] = shot
        df["COND"] = info.get("COND", None)
        df["StartPos_cm"] = start

        # If start is unknown, keep total distance as NaN
        df[TOTAL_POS_COL] = df[POS_COL] + float(start) if start is not None else np.nan
        df[TOTAL_POS_COL] =  df[TOTAL_POS_COL] - 4.0  # adjust for something

        return df[[TOTAL_POS_COL, NE_COL, "Shot", "COND", "StartPos_cm"]]

    for f in files:
        shot, rep = shot_and_rep(f)
        by_shot.setdefault(shot, []).append((rep, f))
    frames: list[pd.DataFrame] = []

    for shot in sorted(by_shot):
        entries = sorted(by_shot[shot], key=lambda x: x[0])
        for rep, f in entries:
            frames.append(load_profile(f, shot=shot))

    total_dataframe = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=[TOTAL_POS_COL, NE_COL, "Shot", "COND", "StartPos_cm"])
    )

    return total_dataframe


def main():
    ## SET UP RUN DIRECTORY AND LOGGING
    simIO = out.IOHandler(OUTPUT_DIRECTORY_NAME) 
    simIO.startLog()
    ## LOAD SCALAR FIELDS FOR ALL CONDITIONS
    psi_iota3_dflt = Mesh(R0=0.72, a=0.19)
    psi_iota3_dflt.loadScalarField('input_files/big_grid_linearTEST.npy', period=np.array([0, 1, 1]))
    psi_iota3_rev = Mesh(R0=0.72, a=0.19)
    psi_iota3_rev.loadScalarField('input_files/psiNorm_i3rev.npy', period=np.array([0, 1, 1]))
    psi_iota4_dflt = Mesh(R0=0.72, a=0.19)
    psi_iota4_dflt.loadScalarField('input_files/psiNorm_i4dflt.npy', period=np.array([0, 1, 1]))
    psi_iota4_rev = Mesh(R0=0.72, a=0.19)
    psi_iota4_rev.loadScalarField('input_files/psiNorm_i4rev.npy', period=np.array([0, 1, 1]))

    ## LOAD RLP MEASUREMENT DATA
    ALL_RLP_DATA = rlp_data_loader(root=DATA_PATH)

    ## GENERATE PSI PROFILES ALONG RLP PATH
    PHI_GEN_RAD = np.radians(306.) # RLP phi-location
    DIST_PLOT = np.arange(0.0, 0.38, 0.005) # RLP radius location
    iota3_dflt_profile = np.zeros(len(DIST_PLOT))
    iota3_rev_profile = np.zeros(len(DIST_PLOT))
    iota4_dflt_profile = np.zeros(len(DIST_PLOT))
    iota4_rev_profile = np.zeros(len(DIST_PLOT))
    for i, dist in enumerate(DIST_PLOT):
        if dist < psi_iota3_dflt.a:
            theta = 0.0
            rad = psi_iota3_dflt.a - dist
        else:
            theta = np.pi
            rad = dist - psi_iota3_dflt.a

        rtp_point = np.array([rad, theta, PHI_GEN_RAD])

        iota3_dflt_profile[i] = psi_iota3_dflt.interpScalarField(rtp_point, Cart=False)[0]
        iota3_rev_profile[i] = psi_iota3_rev.interpScalarField(rtp_point, Cart=False)[0]
        iota4_dflt_profile[i] = psi_iota4_dflt.interpScalarField(rtp_point, Cart=False)[0]
        iota4_rev_profile[i] = psi_iota4_rev.interpScalarField(rtp_point, Cart=False)[0]

    # plotting
    for cond in CONDITIONS:
        if cond == 'iota3_dflt':
            this_profile = iota3_dflt_profile
            this_label = 'iota3_dflt Psi Profile'
        elif cond == 'iota3_rev':
            this_profile = iota3_rev_profile
            this_label = 'iota3_rev Psi Profile'
        elif cond == 'iota4_dflt':
            this_profile = iota4_dflt_profile
            this_label = 'iota4_dflt Psi Profile'
        elif cond == 'iota4_rev':
            this_profile = iota4_rev_profile
            this_label = 'iota4_rev Psi Profile'

        peak_ne = 1.0
        reshaped_psi = (1 - (1 - this_profile)**(INPUT_ALPHA)) 
        print(f'{reshaped_psi=}' )
        this_data = ALL_RLP_DATA[ALL_RLP_DATA['COND'] == cond]

        # Loop through each Shot within this condition
        plt.figure()
        for shot, shot_df in this_data.groupby("Shot"):
            shot_df = shot_df.sort_values(TOTAL_POS_COL)
            pos = shot_df[TOTAL_POS_COL].to_numpy()
            ne = shot_df[NE_COL].to_numpy()
            plt.plot(pos, ne, ':x', color='lightgrey', markersize=2.0, linewidth=0.5, label=f'RLP Data (Shot {shot})')
            
            p = Polynomial.fit(pos, ne, 9)
            ne_fit = p(pos)
            plt.plot(pos, ne_fit, linewidth=1, label=f'Poly Fit (Shot {shot})')

            peak_ne = max(peak_ne, float(np.max(ne_fit)))

        print(f'{peak_ne=}')
        scaled_profile = reshaped_psi * peak_ne
        plt.plot(DIST_PLOT * 100, scaled_profile, ':b', linewidth=1.5, label=this_label)

        plt.xticks(np.arange(0, 39, 2))
        plt.xlabel('Distance from Outer Wall [cm]', fontsize=10)
        plt.ylabel('$n_e$ [m$^{-3}$]', fontsize=10)
        plt.legend(loc='upper right', fontsize=8)
        plt.grid(which='both')
        plt.tick_params(axis='both', labelsize=8)
        simIO.saveFig(cond + '_' + TAG + '_psi_profile.png', dpi=300)

if __name__ == "__main__":
    ## SET SIMULATION INPUTS:
    CONDITIONS = ['iota3_dflt', 'iota3_rev', 'iota4_dflt', 'iota4_rev']
    DATA_PATH = Path("input_files") / "RLP_Results"
    POS_COL = "Position (cm)"
    NE_COL = "ne (m-3)"
    TOTAL_POS_COL = "Total distance (cm)"

    INPUT_ALPHA =1.0 # manual value of alpha psi profile scaling exponent
    OUTPUT_DIRECTORY_NAME = "RLP_FITTING_ALL"
    TAG = 'alpha1p0'

    main()