
import os
import sys
#from pathlib import Path
# Allow running from any subdirectory: resolve the project root relative to this file
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

## IMPORTS
import classes.class_outputHandler as out
from classes.meshNew import Mesh
import plot_funcs.plotFuncs as plotFuncs


OUTPUT_DIRECTORY_NAME = "It-0486_Ih-0900_noErr_1500sp_LSODA1e8"
TAG = "Lithium_FS86_1p0ms_TRACK72x40"

sim40 = out.IOHandler(OUTPUT_DIRECTORY_NAME)
sim86 = out.IOHandler(OUTPUT_DIRECTORY_NAME)
simOut = out.IOHandler(OUTPUT_DIRECTORY_NAME)
simOut.startLog()

trace_sources = [
    {
        'path': os.path.join(sim40.data_dir, "Ion_traces_0mm_LCFS40_2eV_60V_Z1_Lithium_FS40_1p0ms_TRACK72x40.npy"),
        'linecolor': plotFuncs.UIUC['il_orange'],
        'markercolor': plotFuncs.UIUC['il_blue'],
    },
    {
        'path': os.path.join(sim86.data_dir, "Ion_traces_0mm_LCFS86_2eV_60V_Z1_Lithium_FS86_1p0ms_TRACK72x40.npy"),
        'linecolor': plotFuncs.UIUC['il_orange'],
        'markercolor': plotFuncs.UIUC['il_blue'],
    },
]

b_hidra = Mesh(R0=0.72, a=0.19)

plotFuncs.boris_plotTraceAnim(
    None,
    b_hidra,
    runString="2eV_60V_Z1_Lithium_FS40_FS86_posterOverdrivePlus_1440p4000",
    simIO=simOut,
    interval=1000/120,
    stride=13,
    max_frames=4000,
    linewidth=4.0,
    line_alpha=0.25,
    line_window=40,
    trail_length=10,
    markersize=11.0,
    parallel=True,
    n_workers=16,
    resolution="1440p",
    trace_sources=trace_sources,
    parallel_chunk_size=50,
    style='poster_overdrivePlus',
)