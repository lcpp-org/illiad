
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

simOut = out.IOHandler(OUTPUT_DIRECTORY_NAME)
simOut.startLog()

animation_jobs = [
    # {
    # 'trace_file': ["Ion_traces_0mm_LCFS86_2eV_80V_Z1_Lithium_TRACK60x40_IdealIota3_collisionTest_both_TypicalOperation.npy",
    #                    "Ion_traces_0mm_LCFS40_2eV_80V_Z1_Lithium_TRACK60x40_IdealIota3_collisionTest_both_TypicalOperation.npy"],
    # 'runString': "2eV_80V_Z1_Lithium_TRACK60x40_IdealIota3_collisionTest_both_TypicalOperation_LCFS86_LCFS40_1440p_3",
    # },
    {
    'trace_file': [#"Ion_traces_0mm_LCFS86_2eV_120V_Z1_Lithium_TRACK60x40_IdealIota3_collisionTest_both_LiEvaporation_stride13.npy",
                   "Ion_traces_0mm_LCFS40_2eV_120V_Z1_Lithium_TRACK60x40_IdealIota3_collisionTest_both_LiEvaporation_stride13.npy"],
    'runString': "2eV_120V_Z1_Lithium_TRACK60x40_IdealIota3_collisionTest_both_LiEvaporation_1080p_stride13",
    }
]

b_hidra = Mesh(R0=0.72, a=0.19)

for job in animation_jobs:
    trace_files = job['trace_file']
    if isinstance(trace_files, str):
        trace_files = [trace_files]

    trace_sources = [
        {
            'path': os.path.join(simOut.data_dir, trace_file),
            'linecolor': plotFuncs.UIUC['il_orange'],
            'markercolor': plotFuncs.UIUC['il_blue'],
        }
        for trace_file in trace_files
    ]

    plotFuncs.boris_plotTraceAnim(
        None,
        b_hidra,
        runString=job['runString'],
        simIO=simOut,
        interval=1000/120,
        stride=1,#13,
        max_frames=4500,
        linewidth=2.5,
        line_alpha=0.25,
        line_window=40,
        trail_length=10,
        markersize=7.0,
        parallel=True,
        n_workers=14,#18,
        resolution="1080p",
        trace_sources=trace_sources,
        parallel_chunk_size=10,
        style='poster_manual_1',#'research_clean'#
    )
