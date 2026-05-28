import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np

import classes.class_outputHandler as out
from classes.meshNew import Mesh
import plot_funcs.plotFuncs as plotFuncs


OUTPUT_DIRECTORY_NAME = "It-0486_Ih-0900_noErr_1500sp_LSODA1e8"
PREVIEW_SUBDIR = "style_previews"

# Preview frame index is in animation-frame units after striding.
FRAME_INDEX = 220
# Use a larger preview frame so marker and title scale reads closer to final output.
RESOLUTION = "720p"
RENDER_DPI = 100

# Match the animation settings you care about while tweaking style.
STRIDE = 13
STEPS_PER_FRAME = 1
LINEWIDTH = 4.0
LINE_ALPHA = 0.25
LINE_WINDOW = 40
TRAIL_LENGTH = 10
MARKERSIZE = 6.0

TRACE_SOURCES = [
    {
        'filename': "Ion_traces_0mm_LCFS40_2eV_60V_Z1_Lithium_FS40_1p0ms_TRACK72x40.npy",
        'linecolor': plotFuncs.UIUC['il_orange'],
        'markercolor': plotFuncs.UIUC['il_blue'],
    },
    {
        'filename': "Ion_traces_0mm_LCFS86_2eV_60V_Z1_Lithium_FS86_1p0ms_TRACK72x40.npy",
        'linecolor': plotFuncs.UIUC['il_orange'],
        'markercolor': plotFuncs.UIUC['il_blue'],
    },
]

STYLE_GROUPS = {
    'core': [
        {
            'slug': 'research_clean',
            'title': 'Research Clean',
            'style': 'research_clean',
        },
        {
            'slug': 'conference_slide',
            'title': 'Conference Slide',
            'style': 'conference_slide',
        },
        {
            'slug': 'cinematic_uiuc',
            'title': 'Cinematic UIUC',
            'style': 'cinematic_uiuc',
        },
    ],
    'aggressive': [
        {
            'slug': 'cinematic_thin',
            'title': 'Cinematic Thin',
            'style': 'cinematic_uiuc',
            'style_overrides': {
                'torus_alpha': 0.11,
                'torus_linewidth': 0.020,
                'torus_edgecolor': '#4E5B67',
                'axes_zoom': 1.05,
            },
        },
        {
            'slug': 'cinematic_close',
            'title': 'Cinematic Close',
            'style': 'cinematic_uiuc',
            'style_overrides': {
                'camera_dist': 2.55,
                'camera_elev': 16,
                'camera_azim': -40,
                'camera_fov_deg': 62,
                'torus_alpha': 0.15,
                'torus_linewidth': 0.024,
                'axes_zoom': 1.08,
            },
        },
        {
            'slug': 'cinematic_titlecard',
            'title': 'Cinematic Titlecard',
            'style': 'cinematic_uiuc',
            'style_overrides': {
                'background_color': '#050D14',
                'torus_alpha': 0.09,
                'torus_linewidth': 0.018,
                'camera_dist': 2.35,
                'camera_elev': 12,
                'camera_azim': -26,
                'camera_fov_deg': 56,
                'axes_zoom': 1.10,
                'title_color': '#FFF3EA',
                'title_fontsize': 22,
            },
            'title_kwargs': {
                'x': 0.50,
                'ha': 'center',
            },
        },
    ],
    'sizzle': [
        {
            'slug': 'wide_punch',
            'title': 'Wide Punch',
            'style': 'cinematic_uiuc',
            'style_overrides': {
                'camera_dist': 2.20,
                'camera_elev': 24,
                'camera_azim': -56,
                'camera_fov_deg': 76,
                'axes_zoom': 1.12,
                'torus_alpha': 0.11,
                'torus_linewidth': 0.018,
                'torus_edgecolor': '#4B5966',
            },
        },
        {
            'slug': 'orbital_dive',
            'title': 'Orbital Dive',
            'style': 'cinematic_uiuc',
            'style_overrides': {
                'background_color': '#040A11',
                'camera_dist': 2.00,
                'camera_elev': 34,
                'camera_azim': -78,
                'camera_fov_deg': 88,
                'axes_zoom': 1.14,
                'torus_alpha': 0.10,
                'torus_linewidth': 0.016,
                'title_color': '#F6EEE7',
            },
        },
        {
            'slug': 'telephoto_blade',
            'title': 'Telephoto Blade',
            'style': 'cinematic_uiuc',
            'style_overrides': {
                'camera_dist': 2.30,
                'camera_elev': 9,
                'camera_azim': -14,
                'camera_fov_deg': 42,
                'axes_zoom': 1.16,
                'torus_alpha': 0.08,
                'torus_linewidth': 0.016,
                'title_color': '#FFF0E3',
            },
            'title_kwargs': {
                'x': 0.06,
                'y': 0.91,
            },
        },
    ],
    'poster': [
        {
            'slug': 'poster_left_space',
            'title': 'Poster Left Space',
            'style': 'cinematic_uiuc',
            'style_overrides': {
                'background_color': '#040B12',
                'camera_dist': 1.54,
                'camera_elev': 17,
                'camera_azim': -60,
                'camera_fov_deg': 118,
                'axes_zoom': 1.18,
                'allow_scene_clip': True,
                'limits_scale': 0.86,
                'limits_offset': (0.18, 0.05, 0.00),
                'torus_alpha': 0.10,
                'torus_linewidth': 0.015,
                'torus_edgecolor': '#3D4B58',
                'title_color': '#FFF1E7',
                'title_fontsize': 28,
            },
            'title_kwargs': {
                'x': 0.93,
                'y': 0.87,
                'ha': 'right',
            },
        },
        {
            'slug': 'poster_right_space',
            'title': 'Poster Right Space',
            'style': 'cinematic_uiuc',
            'style_overrides': {
                'background_color': '#030A10',
                'camera_dist': 1.50,
                'camera_elev': 18,
                'camera_azim': -20,
                'camera_fov_deg': 120,
                'axes_zoom': 1.18,
                'allow_scene_clip': True,
                'limits_scale': 0.85,
                'limits_offset': (-0.19, -0.01, 0.00),
                'torus_alpha': 0.09,
                'torus_linewidth': 0.015,
                'torus_edgecolor': '#43515D',
                'title_color': '#FFF0E3',
                'title_fontsize': 28,
            },
            'title_kwargs': {
                'x': 0.07,
                'y': 0.88,
                'ha': 'left',
            },
        },
        {
            'slug': 'poster_overdrive',
            'title': 'Poster Overdrive',
            'style': 'cinematic_uiuc',
            'style_overrides': {
                'background_color': '#02070D',
                'camera_dist': 1.36,
                'camera_elev': 34,
                'camera_azim': -86,
                'camera_fov_deg': 126,
                'axes_zoom': 1.24,
                'allow_scene_clip': True,
                'limits_scale': 0.80,
                'limits_offset': (0.10, 0.10, 0.00),
                'torus_alpha': 0.07,
                'torus_linewidth': 0.014,
                'torus_edgecolor': '#334450',
                'title_color': '#FFF4EA',
                'title_fontsize': 30,
            },
            'title_kwargs': {
                'x': 0.92,
                'y': 0.14,
                'ha': 'right',
                'va': 'bottom',
            },
        },
        {
            'slug': 'poster_overdrive_plus',
            'title': 'Poster Overdrive+',
            'style': 'poster_overdrivePlus',
            'title_kwargs': {
                'x': 0.93,
                'y': 0.12,
                'ha': 'right',
                'va': 'bottom',
            },
        },
    ],
}


def build_trace_sources(sim_io, source_specs):
    trace_sources = []
    for spec in source_specs:
        trace_sources.append({
            'path': os.path.join(sim_io.data_dir, spec['filename']),
            'linecolor': spec['linecolor'],
            'markercolor': spec['markercolor'],
        })
    return trace_sources


def save_contact_sheet(sheet_path, styles_with_paths, title):
    n_styles = len(styles_with_paths)
    fig, axes = plt.subplots(1, n_styles, figsize=(6 * n_styles, 5.8), dpi=140, facecolor='#0E1720')
    if n_styles == 1:
        axes = [axes]

    for ax, (spec, image_path) in zip(axes, styles_with_paths):
        image = plt.imread(image_path)
        ax.imshow(image)
        ax.set_title(spec['title'], color='#F2F6FA', fontsize=15, pad=12)
        ax.axis('off')

    fig.suptitle(title, color='#F2F6FA', fontsize=20, y=0.98)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.03, wspace=0.04)
    fig.savefig(sheet_path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_group(group_name, style_specs, trace_sources, b_hidra, preview_root):
    group_dir = os.path.join(preview_root, group_name)
    os.makedirs(group_dir, exist_ok=True)

    rendered = []
    for spec in style_specs:
        save_path = os.path.join(group_dir, f"{spec['slug']}.png")
        plotFuncs.boris_saveTracePreviewFrame(
            None,
            b_hidra,
            FRAME_INDEX,
            save_path,
            trace_sources=trace_sources,
            stride=STRIDE,
            steps_per_frame=STEPS_PER_FRAME,
            linewidth=LINEWIDTH,
            line_alpha=LINE_ALPHA,
            line_window=LINE_WINDOW,
            trail_length=TRAIL_LENGTH,
            markersize=MARKERSIZE,
            resolution=RESOLUTION,
            render_dpi=RENDER_DPI,
            style=spec['style'],
            style_overrides=spec.get('style_overrides'),
            title=spec['title'],
            title_kwargs=spec.get('title_kwargs'),
        )
        rendered.append((spec, save_path))

    contact_sheet = os.path.join(group_dir, f"{group_name}_contact_sheet.png")
    save_contact_sheet(contact_sheet, rendered, f"UIUC Trace Style Previews: {group_name}")
    return rendered, contact_sheet


def main():
    sim_io = out.IOHandler(OUTPUT_DIRECTORY_NAME)
    sim_io.startLog()

    preview_root = os.path.join(sim_io.plot_dir, PREVIEW_SUBDIR)
    os.makedirs(preview_root, exist_ok=True)

    trace_sources = build_trace_sources(sim_io, TRACE_SOURCES)
    b_hidra = Mesh(R0=0.72, a=0.19)

    print(f"Preview root: {preview_root}")
    for group_name, style_specs in STYLE_GROUPS.items():
        rendered, sheet_path = render_group(group_name, style_specs, trace_sources, b_hidra, preview_root)
        print(f"\nGroup: {group_name}")
        for spec, image_path in rendered:
            print(f"  {spec['title']}: {image_path}")
        print(f"  Contact sheet: {sheet_path}")


if __name__ == '__main__':
    main()