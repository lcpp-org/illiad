import argparse
import json
import os
import re
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import matplotlib
matplotlib.use('QtAgg')

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox
import numpy as np

from classes.meshNew import Mesh
import plot_funcs.plotFuncs as plotFuncs


POSTER_OVERDRIVE_BASELINE = {
    'background_color': '#02070D',
    'camera_dist': 1.36,
    'camera_elev': 34.0,
    'camera_azim': -86.0,
    'camera_fov_deg': 126.0,
    'axes_zoom': 1.24,
    'allow_scene_clip': True,
    'limits_scale': 0.80,
    'limits_offset': (0.10, 0.10, 0.00),
    'torus_alpha': 0.07,
    'torus_linewidth': 0.014,
    'torus_edgecolor': '#334450',
}

SAVE_DIRECTORY = os.path.join(_PROJECT_ROOT, 'output', 'framing_tuner_saved_styles')


def parse_args():
    parser = argparse.ArgumentParser(description='Interactive Boris torus framing tuner.')
    parser.add_argument(
        '--resolution',
        default='1440p',
        choices=sorted(plotFuncs._RESOLUTION_MAP),
        help='Target preview resolution. Default: 1440p.',
    )
    parser.add_argument(
        '--title',
        default='Poster Overdrive Framing Tuner',
        help='Window title text.',
    )
    parser.add_argument(
        '--name',
        default='Poster Manual 1',
        help='Default custom name used by the Save button.',
    )
    return parser.parse_args()


def _build_style_config(values):
    return plotFuncs._resolve_trace_anim_style(
        'cinematic_uiuc',
        {
            'background_color': POSTER_OVERDRIVE_BASELINE['background_color'],
            'camera_dist': values['camera_dist'],
            'camera_elev': values['camera_elev'],
            'camera_azim': values['camera_azim'],
            'camera_fov_deg': values['camera_fov_deg'],
            'axes_zoom': values['axes_zoom'],
            'allow_scene_clip': True,
            'limits_scale': values['limits_scale'],
            'limits_offset': (
                values['offset_x'],
                values['offset_y'],
                values['offset_z'],
            ),
            'torus_alpha': POSTER_OVERDRIVE_BASELINE['torus_alpha'],
            'torus_linewidth': POSTER_OVERDRIVE_BASELINE['torus_linewidth'],
            'torus_edgecolor': POSTER_OVERDRIVE_BASELINE['torus_edgecolor'],
        },
    )


def _render_preview_image(mesh, resolution, style_config, render_dpi=100):
    resolution_key = str(resolution).strip().lower()
    w_px, h_px = plotFuncs._RESOLUTION_MAP[resolution_key]
    scene_fig = plt.figure(figsize=(h_px / render_dpi, h_px / render_dpi), dpi=render_dpi)
    plotFuncs._setup_trace_anim_axes(scene_fig, mesh.R0, mesh.a, style_config=style_config)
    crop_bbox = plotFuncs._measure_trace_frame_bbox(scene_fig, style_config['background_color'])
    image = plotFuncs._compose_trace_frame(
        scene_fig,
        (w_px, h_px),
        style_config['background_color'],
        crop_bbox,
    )
    plt.close(scene_fig)
    return image


def _format_settings(values):
    return (
        "camera_dist={camera_dist:.2f}, camera_elev={camera_elev:.1f}, "
        "camera_azim={camera_azim:.1f}, camera_fov_deg={camera_fov_deg:.1f}, "
        "axes_zoom={axes_zoom:.2f}, limits_scale={limits_scale:.2f}, "
        "limits_offset=({offset_x:.2f}, {offset_y:.2f}, {offset_z:.2f})"
    ).format(**values)


def _slugify(name):
    slug = re.sub(r'[^a-z0-9]+', '_', str(name).strip().lower()).strip('_')
    return slug or 'untitled_style'


def _build_saved_style_payload(name, values):
    return {
        'slug': _slugify(name),
        'title': str(name).strip() or 'Untitled Style',
        'style': 'cinematic_uiuc',
        'style_overrides': {
            'background_color': POSTER_OVERDRIVE_BASELINE['background_color'],
            'camera_dist': round(values['camera_dist'], 2),
            'camera_elev': round(values['camera_elev'], 2),
            'camera_azim': round(values['camera_azim'], 2),
            'camera_fov_deg': round(values['camera_fov_deg'], 2),
            'axes_zoom': round(values['axes_zoom'], 2),
            'allow_scene_clip': True,
            'limits_scale': round(values['limits_scale'], 2),
            'limits_offset': [
                round(values['offset_x'], 2),
                round(values['offset_y'], 2),
                round(values['offset_z'], 2),
            ],
            'torus_alpha': POSTER_OVERDRIVE_BASELINE['torus_alpha'],
            'torus_linewidth': POSTER_OVERDRIVE_BASELINE['torus_linewidth'],
            'torus_edgecolor': POSTER_OVERDRIVE_BASELINE['torus_edgecolor'],
        },
        'title_kwargs': {
            'x': 0.92,
            'y': 0.14,
            'ha': 'right',
            'va': 'bottom',
        },
    }


def main():
    args = parse_args()
    mesh = Mesh(R0=0.72, a=0.19)

    initial_values = {
        'camera_dist': POSTER_OVERDRIVE_BASELINE['camera_dist'],
        'camera_elev': POSTER_OVERDRIVE_BASELINE['camera_elev'],
        'camera_azim': POSTER_OVERDRIVE_BASELINE['camera_azim'],
        'camera_fov_deg': POSTER_OVERDRIVE_BASELINE['camera_fov_deg'],
        'axes_zoom': POSTER_OVERDRIVE_BASELINE['axes_zoom'],
        'limits_scale': POSTER_OVERDRIVE_BASELINE['limits_scale'],
        'offset_x': POSTER_OVERDRIVE_BASELINE['limits_offset'][0],
        'offset_y': POSTER_OVERDRIVE_BASELINE['limits_offset'][1],
        'offset_z': POSTER_OVERDRIVE_BASELINE['limits_offset'][2],
    }

    fig = plt.figure(figsize=(16, 9), facecolor='#101418')
    manager = getattr(fig.canvas, 'manager', None)
    if manager is not None:
        try:
            manager.set_window_title(args.title)
        except Exception:
            pass

    preview_ax = fig.add_axes([0.05, 0.33, 0.90, 0.62])
    preview_ax.axis('off')

    style_config = _build_style_config(initial_values)
    preview_image = _render_preview_image(mesh, args.resolution, style_config)
    image_artist = preview_ax.imshow(preview_image)
    preview_ax.set_title(f'{args.title} ({args.resolution})', color='#F2F6FA', fontsize=16, pad=10)

    status_text = fig.text(
        0.05,
        0.285,
        _format_settings(initial_values),
        color='#E6EDF3',
        fontsize=10,
        family='monospace',
    )
    help_text = fig.text(
        0.05,
        0.255,
        'Adjust sliders below. Press p to print current settings, r to reset, s to save the current style snapshot.',
        color='#AAB6C2',
        fontsize=10,
    )
    help_text.set_wrap(True)
    save_status_text = fig.text(
        0.05,
        0.225,
        f'Save target: {os.path.join(SAVE_DIRECTORY, _slugify(args.name) + ".json")}',
        color='#8FB7D9',
        fontsize=9,
        family='monospace',
    )

    slider_specs = [
        ('camera_azim', 'Azim', (-180.0, 180.0), 1.0),
        ('camera_elev', 'Elev', (-90.0, 90.0), 1.0),
        ('camera_fov_deg', 'FOV', (20.0, 160.0), 1.0),
        ('camera_dist', 'Dist', (0.8, 4.0), 0.01),
        ('axes_zoom', 'Zoom', (0.6, 1.6), 0.01),
        ('limits_scale', 'LimitScale', (0.4, 1.4), 0.01),
        ('offset_x', 'Offset X', (-0.4, 0.4), 0.01),
        ('offset_y', 'Offset Y', (-0.4, 0.4), 0.01),
        ('offset_z', 'Offset Z', (-0.3, 0.3), 0.01),
    ]

    sliders = {}
    left_x = 0.08
    right_x = 0.54
    top_y = 0.18
    row_gap = 0.045
    width = 0.34
    height = 0.022

    for idx, (key, label, bounds, step) in enumerate(slider_specs):
        column_x = left_x if idx < 5 else right_x
        row_idx = idx if idx < 5 else idx - 5
        slider_ax = fig.add_axes([column_x, top_y - row_idx * row_gap, width, height], facecolor='#1B222B')
        sliders[key] = Slider(
            slider_ax,
            label,
            bounds[0],
            bounds[1],
            valinit=initial_values[key],
            valstep=step,
            color='#4C8EDA',
        )

    name_box_ax = fig.add_axes([0.05, 0.06, 0.44, 0.05])
    reset_button_ax = fig.add_axes([0.54, 0.06, 0.12, 0.05])
    save_button_ax = fig.add_axes([0.70, 0.06, 0.12, 0.05])
    print_button_ax = fig.add_axes([0.84, 0.06, 0.11, 0.05])
    name_box = TextBox(name_box_ax, 'Custom name', initial=args.name)
    print_button = Button(print_button_ax, 'Print')
    reset_button = Button(reset_button_ax, 'Reset')
    save_button = Button(save_button_ax, 'Save')

    def current_values():
        return {key: slider.val for key, slider in sliders.items()}

    def redraw(_event=None):
        values = current_values()
        style = _build_style_config(values)
        image_artist.set_data(_render_preview_image(mesh, args.resolution, style))
        status_text.set_text(_format_settings(values))
        fig.canvas.draw_idle()

    def print_settings(_event=None):
        values = current_values()
        print(_format_settings(values))

    def current_name():
        name = name_box.text.strip()
        return name or 'Untitled Style'

    def update_save_status(message=None):
        target_path = os.path.join(SAVE_DIRECTORY, _slugify(current_name()) + '.json')
        text = f'Save target: {target_path}'
        if message:
            text = f'{text} | {message}'
        save_status_text.set_text(text)
        fig.canvas.draw_idle()

    def save_settings(_event=None):
        os.makedirs(SAVE_DIRECTORY, exist_ok=True)
        payload = _build_saved_style_payload(current_name(), current_values())
        save_path = os.path.join(SAVE_DIRECTORY, payload['slug'] + '.json')
        with open(save_path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)
            handle.write('\n')
        print(f'Saved framing snapshot to {save_path}')
        update_save_status('saved')

    def reset_settings(_event=None):
        for key, slider in sliders.items():
            slider.reset()

    def on_key(event):
        if event.key == 'p':
            print_settings()
        elif event.key == 'r':
            reset_settings()
        elif event.key == 's':
            save_settings()

    def on_name_submit(_text):
        update_save_status()

    for slider in sliders.values():
        slider.on_changed(redraw)
    print_button.on_clicked(print_settings)
    reset_button.on_clicked(reset_settings)
    save_button.on_clicked(save_settings)
    name_box.on_submit(on_name_submit)
    fig.canvas.mpl_connect('key_press_event', on_key)

    print('Opened interactive framing tuner.')
    print(_format_settings(initial_values))
    plt.show()


if __name__ == '__main__':
    main()