import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from metadata import get_files_and_metadata


def slugify(s):
    return s.lower().replace(" ", "_").replace("(", "").replace(")", "")

def set_ylim0(values):
    _, maximum = plt.ylim()
    plt.ylim(0, max(maximum, np.max(values)))

def plot_with_regression(x, y, color, **kwargs):
    coef = np.polyfit(x,y,2)
    # poly1d_fn is a function which takes in x and returns an estimate for y
    poly1d_fn = np.poly1d(coef) 
    kwargs.pop("linestyle", None)
    kwargs.pop("marker", None)
    # label only here
    plt.plot(x, y, color=color, marker='o', linestyle='', **kwargs)
    kwargs.pop("label", None)
    # plt.plot(x, y, color+'o', x, poly1d_fn(x), '--'+color, **kwargs)
    plt.plot(x, poly1d_fn(x), color=color, linestyle="--", **kwargs)

_colors = {
    "simple": [
        "green",
        "lightgreen",
        "olive",
    ],
    "pyannote": [
        "red",
        "orange",
        "gold",
    ],
    "azure": [
        "blue",
        "lightblue",
    ]
}
_used_colors = {}
_system_to_colors = {}

def get_color_engine(system_name):
    if system_name in _system_to_colors:
        return _system_to_colors[system_name]
    for k, colors in _colors.items():
        if k in system_name:
            for c in colors:
                if c not in _used_colors:
                    _used_colors[c] = system_name
                    _system_to_colors[system_name] = c
                    return c
    raise NotImplementedError(f"Could not find a color for {system_name} (already used: {_used_colors})")

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Plot memory consumption and processing times for the different engines')
    parser.add_argument('output', default = None, type=str, help='Output folder name to save figures', nargs='?')
    parser.add_argument('--ignore-old-versions', action='store_true', help='Ignore old versions of the engines')
    parser.add_argument('--spk-settings', default = "unknown", type=str, choices=["unknown", "known", "all"], help='Which speaker settings to plot')
    parser.add_argument('--only', type=str, help='Only plot the engine that contains that string')
    args = parser.parse_args()

    metadata = get_files_and_metadata()
    

    FIGSIZE = (10, 6)
    IGNORE_OLD_VERSIONS = args.ignore_old_versions
    SPK_SETTINGS = r"*" if args.spk_settings == "all" else args.spk_settings

    index_engine = 0
    engine_color = {}
    figures = {}

    for use_gpu, path_to_results in [
        (False, 'results/cpu'),
        (True, 'results/cuda'),
    ]:
        if not os.path.isdir(path_to_results):
            continue

        subfolders = [f for f in os.scandir(path_to_results) if f.is_dir()]

        ind = -1
        for path_setting_spk in sorted(subfolders, key=lambda x: x.name):

            setting_spk = os.path.basename(path_setting_spk)

            if not re.match(SPK_SETTINGS, setting_spk):
                continue
            ind += 1

            sub_sub_folders = [f for f in os.scandir(path_setting_spk)]
            sub_sub_folders = sorted(sub_sub_folders, key=lambda x: x.name)

            print_spk_number = (ind == 0)

            engine_versions = {}

            for path_engine_name in sub_sub_folders:

                engine_name_version = os.path.basename(path_engine_name)

                if args.only and args.only not in engine_name_version:
                    continue

                if IGNORE_OLD_VERSIONS:
                    # Remove the version number
                    engine_name = "-".join(engine_name_version.split("-")[:-1])
                else:
                    engine_name = engine_name_version

                if "ignore" in engine_name:
                    continue

                # Ignore old versions
                if engine_name in engine_versions:
                    print(f"Ignoring {engine_name_version} because it is an old version than {engine_versions[engine_name]}")
                    continue
                engine_versions[engine_name] = engine_name_version

                label= f"{setting_spk} / {engine_name}" if SPK_SETTINGS == "*" else f"{engine_name}"

                perf_files = [pos_json for pos_json in os.listdir(path_engine_name) if pos_json.endswith('.txt')]

                durations = []
                nb_speakers = []
                times = []
                memories = []
                vrams = []
                for index, js in enumerate(perf_files):

                    recname = os.path.splitext(js)[0]
                    recname = recname.split('.', 1)[0]
                    recname = recname+".wav"                    
                    assert recname in metadata.keys(), f"Could not find information for {path_engine_name.path}/{js}\n --> Look for '{recname}' amoung possible values {metadata.keys()}"

                    if metadata[recname]["group"] not in [
                        "",
                        "LINAGORA",
                        "ETAPE",
                        # "SUMM-RE",
                    ]:
                        continue

                    duration = metadata[recname]["duration"]
                    durations.append(duration)
                    nb_speakers.append(metadata[recname]["num_speakers"])

                    time = memory = vram = None
                    def _get_value(line):
                        try:
                            return float(line.split(":")[-1].strip().split()[0])
                        except Exception as err:
                            print(f"Error while parsing line '{line}' in {path_engine_name.path}/{js}")
                            raise err
                    
                    input_filename = os.path.join(path_engine_name, js)
                    try:
                        with open(input_filename, "r") as input_file:
                            lines = input_file.readlines()
                            for line in lines:
                                if "Time:" in line:
                                    time = _get_value(line)
                                elif "Memory Peak:" in line:
                                    memory = _get_value(line)
                                elif "VRAM Peak:" in line:
                                    vram = _get_value(line)
                    except Exception as err:
                        raise RuntimeError(f"Could not read {input_filename}") from err
                    assert time is not None
                    assert memory is not None
                    if use_gpu:
                        assert vram is not None, f"Could not find information about VRAM in {input_filename}"
                    else:
                        assert vram is None, f"Found unexpected information about VRAM in {input_filename}"

                    times.append(time)
                    memories.append(memory)
                    vrams.append(vram)
                
                if not len(durations):
                    continue

                # Sort by duration
                (durations, nb_speakers, times, memories) = zip(*sorted(zip(durations, nb_speakers, times, memories)))

                durations_minutes = [d/60 for d in durations]
                rtfs = [t/d for t,d in zip(times, durations)]

                if engine_name in engine_color:
                    color = engine_color[engine_name]
                else:
                    color = engine_color[engine_name] = get_color_engine(engine_name)
                    index_engine += 1

                plot_opts = {
                    "linestyle": "-" if ind == 0 else ":",
                    "marker": "+" if ind == 0 else "x",
                    "color": color,
                }
                def _text_color(spk):
                    return plt.cm.jet(int(spk * 256 / 15))

                idx_figure = 1 + use_gpu * 2
                title = f"Memory consumption ({'GPU' if use_gpu else 'CPU'})"
                figures[idx_figure] = title

                plt.figure(idx_figure, figsize=FIGSIZE)
                if use_gpu:
                    plt.subplot(2,1,1)
                    plot_with_regression(durations_minutes, vrams, label=f"(max={int(round(np.max(vrams),-2)):5}) {label}", **plot_opts)
                    if print_spk_number:
                        for i, spk in enumerate(nb_speakers):
                            plt.text(durations_minutes[i], 0, str(spk), color = _text_color(spk))
                    set_ylim0(vrams)
                    plt.legend()
                    plt.ylabel("VRAM Peak (MB)")
                    plt.subplot(2,1,2)
                plt.suptitle(title)
                plot_with_regression(durations_minutes, memories, label=f"(max={int(round(np.max(memories),-2)):5}) {label}", **plot_opts)
                if print_spk_number:
                    for i, spk in enumerate(nb_speakers):
                        plt.text(durations_minutes[i], 0, str(spk), color = _text_color(spk))
                set_ylim0(memories)
                plt.legend()
                plt.xlabel("Audio duration (min)")
                plt.ylabel("RAM Peak (MB)")
                
                idx_figure = 2 + use_gpu * 2
                title = f"Real Time Factor ({'GPU' if use_gpu else 'CPU'})"
                figures[idx_figure] = title
                plt.figure(2 + use_gpu * 2, figsize=FIGSIZE)
                plt.suptitle(title)
                plot_with_regression(durations_minutes, rtfs, label=f"(mean={np.mean(rtfs):.3f}) {label}", **plot_opts)
                if print_spk_number:
                    for i, spk in enumerate(nb_speakers):
                        plt.text(durations_minutes[i], 0, str(spk), color = _text_color(spk))
                set_ylim0(rtfs)
                plt.legend()
                plt.xlabel("Duration (min)")
                plt.ylabel("RTF")

    if args.output:
        for idx_figure, title in figures.items():
            output_filename = os.path.join(args.output, slugify(title) + ".png")
            print("Writing:", output_filename)
            os.makedirs(args.output, exist_ok=True)
            plt.figure(idx_figure)
            plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()