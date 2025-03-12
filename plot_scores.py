import json
import os
import subprocess
import re
import matplotlib.pyplot as plt
from slugify import slugify
import numpy as np

from remove_overlaps import *
from remove_overlaps import create_rttm_without_overlap
from convert_json2rttm import json2rttm
from metadata import get_files_and_metadata
from plot_memory_time import get_color_engine

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "CDER_Metric"))
from diarization import CSSDErrorRate, DiarizationErrorRate, DiarizationPurity,DiarizationCoverage, DiarizationPurityCoverageFMeasure
from rttm_io import rttm_read

sys.path.append(os.path.join(os.path.dirname(__file__), "BER"))
from ber_score import main as ber_score


def compute_number_of_speakers(ref, hyp, *kargs, **kwargs):
    def _get_speaker_names(annotation):
        return len(annotation.chart())
    return (_get_speaker_names(ref), _get_speaker_names(hyp))

def wrap_rttm_read(func):
    def wrapper(ref, hyp, *kargs, **kwargs):

        ref = rttm_read(ref)
        hyp = rttm_read(hyp)

        assert len(ref) == 1
        assert len(hyp) <= 1
        assert ref.keys() == hyp.keys() or len(ref) == 0 or len(hyp) == 0
        ref = list(ref.values())[0][1]
        hyp = list(hyp.values())[0][1] if hyp else empty_segmentation()

        return func(ref, hyp, *kargs, **kwargs)
    return wrapper

def empty_segmentation():
    raise NotImplementedError("Empty segmentation")

BER_SCORES = {}
def get_ber_score(ref, hyp, score):
    global BER_SCORES
    if hyp not in BER_SCORES:    
        sys.argv = ["", "-s", hyp, "-r", ref]
        jer, ser, ref_spk_ber, fa_duraion, fa_segnum, fa_mean, ber = ber_score()
        BER_SCORES[hyp] = {
            "JER": jer,
            "SER": ser,
            "BER": ber,
            "FA_DURATION": fa_duraion,
            "FA_SEGNUM": fa_segnum,
            "FA_MEAN": fa_mean,
            "REF_SPK_BER": ref_spk_ber,
        }
    return BER_SCORES[hyp][score]

def getter_ber_score(score):
    return lambda ref, hyp: get_ber_score(ref, hyp, score)


def get_scores_from_perl_script(ref, hyp, collar=0):
    exename = os.path.join(os.path.dirname(__file__), "CDER_Metric", "md-eval-v21.pl")
    command = f"{exename} -c {collar} -r {ref} -s {hyp}"
    output = subprocess.check_output(command, shell=True).decode("utf-8")
    scores = {}
    detail_times = {}
    for line in output.splitlines():
        if line.startswith(" OVERALL SPEAKER DIARIZATION ERROR ="):
            scores["DER"] = float(line.split("=")[1].strip().split()[0]) / 100.0
        for what in ["SCORED SPEAKER", "MISSED SPEAKER", "FALARM SPEAKER", "SPEAKER ERROR"]:
            if line.lstrip().startswith(what+ " TIME"):
                detail_times[what] = float(line.split("=")[1].strip().split()[0])
    assert "DER" in scores
    total = detail_times["SCORED SPEAKER"]
    scores["Subs"] = detail_times["SPEAKER ERROR"] / total
    scores["Del"] = detail_times["MISSED SPEAKER"] / total
    scores["Ins"] = detail_times["FALARM SPEAKER"] / total
    scores["ConfusionRate"] = (scores["Subs"] + scores["Del"])
    scores[f"DER_{collar}"] = scores["DER"]
    return scores

def format_system_name(name):
    if name.startswith("pyannote") or name.startswith("simple"):
        name = "linto-" + name
    if "linto-pyannote" in name:
        if "1.0.0" in name:
            name += " (2.1)"
        elif "1.1.0" in name:
            name += " (3.1)"
    return name

TO_MAXIMIZE = ["Purity", "Coverage", "FMeasure"]

def custom_violinplot(data, positions=None, color="red", plot_quartiles=True, **kwargs):

    if positions is None:
        positions = range(1, len(data) + 1)

    if isinstance(color, list):
        assert len(color) == len(data), f"{len(color)=} {len(data)=}"
        assert len(color) == len(positions)
        for x, y, c in zip(positions, data, color):
            if not len(y): continue
            custom_violinplot([y], positions=[x], color=c, plot_quartiles=plot_quartiles, **kwargs)
        return
    
    alpha = kwargs.pop("alpha", 1)

    parts = plt.violinplot(
        data,
        positions=positions,
        showmedians=plot_quartiles,
        showmeans=False,
        quantiles=([[0.25, 0.75]] * len(data)) if plot_quartiles else [],
        showextrema=plot_quartiles,
        **kwargs)

    for pc in parts['bodies']:
        # pc.set_facecolor('#D43F3A')
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.5 * alpha)

    if not plot_quartiles:
        # parts.keys()
        # for key in ['bodies', 'cmaxes', 'cmins', 'cbars', 'cmedians', 'cquantiles']:
        #     for pc in parts[key]:
        #         pc.set_linewidth(0)
        #         pc.set_alpha(0)
        return parts

    means = [np.mean(d) for d in data]
    quartiles = [np.percentile(d, [25, 50, 75]) for d in data]
    quartile1, medians, quartile3 = zip(*quartiles)
    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    plt.scatter(positions, means, marker='o', color=color, s=30, zorder=3)
    plt.scatter(positions, medians, marker='o', color='k', s=30, zorder=3)
    plt.vlines(positions, quartile1, quartile3, color='k', linestyle='-', lw=5)
    plt.vlines(positions, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Plot memory consumption and processing times for the different engines')
    parser.add_argument('output', default = None, type=str, help='Output folder name to save figures', nargs='?')
    parser.add_argument('--skip-overlap', action='store_true', default=False, help='To skip overlaps when evaluating')
    parser.add_argument('--collar', type=float, default=0, help='Collar for overall DER reported')
    parser.add_argument('--use-mdeval', action='store_true', default=False, help='Use md-eval perl script to compute DER')
    parser.add_argument('--complete', action='store_true', default=False, help='Compute all the metrics (not only DER and JER)')
    parser.add_argument('--ignore-old-versions', action='store_true', help='Ignore old versions of the engines')
    parser.add_argument('--recompute', action='store_true', default=False, help='To force recompute all the DERs')
    parser.add_argument('--plot_distribution', default="violin", type=str, help='Type of plot for distribution of DERs', choices=["violin", "boxplot"])
    parser.add_argument('--verbose', action='store_true', default=False, help='Print more information (number of speakers...)')
    args = parser.parse_args()

    _score_funcs = {
        "JER": getter_ber_score("JER"),
        # "SpeakerConfusionErrorRate": wrap_rttm_read(DiarizationErrorRate(collar=args.collar, count_false_alarm=False)),
        "NumberOfSpeakers": wrap_rttm_read(compute_number_of_speakers),
    }

    key_DER = "DER" if args.collar == 0 else f"DER_{args.collar}"

    if args.complete:
        _score_funcs = _score_funcs | {
            "CSSDER": wrap_rttm_read(CSSDErrorRate(collar=args.collar)),
            "Purity": wrap_rttm_read(DiarizationPurity(collar=args.collar)),
            "Coverage": wrap_rttm_read(DiarizationCoverage(collar=args.collar)),
            "FMeasure": wrap_rttm_read(DiarizationPurityCoverageFMeasure(collar=args.collar)),
            "SER": getter_ber_score("SER"),
            "BER": getter_ber_score("BER"),
        }

    perfnames_all = list(_score_funcs.keys()) + [key_DER]

    if args.use_mdeval:
        perfnames_all += [
            f"DER_{args.collar}",
            "ConfusionRate",
            "Subs",
            "Del",
            "Ins",
        ]   
    else:
        _score_funcs[key_DER] = wrap_rttm_read(DiarizationErrorRate(collar=args.collar))

    perfnames_final = [p for p in perfnames_all if p not in [
        "Subs",
        "Del",
        "Ins",
    ] and not p.startswith("DER_")]

    IGNORE_OLD_VERSIONS = args.ignore_old_versions
    RECOMPUTE_DER = args.recompute
    REMOVE_OVERLAPS = args.skip_overlap
    COMPUTE_DER_WITH_MDEVAL = args.use_mdeval

    _warned_if_recompute = False
    
    path_to_rttm_ref = "data/rttm/"
    metadata = get_files_and_metadata()

    dataset_names = sorted(list(set([m["group"] for m in metadata.values() if m["group"]])))

    # First collect all the files on which to compute DER
    all_files = {}
    for path_to_results in "results/cuda", "results/cpu", :
        subfolders = [f for f in os.scandir(path_to_results) if f.is_dir()]
        for ind, path_setting_spk in enumerate(sorted(subfolders, key=lambda x: x.name)):
            engine_versions = {}
            setting_spk = os.path.basename(path_setting_spk)
            sub_sub_folders = [f for f in os.scandir(path_setting_spk)]
            sub_sub_folders = sorted(
                sub_sub_folders, key=lambda x: x.name.split("-")[-1], reverse=True)
            for path_engine_name in sub_sub_folders:
                engine_name_version = os.path.basename(path_engine_name)
                if IGNORE_OLD_VERSIONS:
                    # Remove the version number
                    engine_name = "-".join(engine_name_version.split("-")[:-1])
                else:
                    engine_name = engine_name_version
                # Ignore old versions
                if engine_name in engine_versions:
                    print(
                        f"Ignoring {engine_name_version} because it is an old version than {engine_versions[engine_name]}")
                    continue

                # Format engine name for future xticks labels
                engine_name = engine_name.replace("-", "\n").replace("_", "\n")
                setting_spk = {
                    "known_spk": "Given # of speakers",
                    "unknown_spk": "Unknown # of speakers",
                }.get(setting_spk, setting_spk)

                engine_versions[engine_name] = engine_name_version
                json_files = [
                    pos_json for pos_json in os.listdir(path_engine_name) if pos_json.endswith('.json') and not pos_json.startswith("test")
                ]
                all_files[setting_spk] = all_files.get(setting_spk, {})
                all_files[setting_spk][engine_name] = all_files[setting_spk].get(engine_name, {})
                for fn in sorted(json_files):
                    if fn not in all_files[setting_spk][engine_name]:
                        all_files[setting_spk][engine_name][fn] = os.path.join(path_engine_name, fn)
                if setting_spk == "unknown_spk" and engine_name == "pyannote-1.1.0":
                    all_files[setting_spk][engine_name]["004a_PARH.json"]

    all_engine_names = set()
    for setting_spk, all_files_for_spk_setting in all_files.items():
        for engine_name in all_files_for_spk_setting.keys():
            if "ignore" in engine_name:
                continue
            all_engine_names.add(engine_name)
    all_engine_names = sorted(all_engine_names, key= lambda x: ("streaming" not in x, x))

    num_done = 0
    figures = {}

    for ind, (setting_spk, all_files_for_spk_setting) in enumerate(all_files.items()):

        all_accuracies = {}
        all_ders = {}
        for engine_name in all_engine_names:

            json_files = all_files_for_spk_setting.get(engine_name, {}).values()
            for dataset_name in dataset_names:
                for k in perfnames_final:
                    all_accuracies[k] = all_accuracies.get(k, {})
                    all_accuracies[k][dataset_name] = all_accuracies[k].get(dataset_name, {})
                    all_accuracies[k][dataset_name][engine_name] = all_accuracies[k][dataset_name].get(engine_name, [])

            all_rttm_ref = {}
            all_rttm_hyp = {}
            for index, json_hyp in enumerate(json_files):

                recname = os.path.basename(os.path.splitext(json_hyp)[0])
                recname = recname.split('.', 1)[0]

                if recname+".wav" in metadata:
                    dataset_name = metadata[recname+".wav"]["group"]
                else:
                    continue
                    # assert False, f"ERROR: {recname} has no group (file {json_hyp})"
                    # dataset_name = "UNK"
                assert dataset_name, f"ERROR: {recname} has no group (file {json_hyp})"
                if dataset_name not in dataset_names:
                    continue

                scores = perf_file = None
                if RECOMPUTE_DER is not None:
                    perf_file = os.path.splitext(json_hyp)[0] + ".der"
                    if not REMOVE_OVERLAPS:
                        perf_file += "_overlap"
                    # if COMPUTE_DER_WITH_MDEVAL:
                    #     perf_file += "_mdeval"
                    if not RECOMPUTE_DER and os.path.exists(perf_file):
                        with open(perf_file) as f:
                            scores = json.load(f)
                        if sorted(scores.keys()) != sorted(perfnames_all):
                            if not _warned_if_recompute:
                                print(f"WARNING: force to recompute performances for {perf_file} (and maybe others...) because the keys are different ({sorted(scores.keys())}) != ({sorted(perfnames_all)})")
                                _warned_if_recompute = True
                            # Force recomputation
                            scores = None

                rttm_ref = recname+".rttm"
                rttm_ref = os.path.join(path_to_rttm_ref, rttm_ref)

                if not os.path.exists(rttm_ref):
                    print(f"WARNING: ignored {recname} because could not find {rttm_ref}")
                    continue

                # print(f"Computing... {perf_file if perf_file else json_hyp}")

                rttm_hyp = os.path.splitext(json_hyp)[0] + ".rttm" #    tempfile.mktemp(suffix=".rttm")
                rttm_ref_nooverlap = rttm_ref + "_nooverlap.rttm" if REMOVE_OVERLAPS else rttm_ref
                rttm_hyp_nooverlap = rttm_hyp + "_nooverlap.rttm" if REMOVE_OVERLAPS else rttm_hyp

                json2rttm(json_hyp, rttm_hyp)

                if REMOVE_OVERLAPS:
                    create_rttm_without_overlap(
                        rttm_ref, rttm_hyp,
                        rttm_ref_nooverlap, rttm_hyp_nooverlap,
                    )

                all_rttm_ref[dataset_name] = all_rttm_ref.get(dataset_name, []) + [rttm_ref_nooverlap]
                all_rttm_hyp[dataset_name] = all_rttm_hyp.get(dataset_name, []) + [rttm_hyp_nooverlap]

                if scores is None:

                    # Compute DER and co
                    try:
                        scores = dict((k, v(rttm_ref_nooverlap, rttm_hyp_nooverlap))
                            for k, v in _score_funcs.items())
                    except Exception as err:
                        raise RuntimeError(f"Error processing {rttm_hyp_nooverlap}") from err

                    if COMPUTE_DER_WITH_MDEVAL:
                        other_scores = get_scores_from_perl_script(rttm_ref_nooverlap, rttm_hyp_nooverlap, collar=args.collar)
                        scores.update(other_scores)

                    if perf_file is not None:
                        with open(perf_file, "w") as f:
                            json.dump(scores, f, indent=4)

                    # Remove temporary files
                    # for tmpfile in (rttm_hyp, rttm_ref_nooverlap, rttm_hyp_nooverlap,) if REMOVE_OVERLAPS else (rttm_hyp,):
                    #     os.remove(tmpfile)

                for k, val in scores.items():
                    if k not in perfnames_final:
                        continue
                    if k != "NumberOfSpeakers":
                        val = 100 * val # Convert to percent here 
                    else:
                        if abs(val[1] - val[0]) > 10 or setting_spk == "known_spk" and val[0] != val[1]:
                            if args.verbose:
                                print(f"{json_hyp} -- number of speakers: real= {val[0]}, predicted= {val[1]}")
                    assert k in all_accuracies, f"ERROR: {k} not in {all_accuracies.keys()}"
                    all_accuracies[k][dataset_name][engine_name] = all_accuracies[k][dataset_name].get(engine_name, []) + [val]


            # Make overall DER
            for dataset_name in all_rttm_ref:
            
                # Concatenate all RTTM files
                # cumulated_ref = tempfile.mktemp(suffix=".rttm")
                # cumulated_hyp = tempfile.mktemp(suffix=".rttm")
                cumulated_ref = os.path.join(path_to_rttm_ref, f"{dataset_name}.rttm")
                cumulated_hyp = os.path.join(path_engine_name, f"{dataset_name}.rttm")
                def converted_lines(lines, prefix=dataset_name+"-", from_linto=False):
                    for line in lines.splitlines():
                        f = line.split()
                        spk = f[7]
                        if from_linto and spk.startswith("spk"):
                            num_speaker = re.search(r"spk(\d+)", spk).group(1)
                            num_speaker = int(num_speaker)-1
                            if num_speaker >= 0:
                                spk = f"spk{num_speaker:02d}"
                            else:
                                # -1 -> 0
                                spk = f"{num_speaker+1:02d}"
                        spk = prefix + spk
                        f[7] = spk
                        yield " ".join(f)
                with open(cumulated_ref, "w") as fp:
                    for ref in sorted(all_rttm_ref[dataset_name]):
                        with open(ref) as f:
                            empty = True
                            for line in converted_lines(f.read()):
                                fp.write(line + "\n")
                                empty = False
                            assert not empty, f"ERROR: empty file {ref}"
                with open(cumulated_hyp, "w") as fp:
                    for hyp in sorted(all_rttm_hyp[dataset_name]):
                        with open(hyp) as f:
                            empty = True
                            for line in converted_lines(f.read(), from_linto=True):
                                fp.write(line + "\n")
                                empty = False
                            if empty and args.verbose:
                                print(f"WARNING: empty file {hyp}")
                # Compute DER
                # scores = _score_funcs["DER"](cumulated_ref, cumulated_hyp)
                scores = get_scores_from_perl_script(cumulated_ref, cumulated_hyp, collar=args.collar)
                scores = scores["DER"] * 100

                # Collect DER
                all_ders[engine_name] = all_ders.get(engine_name, {})
                all_ders[engine_name][dataset_name] = scores
                os.remove(cumulated_ref)
                os.remove(cumulated_hyp)

        for iperf, (perfname, perfs) in enumerate(all_accuracies.items()):

            num_plots = len(subfolders)
            ind0 = ind

            is_number_of_speaker = (perfname == "NumberOfSpeakers")
            if is_number_of_speaker:
                if "unknown" not in setting_spk.lower():
                    continue
                num_plots = 1
                ind0 = 0

            idx_figure = iperf + 1
            figures[idx_figure] = f"Accuracy: {perfname}"

            plt.figure(idx_figure, figsize=(20, 10))
            if not dataset_names:
                continue

            for ilabel, dataset_name in enumerate(dataset_names):
                if dataset_name not in perfs:
                    continue
                
                DER_result = perfs[dataset_name]

                plt.subplot(num_plots, len(dataset_names), ilabel+1 + ind0 * len(dataset_names))

                if not is_number_of_speaker:
                    values = list(DER_result.values())

                else:
                    new_values = []
                    for vals in DER_result.values():
                        for v in vals:
                            assert len(v) == 2
                    values = [[v[1] - v[0] for v in vals] for vals in DER_result.values()]

                # Remove empty ones
                positions = list(range(1,1+len(values)))
                ticks = list(DER_result.keys())
                for i in positions[::-1]:
                    i -= 1
                    if not values[i]:
                        del positions[i]
                        del values[i]
                        ticks[i] = ""

                if args.plot_distribution == "violin":
                    colors = [get_color_engine(t) for t in ticks if t]
                    violin_parts = custom_violinplot(
                        values,
                        positions=positions,
                        color=colors,
                    )
                else:
                    plt.boxplot(values, positions=positions, whis=100)

                nonentiles = []
                for v in values:
                    nonentiles.append(np.percentile(v, 90))

                plt.xticks(range(1, len(DER_result) + 1), ticks, rotation=10)
                plt.xlim(0.5, len(DER_result) + 0.5)

                # Rescale y-axis
                if is_number_of_speaker:
                    plt.axhline(y=0, color='black', linestyle='--')
                else:
                    xmin, xmax, ymin, ymax = plt.axis()
                    if perfname not in TO_MAXIMIZE:
                        plt.axis([xmin, xmax, 0, max(nonentiles)])
                    else:
                        plt.axis([xmin, xmax, ymin, 100])

                if ilabel == 0:
                    plt.ylabel(setting_spk)
                if ind0 == 0:
                    plt.title(dataset_name)
            
            perfname_ = "Difference in number of speakers" if is_number_of_speaker else (perfname + " (%)")
            plt.suptitle(perfname_)


        print(f"DER {setting_spk.replace('_', ' ')} (collar={args.collar}):")
        all_datasets = set()
        for engine_name in all_ders:
            for dataset_name in all_ders[engine_name]:
                all_datasets.add(dataset_name)
        all_datasets = sorted(all_datasets)
        s = f"| {'Engine':28} |"
        line = f"|{'-'*30}|"
        for dataset_name in all_datasets:
            s += f" {dataset_name:>11} |"
            line += f"{'-'*13}|"
        print(s)
        print(line)
        for engine_name in all_ders:
            name = engine_name.replace("\n", " ")
            name = format_system_name(name)
            s = f"| {name:28} |"
            for dataset_name in all_datasets:
                scores = all_ders[engine_name].get(dataset_name)
                if scores is None:
                    s += f" " + (" " * (11-5) + "_" * 5) + " |"
                else:
                    s += f" {scores:>11.2f} |"
            print(s)

    if args.output:
        for idx_figure, title in figures.items():            
            output_filename = os.path.join(args.output, slugify(title) + ".png")
            print("Writing:", output_filename)
            os.makedirs(args.output, exist_ok=True)
            plt.figure(idx_figure)
            plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()