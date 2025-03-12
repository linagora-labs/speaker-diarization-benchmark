import json
import os
import re
import matplotlib.pyplot as plt
from slugify import slugify
import numpy as np
import itertools

from remove_overlaps import create_rttm_without_overlap
from convert_json2rttm import json2rttm
from metadata import get_files_and_metadata
from plot_memory_time import get_color_engine
from plot_scores import (
    wrap_rttm_read,
    compute_number_of_speakers,
    custom_violinplot,
    format_system_name,
)
from metadata_identification import DEFAULT_CSV
from run_benchmark_identification import (
    target_spk_setting_to_indices_for_sorting,
    target_spk_setting_to_label,
    get_candidate_speaker_for_setting,
)

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "CDER_Metric"))
from rttm_io import rttm_read


from collections import defaultdict
from typing import Sequence, Dict, Tuple
from pyannote.core import Annotation, Segment
from pyannote.metrics.identification import IdentificationErrorRate
from pyannote.metrics.diarization import DiarizationErrorRate

RTTM = Dict[str, Sequence[Tuple[str, float, float]]]
def load_rttm(file: str) -> RTTM:
    rttm = defaultdict(list)
    with open(file, "r") as f:
        for line in f:
            parts = line.strip().split()
            assert len(parts) > 7, f"Error in line: '{line}' of {file}"
            file_id = parts[1]
            spk = parts[7]
            start = float(parts[3])
            end = start + float(parts[4])
            rttm[file_id].append((spk, start, end))
    return rttm
 
def get_pyannote_score(scorer, ref, hyp):
    ref = conform_to_pyannote(ref)
    hyp = conform_to_pyannote(hyp)
    score = scorer(ref, hyp)
    return score

def generate_mappings(list1, list2):
    if len(list2) > len(list1):
        # All ordered sublist of list2
        return [list(sublist) for sublist in itertools.permutations(list2, len(list1))]

    # Number of elements to be mapped to themselves
    num_self_maps = len(list1) - len(list2)
    
    # Generate permutations of list2
    list2_permutations = list(itertools.permutations(list2))
    
    # Possible positions for self-mapping
    positions_for_self_maps = list(itertools.combinations(range(len(list1)), num_self_maps))
    
    all_possible_assignments = []
    
    for perm in list2_permutations:
        for positions in positions_for_self_maps:
            mapping = [None] * len(list1)
            perm_index = 0
            
            for i in range(len(list1)):
                if i in positions:
                    mapping[i] = list1[i]
                else:
                    mapping[i] = perm[perm_index]
                    perm_index += 1
            
            all_possible_assignments.append(mapping)
    
    return all_possible_assignments

_converted_hypothesis = {}

def get_pyannote_score_replacing_unknown_speakers(
    scorer,
    ref_file,
    hyp_file,
    candidate_speakers=None,
    is_unknown=lambda x: x.startswith("spk"),
    ):
    """
    Get best score by replacing unknown speakers by best known speakers that was not in the candidates

    Args:
        scorer: PyAnnote scorer function
        ref_file: reference file
        hyp_file: hypothesis file
        is_unknown: function to tell if a speaker is unknown (in linto-diarization, unknown speakers are named spkXX where XX is a number)
    """
    global _converted_hypothesis
    cache_key_prefix = os.path.dirname(hyp_file)
    ref = load_rttm(ref_file)
    hyp = load_rttm(hyp_file)

    for audio in hyp:
        # Cache to avoid computing twice the best hypothesis
        cache_key = (cache_key_prefix, audio)
        if cache_key in _converted_hypothesis:
            hyp[audio] = _converted_hypothesis[cache_key]
            continue

        # Check if there are unknown speakers
        unknown_speakers = set()
        known_speakers = set()
        hyp_audio = hyp[audio]
        for spk, _, _ in hyp_audio:
            if is_unknown(spk):
                unknown_speakers.add(spk)
            else:
                known_speakers.add(spk)

        # Sanity check
        if candidate_speakers is not None:
            assert not (known_speakers - set(candidate_speakers)), \
                f"Some detected speakers are not among the candidate speakers in {hyp_file} : {known_speakers=} {candidate_speakers=} => {(known_speakers - set(candidate_speakers))}"

        if unknown_speakers:

            if candidate_speakers is None:
                continue
            # assert candidate_speakers is not None, \
            #     f"ERROR: candidate_speakers must be provided for {audio} (available: {list(_converted_hypothesis.keys())})"

            # Collect all the target speakers
            target_speakers = set()
            for spk, _, _ in ref[audio]:
                target_speakers.add(spk)

            # Check if there are unknown target speakers
            unknown_speakers = list(unknown_speakers)
            target_speakers = list(target_speakers - set(candidate_speakers))
            if not target_speakers:
                continue

            # Try all the possible assignments (unknown speaker -> target speaker)
            best_score = None
            best_hyp = None
            # all_possible_assignments = [list(pair) for pair in itertools.product(target_speakers, repeat=len(unknown_speakers))]
            all_possible_assignments = generate_mappings(unknown_speakers, target_speakers)
            for assignment in all_possible_assignments:
                hyp_ = {
                    audio:[(assignment[unknown_speakers.index(spk)] if spk in unknown_speakers else spk, start, end) for spk, start, end in hyp_audio]
                }
                ref_ = {
                    audio:ref[audio]
                }
                score = get_pyannote_score(scorer, ref_, hyp_)
                if best_score is None or score < best_score:
                    best_score = score
                    best_hyp = hyp_
            hyp[audio] = best_hyp[audio]
            _converted_hypothesis[cache_key] = hyp[audio]

    return get_pyannote_score(scorer, ref, hyp)

def conform_to_pyannote(annot):
    if isinstance(annot, Annotation):
        return annot
    if isinstance(annot, str):
        annot = load_rttm(annot)
    return rttm_to_pyannote(annot)

def rttm_to_pyannote(rttm: RTTM) -> "Annotation":
    reference = Annotation()
    segments = list(rttm.values())[0]
    for segment in segments:
        label, start, end = segment
        reference[Segment(start, end)] = label
    return reference

def getter_ier_score(collar=0, skip_overlap=False):
    ier_scorer = IdentificationErrorRate(
        collar = collar,
        skip_overlap = skip_overlap,
    )
    return lambda ref, hyp, candidate_speakers: get_pyannote_score_replacing_unknown_speakers(ier_scorer, ref, hyp, candidate_speakers)


TO_MAXIMIZE = ["Purity", "Coverage", "FMeasure"]


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Plot memory consumption and processing times for the different engines')
    parser.add_argument('output', default = None, type=str, help='Output folder name to save figures', nargs='?')
    parser.add_argument('--skip-overlap', action='store_true', default=False, help='To skip overlaps when evaluating')
    parser.add_argument('--collar', type=float, default=0, help='Collar for overall DER reported')
    parser.add_argument('--recompute', action='store_true', default=False, help='To force recompute all the IERs')
    parser.add_argument('--plot_distribution', default="violin", type=str, help='Type of plot for distribution of IERs', choices=["violin", "boxplot"])
    parser.add_argument('--yscale', default="100", type=str, help='How to scale y axis ("100": in 0-100, "auto": automatic)', choices=["100", "auto"])
    parser.add_argument('--add_der', action='store_true', default=False, help='Display DER besides IER (in transparency)')
    parser.add_argument('--verbose', action='store_true', default=False, help='Print more information (number of speakers...)')
   
    args = parser.parse_args()

    key_IER = "IER" if args.collar == 0 else f"IER_{args.collar}"
    key_DER = "DER" if args.collar == 0 else f"DER_{args.collar}"

    der_func = DiarizationErrorRate(collar=args.collar, skip_overlap=args.skip_overlap)

    _score_funcs = {
        key_IER: getter_ier_score(collar=args.collar, skip_overlap=args.skip_overlap),
        key_DER: wrap_rttm_read(der_func),
        "NumberOfSpeakers": wrap_rttm_read(compute_number_of_speakers),
    }

    perfnames_all = list(_score_funcs.keys())
    perfnames_final = list(set(perfnames_all) - {key_DER, "NumberOfSpeakers"})

    RECOMPUTE_IER = args.recompute
    REMOVE_OVERLAPS = args.skip_overlap

    _warned_if_recompute = False
    
    path_to_rttm_ref = "data/rttm_identification/"
    metadata = get_files_and_metadata(DEFAULT_CSV)

    # First collect all the files on which to compute DER
    all_files = {}
    for path_to_results in "results_identification/cuda", "results_identification/cpu", :
        if not os.path.isdir(path_to_results):
            print(f"WARNING: {path_to_results} does not exist")
            continue
        subfolders = [f for f in os.scandir(path_to_results) if f.is_dir()]        
        for ind, path_setting_spk in enumerate(sorted(subfolders, key=lambda x: x.name)):
            engine_versions = {}
            setting_spk = os.path.basename(path_setting_spk)
            sub_sub_folders = [f for f in os.scandir(path_setting_spk)]                           
            for ind, path_setting_engine in enumerate(sorted(sub_sub_folders, key=lambda x: x.name)):  
                setting_target_spk = os.path.basename(path_setting_engine)
                sub_sub_sub_folders = [f for f in os.scandir(path_setting_engine)]                
                sub_sub_sub_folders = sorted(
                sub_sub_sub_folders, key=lambda x: x.name.split("-")[-1], reverse=True)                                
                for path_engine_name in sub_sub_sub_folders:                                        
                    engine_name_version = os.path.basename(path_engine_name)                            
                    
                    engine_name = engine_name_version                                                     

                    # Format engine name for future xticks labels
                    #engine_name = engine_name.replace("-", "\n").replace("_", "\n")
                    setting_spk = {
                        "known_spk": "Given # of speakers",
                        "unknown_spk": "Unknown # of speakers",
                    }.get(setting_spk, setting_spk)

                    engine_versions[engine_name] = engine_name_version                    
                    json_files = [
                        pos_json for pos_json in os.listdir(path_engine_name) if pos_json.endswith('.json') and not pos_json.startswith("test")
                    ]                    
                    all_files[setting_spk] = all_files.get(setting_spk, {})
                    all_files[setting_spk][setting_target_spk] = all_files[setting_spk].get(setting_target_spk, {})
                    all_files[setting_spk][setting_target_spk][engine_name]=all_files[setting_spk][setting_target_spk].get(engine_name, {})
                    for fn in sorted(json_files):
                        if fn not in all_files[setting_spk][setting_target_spk][engine_name]:
                            all_files[setting_spk][setting_target_spk][engine_name][fn] = os.path.join(path_engine_name, fn)
                    if setting_spk == "unknown_spk" and engine_name == "pyannote-1.1.0":
                        all_files[setting_spk][engine_name]["004a_PARH.json"]
    
    all_engine_names = set()
    for setting_spk, all_files_for_spk_setting in all_files.items():
        for setting_target_spk,all_files_for_setting_engine in all_files_for_spk_setting.items():

            for engine_name in all_files_for_setting_engine.keys():
                if "ignore" in engine_name:
                    continue
                all_engine_names.add(engine_name)
    
    all_engine_names = sorted(all_engine_names, key= lambda x: ("streaming" not in x, x))    
    num_done = 0
    figures = {}
    all_iers = {}

    num_rows = len(all_files)

    for ind, (setting_spk, all_files_for_speaker_setting) in enumerate(all_files.items()):
        num_cols = len(all_files_for_speaker_setting)
        sorted_settings = sorted(all_files_for_speaker_setting.keys(), key=target_spk_setting_to_indices_for_sorting, reverse=True)
        for icolumn, setting_target_spk in enumerate(sorted_settings):
            all_files_for_spk_setting = all_files_for_speaker_setting[setting_target_spk]

            all_accuracies = {}
            for engine_name in all_engine_names:

                json_files = all_files_for_spk_setting.get(engine_name, {}).values()
                if not json_files:
                    continue
                for k in perfnames_all:
                    all_accuracies[k] = all_accuracies.get(k, {})
                    all_accuracies[k][setting_target_spk] = all_accuracies[k].get(setting_target_spk, {})
                    all_accuracies[k][setting_target_spk][engine_name] = all_accuracies[k][setting_target_spk].get(engine_name, [])

                all_rttm_ref = {}
                all_rttm_hyp = {}
                for index, json_hyp in enumerate(json_files):

                    recname = os.path.basename(os.path.splitext(json_hyp)[0])
                    recname = recname.split('.', 1)[0]

                    scores = perf_file = None
                    if RECOMPUTE_IER is not None:
                        perf_file = os.path.splitext(json_hyp)[0] + ".ier"
                        if not REMOVE_OVERLAPS:
                            perf_file += "_overlap"
                        if not RECOMPUTE_IER and os.path.exists(perf_file):
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

                    json2rttm(json_hyp, rttm_hyp)

                    all_rttm_ref[setting_target_spk] = all_rttm_ref.get(setting_target_spk, []) + [rttm_ref]
                    all_rttm_hyp[setting_target_spk] = all_rttm_hyp.get(setting_target_spk, []) + [rttm_hyp]

                    if scores is None:

                        # Get the candidate speakers
                        assert recname+".wav" in metadata, f"ERROR: {recname}.wav not in {metadata.keys()}"
                        all_speakers = eval(metadata[recname+".wav"]["speakers"])
                        candidate_speakers = get_candidate_speaker_for_setting(setting_target_spk, all_speakers)

                        # Compute IER
                        try:
                            scores = dict((k, v(rttm_ref, rttm_hyp, candidate_speakers))
                                for k, v in _score_funcs.items())
                        except Exception as err:
                            raise RuntimeError(f"Error processing {rttm_hyp}") from err

                        if perf_file is not None:
                            with open(perf_file, "w") as f:
                                json.dump(scores, f, indent=4)

                    for k, val in scores.items():
                        if k == "NumberOfSpeakers":
                            if abs(val[1] - val[0]) > 10 or setting_spk == "known_spk" and val[0] != val[1]:
                                if args.verbose:
                                    print(f"{json_hyp} -- number of speakers: real= {val[0]}, predicted= {val[1]}")
                        elif isinstance(val, float):
                            val = 100 * val # Convert to percent here 
                        assert k in all_accuracies, f"ERROR: {k} not in {all_accuracies.keys()}"                        
                        all_accuracies[k][setting_target_spk][engine_name] = all_accuracies[k][setting_target_spk].get(engine_name, []) + [val]
                        
                # Make overall IER
                path_engine_name = os.path.dirname(list(json_files)[0])
                for setting_target_spk in all_rttm_ref:
                
                    # Caching mechanism
                    perf_file =  os.path.join(path_engine_name, f"_overall.ier")
                    if not REMOVE_OVERLAPS:
                        perf_file += "_overlap"
                    score = None
                    if not RECOMPUTE_IER and os.path.exists(perf_file):
                        with open(perf_file) as f:
                            score = json.load(f)

                    if score is None:

                        # Concatenate all RTTM files
                        cumulated_ref = os.path.join(path_to_rttm_ref, f"_overall_{setting_target_spk}.rttm")
                        cumulated_hyp = os.path.join(path_engine_name, f"_overall.rttm")
                        with open(cumulated_ref, "w") as fp:
                            for ref in sorted(all_rttm_ref[setting_target_spk]):
                                with open(ref) as f:
                                    empty = True
                                    for line in f.readlines():
                                        fp.write(line.strip() + "\n")
                                        empty = False
                                    assert not empty, f"ERROR: empty file {ref}"
                        with open(cumulated_hyp, "w") as fp:
                            for hyp in sorted(all_rttm_hyp[setting_target_spk]):
                                with open(hyp) as f:
                                    empty = True
                                    for line in f.readlines():
                                        fp.write(line.strip() + "\n")
                                        empty = False
                                    if empty and args.verbose:
                                        print(f"WARNING: empty file {hyp}")
                        # Compute average IER
                        score = _score_funcs[key_IER](cumulated_ref, cumulated_hyp, None) * 100
                        os.remove(cumulated_ref)
                        os.remove(cumulated_hyp)

                        if perf_file is not None:
                            with open(perf_file, "w") as f:
                                json.dump(score, f, indent=4)

                    # Collect IER
                    all_iers[engine_name] = all_iers.get(engine_name, {})
                    all_iers[engine_name][setting_target_spk] = score

            for iperf, (perfname, perfs) in enumerate(all_accuracies.items()):
                if perfname not in perfnames_final:
                    continue

                idx_figure = iperf + 1
                figures[idx_figure] = perfname
                plt.figure(idx_figure, figsize=(20, 10))

                num_plots = len(subfolders)
                irow = ind

                is_number_of_speaker = (perfname == "NumberOfSpeakers")
                if is_number_of_speaker:
                    if "unknown" not in setting_spk.lower():
                        continue
                    num_plots = 1
                    irow = 0
                    
                perf = perfs[setting_target_spk]
                
                plt.subplot(num_rows, num_cols, icolumn+1 + irow * num_cols)

                if not is_number_of_speaker:
                    values = list(perf.values())

                else:
                    new_values = []
                    for vals in perf.values():
                        for v in vals:
                            assert len(v) == 2
                    values = [[v[1] - v[0] for v in vals] for vals in perf.values()]

                # Remove empty ones
                positions = list(range(1,1+len(values)))
                ticks = list(perf.keys())
                for i in positions[::-1]:
                    i -= 1
                    if not values[i]:
                        del positions[i]
                        del values[i]
                        ticks[i] = ""

                if not values:
                    continue

                colors = [get_color_engine(t) for t in ticks if t]

                if perfname == key_IER and args.add_der:
                    # Also plot DER in the same figure
                    der_values = []
                    for engine_name in ticks:
                        der_details = all_accuracies[key_DER][setting_target_spk][engine_name]
                        der_values.append([v["diarization error rate"] * 100 for v in der_details])
                    if args.plot_distribution == "violin":
                        custom_violinplot(
                            der_values,
                            positions=positions,
                            color=colors,
                            alpha=0.5,
                            plot_quartiles=False,
                        )
                    else:
                        raise NotImplementedError("Diplay DER above IER Not implemented yet")
                    

                if args.plot_distribution == "violin":
                    custom_violinplot(
                        values,
                        positions=positions,
                        color=colors,
                    )
                else:
                    plt.boxplot(values, positions=positions, whis=100)

                nonentiles = []
                for v in values:
                    nonentiles.append(np.percentile(v, 90))

                plt.xlim(0.5, len(perf) + 0.5)

                # Rescale y-axis
                if is_number_of_speaker:
                    plt.axhline(y=0, color='black', linestyle='--')
                else:
                    xmin, xmax, ymin, ymax = plt.axis()
                    if args.yscale == "100":
                        plt.ylim(0, 100)
                    elif args.yscale == "auto":
                        if perfname not in TO_MAXIMIZE:
                            plt.axis([xmin, xmax, 0, max(nonentiles)])
                        else:
                            plt.axis([xmin, xmax, ymin, 100])
                    else:
                        raise NotImplementedError(f"Unknown yscale: {args.yscale}")
                if icolumn == 0:
                    plt.ylabel({
                        key_IER: "Identification Error Rate (IER)"
                    }.get(perfname, perfname))
                if irow == 0:
                    title = target_spk_setting_to_label(setting_target_spk)
                    plt.title(title)
                    # plt.xlabel(title)
                if icolumn == round((num_cols - 1) / 2):
                    plt.xlabel(setting_spk)
                if irow == num_rows - 1:
                    plt.xticks(range(1, len(perf) + 1), ticks, rotation=10)
                else:
                    plt.xticks(range(1, len(perf) + 1), "" * len(ticks))

    print(f"IER {setting_spk.replace('_', ' ')} (collar={args.collar}):")
    all_settings_target_spk = set()
    for engine_name in all_iers:
        for setting_target_spk in all_iers[engine_name]:
            all_settings_target_spk.add(setting_target_spk)
    all_settings_target_spk = sorted(all_settings_target_spk, key=target_spk_setting_to_indices_for_sorting, reverse=True)
    s = f"| {'Engine':28} |"
    line = f"|{'-'*30}|"
    for setting_target_spk in all_settings_target_spk:
        s += f" {setting_target_spk:>11} |"
        line += f"{'-'*18}|"
    print(s)
    print(line)
    for engine_name in all_iers:
        name = engine_name.replace("\n", " ")
        name = format_system_name(name)
        s = f"| {name:28} |"
        for setting_target_spk in all_settings_target_spk:
            score = all_iers[engine_name].get(setting_target_spk)
            if score is None:
                s += f" " + (" " * (16-5) + "_" * 5) + " |"
            else:
                s += f" {score:>16.2f} |"
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