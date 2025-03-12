############################################

import time
import argparse
import json
import requests
import os
import math
import re

from metadata import get_files_and_metadata
from run_benchmark import (
    launch_docker,
    monitor_memory,
    get_ram_peak,
    get_vram_peak,
    get_gpus,
    convert_audio,
)
from metadata_identification import DEFAULT_CSV
_metadata = get_files_and_metadata(csv_file=DEFAULT_CSV)

######################
# VARIABLES AND PATH

MAX_SPEAKER = 25
LAST_TAG = "1.0.1"

folder_output_cpu = "results_identification/cpu"
folder_output_gpu = "results_identification/cuda"

default_folder_input = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/benchmark_identification/wav")
default_folder_speakers_samples = os.path.join(os.path.dirname(__file__), "data/benchmark_identification/speakers_samples")

######################
# HELPERS (also used in the plotting script, that needs to recover some information ...)

def list_to_string(l):
    """
    Helper for converting a list to a string, with double quotes, as required by the API (for speaker_names)
    """
    return str(l).replace("'", '"')

def get_all_speaker_settings(speakers_in_audio, all_speakers):
    num_speakers_in_database = len(all_speakers)
    # Caution : Assuming here (almost) 4 speakers per audio
    ratio_absent_with_all_speakers = (num_speakers_in_database - 4) / num_speakers_in_database
    ratio_absent_with_all_speakers = str(round(ratio_absent_with_all_speakers * 100))
    ratio_absent_with_all_expect_two = (num_speakers_in_database - 4) / (num_speakers_in_database - 2)
    ratio_absent_with_all_expect_two = str(round(ratio_absent_with_all_expect_two * 100))

    half_speakers = speakers_in_audio[:math.ceil(len(speakers_in_audio)/2)]
    speakers_difference = list(set(all_speakers) - set(speakers_in_audio))
    half_speakers_with_difference = speakers_difference + half_speakers

    ### list of possible speakers name for eval
    return {
        "100known_0absent": list_to_string(speakers_in_audio),
        "50known_0absent": list_to_string(half_speakers),
        f"100known_{ratio_absent_with_all_speakers}absent": list_to_string(all_speakers),
        f"50known_{ratio_absent_with_all_expect_two}absent": list_to_string(half_speakers_with_difference),
        "0known_100absent": list_to_string(speakers_difference),
    }

def get_candidate_speaker_for_setting(s, speakers_in_audio):
    settings = get_all_speaker_settings(speakers_in_audio, get_all_speakers_in_metadata())
    def norm_key(k):
        _, ratio_absent = target_spk_setting_to_indices(k)
        if ratio_absent not in (0, 100):
            return k.split("_")[0]
        return k
    settings = {norm_key(k): v for k, v in settings.items()}
    return eval(settings[norm_key(s)])

def get_all_speakers_in_metadata():
    all_speakers = [eval(v["speakers"]) for v in _metadata.values()]
    all_speakers = [item for sublist in all_speakers for item in sublist]
    return list(set(all_speakers)) + ["Amine"] # WTF

def target_spk_setting_to_indices(s):
    xs = s.split("_")
    xs = [re.sub(r"[a-zA-Z]", "", x) for x in xs]
    return tuple(map(int, xs))

def target_spk_setting_to_indices_for_sorting(s):
    ratio_knwown, ratio_absent = target_spk_setting_to_indices(s)
    return ratio_knwown, -ratio_absent

def target_spk_setting_to_label(s):
    what = s.split("_")
    what = [re.sub(r"[0-9\.]", "", x) for x in what]
    what = [f"{x} speakers" for x in what]
    ratios = target_spk_setting_to_indices(s)
    assert len(what) == len(ratios)
    return "\n".join(f"{r}% {x}" for x, r in zip(what, ratios))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='speaker diarization benchmark')
    parser.add_argument('folder_input', type=str, default=default_folder_input, help='folder containing the audio files to process', nargs='?')
    parser.add_argument('folder_speakers_samples', type=str, default=default_folder_speakers_samples, help='folder containing the samples for target speakers', nargs='?')
    parser.add_argument('--name', type=str, default="linto-diarization-simple", help='name of the docker image to use')
    parser.add_argument('--tag', type=str, default=LAST_TAG, help='tag of the docker image to use, with numbers (ex: 1.0.1, 2.0.0, ...)')
    parser.add_argument('--convert_audio', default=False, action='store_true', help='convert audio to wav in 16kHz before processing')
    parser.add_argument('--overwrite', default=False, action='store_true', help='overwrite existing results (by default, existing experiments will be skipped)')
    parser.add_argument('--cache_diarization', default=False, action='store_true', help='cache diarization results')
    args = parser.parse_args()

    assert os.path.isdir(args.folder_input), f"Folder {args.folder_input} does not exist"
    assert os.path.isdir(args.folder_speakers_samples), f"Folder {args.folder_speakers_samples} does not exist"

    all_speakers = os.listdir(args.folder_speakers_samples)
    
    print(f"Running identification benchmark with max {len(all_speakers)} target speakers")

    docker_options = f"""\
 -v {args.folder_speakers_samples}:/opt/speaker_samples\
 --env CACHE_DIARIZATION_RESULTS=1\
    """
    if args.cache_diarization:
        folder_cache_parent = os.path.dirname(os.path.realpath(__file__))
        folder_cache_diarization = os.path.join(folder_cache_parent, f"tmp_cache_diarization_{args.name}")
        folder_cache_precomputed = os.path.join(folder_cache_parent, f"tmp_cache_speaker_precomputed_{args.name}_{args.tag}")
        os.makedirs(folder_cache_diarization, exist_ok=True)
        os.makedirs(folder_cache_precomputed, exist_ok=True)
        docker_options += f" -v {folder_cache_diarization}:/opt/cache_diarization"
        docker_options += f" -v {folder_cache_precomputed}:/opt/speaker_precomputed"

    docker = launch_docker(
        args.tag,
        name=args.name,
        options=docker_options,
    )
    url = docker["url"]
    pids = docker["pids"]
    dockername = docker["dockername"]
    system_name = docker["system_name"]
    if docker["device"] == "cpu":
        folder_output = folder_output_cpu
    else:
        folder_output = folder_output_gpu
    headers = {'accept': 'application/json'}

    print("Will monitor memory of processes:", pids)

    ############################################
    # Main loop

    first_run = True
    try:
        for use_spk_number in False, True, :

            # Process from the shortest to the longest audio
            for file in sorted(_metadata.keys(), key=lambda x: _metadata[x]["duration"]):

                _,recname= os.path.split(file)    
                recname = os.path.splitext(recname)[0]

                spk_number = _metadata[file]["num_speakers"]
                speakers_in_audio = eval(_metadata[file]["speakers"])
                assert len(speakers_in_audio) == spk_number, f"Error in metadata for {file}: spk_number should be {len(speakers_in_audio)}"
                assert isinstance(speakers_in_audio, list), f"Error in metadata for {file}: speakers should be a list"
                assert set(speakers_in_audio) <= set(all_speakers), f"Missing some speaker(s) for {file} among {speakers_in_audio}: all speakers should be in {all_speakers}"

                ### list of possible speakers name for eval
                speakers_list_eval = get_all_speaker_settings(speakers_in_audio, all_speakers)
                
                output_dir_name = f"{folder_output}/{'known_spk' if use_spk_number else 'unknown_spk'}"
                for speakers_list_config, speakers_list in speakers_list_eval.items():
                    
                    output_dir_name_ = f"{output_dir_name}/{speakers_list_config}/{system_name}"
                    output_filename = os.path.join(
                        output_dir_name_, recname)
                    output_filename_json = output_filename + ".json"
                    output_filename_perfs = output_filename + ".perfs.txt"
                    if os.path.exists(output_filename_json) and not args.overwrite:
                        print("Skipping", file, "/", speakers_list_config)
                        continue

                    if not use_spk_number:
                        spk_number = None

                    file = os.path.join(args.folder_input, file)
                    if not os.path.isfile(file):
                        print(f"WARNING: File {file} does not exist")
                        continue
                    assert os.path.isfile(file), f"File {file} does not exist"

                    if args.convert_audio:
                        file = convert_audio(file)

                    os.makedirs(output_dir_name_, exist_ok=True)

                    print("=====================================")
                    print("Generating", output_filename_json)
                    print("Processing", file, "with", spk_number, "speakers")
                    files = {'file': open(file, 'rb')}
                    data = {'spk_number': spk_number, 'max_speaker': MAX_SPEAKER, 'speaker_names':f"{speakers_list}" }                

                    # Maybe something like this needed to avoid errors at the first run when the docker is not ready yet
                    slept_time = 0
                    max_sleep_time = 120 if first_run else -1
                    first_run = False
                    while True:
                        try:
                            start = time.time()
                            with monitor_memory(pids):
                                response = requests.post(
                                    url, headers=headers, data=data, files=files)
                            break
                        except Exception as err:
                            import traceback
                            print(traceback.format_exc())
                            if slept_time > max_sleep_time:
                                raise err
                            print("Warning: retrying http request in 30 sec...")
                            slept_time += 30
                            time.sleep(30)

                    with open(output_filename_perfs, "w") as f:
                        print(f"Time: {time.time() - start:.2f} sec", file=f)
                        print(f"Memory Peak: {get_ram_peak()} MB", file=f)
                        vram = get_vram_peak()
                        gpus = get_gpus()
                        if vram:
                            print(f"VRAM Peak: {vram} MB", file=f)
                            print(f"GPU(s): {gpus}", file=f)

                    if response.status_code != 200:
                        print('Error:', response.status_code, response.reason)
                        raise RuntimeError("Error while calling the API")

                    result = json.loads(response.content.decode('utf-8'))
                    json.dump(result, open(output_filename_json, "w"), indent=2)

    finally:
        os.system(f"docker stop {dockername} 2> /dev/null")
