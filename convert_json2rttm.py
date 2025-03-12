#!/usr/bin/env python3

import json
import os

def json2rttm(input_json, output_rttm, channel=1, prefix_speaker="", warn_if_overlap=False):

    with open(input_json) as json_file:
        result = json.load(json_file)
        
    _,recname= os.path.split(input_json)
    
    recname = os.path.splitext(recname)[0]
    rttm_line = "SPEAKER {} {} {} {} <NA> <NA> {} <NA> <NA>\n"

    possible_keys_start = ["seg_begin", "start"]
    possible_keys_end = ["seg_end", "end"]
    key_start = None
    key_end = None
    has_given_warning_about_overlap = not warn_if_overlap

    with open(output_rttm, "w") as fp:
        previous_end = 0
        previous_start = 0
        for seg in result["segments"]:
            if key_start is None:
                for k in possible_keys_start:
                    if k in seg:
                        key_start = k
                        break
                assert key_start is not None, f"Could not find start key in {seg.keys()} (among {possible_keys_start})"
            if key_end is None:
                for k in possible_keys_end:
                    if k in seg:
                        key_end = k
                        break
                assert key_end is not None, f"Could not find end key in {seg.keys()} (among {possible_keys_end})"
            start = seg[key_start]
            end = seg[key_end]
            duration = end - start
            label = seg["spk_id"]
            if prefix_speaker:
                label = prefix_speaker + label


            if start < previous_end and not has_given_warning_about_overlap:
                has_given_warning_about_overlap = True
                print(f"Warning: Got overlapping segments")
            assert start >= previous_start, f"Got start {start} <= previous_start {previous_start}"
            previous_end = end
            previous_start = start

            start = f"{start:.3f}"
            duration = f"{duration:.3f}"
            line = rttm_line.format(recname, channel, start, duration, label)
            fp.write(line)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", default="results", nargs="?", help="Input folder or file")
    parser.add_argument("--prefix_speaker", default="", help="Prefix to add for speaker names")
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        inputs = []
        for root, dirs, files in os.walk(args.input):
            for name in files:
                if name.endswith(".json"):
                    input_json = os.path.join(root, name)
                    output_rttm= os.path.splitext(input_json)[0]+ ".rttm"
                    inputs.append((input_json, output_rttm))
    else:
        input_json = args.input
        output_rttm= os.path.splitext(input_json)[0]+ ".rttm"
        inputs = [(input_json, output_rttm)]

    for input_json, output_rttm in inputs:
        print("Converting", input_json, "to", output_rttm, "...")
        json2rttm(input_json, output_rttm, prefix_speaker=args.prefix_speaker)
