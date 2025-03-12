#!/usr/bin/env python3

import json
import os
import pprint

def import_rttm(file, sep = None):
    if sep is None:
        with open(file, 'r') as f:
            first_line = f.readline()
            if "\t" in first_line:
                sep = "\t"
            else:
                sep = " "
        return import_rttm(file, sep)
    with open(file, 'r') as f:
        return [import_rttm_line(l, sep) for l in f.readlines()]

def import_rttm_line(line, sep):
    fields = line.strip().split(sep)
    if sep == " ":
        fields = [sep.join(fields[:2]), *fields[2:]]
    assert len(fields) in [8,9], "Invalid line: {}".format(line)
    ID = fields[:2]
    start = float(fields[2])
    duration = float(fields[3])
    spk = fields[6]
    EXTRA = (fields[4:6], fields[7:])
    return (ID, start, duration, spk, EXTRA)



def rttm2json(input_rttm, output_json):

    
    result = import_rttm(input_rttm)
        
    _,recname= os.path.split(input_rttm)
    
    recname = os.path.splitext(recname)[0]
    json_result = {}
    _segments = []
    _speakers = {}
    seg_id = 1
    
    
    with open(output_json, "w") as fp:
            for (ID, start, duration, spk, EXTRA) in result:
                
                spk_id =spk
                segment = {}
                segment["seg_id"] = seg_id

                

                
                segment["spk_id"] = spk
                segment["seg_begin"] = start
                segment["seg_end"] = start+duration

                if spk_id not in _speakers:
                    _speakers[spk_id] = {}
                    _speakers[spk_id]["spk_id"] = spk
                    _speakers[spk_id]["duration"] = segment["seg_end"]-segment["seg_begin"]
                    _speakers[spk_id]["nbr_seg"] = 1
                else:
                    _speakers[spk_id]["duration"] += segment["seg_end"]-segment["seg_begin"]
                    _speakers[spk_id]["nbr_seg"] += 1

                _segments.append(segment)
                seg_id += 1

            for spkstat in _speakers.values():
                spkstat["duration"] = spkstat["duration"]

            json_result["segments"] = _segments
            json_result["speakers"] = list(_speakers.values())
            
            #pprint.pprint(json) 
            json_object = json.dumps(json_result, indent=4)
            fp.write(json_object)

#json_fns="Linagora_C2_3.json"
#rttm_fns="Linagora_C2_3.rttm"
#ref = rttm2json(rttm_fns,json_fns)  


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", default="output", nargs="?", help="Input folder or file")
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        inputs = []
        for root, dirs, files in os.walk(args.input):
            for name in files:
                if name.endswith(".rttm"):
                    input_rttm = os.path.join(root, name)
                    output_json= os.path.splitext(input_rttm)[0]+ ".json"
                    inputs.append((input_rttm, output_json))
    else:
        input_rttm = args.input
        output_json= os.path.splitext(input_rttm)[0]+ ".json"
        inputs = [(input_rttm, output_json)]

    print(inputs)
    for input_rttm, output_json in inputs:
        print("Converting", input_rttm, "to", output_json, "...")
        rttm2json(input_rttm, output_json)
       
