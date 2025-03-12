

def cut_diarization(d, start=None, end=None):
    if isinstance(start, float):
        start = round(start, 3)
    else:
        assert start is None
    if isinstance(end, float):
        end = round(end, 3)
    else:
        assert end is None
    new_segments = []
    speakers = {}
    for segment in d["segments"]:
        seg_start = segment["seg_begin"]
        seg_end = segment["seg_end"]
        spk = segment["spk_id"]
        assert isinstance(seg_start, float)
        assert isinstance(seg_end, float)
        assert seg_start <= seg_end
        if isinstance(start, float):
            if seg_start < start:
                if seg_end > start:
                    seg_start = segment["seg_begin"] = start
                else:
                    continue
        if isinstance(end, float):
            if seg_end > end:
                if seg_start < end:
                    seg_end = segment["seg_end"] = end
                else:
                    continue
        assert seg_start <= seg_end
        new_segments.append(segment)
        duration = seg_end - seg_start
        if spk in speakers:
            speakers[spk]["duration"] += duration
            speakers[spk]["nbr_seg"] += 1
        else:
            speakers[spk] = {
                "duration": duration,
                "nbr_seg": 1,
                "spk_id": spk
            }
    return {
        "segments": new_segments,
        "speakers": [{
            "duration": round(speakers[spk]["duration"], 3),
            "nbr_seg": speakers[spk]["nbr_seg"],
            "spk_id": spk,
        } for spk in sorted(speakers.keys())]
    }

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input JSON file", nargs='+')
    parser.add_argument("--start", type=float, help="Start time in seconds", default=None)
    parser.add_argument("--end", type=float, help="End time in seconds", default=None)
    args = parser.parse_args()

    for input_file in args.input:
        with open(input_file, "r") as f:
            d = json.load(f)
        d = cut_diarization(d, args.start, args.end)
        json.dump(d, open(input_file, "w"), indent=2)