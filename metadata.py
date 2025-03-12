import subprocess
import os
import csv
import re

FILENAME_TO_NUM_SPEAKERS = dict([
    ("test3.wav", 5),
    ("test2.wav", 8),
    ("test.wav", 5),
    ("Linagora_C2_3.wav", 5), # 7 speakers, but 2 are only involved in small segments with overlap
])

DEFAULT_CSV = "data/metadata.csv"

def get_audio_metadata(
    filename,
    folder_input_rttm=None,
    include_speakers_name=False,
    verbose=False,
    ):
    """
    Get all the metadata on a filename that soxi provides
    """
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "CDER_Metric"))
    from rttm_io import rttm_read

    if isinstance(filename, str) and os.path.isdir(filename):
        return get_audio_metadata([os.path.join(filename, f) for f in os.listdir(filename)], folder_input_rttm=folder_input_rttm, include_speakers_name=include_speakers_name, verbose=verbose)
    if isinstance(filename, list):
        return [get_audio_metadata(f, folder_input_rttm=folder_input_rttm, include_speakers_name=include_speakers_name, verbose=verbose) for f in sorted(filename)]
    assert isinstance(filename, str), "Input argument must be a filename, folder name or list of files"
    assert os.path.isfile(filename), "File or folder does not exist: {}".format(filename)
    
    cmd = ["soxi", filename]
    if verbose :print(" ".join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    out = out.decode("utf-8")
    err = err.decode("utf-8")
    if verbose :print(out)
    if err: print(err)

    meta = get_dict(
        out,
        ["input_file", "duration", "file_size", "channels", "sample_rate", "precision", "bit_rate", "sample_encoding"],
        func={
            # Convert 00:10:03.52 to seconds
            "duration": lambda x: to_sec(x.split()[0]),
            # Ensure integers
            "sample_rate": int,
            "channels": int,
            # Remove folder
            "input_file": lambda x: os.path.basename(x[1:-1]),
        }
    )

    input_file = meta["input_file"]

    # Add name of dataset from which is taken the audio
    group = None
    if input_file.startswith("0") or input_file.startswith("1"):
        group = "SUMM-RE"
    elif input_file.startswith("LCP"):
        group = "ETAPE"
    elif input_file.startswith("Linagora") or input_file.startswith("meeting_RAP"):
        group = "LINAGORA"
    elif re.match(r"[a-z]{5}.wav$", input_file):
        group = "VoxConverse"
    elif input_file.startswith("dj_") or input_file.startswith("reu_"):
        group = "Simsamu"
    meta["group"] = group

    # Add number of speakers
    if folder_input_rttm:
        rttm_filename = os.path.splitext(os.path.basename(input_file))[0] + ".rttm"
        rttm_file1 = os.path.join(folder_input_rttm, rttm_filename)
        rttm_file = rttm_file1
        if not os.path.isfile(rttm_file):
            rttm_file = os.path.join(folder_input_rttm, rttm_filename)
        if os.path.isfile(rttm_file):
            rttm = rttm_read(rttm_file)
            assert len(rttm) == 1
            rttm = rttm[list(rttm.keys())[0]]
            speakers = rttm[-1].chart()
            speakers = sorted([s[0] for s in speakers])
            num_speakers = len(speakers)
            meta["num_speakers"] = num_speakers
            if include_speakers_name:
                meta["speakers"] = speakers
    if "num_speakers" not in meta or input_file in FILENAME_TO_NUM_SPEAKERS:
        if input_file not in FILENAME_TO_NUM_SPEAKERS:
            raise ValueError(f"Number of speakers not found for file '{input_file}'. Available: {sorted(FILENAME_TO_NUM_SPEAKERS.keys())} {rttm_file1} and {rttm_file} not found)")
        if include_speakers_name:
            raise NotImplementedError("Speakers name not implemented for file '{}'".format(input_file))
        meta["num_speakers"] = FILENAME_TO_NUM_SPEAKERS[input_file]

    
    return meta


def to_sec(duration_str):
    """
    Convert 00:10:03.52 to seconds
    """
    try:
        import datetime
        hours, minutes, seconds = duration_str.split(':')
        seconds, milliseconds = seconds.split('.')
        duration = datetime.timedelta(hours=int(hours), minutes=int(minutes), seconds=int(seconds), milliseconds=int(milliseconds)*10)
        return duration.total_seconds()
    except Exception as err:
        raise RuntimeError("Could not convert duration to seconds string '{}'\n{}".format(duration_str, err))

# Convert txt output to dict
# Example input:
# # Input File     : '/home/jlouradour/data/audio/FILE.wav'
# # Channels       : 1
# # Sample Rate    : 16000
# # Precision      : 16-bit
# # Duration       : 00:00:00.50 = 8000 samples ~ 37.5 CDDA sectors
# # File Size      : 16.0k
# # Bit Rate       : 257k
# # Sample Encoding: 16-bit Signed Integer PCM
def get_dict(txt, key, func = lambda x:x):
    """
    Get a dictionary from a txt output of soxi
    """
    if isinstance(key, list):
        assert isinstance(func, dict)
        return dict([(k, get_dict(txt, k, func.get(k, lambda x:x))) for k in key])
    assert isinstance(key, str), "Key must be a string"
    lkey = key.lower().replace(" ", "").replace("_", "")
    for line in txt.splitlines():
        if line.strip().lower().replace(" ", "").startswith(lkey+":"):
            return func(":".join(line.split(":")[1:]).strip())
    raise ValueError("Key not found: {}\nin:{}\n".format(key, txt))

def write_csv_from_list_of_dict(
    list_of_dicts,
    filename,
    update_if_exists=True,
):
    """
    Write a CSV file from a list of dictionaries
    """
    assert isinstance(list_of_dicts, list), "Input argument must be a list of dictionaries"
    assert len(list_of_dicts) > 0, "Input argument must be a not-empty list (of dictionaries)"
    assert isinstance(list_of_dicts[0], dict), "Input argument must be a list of dictionaries"
    assert isinstance(filename, str), "Input argument must be a filename"
    
    write_header = True
    keys = None
    if os.path.isfile(filename):
        if update_if_exists:
            write_header = False
            with open(filename, "r") as f:
                reader = csv.DictReader(f)
                keys = reader.fieldnames
                list_of_dicts = [d for d in list_of_dicts if d not in reader]
        else:
            os.remove(filename)

    if keys is not None:
        assert list(list_of_dicts[0].keys()) == keys, f"Keys do not match: {list(list_of_dicts[0].keys())} != {keys}"
    keys = list_of_dicts[0].keys()
    with open(filename, "a") as f:
        dict_writer = csv.DictWriter(f, keys)
        if write_header:
            dict_writer.writeheader()
        dict_writer.writerows(list_of_dicts)

    print(f"Wrote {len(list_of_dicts)} lines to {filename}")

def get_files_and_metadata(csv_file=DEFAULT_CSV):
    """
    Get a list of files and metadata from a CSV file
    """
    assert os.path.isfile(csv_file), "CSV file does not exist: {}".format(csv_file)

    def to_dict(row):
        d = dict(row)
        for key, type in [
            ("duration", float),
            ("num_speakers", int),
            ("sample_rate", int),
            ("channels", int),
        ]:
            d[key] = type(d[key])
        return (d["input_file"], d)
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        return dict(to_dict(row) for row in reader)

if __name__ == "__main__":
    
    from run_benchmark import default_folder_input as folder_input_wav
    folder_input_rttm = os.path.join(os.path.dirname(folder_input_wav), "rttm")

    import argparse
    parser = argparse.ArgumentParser(
        "Get meta data from audio files (using soxi) and write a CSV file with all the metadata",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", type=str, help="Input file, directory, or list of files", nargs="*")
    parser.add_argument("--output", type=str, help="Output CSV file", default=DEFAULT_CSV)
    args = parser.parse_args()

    if not args.input:
        args.input = [os.path.join(folder_input_wav, f) for f in os.listdir(folder_input_wav) if f.lower().endswith(".wav")]
        if not args.input:
            raise ValueError(f"No WAV files found in {folder_input_wav}")

    write_csv_from_list_of_dict(
        get_audio_metadata(args.input, folder_input_rttm=folder_input_rttm),
        args.output
    )

