import os
from metadata import (
    write_csv_from_list_of_dict,
    get_audio_metadata,
)

FILENAME_TO_NUM_SPEAKERS = dict([
    ("meeting_RAP_1.wav", 4),    
    ("004b_PADH.wav", 4),
    ("004c_PAPH.wav", 4),    
    ("009b_PBDZ.wav", 4),
    ("009c_PBPZ.wav", 4),
    ("097c_EBPH.wav",4),
    ("081c_EBPH.wav",3),
    ("069c_EEPL.wav",4),
    ("033c_EBPH.wav",4),
    ("032b_EADH.wav",4),
    ("021b_EADD.wav",4),
    ("020c_EBPZ.wav",4),
    ("018b_EADZ.wav",4),
])

DEFAULT_CSV = "data/metadata_identification.csv"


if __name__ == "__main__":
    
    from run_benchmark_identification import default_folder_input as folder_input_wav
    folder_input_rttm = os.path.join("data", "rttm_identification")
    
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
        get_audio_metadata(args.input, folder_input_rttm=folder_input_rttm, include_speakers_name=True),
        args.output
    )

