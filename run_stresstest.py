import requests
import json
import os

from run_benchmark import launch_docker, LAST_TAG, MAX_SPEAKER

data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='speaker diarization benchmark')
    parser.add_argument('--name', type=str, default="linto-diarization-simple", help='name of the docker image to use')
    parser.add_argument('--tag', type=str, default=LAST_TAG, help='tag of the docker image to use, with numbers (ex: 1.0.1, 2.0.0, ...)')
    args = parser.parse_args()

    docker = launch_docker(args.tag, name=args.name, prefix = "diarization_stresstest")

    dockername = docker["dockername"]
    url = docker["url"]
    headers = {'accept': 'application/json'}

    failures = []

    try:

        # run stresstest
        for filename, real_nb_spk in [
                ("empty.wav", 0),
                ("bonjour.wav", 1),
            ]:
            filepath = os.path.join(data_folder, filename)

            for spk_number in None, 1, 2,:

                print("> Processing", filepath, "with", spk_number, "speakers")
                files = {'file': open(filepath, 'rb')}
                data = {'spk_number': spk_number, 'max_speaker': MAX_SPEAKER if not spk_number else None}
                response = requests.post(url, headers=headers, data=data, files=files)

                if response.status_code != 200:
                    print('Error:', response.status_code, response.reason)
                    raise RuntimeError("Error while calling the API")

                result = json.loads(response.content.decode('utf-8'))

                num_speakers = len(result['speakers'])

                if spk_number:
                    if num_speakers > spk_number:
                        failures.append(
                            f"Error: {filepath} with {spk_number} speakers: expected at most {spk_number} speakers, got {num_speakers}"
                        )
                    elif num_speakers == 0 and "empty" not in filename:
                        failures.append(
                            f"Error: {filepath} with {spk_number} speakers: got no speaker"
                        )

                if real_nb_spk is not None:
                    if num_speakers != real_nb_spk:
                        failures.append(
                            f"Warning: {filepath} with {spk_number} speakers: expected {real_nb_spk} speakers, got {num_speakers}"
                        )

    finally:

        os.system(f"docker stop {dockername} 2> /dev/null")

        for failure in failures:
            print(failure)
        print(len(failures), "failures")


