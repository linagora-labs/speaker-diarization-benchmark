############################################

import time
import argparse
import json
import requests
import sys
import os
import psutil
import multiprocessing
import tempfile
import py3nvml.py3nvml as pynvml # GPU management
import random
import subprocess

from metadata import get_files_and_metadata

############################################
# VARIABLES

NEW_BUILD_SYSTEM = True
MAX_SPEAKER = 25
LAST_TAG = "1.0.1" if NEW_BUILD_SYSTEM else "3.0.4"

folder_output_cpu = "results/cpu"
folder_output_gpu = "results/cuda"
system_nicknames = {
    1: "pybk",
    2: "pyannote",
    3: "simple_diarizer",
}

def engine_afford_gpu(system_nickname):
    return system_nickname not in ["pybk"]

############################################
# Helpers to monitor GPU VRAM

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # To have GPU in the right order

def get_num_gpus():
    try:
        pynvml.nvmlInit() # Can throw pynvml.NVMLError_DriverNotLoaded if driver problem
    except:
        return 0
    return pynvml.nvmlDeviceGetCount()
    
def has_gpu():
    return get_num_gpus() > 0

# Get VRAM usage (GPU)
def get_vram_usage(job_index = None, gpu_index = None, minimum = 10):
    """
    Args:
        job_index: Job index
        gpu_index: GPU index
        minimum: Minimum memory usage to report the mem usage (per GPU)
    """
    if isinstance(job_index, int):
        job_index = [job_index]
    assert job_index is None or isinstance(job_index, list), "job_index must be None, int or list of int"
    if isinstance(gpu_index, int):
        gpu_index = [gpu_index]

    indices = range(get_num_gpus())
    if isinstance(gpu_index, list):
        for i in gpu_index:
            assert i in indices, "Got gpu_index %d but only %d GPUs available" % (i, indices)
        indices = gpu_index
    else:
        assert gpu_index is None, "gpu_index must be None, int or list of int"
    result = {}
    for igpu in indices:
        handle = pynvml.nvmlDeviceGetHandleByIndex(igpu)
        jobs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        if not len(jobs):
            continue
        if job_index:
            found_job = False
            for job in jobs:
                if job.pid in job_index:
                    found_job = True
                    break
            if not found_job:
                continue
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpuname = pynvml.nvmlDeviceGetName(handle)
        # use = pynvml.nvmlDeviceGetUtilizationRates(handle) # This info does not seem to be reliable
        memused = info.used // 1024**2
        memtotal = info.total // 1024**2
        if memused >= minimum: # There is always a residual GPU memory used (1 or a few MB). Less than 10 MB usually means nothing.
            result[gpuname] = memused
                
    return result

def get_free_gpu_index():
    """
    Returns:
        int: Index of the first free GPU, or None if no GPU is free
    """
    num_gpus = get_num_gpus()
    for i in range(num_gpus-1, -1, -1):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        jobs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        if not len(jobs):
            return i
    return None

############################################
# Helpers to monitor RAM memory

# Get list of running processes
def get_processes(pids_to_ignore=[]):
    for p in psutil.process_iter(attrs=[]):
        if p.info["pid"] in pids_to_ignore:
            continue
        cmdline = p.info["cmdline"]
        if len(cmdline) >= 2 and cmdline[0].startswith("python") and cmdline[1] == "http_server/ingress.py":
            yield p

def get_processes_pid(pids_to_ignore=[]):
    return [p.info["pid"] for p in get_processes(pids_to_ignore=pids_to_ignore)]

# Get RAM usage
def get_ram_usage(pids):
    return sum([psutil.Process(pid=p).memory_info().rss / (1024 * 1024) for p in pids])

############################################
# Helpers to monitor memory (both RAM and VRAM)

# see monitor_memory
LOG_FILE_RAM = None
LOG_FILE_VRAM = None
LOG_FILE_GPUS = None

class monitor_memory(object):

    def __init__(self, pids, verbose=False):
        self.pids = pids
        self.verbose = verbose

    def __enter__(self):
        # We have to use files because global variables are not shared between processes
        global LOG_FILE_RAM, LOG_FILE_VRAM, LOG_FILE_GPUS
        LOG_FILE_RAM = tempfile.mktemp()
        self.file_ram = open(LOG_FILE_RAM, "w")
        LOG_FILE_VRAM = tempfile.mktemp()
        self.file_vram = open(LOG_FILE_VRAM, "w")
        LOG_FILE_GPUS = tempfile.mktemp()
        self.file_gpus = open(LOG_FILE_GPUS, "w")
        self.gpus = []
        self.p = multiprocessing.Process(target=self.continuously_monitor_memory)
        self.p.start()
        return self
    
    def continuously_monitor_memory(self, sleep_time=0.5):
        while True:
            memory = get_ram_usage(self.pids)
            if self.verbose:
                print(f"Memory usage: {memory} MB")
            self.file_ram.write(f"{memory}\n")
            self.file_ram.flush()
            
            gpu_memory = get_vram_usage(self.pids)
            if gpu_memory:
                total_memory = sum(gpu_memory.values())
                for gpu in gpu_memory:
                    if gpu not in self.gpus:
                        self.gpus.append(gpu)
                        if self.verbose:
                            print(f"Using GPU: {gpu}")
                        self.file_gpus.write(f"{gpu}\n")
                        self.file_gpus.flush()
                if self.verbose:
                    print(f"GPU memory usage {list(gpu_memory.keys())}: {total_memory} MB")
                self.file_vram.write(f"{total_memory}\n")
                self.file_vram.flush()
            
            time.sleep(sleep_time)

    def __exit__(self, *_):        
        self.p.terminate()
        self.file_ram.close()
        self.file_vram.close()
        self.file_gpus.close()

def _get_peak(filename):
    with open(filename, "r") as f:
        values = [float(line) for line in f]
        res = max(values) if len(values) else None
    os.remove(filename)
    return res

def get_ram_peak(): return _get_peak(LOG_FILE_RAM)
def get_vram_peak(): return _get_peak(LOG_FILE_VRAM)
def get_gpus():
    with open(LOG_FILE_GPUS, "r") as f:
        res = f.read().splitlines()
    os.remove(LOG_FILE_GPUS)
    return res

############################################
# Docker

def launch_docker(tag, name="linto-platform-diarization", prefix = "diarization_bench", options=""):

    main_version = int(tag.split(".")[0])
    port = 8080
    port += random.randint(0, 120)
    if name.startswith("linto-platform"):
        system_nickname = system_nicknames.get(main_version, None)
    else:
        # NEW_BUILD_SYSTEM
        system_nickname = name.split('-')[-1]
    system_name = system_nickname + f"-{tag}"

    use_gpu = get_num_gpus() > 0
    if use_gpu and not engine_afford_gpu(system_nickname):
        print(f"System {system_nickname} does not support GPU, using CPU")
        use_gpu = False
    device = "cuda" if use_gpu > 0 else "cpu"

    docker_image_name = f"{name}:{tag}"
    success = not os.system(f"docker inspect {docker_image_name} > /dev/null 2>&1")        
    if not success:
        docker_image_name = f"lintoai/{name}:{tag}"
        success = not os.system(f"docker pull {docker_image_name} 2> /dev/null")
        if not success:
            raise RuntimeError(f"Could not find docker image {docker_image_name} (neither locally nor on lintoai)")

    pids_to_ignore = get_processes_pid()

    print("Launching docker image and waiting...")
    dockername = f"{prefix}_{system_name}_{port}"
    command = f"docker run --rm --name {dockername}"
    command += f" -p {port}:80"
    command += " --env SERVICE_MODE=http"
    command += " --env CONCURRENCY=0"
    command += " --env NUM_THREADS=4"
    command += f" --env DEVICE={device}"
    command += " --shm-size=1gb"
    command += " --tmpfs /run/user/0"
    # home = os.path.expanduser('~')
    # command += f" -v {home}/.cache:/root/.cache"
    if use_gpu: # docker needs something like: command += f" --gpus all"
        # Restrict to one (free) GPU
        i = get_free_gpu_index()
        assert i is not None, "No free GPU found"
        command += f" --gpus '\"device={i}\"'"
    if options:
        command += f" {options}"
    command += f" {docker_image_name}"
    print(command)
    os.system(f"docker stop {dockername} 2> /dev/null")
    os.system(command+" &")
    print("Waiting for the docker to start...")
    time.sleep(60)

    pids = get_processes_pid(pids_to_ignore=pids_to_ignore)
    if len(pids) == 0:
        raise RuntimeError("No process found. Probably the docker did not start correctly.")

    return {
        "port": port,
        "pids": pids,
        "dockername": dockername,
        "system_name": system_name,
        "device": device,
        "url": f"http://127.0.0.1:{port}/diarization", # 127.0.0.1 stands for localhost in most cases
    }

############################################
# Utils

def convert_audio(input_file, sample_rate=16000):
    output_file = os.path.join(tempfile.gettempdir(), os.path.basename(file))
    if not os.path.exists(output_file):
        command = f"ffmpeg -i {input_file} -y -acodec pcm_s16le"
        command += f" -ac 1"
        command += f" -ar {sample_rate} {output_file}"
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if not os.path.isfile(output_file):
            stderr = stderr.decode("utf-8")
            raise Exception(f"Failed transcoding (command: {command}):\n{stderr}")
    return output_file

############################################
# Main loop

default_folder_input = os.path.dirname(os.path.realpath(__file__)) + "/data/benchmark/wav"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='speaker diarization benchmark')
    parser.add_argument('folder_input', type=str, default=default_folder_input, help='folder containing the audio files to process', nargs='?')
    parser.add_argument('--name', type=str, default="linto-diarization-simple" if NEW_BUILD_SYSTEM else "linto-platform-diarization", help='name of the docker image to use')
    parser.add_argument('--tag', type=str, default=LAST_TAG, help='tag of the docker image to use, with numbers (ex: 1.0.1, 2.0.0, ...)')
    parser.add_argument('--convert_audio', default=False, action='store_true', help='convert audio to wav in 16kHz before processing')
    parser.add_argument('--overwrite', default=False, action='store_true', help='overwrite existing results (by default, existing experiments will be skipped)')
    args = parser.parse_args()

    assert os.path.isdir(args.folder_input), f"Folder {args.folder_input} does not exist"

    metadata = get_files_and_metadata()

    docker = launch_docker(args.tag, name=args.name)
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


    first_run = True
    try:
        for use_spk_number in False, True, :

            # Process from the shortest to the longest audio
            for file in sorted(metadata.keys(), key=lambda x: metadata[x]["duration"]): 
            # # Process in alphabetical order
            # for file in sorted(metadata.keys()):

                spk_number = metadata[file]["num_speakers"]

                output_dir_name = f"{folder_output}/{'known_spk' if use_spk_number else 'unknown_spk'}/{system_name}"
                output_filename = os.path.join(
                    output_dir_name, os.path.splitext(file)[0].replace("/", "--"))
                output_filename_json = output_filename + ".json"
                output_filename_perfs = output_filename + ".perfs.txt"
                if os.path.exists(output_filename_json) and not args.overwrite:
                    print("Skipping", file)
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

                os.makedirs(output_dir_name, exist_ok=True)

                print("=====================================")
                print("Generating", output_filename_json)
                print("Processing", file, "with", spk_number, "speakers")
                files = {'file': open(file, 'rb')}
                data = {'spk_number': spk_number, 'max_speaker': MAX_SPEAKER}

                # start = time.time()
                # with monitor_memory(pids):
                #     response = requests.post(
                #         url, headers=headers, data=data, files=files)

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

                ram_peak = get_ram_peak()
                assert ram_peak is not None, "Something went wrong when monitoring RAM memory"
                with open(output_filename_perfs, "w") as f:
                    print(f"Time: {time.time() - start:.2f} sec", file=f)
                    print(f"Memory Peak: {ram_peak} MB", file=f)
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
