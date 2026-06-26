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

from metadata import get_files_and_metadata, MEMORY_TIME_GROUPS

############################################
# VARIABLES

NEW_BUILD_SYSTEM = True
MAX_SPEAKER = 25
LAST_TAG = "2.3.0" if NEW_BUILD_SYSTEM else "3.0.4"

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
        if cmdline and len(cmdline) >= 2 and cmdline[0].startswith("python") and cmdline[1] == "http_server/ingress.py":
            yield p

def get_processes_pid(pids_to_ignore=[]):
    return [p.info["pid"] for p in get_processes(pids_to_ignore=pids_to_ignore)]

# Get RAM usage (sum of the RSS of a fixed set of PIDs). Kept as a fallback for
# when the container memory cgroup cannot be resolved.
def get_ram_usage(pids):
    return sum([psutil.Process(pid=p).memory_info().rss / (1024 * 1024) for p in pids])

def _read_cgroup_int(path):
    """Read a single-integer cgroup file (e.g. memory.current). None on failure."""
    try:
        with open(path) as f:
            return int(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None

def _resolve_container_cgroup(pids):
    """Locate the container's memory cgroup from one of its (host-visible) PIDs.

    The monitored PIDs run inside the container, so their cgroup is the
    container's -- which accounts for the WHOLE container process tree (the
    server plus any worker/child it spawns per request) via the kernel's own
    memory accounting, the same the OOM killer uses. This fixes the previous
    approach that summed the RSS of a fixed PID list captured at startup and so
    missed per-request child processes and spikes.

    Returns (current_path, peak_path, version):
      - current_path: instantaneous usage file (memory.current / memory.usage_in_bytes)
      - peak_path: resettable high-water-mark file (memory.peak /
        memory.max_usage_in_bytes), or None if unavailable
      - version: 2, 1, or None when resolution failed.
    Reading these files needs no privileges; resetting peak_path may need root.
    """
    for pid in pids:
        try:
            with open(f"/proc/{pid}/cgroup") as f:
                entries = f.read().splitlines()
        except OSError:
            continue
        # cgroup v2: a single unified hierarchy line "0::/<path>"
        for line in entries:
            hierarchy_id, controllers, path = line.split(":", 2)
            if hierarchy_id == "0" and controllers == "":
                base = "/sys/fs/cgroup" + path
                current = os.path.join(base, "memory.current")
                if os.path.exists(current):
                    peak = os.path.join(base, "memory.peak")
                    return current, (peak if os.path.exists(peak) else None), 2
        # cgroup v1: the line carrying the "memory" controller
        for line in entries:
            hierarchy_id, controllers, path = line.split(":", 2)
            if "memory" in controllers.split(","):
                base = "/sys/fs/cgroup/memory" + path
                current = os.path.join(base, "memory.usage_in_bytes")
                if os.path.exists(current):
                    peak = os.path.join(base, "memory.max_usage_in_bytes")
                    return current, (peak if os.path.exists(peak) else None), 1
    return None, None, None

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

    def _current_ram_mb(self):
        """Current container RAM usage (MB), with a PID-RSS fallback."""
        if self.cg_current_path is not None:
            value = _read_cgroup_int(self.cg_current_path)
            if value is not None:
                return value / (1024 * 1024)
        # Fallback when the cgroup could not be resolved/read.
        try:
            return get_ram_usage(self.pids)
        except Exception:
            return None

    def __enter__(self):
        # We have to use files because global variables are not shared between processes
        global LOG_FILE_RAM, LOG_FILE_VRAM, LOG_FILE_GPUS

        # Resolve the container memory cgroup so we monitor the whole container,
        # not just a fixed set of PIDs captured at startup.
        self.cg_current_path, peak_path, self.cg_version = _resolve_container_cgroup(self.pids)

        LOG_FILE_RAM = tempfile.mktemp()
        self.file_ram = open(LOG_FILE_RAM, "w")
        LOG_FILE_VRAM = tempfile.mktemp()
        self.file_vram = open(LOG_FILE_VRAM, "w")
        LOG_FILE_GPUS = tempfile.mktemp()
        self.file_gpus = open(LOG_FILE_GPUS, "w")
        self.gpus = []

        # Best effort: reset the kernel peak counter so it reflects only this
        # request. The same file descriptor is kept open and read back on exit
        # (cgroup v2 'memory.peak' tracks the max per open fd since its last
        # reset). This captures spikes between samples and survives a child
        # being OOM-killed. Resetting usually needs root; on failure we fall
        # back transparently to the sampled maximum of memory.current.
        self.peak_fd = None
        if peak_path is not None:
            fd = None
            try:
                fd = os.open(peak_path, os.O_RDWR)
                os.write(fd, b"0")
                self.peak_fd = fd
            except OSError:
                if fd is not None:
                    os.close(fd)
                self.peak_fd = None

        self.p = multiprocessing.Process(target=self.continuously_monitor_memory)
        self.p.start()
        return self
    
    def continuously_monitor_memory(self, sleep_time=0.2):
        while True:
            memory = self._current_ram_mb()
            if memory is not None:
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
        self.p.join()

        # Always record one final sample so the peak is defined even for very
        # fast requests the background sampler may not have caught.
        final = self._current_ram_mb()
        if final is not None:
            self.file_ram.write(f"{final}\n")

        # Fold in the kernel-tracked high-water mark (true peak across the whole
        # container, including spikes between samples) when we could reset it.
        if self.peak_fd is not None:
            try:
                os.lseek(self.peak_fd, 0, os.SEEK_SET)
                raw = os.read(self.peak_fd, 64).decode().strip()
                self.file_ram.write(f"{int(raw) / (1024 * 1024)}\n")
            except (OSError, ValueError):
                pass
            finally:
                os.close(self.peak_fd)
                self.peak_fd = None

        self.file_ram.flush()
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

def launch_docker(tag, name="linto-diarization-pyannote", prefix = "diarization_bench", options=""):

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
    # Tee the container output to a logfile, so the crash reason survives even with --rm
    # (which removes the container, and thus its logs, as soon as it exits).
    logfile = os.path.join(tempfile.gettempdir(), f"{dockername}.log")
    os.system(f"{command} > {logfile} 2>&1 &")
    print(f"Waiting for the docker to start... (logs: {logfile})")

    # Poll for the server process to appear, instead of waiting a fixed amount of time:
    # startup can be long (recursive chown of mounted volumes, model download, ingestion
    # of speaker samples into Qdrant, ...), but can also be quick when everything is cached.
    max_wait = 300
    pids = []
    waited = 0
    container_was_alive = False
    while waited < max_wait:
        # 'docker run' is launched asynchronously (with '&'), so on the first
        # iterations the container may not show up in 'docker ps' yet because it is
        # still being created. We must not confuse "not appeared yet" with "exited":
        # only treat a missing container as a crash once we have seen it alive.
        container_alive = not os.system(f"docker ps --format '{{{{.Names}}}}' | grep -qx {dockername}")
        container_was_alive = container_was_alive or container_alive
        pids = get_processes_pid(pids_to_ignore=pids_to_ignore)
        if pids:
            break
        if container_was_alive and not container_alive:
            os.system(f"tail -n 40 {logfile}")
            raise RuntimeError(f"Container {dockername} exited before the server started. See logs above ({logfile}).")
        time.sleep(5)
        waited += 5

    if len(pids) == 0:
        os.system(f"tail -n 40 {logfile}")
        raise RuntimeError(f"No process found after {max_wait}s. Probably the docker did not start correctly (see logs above, {logfile}).")

    # The server process exists now, but it still has to load the model and bind the
    # port before it can answer. Poll /healthcheck until it is ready, otherwise the
    # first request races startup and fails with "connection reset by peer".
    health_url = f"http://127.0.0.1:{port}/healthcheck"
    ready = False
    while waited < max_wait:
        try:
            if requests.get(health_url, timeout=5).status_code == 200:
                ready = True
                break
        except requests.exceptions.RequestException:
            pass
        # if the container died during startup, surface the logs instead of looping
        if container_was_alive and os.system(f"docker ps --format '{{{{.Names}}}}' | grep -qx {dockername}"):
            os.system(f"tail -n 40 {logfile}")
            raise RuntimeError(f"Container {dockername} exited during startup. See logs above ({logfile}).")
        time.sleep(3)
        waited += 3

    if not ready:
        os.system(f"tail -n 40 {logfile}")
        raise RuntimeError(f"Server not ready (no 200 from {health_url}) after {max_wait}s (see logs above, {logfile}).")

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

    parser = argparse.ArgumentParser(description='speaker diarization benchmark', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('folder_input', type=str, default=default_folder_input, help='folder containing the audio files to process', nargs='?')
    parser.add_argument('--name', type=str, default="linto-diarization-pyannote" if NEW_BUILD_SYSTEM else "linto-platform-diarization", help='name of the docker image to use')
    parser.add_argument('--tag', type=str, default=LAST_TAG, help='tag of the docker image to use, with numbers (ex: 1.0.1, 2.0.0, ...)')
    parser.add_argument('--env', action='append', default=[], metavar='KEY=VALUE',
        help='extra environment variable(s) to set in the container; repeatable '
             '(e.g. --env PYANNOTE_SEGMENTATION_STEP=0.25)')
    parser.add_argument('--suffix', type=str, default="",
        help='suffix appended to the results folder name, to keep tuned runs separate '
             '(e.g. --suffix _SEGSTEP0.25 -> results in pyannote-2.3.0_SEGSTEP0.25)')
    parser.add_argument('--convert_audio', default=False, action='store_true', help='convert audio to wav in 16kHz before processing')
    parser.add_argument('--overwrite', default=False, action='store_true', help='overwrite existing results (by default, existing experiments will be skipped)')
    parser.add_argument('--groups', type=str, default=None,
        help='comma-separated list of dataset groups to process (e.g. "LINAGORA,ETAPE,SUMM-RE"). '
             'An empty entry matches files without a group (e.g. ",LINAGORA"). By default, all files are processed.')
    parser.add_argument('--memory-time-only', default=False, action='store_true',
        help='only process the files used by plot_memory_time.py for the memory/RTF analysis '
             f'(groups {MEMORY_TIME_GROUPS}). Shortcut to avoid processing files that are not plotted.')
    args = parser.parse_args()

    assert os.path.isdir(args.folder_input), f"Folder {args.folder_input} does not exist"

    if args.memory_time_only:
        assert args.groups is None, "Cannot use both --groups and --memory-time-only"
        groups_filter = set(MEMORY_TIME_GROUPS)
    elif args.groups is not None:
        groups_filter = set(g.strip() for g in args.groups.split(","))
    else:
        groups_filter = None

    metadata = get_files_and_metadata()

    options = " ".join(f"--env {e}" for e in args.env)

    docker = launch_docker(args.tag, name=args.name, options=options)
    url = docker["url"]
    pids = docker["pids"]
    dockername = docker["dockername"]
    system_name = docker["system_name"] + args.suffix
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

                if groups_filter is not None and metadata[file].get("group", "") not in groups_filter:
                    continue

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
                fh = open(file, 'rb')
                data = {'spk_number': spk_number, 'max_speaker': MAX_SPEAKER}

                # start = time.time()
                # with monitor_memory(pids):
                #     response = requests.post(
                #         url, headers=headers, data=data, files=files)

                # Maybe something like this needed to avoid errors at the first run when the docker is not ready yet
                slept_time = 0
                max_sleep_time = 120 if first_run else -1
                first_run = False
                try:
                    while True:
                        try:
                            # Rewind before each attempt: a previous (failed) attempt may have
                            # consumed the stream, and reusing an exhausted handle would upload an
                            # empty/truncated body, which the server rejects as "Invalid data found".
                            fh.seek(0)
                            start = time.time()
                            with monitor_memory(pids):
                                response = requests.post(
                                    url, headers=headers, data=data, files={'file': fh})
                            break
                        except Exception as err:
                            import traceback
                            print(traceback.format_exc())
                            if slept_time > max_sleep_time:
                                raise err
                            print("Warning: retrying http request in 30 sec...")
                            slept_time += 30
                            time.sleep(30)
                finally:
                    fh.close()

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
