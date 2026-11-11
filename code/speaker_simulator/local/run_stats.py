import atexit
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


def start_profile():
    target = os.environ.get('SIMULATOR_PROFILE')
    stage = os.environ.get('SIMULATOR_STAGE')
    if not target or (Path(target).suffix != '.json' and not stage):
        return
    output = Path(target)
    if output.suffix != ".json":
        output = output / f'{os.environ.get("SIMULATOR_CLIP_ID")}_{stage}_{time.time_ns()}.json'
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started, samples, stopped = time.perf_counter(), [], threading.Event()
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    report = {'clip_id': os.environ.get('SIMULATOR_CLIP_ID'), 'stage': stage, 'command': sys.argv, 'gpu': gpu, 'returncode': 0, 'time_scope': 'entry body after Python imports'}

    def sample():
        while not stopped.is_set():
            if gpu:
                try:
                    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                        capture_output=True, text=True, timeout=3)
                    for line in result.stdout.splitlines():
                        pid, memory = line.split(',')
                        if int(pid) == os.getpid():
                            samples.append(float(memory))
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    pass
            stopped.wait(1)

    def failed(kind, value, traceback):
        report['returncode'] = 1
        report['error'] = str(value)
        sys.__excepthook__(kind, value, traceback)

    def finish():
        stopped.set()
        report.update(seconds=time.perf_counter() - started,
                      peak_process_gpu_mib=max(samples) if samples else None,
                      gpu_memory_samples=len(samples), sampling_interval_seconds=1,
                      memory_scope='nvidia-smi memory of this process; sampled peak')
        temporary = output.with_suffix('.partial')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(output)

    sys.excepthook = failed
    atexit.register(finish)
    threading.Thread(target=sample, daemon=True).start()
