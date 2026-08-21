# Shepherd: run all six encoders as PARALLEL worker processes (one core each,
# GPU shared by the codec workers), then run Gate 1 via job_b_run — whose
# encode_corpus pass sees a warm cache, skips everything, and goes straight to
# the gate. Kill this pid with /T to stop the whole flock; resume is free.
import subprocess
import sys
import time
from pathlib import Path

import job_b_run

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
WORKER = str(ROOT / "tools" / "encode_worker.py")
ENCODERS = ["logmel", "energy", "encodec24k", "wavlm", "dac44k", "mimi"]


def main() -> None:
    t0 = time.time()
    procs = {}
    for name in ENCODERS:
        log = open(ROOT / "data" / f"worker_{name}.log", "ab")
        procs[name] = (subprocess.Popen([PY, "-u", WORKER, name],
                                        stdout=log, stderr=subprocess.STDOUT, cwd=ROOT), log)
        print(f"spawned {name} pid={procs[name][0].pid}", flush=True)

    failed = []
    for name, (p, log) in procs.items():
        rc = p.wait()
        log.close()
        print(f"{name} exited rc={rc} at +{(time.time() - t0) / 60:.0f} min", flush=True)
        if rc != 0:
            failed.append(name)

    if failed:
        print(f"FAILED workers: {failed} — fix and rerun (cache resumes); NOT running the gate.", flush=True)
        sys.exit(1)

    print(f"all encoders done in {(time.time() - t0) / 3600:.2f} h — running Gate 1", flush=True)
    job_b_run.main()  # warm cache -> encode pass skips -> gate + report


if __name__ == "__main__":
    main()
