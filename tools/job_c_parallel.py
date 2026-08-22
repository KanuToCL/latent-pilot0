# Shepherd for Phase 7: run all six COMBO workers as parallel processes (one core
# each, GPU shared by the codec workers), then run the RQ2 combo analysis via
# job_c_run - whose encode_combos pass sees a warm cache, skips everything, and goes
# straight to additivity + transfer. Kill this pid with /T to stop the whole flock;
# resume is free (every cell is skipped on the next pass).
import subprocess
import sys
import time
from pathlib import Path

import job_c_run

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
WORKER = str(ROOT / "tools" / "combo_worker.py")
ENCODERS = job_c_run.ENCODERS  # same six as Job B, via job_b_run


def main() -> None:
    t0 = time.time()
    procs = {}
    for name in ENCODERS:
        log = open(ROOT / "data" / f"combo_worker_{name}.log", "ab")
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
        print(f"FAILED workers: {failed} - fix and rerun (cache resumes); NOT running the analysis.", flush=True)
        sys.exit(1)

    print(f"all combo encoders done in {(time.time() - t0) / 3600:.2f} h - running RQ2 combo analysis", flush=True)
    job_c_run.main()  # warm cache -> encode pass skips -> additivity + transfer + report


if __name__ == "__main__":
    main()
