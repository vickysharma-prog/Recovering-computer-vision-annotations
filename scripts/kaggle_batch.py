"""
Run the recovery pipeline over thousands of frames on Kaggle, then merge the results.

The laptop writes code and reads results; Kaggle does the downloading and the pipeline.
A run is a split (`data/cache/set_dev.csv` or `set_test.csv`) cut into chunks, one
Kaggle CPU kernel per chunk, and inside each kernel the chunk is cut again into one
export process per core. `scripts/export_dataset.py` is serial per frame, so the second
cut is where the speed comes from.

The code goes up as a private Kaggle *dataset* rather than a git clone, so a run uses
exactly the working tree it was launched from, committed or not. Every run is tagged
and records which snapshot version it used.

Usage:
    python scripts/kaggle_batch.py snapshot                      # upload src/, scripts/, sets
    python scripts/kaggle_batch.py push --tag b0 --split dev --chunks 5
    python scripts/kaggle_batch.py push --tag rate --split dev --chunks 1 --limit 200
    python scripts/kaggle_batch.py status --tag b0 --split dev --chunks 5
    python scripts/kaggle_batch.py pull --tag b0 --split dev --chunks 5
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, "results", "kaggle")
USER = "vickysynapsev"
CODE_DS = f"{USER}/bird-recovery-code"
SHIP = ["src", "scripts", "config.yaml"] + [os.path.relpath(p, ROOT) for p in
        glob.glob(os.path.join(ROOT, "data", "cache", "set_*.csv"))
        + glob.glob(os.path.join(ROOT, "data", "cache", "align_*.json"))]

KERNEL = r'''"""Recovery run {tag}, split {split}, chunk {chunk} of {chunks}."""
import os
os.environ["OMP_THREAD_LIMIT"] = "1"   # tesseract's own threads choke parallel calls
import glob, shutil, subprocess, sys, time, urllib.error, urllib.parse, urllib.request
os.environ.update({env})                # the experiment's flags
from concurrent.futures import ThreadPoolExecutor
t0 = time.time()
def sh(c): print("$", c, flush=True); subprocess.run(c, shell=True, check=True)
sh("apt-get -qq update > /dev/null && apt-get -qq install -y tesseract-ocr > /dev/null")
sh(f"{{sys.executable}} -m pip install -q pytesseract")
src = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/src/align.py", recursive=True)[0]))
REPO = "/kaggle/tmp/repo"; shutil.copytree(src, REPO, dirs_exist_ok=True); os.chdir(REPO)
got = open("SNAPSHOT_ID").read().strip() if os.path.exists("SNAPSHOT_ID") else "none"
print("code snapshot", got, "expected", "{snap}", flush=True)
if got != "{snap}":
    raise SystemExit(f"wrong code snapshot mounted: {{got}} != {snap}; re-push after the dataset finishes processing")
import pandas as pd
frames = pd.read_csv("data/cache/set_{split}.csv").iloc[{chunk}::{chunks}]
if {limit}: frames = frames.head({limit})
BUCKET = "https://twi-aviandata.s3.amazonaws.com/"
def get(key):
    path = os.path.join("data/fixtures/pairs", key.replace("avian_monitoring/", ""))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for attempt in range(6):          # 5 kernels share S3; back off instead of giving up
        try:
            data = urllib.request.urlopen(BUCKET + urllib.parse.quote(key), timeout=120).read()
            open(path, "wb").write(data); return True
        except urllib.error.HTTPError as e:
            if e.code in (403, 404): return False     # not in the bucket; retrying won't help
            time.sleep(2 ** attempt)
        except Exception: time.sleep(2 ** attempt)
    return False
keys = list(frames.screenshot_key) + list(frames.highres_key)
with ThreadPoolExecutor(8) as ex: got = sum(ex.map(get, keys))
print(f"downloaded {{got}}/{{len(keys)}} in {{time.time()-t0:.0f}}s", flush=True)
OUT = "/kaggle/working"; procs = []
for j in range({procs}):
    sub = f"/kaggle/tmp/sub{{j}}.csv"; frames.iloc[j::{procs}].to_csv(sub, index=False)
    # Inherit stdout: every frame's line lands in the Kaggle log, so a live run shows progress.
    procs.append(subprocess.Popen([sys.executable, "scripts/export_dataset.py", "--benchmark", sub,
        "--out", f"{{OUT}}/out_{{j}}", "--align-cache", "{align}",
        "--save-align", f"{{OUT}}/out_{{j}}/align.json"]))
codes = [p.wait() for p in procs]
print("exit codes", codes, f"total {{time.time()-t0:.0f}}s for {{len(frames)}} frames", flush=True)
'''


def kaggle(*args: str) -> str:
    out = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    return (out.stdout + out.stderr).strip()


def kernel_id(tag: str, split: str, chunk: int) -> str:
    return f"{USER}/bird-recovery-{tag}-{split}-{chunk}".lower()


def code_fingerprint() -> str:
    """Hash of every shipped source file. A kernel compares it with the one it was
    pushed for, because Kaggle can mount the previous dataset version while a new
    one is still processing — on 2026-11-23 that silently ran E4 on E3's code."""
    import hashlib
    h = hashlib.sha1()
    for d in ("src", "scripts"):
        for f in sorted(glob.glob(os.path.join(ROOT, d, "*.py"))):
            h.update(open(f, "rb").read())
    return h.hexdigest()[:12]


def snapshot() -> None:
    stage = os.path.join(WORK, "code_snapshot")
    shutil.rmtree(stage, ignore_errors=True)
    repo = os.path.join(stage, "repo")
    for item in SHIP:
        src = os.path.join(ROOT, item)
        dst = os.path.join(repo, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc",
                                                                    "count_ocr_debug", "shape_debug"))
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
    fp = code_fingerprint()
    open(os.path.join(repo, "SNAPSHOT_ID"), "w").write(fp)
    open(os.path.join(WORK, "last_snapshot_id"), "w").write(fp)
    print("snapshot", fp)
    json.dump({"title": "bird recovery code", "id": CODE_DS,
               "licenses": [{"name": "other"}]},
              open(os.path.join(stage, "dataset-metadata.json"), "w"))
    exists = "bird-recovery-code" in kaggle("datasets", "list", "--mine", "-s", "bird-recovery-code")
    if exists:
        print(kaggle("datasets", "version", "-p", stage, "-m", "snapshot", "-r", "zip"))
    else:
        print(kaggle("datasets", "create", "-p", stage, "-r", "zip"))


def push(tag: str, split: str, chunks: int, limit: int, procs: int,
         env: dict, align: str) -> None:
    for c in range(chunks):
        d = os.path.join(WORK, tag, split, f"kernel_{c}")
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "run.py"), "w", encoding="utf-8").write(KERNEL.format(
            tag=tag, split=split, chunk=c, chunks=chunks, limit=limit, procs=procs,
            env=repr(env), align=align or "/kaggle/tmp/none.json",
            snap=open(os.path.join(WORK, "last_snapshot_id")).read().strip()))
        json.dump({"id": kernel_id(tag, split, c), "title": kernel_id(tag, split, c).split("/")[1],
                   "code_file": "run.py", "language": "python", "kernel_type": "script",
                   "is_private": True, "enable_gpu": False, "enable_internet": True,
                   "dataset_sources": [CODE_DS], "competition_sources": [], "kernel_sources": []},
                  open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
        print(c, kaggle("kernels", "push", "-p", d).splitlines()[-1])


def status(tag: str, split: str, chunks: int) -> None:
    for c in range(chunks):
        print(c, kaggle("kernels", "status", kernel_id(tag, split, c)).split("status")[-1])


def pull(tag: str, split: str, chunks: int) -> None:
    import pandas as pd
    base = os.path.join(WORK, tag, split)
    frames, ann, align = [], [], {}
    for c in range(chunks):
        d = os.path.join(base, f"output_{c}")
        os.makedirs(d, exist_ok=True)
        kaggle("kernels", "output", kernel_id(tag, split, c), "-p", d, "-o")
        for sub in sorted(glob.glob(os.path.join(d, "out_*"))):
            if os.path.exists(os.path.join(sub, "frames.csv")):
                frames.append(pd.read_csv(os.path.join(sub, "frames.csv")))
            if os.path.exists(os.path.join(sub, "annotations_full.csv")):
                ann.append(pd.read_csv(os.path.join(sub, "annotations_full.csv")))
            if os.path.exists(os.path.join(sub, "align.json")):
                align.update(json.load(open(os.path.join(sub, "align.json"))))
    f = pd.concat(frames, ignore_index=True)
    f.to_csv(os.path.join(base, "frames.csv"), index=False)
    if ann:
        pd.concat(ann, ignore_index=True).to_csv(os.path.join(base, "annotations_full.csv"),
                                                 index=False)
    json.dump(align, open(os.path.join(base, "align.json"), "w"))
    print(f"{len(f)} frames, {sum(len(a) for a in ann)} dot rows, {len(align)} alignments "
          f"-> {base}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["snapshot", "push", "status", "pull"])
    ap.add_argument("--tag", default="b0")
    ap.add_argument("--split", default="dev", help="data/cache/set_<split>.csv")
    ap.add_argument("--chunks", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="frames per chunk, 0 = all")
    ap.add_argument("--procs", type=int, default=4, help="export processes per kernel")
    ap.add_argument("--env", nargs="*", default=[], help="KEY=VALUE flags for the run")
    ap.add_argument("--align", default="", help="alignment cache path inside the snapshot,"
                                               " e.g. data/cache/align_dev.json")
    a = ap.parse_args()
    if a.cmd == "snapshot":
        snapshot()
    elif a.cmd == "push":
        push(a.tag, a.split, a.chunks, a.limit, a.procs,
             dict(kv.split("=", 1) for kv in a.env), a.align)
    elif a.cmd == "status":
        status(a.tag, a.split, a.chunks)
    else:
        pull(a.tag, a.split, a.chunks)


if __name__ == "__main__":
    sys.exit(main())
