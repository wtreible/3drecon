#!/usr/bin/env python3
"""Project manifest (recon.json): capture metadata, inputs, recommended defaults, and a
run log. Stdlib-only so it works on the host (where `recon setup` runs) and in containers.

    python3 scripts/manifest.py show <project>
    python3 scripts/manifest.py log  <project> <output> --status ok --options "..." --result path
"""
import argparse, datetime, json, os, sys


def path(proj):
    return os.path.join(proj, "recon.json")


def load(proj):
    p = path(proj)
    return json.load(open(p)) if os.path.exists(p) else {}


def save(proj, m):
    json.dump(m, open(path(proj), "w"), indent=2)


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def log_run(proj, output, options="", result="", status="ok", seq="", runid="", log=""):
    m = load(proj)
    if not m:
        return  # no manifest -> nothing to log
    rec = {"when": now(), "output": output, "options": options, "result": result, "status": status}
    if seq:
        rec["seq"] = seq
    if runid:
        rec["runid"] = runid
    if log:
        rec["log"] = log
    m.setdefault("runs", []).append(rec)
    save(proj, m)


def _cli():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    lg = sub.add_parser("log")
    lg.add_argument("project"); lg.add_argument("output")
    lg.add_argument("--status", default="ok")
    lg.add_argument("--options", default="")
    lg.add_argument("--result", default="")
    lg.add_argument("--seq", default="")
    lg.add_argument("--runid", default="")
    lg.add_argument("--log", default="")
    sh = sub.add_parser("show"); sh.add_argument("project")
    a = ap.parse_args()
    if a.cmd == "log":
        log_run(a.project, a.output, a.options, a.result, a.status, a.seq, a.runid, a.log)
    else:
        m = load(a.project)
        print(json.dumps(m, indent=2) if m else "(no manifest)")


if __name__ == "__main__":
    _cli()
