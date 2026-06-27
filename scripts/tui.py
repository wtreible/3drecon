"""Tiny consistent TUI: whiptail dialogs when available + on a TTY, else clean text prompts.
Gives host-side scripts (setup) the same picker style as `recon menu`. Stdlib only.
"""
import os, shlex, shutil, subprocess, sys

# Zenburn-ish whiptail theme — used when this module is invoked outside bin/recon
# (which exports its own). setdefault so an inherited NEWT_COLORS still wins.
os.environ.setdefault("NEWT_COLORS", "\n".join([
    "root=white,black", "shadow=black,gray", "border=green,black",
    "title=brightcyan,black", "window=white,black", "textbox=white,black",
    "label=white,black", "entry=black,lightgray", "listbox=white,black",
    "actlistbox=black,cyan", "sellistbox=white,black", "actsellistbox=black,cyan",
    "checkbox=green,black", "actcheckbox=black,cyan", "button=white,black",
    "actbutton=black,white", "compactbutton=white,black", "helpline=brightcyan,black",
]))


def _tui():
    return bool(shutil.which("whiptail")) and sys.stdin.isatty()


def have_whiptail():
    """True if a whiptail TUI (incl. the --gauge progress bar) can be shown."""
    return _tui()


def _wt(args):
    # whiptail draws its UI on the terminal and returns the result on fd 3; the
    # 3>&1 1>&2 2>&3 swap puts the result on stdout (captured) and the UI on the tty.
    cmd = "whiptail " + " ".join(shlex.quote(a) for a in args) + " 3>&1 1>&2 2>&3"
    r = subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE)
    return r.returncode, r.stdout.decode().strip()


def menu(title, prompt, options, default=None):
    """Single choice from a list of strings."""
    if _tui():
        args = ["--title", title]
        if default in options:
            args += ["--default-item", default]
        args += ["--menu", prompt, "18", "74", str(len(options))]
        for o in options:
            args += [o, ""]
        rc, out = _wt(args)
        if rc != 0:
            sys.exit("cancelled")
        return out
    print(f"\n{title}")
    for i, o in enumerate(options, 1):
        print(f"  {i}) {o}")
    while True:
        v = input(f"{prompt}" + (f" [{default}]" if default else "") + ": ").strip()
        if not v and default:
            return default
        if v in options:
            return v
        if v.isdigit() and 1 <= int(v) <= len(options):
            return options[int(v) - 1]
        pre = [o for o in options if o.lower().startswith(v.lower())]
        if len(pre) == 1:                      # forgiving prefix match ("medium" -> "medium — q85 …")
            return pre[0]
        print(f"  pick 1-{len(options)} or a name")


def inputbox(title, prompt, default=""):
    if _tui():
        rc, out = _wt(["--title", title, "--inputbox", prompt, "10", "74", default])
        if rc != 0:
            sys.exit("cancelled")
        return out or default
    v = input(f"{prompt}" + (f" [{default}]" if default else "") + ": ").strip()
    return v or default


def yesno(title, prompt, default=False):
    if _tui():
        args = ["--title", title] + ([] if default else ["--defaultno"]) + ["--yesno", prompt, "10", "74"]
        rc, _ = _wt(args)
        return rc == 0
    d = "y" if default else "n"
    return (input(f"{prompt} (y/n) [{d}]: ").strip().lower() or d).startswith("y")
