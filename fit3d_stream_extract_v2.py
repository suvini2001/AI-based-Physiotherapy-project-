"""
Stream fit3d_train.tar.gz and keep ONLY the files you ask for (default: curls).

Shoulder exercises:
    python fit3d_stream_extract_v2.py --out "F:\\fit3d\\shoulder" --cookie "SESSxxxx=yyyy" \
        --keep side_lateral_raise dumbbell_scaptions w_raise rep_ann info readme

The archive is .tar.gz, so all 18 GB pass through the network once, but
only the matching files are written to disk. If the connection drops or
stalls, the script reconnects and asks the server to continue from the
same byte (HTTP Range), so the download does not start over.

Usage (PowerShell):
    pip install requests
    python fit3d_stream_extract.py --out F:\\fit3d\\curls --cookie "SESSxxxx=yyyy"

Do not share the cookie: it logs in as you. Log out of the site when done.
"""
import argparse, os, sys, tarfile, time
import requests

URL = "https://fit3d.imar.ro/data/fit3d_train.tar.gz"
KEEP = ("curl", "rep_ann", "info", "readme")      # parts of names to keep
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/130.0 Safari/537.36",
           "Referer": "https://fit3d.imar.ro/download"}


class ResumableStream:
    """A file-like reader over HTTP that reconnects with a Range header after any network error."""

    def __init__(self, url, headers, timeout=120, max_retries=50):
        self.url, self.headers, self.timeout, self.max_retries = url, headers, timeout, max_retries
        self.done, self.total, self.t0, self.last_print = 0, 0, time.time(), 0
        self.resp = None
        self._connect()

    def _connect(self):
        h = dict(self.headers)
        if self.done:
            h["Range"] = f"bytes={self.done}-"
        r = requests.get(self.url, headers=h, stream=True, timeout=self.timeout)
        if r.status_code in (401, 403):
            sys.exit(f"HTTP {r.status_code}: not logged in (or the cookie expired). "
                     "Log in again, copy a fresh cookie and rerun.")
        if "text/html" in r.headers.get("Content-Type", ""):
            sys.exit("Got a web page, not the archive: pass your login cookie with --cookie.")
        if self.done and r.status_code != 206:
            sys.exit("The server does not support resuming (no HTTP 206). "
                     "Download the file in the browser instead, then extract it locally.")
        r.raise_for_status()
        if not self.done:
            self.total = int(r.headers.get("Content-Length", 0))
        r.raw.decode_content = False
        self.resp = r

    def read(self, n=-1):
        for attempt in range(self.max_retries + 1):
            try:
                b = self.resp.raw.read(n)
                break
            except Exception as e:                      # timeout, reset, broken SSL ...
                if attempt == self.max_retries:
                    raise
                wait = min(60, 5 * (attempt + 1))
                print(f"  connection problem at {self.done / 1e9:.2f} GB ({type(e).__name__}); "
                      f"retry {attempt + 1} in {wait}s", flush=True)
                time.sleep(wait)
                try:
                    self.resp.close()
                except Exception:
                    pass
                try:
                    self._connect()
                except requests.RequestException as e2:
                    print(f"  reconnect failed ({type(e2).__name__}), will retry", flush=True)
        self.done += len(b)
        if self.done - self.last_print > 200e6:
            self.last_print = self.done
            tot = f" of {self.total / 1e9:.1f}" if self.total else ""
            print(f"  {self.done / 1e9:.1f}{tot} GB downloaded, "
                  f"{(time.time() - self.t0) / 60:.0f} min", flush=True)
        return b


def safe_extract(tar, m, out):
    """Extract one member, refusing paths that would land outside `out`."""
    try:
        tar.extract(m, out, filter="data")           # Python 3.10.12+ / 3.11.4+ / 3.12
    except TypeError:                                # older Python: check the path ourselves
        target = os.path.realpath(os.path.join(out, m.name))
        if not target.startswith(os.path.realpath(out) + os.sep):
            print("skipped unsafe path", m.name)
            return
        tar.extract(m, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="folder for the curl files")
    ap.add_argument("--cookie", default=None, help="Cookie header from your logged-in browser")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--keep", nargs="+", default=list(KEEP),
                    help="keep members whose path contains any of these words (lower case)")
    a = ap.parse_args()
    KEEP_WORDS = tuple(k.lower() for k in a.keep)
    print("keeping files whose path contains:", ", ".join(KEEP_WORDS))
    os.makedirs(a.out, exist_ok=True)

    headers = dict(HEADERS)
    if a.cookie:
        headers["Cookie"] = a.cookie
    stream = ResumableStream(a.url, headers)

    kept = 0
    with open(os.path.join(a.out, "fit3d_listing.txt"), "w", encoding="utf-8") as listing, \
         tarfile.open(fileobj=stream, mode="r|gz") as tar:
        for m in tar:
            listing.write(m.name + "\n")
            listing.flush()
            if m.isfile() and any(k in m.name.lower() for k in KEEP_WORDS):
                dest = os.path.join(a.out, m.name)
                if os.path.exists(dest) and os.path.getsize(dest) == m.size:
                    print("already have", m.name, flush=True)
                    continue
                safe_extract(tar, m, a.out)
                kept += 1
                print("kept", m.name, flush=True)
    print(f"Done. {kept} new files kept in {a.out}; full file list in fit3d_listing.txt")


if __name__ == "__main__":
    main()
