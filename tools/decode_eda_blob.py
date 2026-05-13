#!/usr/bin/env python
"""Decode EDA blob from platform Logs UI output.

Usage::

  # Method 1: paste copied Logs UI text into a file, then decode
  pbpaste > /tmp/logs.txt          # macOS clipboard → file
  python tools/decode_eda_blob.py /tmp/logs.txt

  # Method 2: pipe from clipboard
  pbpaste | python tools/decode_eda_blob.py -

Output: pretty JSON to stdout; writes alongside input as
`<input>.eda.json` when a file path is given.
"""
from __future__ import annotations

import base64
import gzip
import json
import re
import sys
from pathlib import Path


BLOB_START = '<<<EDA_BLOB_START>>>'
BLOB_END = '<<<EDA_BLOB_END>>>'
BASE64_ALPHABET = re.compile(r'[A-Za-z0-9+/=]+')


def extract_blob(text: str) -> str:
    if BLOB_START not in text:
        raise SystemExit(
            f'ERROR: {BLOB_START!r} not found. Platform Logs UI may have '
            'tail-truncated the output. Scroll up to check if blob exists at all.'
        )
    after_start = text.split(BLOB_START, 1)[1]
    if BLOB_END not in after_start:
        raise SystemExit(
            f'ERROR: {BLOB_END!r} not found after start sentinel. '
            'Partial blob — platform truncation hit. Consider reducing '
            'blob_chunk_size or splitting scan into 2+ jobs.'
        )
    blob = after_start.split(BLOB_END, 1)[0]
    # Platform Logs UI prefixes every line with '<timestamp> '. Each chunk line
    # therefore looks like '<timestamp-junk> <long-base64-run>'. We take the
    # *longest* base64 run per line and discard shorter spurious matches
    # (timestamps like '2026' or fragments from date/time tokens).
    chunks: list[str] = []
    for line in blob.splitlines():
        matches = BASE64_ALPHABET.findall(line)
        if not matches:
            continue
        longest = max(matches, key=len)
        # Guard against matching a bare timestamp/date fragment (<= 8 chars)
        # when the real chunk line got UI-truncated. Real chunks in our
        # emitter are >= blob_chunk_size (default 120 chars).
        if len(longest) < 16:
            continue
        chunks.append(longest)
    return ''.join(chunks)


def decode(b64_text: str) -> dict:
    raw = base64.b64decode(b64_text)
    gz = gzip.decompress(raw)
    return json.loads(gz.decode('utf-8'))


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    src = sys.argv[1]
    if src == '-':
        text = sys.stdin.read()
        out_path = None
    else:
        text = Path(src).read_text()
        out_path = Path(src).with_suffix('.eda.json')

    blob = extract_blob(text)
    summary = decode(blob)
    pretty = json.dumps(summary, indent=2)
    print(pretty)
    if out_path is not None:
        out_path.write_text(pretty)
        print(f'\n# wrote {out_path}', file=sys.stderr)


if __name__ == '__main__':
    main()
