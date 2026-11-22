"""Restore the original tufazip v2 weights after extracting a compact decoder ZIP."""

import hashlib
import io
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


WEIGHTS_SHA256 = "e94c089f8eb985a87307dca3a7702385fcaccd14d95743811f8788f7da24448d"
PACKED_MAGIC = b"TZW1"


def read_exact(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("Truncated weight data")
    return data


def read_header(stream):
    header = read_exact(stream, 12)
    if header[:4] != b"QPI1":
        raise ValueError("Expected a QPI1 model")
    metadata = read_exact(stream, int.from_bytes(header[4:], "little"))
    return header + metadata, json.loads(metadata)["entries"]


def restore(packed):
    stream = io.BytesIO(packed)
    if read_exact(stream, 4) != PACKED_MAGIC:
        raise ValueError("Unknown packed weight layout")
    header, entries = read_header(stream)
    result = bytearray(header)
    for entry in entries:
        scales = read_exact(stream, entry["scale_bytes"])
        # Half-precision scales were stored with low and high bytes grouped.
        interleaved = bytearray(len(scales))
        half = len(scales) // 2
        interleaved[::2], interleaved[1::2] = scales[:half], scales[half:]
        result.extend(interleaved)

        mode = read_exact(stream, 1)
        size = entry["value_bytes"]
        if mode == b"R":
            result.extend(read_exact(stream, size))
        elif mode == b"C" and entry["bits"] == 1:
            rows, cols = entry["shape"]
            labels = np.frombuffer(read_exact(stream, cols), dtype=np.uint8)
            if np.any(labels > 15) or rows * cols != size * 8:
                raise ValueError("Invalid grouped column layout")
            data = np.frombuffer(read_exact(stream, size), dtype=np.uint8)
            grouped = np.unpackbits(data, bitorder="little").reshape(cols, rows)
            columns = np.empty_like(grouped)
            columns[np.argsort(labels & 7, kind="stable")] = grouped
            columns ^= (labels >> 3)[:, None]
            result.extend(np.packbits(columns.T, bitorder="little").tobytes())
        else:
            raise ValueError("Unknown tensor layout")

    if stream.read(1) or hashlib.sha256(result).hexdigest() != WEIGHTS_SHA256:
        raise ValueError("Restored weights differ from the published v2 model")
    return result


def main():
    root = Path(__file__).resolve().parent
    target = root / "weights.bin"
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != WEIGHTS_SHA256:
            raise SystemExit("An unexpected weights.bin already exists")
        print("weights.bin already matches the v2 SHA-256")
        return
    data = restore((root / "weights.packed").read_bytes())
    with NamedTemporaryFile(dir=root, delete=False) as temp:
        partial = Path(temp.name)
    try:
        partial.write_bytes(data)
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    print("Restored weights.bin and verified the v2 SHA-256")


if __name__ == "__main__":
    main()
