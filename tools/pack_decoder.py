"""Build comparable, reproducible DEFLATE-9 decoder ZIPs for tufazip v2."""

import argparse
import hashlib
import io
from pathlib import Path
import zipfile
import zlib

import numpy as np

from restore_weights import PACKED_MAGIC, WEIGHTS_SHA256, read_exact, read_header, restore


ROOT = Path(__file__).resolve().parents[1]
DECODER_FILES = (
    "LICENSE", "NOTICE", "README.md", "arithmetic_coder.py", "checkpoint_utils.py",
    "enwik9.voc", "nn.py", "preprocess.c", "pyproject.toml", "tufazip", "uv.lock",
)


def group_columns(values, shape):
    rows, cols = shape
    bits = np.unpackbits(np.frombuffer(values, dtype=np.uint8), bitorder="little")
    columns = bits.reshape(rows, cols).T
    ones = columns.sum(axis=1)
    flip = (ones > rows // 2).astype(np.uint8)
    # Eight bins of minority-bit frequencies let DEFLATE see similarly biased
    # columns together. Stable ordering means one byte per column suffices.
    bins = np.minimum(np.minimum(ones, rows - ones) * 16 // rows, 7).astype(np.uint8)
    order = np.argsort(bins, kind="stable")
    grouped = (columns ^ flip[:, None])[order]
    labels = bins | (flip << 3)
    return labels.tobytes() + np.packbits(grouped, bitorder="little").tobytes()


def pack(original):
    if hashlib.sha256(original).hexdigest() != WEIGHTS_SHA256:
        raise ValueError("Expected the published tufazip v2 weights")
    stream = io.BytesIO(original)
    header, entries = read_header(stream)
    packed = bytearray(PACKED_MAGIC + header)
    for entry in entries:
        scales = read_exact(stream, entry["scale_bytes"])
        packed.extend(scales[::2] + scales[1::2])
        values = read_exact(stream, entry["value_bytes"])
        mode = b"R"
        shape = entry["shape"]
        if entry["bits"] == 1 and len(shape) == 2 and shape[0] * shape[1] == len(values) * 8:
            grouped = group_columns(values, shape)
            if len(zlib.compress(grouped, 9)) < len(zlib.compress(values, 9)):
                mode, values = b"C", grouped
        packed.extend(mode + values)
    if stream.read(1) or restore(packed) != original:
        raise ValueError("Weight packing failed its byte-for-byte round trip")
    return packed


def build_zip(path, members):
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in sorted(members.items()):
            info = zipfile.ZipInfo(name)  # Fixed 1980 timestamp.
            info.create_system = 3
            info.external_attr = (0o100755 if name == "tufazip" else 0o100644) << 16
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="output directory for the two decoder ZIPs")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    files = {name: (ROOT / name).read_bytes() for name in DECODER_FILES}
    original = (ROOT / "weights.bin").read_bytes()
    compact = pack(original)
    baseline_path = args.directory / "decoder-baseline.zip"
    compact_path = args.directory / "decoder-compact.zip"
    build_zip(baseline_path, {**files, "weights.bin": original})
    build_zip(compact_path, {
        **files, "weights.packed": compact,
        "restore_weights.py": Path(__file__).with_name("restore_weights.py").read_bytes(),
    })
    baseline, candidate = baseline_path.stat().st_size, compact_path.stat().st_size
    print(f"Baseline decoder ZIP: {baseline:,} bytes")
    print(f"Compact decoder ZIP:  {candidate:,} bytes")
    print(f"Saved: {baseline - candidate:,} bytes ({1 - candidate / baseline:.2%} of the decoder)")
    print(f"Published payload + compact decoder: {100_958_324 + candidate:,} bytes")


if __name__ == "__main__":
    main()
