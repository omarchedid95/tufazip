# Pack the v2 decoder for distribution

Run `python3 tools/pack_decoder.py dist` from a checkout with Python 3 and NumPy. The script writes a baseline ZIP and a compact ZIP using the same file manifest, timestamps, and DEFLATE settings. Both include the vocabulary, dependency lock file, source, and license notices. The compact ZIP also includes the weight restoration program.

The compact format groups columns of one-bit weights by the proportion of ones. It flips columns with more ones than zeros and stores the flip and frequency bin in one byte per column. The builder tests whether the transform helps each matrix. It also groups the low and high bytes of the 16-bit scales. All changes are reversible. Before writing either archive, the builder restores the packed model and compares every byte with the original.

To use the compact archive:

```sh
unzip dist/decoder-compact.zip -d dist/restored
python3 dist/restored/restore_weights.py
cmp weights.bin dist/restored/weights.bin
```

The restoration step needs NumPy and runs on CPU. After restoration, use the upstream decoder instructions in the extracted README. The original enwik9 payload and the GPU requirements for decoding it still apply. The published payload is 100,958,324 bytes. The script adds that size to the measured compact ZIP size. Its locally rebuilt baseline may differ from the original submission ZIP.
