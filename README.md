# gguf-audit

Read a GGUF file's tensor table over HTTP range requests, without downloading the weights.

The header, the metadata and the full tensor-info table sit at the front of a GGUF file, so about 16 MB is enough to list every tensor, its shape and its quantisation type, for a file of any size. `gguf_head.py` does that for one file; `sweep.py` runs it across a list of builds and records, per build, the expert count, the quantisation type of each tensor class across every layer, which block index gets promoted, and the distinct weight types in the file.

These are the tools behind StrideNote's GGUF audits, among them:

- [Eight GGUF builds of one model: the filename does not tell you what is inside](https://stridenote.net/gguf-quant-naming/)
- [The smallest Kimi K3 you can download is 589 GB, and two builds of it disagree by 58](https://stridenote.net/kimi-k3-local-size/)
- [Three publishers, one schedule: the quantisation policy we called a choice is a line of llama.cpp](https://stridenote.net/llamacpp-quant-schedule/)
- [Two Unsloth UD quant names, one GGUF file, and not a single IQ1 or IQ2 tensor inside](https://stridenote.net/unsloth-ud-quant-names/)
- [Nemotron 3.5 Lightning's 4-bit GGUF is 12 percent 16-bit, and the weight is a head most people never switch on](https://stridenote.net/nemotron-lightning-gguf-mtp/)

## Use

```
python3 gguf_head.py "https://huggingface.co/<repo>/resolve/main/<file>.gguf"
```

Prints the metadata keys and a per-tensor table (name, shape, type). Needs Python 3 and `curl` on the path; the script shells out to curl for the range requests because the system Python on macOS may lack a CA bundle.

```
python3 sweep.py
```

Audits the builds listed at the top of the file and caches results in `sweep_cache.json`, so a second run is free. Edit the list to audit your own.

## Type ids

`GGML_TYPES` in `gguf_head.py` was checked against `ggml/include/ggml.h` in llama.cpp on 30 July 2026. If a type prints as `typeNN`, re-check the enum before drawing a conclusion. Guessing that table produced one false finding before the check was added.

## Results

`results/` holds the sweeps behind the published pieces: the Qwen builds, a third batch, and the Nemotron 3.5 Lightning publisher sweep with its full NVFP4 tensor listing. Each is the raw output of `sweep.py` or `gguf_head.py` on the date of the article.

## Licence

MIT. Copyright 2026 StrideNote Studio.
