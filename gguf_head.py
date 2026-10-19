#!/usr/bin/env python3
"""Read a GGUF tensor table over HTTP range requests, without downloading the weights.

The header, metadata and full tensor-info table all sit at the front of the file, so a
few MB is enough to enumerate every tensor, its shape and its quantisation type. About
16 MB per file against weights of any size. This is the tool behind the GGUF-audit
pieces (gguf-quant-naming, kimi-k3-local-size).

Type ids verified against llama.cpp ggml/include/ggml.h on 2026-07-30. If a type
prints as "typeNN", re-check the enum before drawing any conclusion: guessing this
table produced a false finding once already.

Usage:
  python3 gguf_head.py "https://huggingface.co/<repo>/resolve/main/<file>.gguf"
"""
import struct, sys, json, subprocess

GGML_TYPES = {
 0:"F32", 1:"F16", 2:"Q4_0", 3:"Q4_1", 6:"Q5_0", 7:"Q5_1", 8:"Q8_0", 9:"Q8_1",
 10:"Q2_K", 11:"Q3_K", 12:"Q4_K", 13:"Q5_K", 14:"Q6_K", 15:"Q8_K",
 16:"IQ2_XXS", 17:"IQ2_XS", 18:"IQ3_XXS", 19:"IQ1_S", 20:"IQ4_NL", 21:"IQ3_S",
 22:"IQ2_S", 23:"IQ4_XS", 24:"I8", 25:"I16", 26:"I32", 27:"I64", 28:"F64",
 29:"IQ1_M", 30:"BF16", 34:"TQ1_0", 35:"TQ2_0",
 39:"MXFP4", 40:"NVFP4", 41:"Q1_0", 42:"Q2_0",
}

class Reader:
    """Byte reader that pulls more of the file over HTTP as needed."""
    def __init__(self, url, chunk=8 << 20):
        self.url, self.chunk, self.buf, self.pos = url, chunk, b"", 0
        self._fetch(chunk)

    def _fetch(self, upto):
        if len(self.buf) >= upto:
            return
        # macOS system Python may lack a usable CA bundle, so shell out to curl
        p = subprocess.run(
            ["curl", "-sL", "--max-time", "180",
             "-H", f"Range: bytes={len(self.buf)}-{upto-1}",
             "-A", "stridenote-gguf-audit", self.url],
            capture_output=True)
        if p.returncode != 0:
            raise SystemExit(f"fetch failed: {p.stderr[:200]!r}")
        self.buf += p.stdout

    def take(self, n):
        if self.pos + n > len(self.buf):
            self._fetch(max(self.pos + n, len(self.buf) + self.chunk))
        b = self.buf[self.pos:self.pos + n]
        self.pos += n
        return b

    def u32(self):  return struct.unpack("<I", self.take(4))[0]
    def u64(self):  return struct.unpack("<Q", self.take(8))[0]
    def i64(self):  return struct.unpack("<q", self.take(8))[0]
    def f32(self):  return struct.unpack("<f", self.take(4))[0]
    def f64(self):  return struct.unpack("<d", self.take(8))[0]
    def string(self):
        return self.take(self.u64()).decode("utf-8", "replace")

    def value(self, t):
        if t == 8:  return self.string()
        if t == 9:
            et, n = self.u32(), self.u64()
            # long arrays (tokenizer vocab) are skipped, we only need scalars
            if n > 4096:
                for _ in range(n): self.value(et)
                return f"<array {n}>"
            return [self.value(et) for _ in range(n)]
        sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
        if t == 6:  return self.f32()
        if t == 12: return self.f64()
        if t in (10, 11): return self.u64() if t == 10 else self.i64()
        raw = self.take(sizes.get(t, 4))
        return int.from_bytes(raw, "little")

def read_gguf(url):
    r = Reader(url)
    if r.take(4) != b"GGUF":
        raise SystemExit("not a GGUF file")
    ver = r.u32()
    n_tensors = r.u64()
    n_kv = r.u64()
    meta = {}
    for _ in range(n_kv):
        k = r.string()
        t = r.u32()
        meta[k] = r.value(t)
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        ttype = r.u32()
        off = r.u64()
        tensors.append({"name": name, "dims": dims,
                        "type": GGML_TYPES.get(ttype, f"type{ttype}"), "type_id": ttype})
    return {"version": ver, "meta": meta, "tensors": tensors,
            "bytes_read": len(r.buf)}

if __name__ == "__main__":
    out = read_gguf(sys.argv[1])
    print(json.dumps({"version": out["version"],
                      "n_tensors": len(out["tensors"]),
                      "bytes_read": out["bytes_read"],
                      "meta_keys": len(out["meta"])}, indent=1))
