#!/usr/bin/env python3
"""One sweep, many articles.

Audits a list of GGUF builds over HTTP range requests and records, per build:
  - expert count / used / blocks from metadata
  - per-tensor-class quantisation type across every layer, with outliers
  - which block index gets promoted (first, last, or a schedule)
  - distinct weight quant types in the file
  - whether the expert dimension matches the metadata expert_count

Feeds the Group B pieces. Results cache to sweep_cache.json so re-runs are free.
Type ids come from gguf_head.GGML_TYPES, verified against llama.cpp ggml.h.
"""
import collections, json, os, re, sys
from gguf_head import read_gguf

BASE = "https://huggingface.co/{repo}/resolve/main/{f}"
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "sweep_cache.json")
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

def layer_of(name):
    m = re.search(r"blk\.(\d+)\.", name)
    return int(m.group(1)) if m else -1

def audit(repo, fname):
    key = f"{repo}/{fname}"
    if key in cache:
        return cache[key]
    g = read_gguf(BASE.format(repo=repo, f=fname))
    m = g["meta"]
    arch = m.get("general.architecture", "?")
    exp = [t for t in g["tensors"] if "exps" in t["name"] and t["name"].endswith(".weight")]
    body = [t for t in g["tensors"]
            if "exps" not in t["name"] and t["name"].endswith(".weight")
            and t["type"] not in ("F32", "F16", "BF16")]

    classes = {}
    for t in exp:
        cls = re.sub(r"^blk\.\d+\.", "", t["name"]).replace(".weight", "")
        classes.setdefault(cls, []).append((layer_of(t["name"]), t["type"]))

    per_class = {}
    for cls, rows in classes.items():
        rows.sort()
        c = collections.Counter(ty for _, ty in rows)
        majority = c.most_common(1)[0][0]
        outliers = [(ly, ty) for ly, ty in rows if ty != majority]
        per_class[cls] = {"types": dict(c), "majority": majority,
                          "n_layers": len(rows), "outliers": outliers}

    n_exp = m.get(f"{arch}.expert_count")
    all_types = {t["type"] for t in body} | {d["majority"] for d in per_class.values()}
    all_types = {t for t in all_types if t not in ("F32", "F16", "BF16")}

    r = {"arch": arch,
         "expert_count": n_exp,
         "expert_used": m.get(f"{arch}.expert_used_count"),
         "blocks": m.get(f"{arch}.block_count"),
         "file_type": m.get("general.file_type"),
         "general_name": m.get("general.name"),
         "n_tensors": len(g["tensors"]),
         "per_class": per_class,
         "body_types": dict(collections.Counter(t["type"] for t in body)),
         "distinct_weight_types": sorted(all_types),
         "expert_dims_ok": (all(n_exp in t["dims"] for t in exp) if (exp and n_exp) else None),
         "mb_read": g["bytes_read"] // 1048576}
    cache[key] = r
    json.dump(cache, open(CACHE, "w"), indent=1)
    return r

def report(builds, title=""):
    if title:
        print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")
    out = []
    for repo, fname in builds:
        try:
            r = audit(repo, fname)
        except Exception as e:
            print(f"ERR  {repo}/{fname}: {type(e).__name__} {e}")
            continue
        org = repo.split("/")[0]
        print(f"-- {org:<18} {fname}")
        print(f"     arch={r['arch']} experts={r['expert_count']} used={r['expert_used']} "
              f"blocks={r['blocks']} dims_ok={r['expert_dims_ok']} read={r['mb_read']}MB")
        for cls, d in sorted(r["per_class"].items()):
            line = f"     {cls:<20} {d['majority']} x{d['n_layers'] - len(d['outliers'])}"
            if d["outliers"]:
                shown = ", ".join(f"blk{l}={t}" for l, t in d["outliers"][:6])
                more = "" if len(d["outliers"]) <= 6 else f" (+{len(d['outliers'])-6})"
                line += f"   OUTLIER: {shown}{more}"
            print(line)
        print(f"     distinct weight types: {len(r['distinct_weight_types'])} {r['distinct_weight_types']}")
        out.append({"repo": org, "file": fname, **r})
    return out

if __name__ == "__main__":
    QWEN = [
      ("unsloth/Qwen3.5-35B-A3B-GGUF", "Qwen3.5-35B-A3B-Q4_K_M.gguf"),
      ("unsloth/Qwen3.5-35B-A3B-GGUF", "Qwen3.5-35B-A3B-MXFP4_MOE.gguf"),
      ("unsloth/Qwen3.5-35B-A3B-GGUF", "Qwen3.5-35B-A3B-Q3_K_M.gguf"),
      ("bartowski/Qwen_Qwen3.5-35B-A3B-GGUF", "Qwen_Qwen3.5-35B-A3B-IQ2_XXS.gguf"),
      ("bartowski/Qwen_Qwen3.5-35B-A3B-GGUF", "Qwen_Qwen3.5-35B-A3B-IQ3_M.gguf"),
      ("mradermacher/Qwen3.5-35B-A3B-heretic-GGUF", "Qwen3.5-35B-A3B-heretic.Q4_K_M.gguf"),
      ("mradermacher/Qwen3.5-35B-A3B-heretic-GGUF", "Qwen3.5-35B-A3B-heretic.Q2_K.gguf"),
    ]
    res = report(QWEN, "Qwen 3.5 35B-A3B: does the Gemma MoE pattern hold?")
    json.dump(res, open(os.path.join(HERE, "sweep_qwen.json"), "w"), indent=1)
    print(f"\n{len(res)} builds audited")
