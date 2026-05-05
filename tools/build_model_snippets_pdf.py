"""Build a printable PDF of key `baseline/model.py` sections for offline reading.

Produces `docs/99-raw/flight-pdfs/08-model-py-key-sections.pdf`. Each section
has a header with "which paper it corresponds to" and a "what to look for"
checklist, followed by the actual source lines as a code block.

Pipeline:
  SECTIONS (below) --> generated markdown --> pandoc --> typst --> pdf

Shares the iPad-optimized template with `build_md_to_pdf.py`.

Usage:
    python tools/build_model_snippets_pdf.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = PROJECT_ROOT / "tools"
MODEL_PY = PROJECT_ROOT / "baseline" / "model.py"
OUTPUT_PDF = PROJECT_ROOT / "docs" / "99-raw" / "flight-pdfs" / "08-model-py-key-sections.pdf"
PANDOC_TEMPLATE = TOOLS_DIR / "pandoc-typst-template.typ"
TYPST_TEMPLATE = TOOLS_DIR / "flight-reading-template.typ"


# Each tuple: (section_title, start_line, end_line, paper_ref, focus_bullets)
# Line ranges are 1-indexed, inclusive. Uses plain ASCII stars (*) in titles
# so the markdown-level emphasis handling stays predictable; emoji-level
# emphasis is redundant once the template highlights `strong`.
SECTIONS: list[tuple[str, int, int, str, list[str]]] = [
    (
        "§1 ModelInput (数据结构)",
        11,
        19,
        "—",
        ["理解模型输入每个字段的 shape"],
    ),
    (
        "§2 RotaryEmbedding + rotate_half + apply_rope_to_tensor",
        26,
        92,
        "论文 2 RoFormer §3.4",
        [
            "inv_freq = 1/base^(2i/d) 即论文的 θ_i",
            "_build_cache 预计算 cos/sin",
            "rotate_half 是论文公式 (15) 的 pair-interleave 简化",
        ],
    ),
    (
        "§3 SwiGLU",
        100,
        114,
        "LLaMA/PaLM 系列惯例，baseline 默认激活",
        [
            "为什么 SiLU 比 ReLU/GELU 更稳定",
            "gate 后为什么再一次 Linear（fc_out）",
        ],
    ),
    (
        "§4 RoPEMultiheadAttention ★",
        117,
        243,
        "论文 1 Attention + 论文 2 RoFormer",
        [
            "W_q/W_k/W_v/W_o 四投影 + W_g 是仓库扩展（gate）",
            "forward 里 reshape (B, L, D) → (B, h, L, d/h) 的维度变换",
            "RoPE 是在 Q/K 投影完成 reshape 后、dot-product 之前注入",
            "F.scaled_dot_product_attention 是 PyTorch 2.0+ 融合核",
        ],
    ),
    (
        "§5 CrossAttention",
        244,
        314,
        "HyFormer Query Decoding",
        [
            "和 self-attention 的唯一区别：Q 来源 vs K/V 来源不同",
            "HyFormer 中 Q = query tokens, K/V = sequence tokens",
        ],
    ),
    (
        "§6 RankMixerBlock ★ (Token Mixing 灵魂)",
        315,
        414,
        "论文 4 RankMixer",
        [
            "token_mixing: (B,T,D)→(B,T,T,d_sub)→transpose→(B,T,D) 无参数",
            "⚠ 仓库用 shared FFN (fc1/fc2 各一份)，论文原版是 Per-token FFN",
            "手画 T=4, d_sub=2 的 reshape 过程验证理解",
        ],
    ),
    (
        "§7 MultiSeqQueryGenerator",
        415,
        557,
        "论文 3 HyFormer §3.4 Multi-Sequence",
        [
            "每个 sequence 有 num_queries 个独立 FFN，生成 Q tokens",
            "GlobalInfo = Concat(NS 均池, Seq 均池)",
            "item_conditioned 分支是仓库 DIN-style 扩展（论文无）",
        ],
    ),
    (
        "§8 SwiGLUEncoder (baseline 默认，3.5× 更快)",
        558,
        598,
        "—",
        [
            "为什么它比 Transformer 快 3.5×（attention-free）",
            "Pre-LN + SwiGLU + 残差的极简结构",
        ],
    ),
    (
        "§9 TransformerEncoder (对照)",
        600,
        671,
        "论文 1 Attention",
        [
            "标准 Pre-LN Transformer encoder layer",
            "对比 SwiGLUEncoder 理解 attention 带来的代价",
        ],
    ),
    (
        "§10 LongerEncoder",
        672,
        905,
        "论文 7 LONGER",
        [
            "L > top_k: Cross-Attn，Q = 最新 top_k，K/V = 全部",
            "L ≤ top_k: Self-Attn",
            "复杂度 O(L²) → O(L_H · L_S)",
        ],
    ),
    (
        "§11 MultiSeqHyFormerBlock ★★ (HyFormer 核心 block)",
        906,
        1043,
        "论文 3 HyFormer §3.2-3.4",
        [
            "Step 1 Sequence Evolve: 每个序列独立 encoder",
            "Step 2 Query Decode: 每个 Q 集合 cross-attend 到对应序列",
            "Step 3 Query Boost: 所有 Q + NS 拼起来过 RankMixer",
            "n_total = num_queries * S + num_ns，决定 d_model 约束",
        ],
    ),
    (
        "§12 OneTransMixedAttn ★★ (Mixed Parameterization)",
        1044,
        1217,
        "论文 5 OneTrans §Mixed Attn",
        [
            "W_Q_s/W_K_s/W_V_s/W_O_s: S-tokens 共享 4×d² 参数",
            "W_Q_ns[j]/.../W_O_ns[j]: 每个 NS token 独立 4×d²×M 参数",
            "⚠ 仓库 S-S 用 BIDIRECTIONAL 不用 causal（多 domain 按字母序拼，无时序）",
            "_build_causal_mixed_mask: 手画 L=4, M=3 的 7x7 mask 矩阵",
        ],
    ),
    (
        "§13 OneTransMixedFFN",
        1218,
        1280,
        "论文 5 OneTrans §Mixed FFN",
        [
            "S 共享 FFN / 每个 NS 独立 FFN",
            "参数量: FFN_s (1 份) + FFN_ns[j] (M 份)",
        ],
    ),
    (
        "§14 OneTransNSConditioner ★★★ (EXP-013 +0.010 AUC 源头)",
        1281,
        1349,
        "仓库自研扩展（有创新奖潜质）",
        [
            "在 OneTrans 主 block 前，给 NS token 加序列 mean-pool 信息",
            "每个 NS 独立 conditioning FFN（mixed-param 一致）",
            "v2 无此模块: 0.848712 full train；v3 有: 0.859408 (+0.010)",
            "飞机上想清楚：为什么这个能带来 +0.010？类比 HyFormer 的 MultiSeqQueryGenerator（都是给跨特征交互提供 warm-start）",
        ],
    ),
    (
        "§15 OneTransBlock",
        1350,
        1484,
        "论文 5 OneTrans (整体装配)",
        [
            "整体对比 MultiSeqHyFormerBlock 的三步流程",
            "OneTrans: S+NS 拼成一个 attention 矩阵 + mixed param",
            "HyFormer: cross-attn 桥接 + RankMixer boost",
        ],
    ),
    (
        "§16 GroupNSTokenizer (原 baseline 方式)",
        1485,
        1566,
        "—",
        ["怎么把 sparse features 按 group 聚成 NS tokens"],
    ),
    (
        "§17 RankMixerNSTokenizer (当前默认)",
        1567,
        1688,
        "论文 4 RankMixer NS 部分",
        [
            "为什么当前默认选它而非 Group",
            "user_ns_tokens=5, item_ns_tokens=2 的设计动机",
        ],
    ),
    (
        "§18 PCVRHyFormer ★ (顶层装配)",
        1689,
        2396,
        "整体装配（所有论文的综合体现）",
        [
            "先过一遍数据流：输入 → NS tokenize → Query gen → blocks → 分类",
            "理解 use_onetrans 开关切换 HyFormer vs OneTrans 的位置",
            "enable_dense_bypass / item_conditioned_query / seq_hash_vocab 扩展参数",
        ],
    ),
]


def _read_code_range(path: Path, start: int, end: int) -> str:
    """Return the inclusive 1-indexed line range as code, with line numbers."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if start < 1 or end > len(lines) or start > end:
        raise ValueError(f"Invalid range {start}-{end} for {path} ({len(lines)} lines)")
    selected = lines[start - 1 : end]
    width = len(str(end))
    return "\n".join(f"{str(start + i).rjust(width)} | {line}" for i, line in enumerate(selected))


def _build_markdown() -> str:
    parts: list[str] = []
    parts.append(
        "> 配套文档：`docs/10-research/flight-reading-list-2026-05-06.md` · "
        "`docs/00-context/model-py-section-map.md`  \n"
        "> ★ 重点段落；★★ 核心必读；★★★ EXP-013 关键差异源头\n"
    )

    parts.append("## 目录\n")
    for idx, (title, start, end, paper, _focus) in enumerate(SECTIONS, start=1):
        parts.append(f"{idx}. **{title}** · L{start}-L{end} · 对应 {paper}")
    parts.append("")

    for idx, (title, start, end, paper, focus) in enumerate(SECTIONS, start=1):
        parts.append(f"## {idx:02d}. {title}\n")
        parts.append(
            f"**行号范围**：L{start}-L{end}（{end - start + 1} 行） · "
            f"**对应论文**：{paper}\n"
        )
        parts.append("**飞机上读的时候重点关注**：\n")
        for bullet in focus:
            parts.append(f"- {bullet}")
        parts.append("")

        code = _read_code_range(MODEL_PY, start, end)
        parts.append("```python")
        parts.append(code)
        parts.append("```\n")

    return "\n".join(parts)


def _check_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            f"'{name}' not found in PATH. Install it first "
            f"(e.g. `brew install {name}`)."
        )
    return path


def build_pdf() -> None:
    if not MODEL_PY.exists():
        raise FileNotFoundError(f"model.py not found: {MODEL_PY}")

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    md_body = _build_markdown()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        md_file = tmp_path / "snippets.md"
        md_file.write_text(md_body, encoding="utf-8")

        typst_out = OUTPUT_PDF.with_suffix(".typ")
        pandoc_cmd = [
            _check_tool("pandoc"),
            "--standalone",
            "--from=gfm",
            "--to=typst",
            f"--template={PANDOC_TEMPLATE}",
            "--metadata=title:baseline/model.py — 飞行精读关键段",
            "--metadata=subtitle:Paper↔Code side-by-side",
            "--metadata=date:2026-05-04",
            "-o",
            str(typst_out),
            str(md_file),
        ]
        subprocess.run(pandoc_cmd, check=True)

        sibling_template = typst_out.parent / TYPST_TEMPLATE.name
        shutil.copyfile(TYPST_TEMPLATE, sibling_template)

        typst_cmd = [
            _check_tool("typst"),
            "compile",
            "--root",
            str(typst_out.parent),
            str(typst_out),
            str(OUTPUT_PDF),
        ]
        subprocess.run(typst_cmd, check=True)

    size_kb = OUTPUT_PDF.stat().st_size / 1024
    print(f"Generated: {OUTPUT_PDF.relative_to(PROJECT_ROOT)}")
    print(f"Size: {size_kb:.1f} KB, sections: {len(SECTIONS)}")


if __name__ == "__main__":
    try:
        build_pdf()
    except (RuntimeError, FileNotFoundError, ValueError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as err:
        print(f"ERROR: subcommand failed with exit code {err.returncode}", file=sys.stderr)
        sys.exit(err.returncode)
