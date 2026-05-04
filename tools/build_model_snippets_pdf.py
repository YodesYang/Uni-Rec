"""Build a printable PDF of key `baseline/model.py` sections for offline reading.

Produces `docs/99-raw/flight-pdfs/08-model-py-key-sections.pdf` containing the
class/function ranges most relevant to the 5 core papers on the flight reading
list. Each section has a header page with "which paper it corresponds to" and
a "what to look for" checklist, followed by the actual source lines.

Usage:
    python tools/build_model_snippets_pdf.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)


# Register CJK fonts so Chinese text in the doc can render. macOS ships
# STHeiti and Songti in standard locations. Falls back gracefully if fonts
# are missing (text will show as boxes but PDF still generates).
_CJK_FONT_CANDIDATES = [
    ("STHeiti", "/System/Library/Fonts/STHeiti Medium.ttc"),
    ("STHeitiLight", "/System/Library/Fonts/STHeiti Light.ttc"),
    ("Songti", "/System/Library/Fonts/Supplemental/Songti.ttc"),
]

CJK_FONT_NAME: str | None = None
for name, path in _CJK_FONT_CANDIDATES:
    if Path(path).exists():
        try:
            pdfmetrics.registerFont(TTFont(name, path))
            CJK_FONT_NAME = name
            break
        except Exception:  # noqa: BLE001 — reportlab raises generic Exception
            continue

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PY = PROJECT_ROOT / "baseline" / "model.py"
OUTPUT_PDF = PROJECT_ROOT / "docs" / "99-raw" / "flight-pdfs" / "08-model-py-key-sections.pdf"


# Each tuple: (section_title, start_line, end_line, paper_ref, focus_bullets)
# Line ranges are 1-indexed, inclusive.
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
        "§4 RoPEMultiheadAttention ⭐",
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
        "§6 RankMixerBlock ⭐ (Token Mixing 灵魂)",
        315,
        414,
        "论文 4 RankMixer",
        [
            "token_mixing: (B,T,D)→(B,T,T,d_sub)→transpose→(B,T,D) 无参数",
            "⚠️ 仓库用 shared FFN (fc1/fc2 各一份)，论文原版是 Per-token FFN",
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
        "§11 MultiSeqHyFormerBlock ⭐⭐ (HyFormer 核心 block)",
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
        "§12 OneTransMixedAttn ⭐⭐ (Mixed Parameterization)",
        1044,
        1217,
        "论文 5 OneTrans §Mixed Attn",
        [
            "W_Q_s/W_K_s/W_V_s/W_O_s: S-tokens 共享 4×d² 参数",
            "W_Q_ns[j]/.../W_O_ns[j]: 每个 NS token 独立 4×d²×M 参数",
            "⚠️ 仓库 S-S 用 BIDIRECTIONAL 不用 causal（多 domain 按字母序拼，无时序）",
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
        "§14 OneTransNSConditioner ⭐⭐⭐ (EXP-013 +0.010 AUC 源头)",
        1281,
        1349,
        "仓库自研扩展（有创新奖潜质）",
        [
            "在 OneTrans 主 block 前，给 NS token 加序列 mean-pool 信息",
            "每个 NS 独立 conditioning FFN（mixed-param 一致）",
            "v2 无此模块: 0.848712 full train；v3 有: 0.859408 (+0.010)",
            "⭐ 飞机上想清楚：为什么这个能带来 +0.010？类比 HyFormer 的 MultiSeqQueryGenerator（都是给跨特征交互提供 warm-start）",
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
        "§18 PCVRHyFormer ⭐ (顶层装配)",
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


def read_lines(path: Path, start: int, end: int) -> str:
    """Read inclusive 1-indexed line range as a single string with line numbers."""
    all_lines = path.read_text(encoding="utf-8").splitlines()
    selected = all_lines[start - 1 : end]
    max_lineno_width = len(str(end))
    numbered = [
        f"{str(start + i).rjust(max_lineno_width)} | {line}"
        for i, line in enumerate(selected)
    ]
    return "\n".join(numbered)


def build_pdf() -> None:
    if not MODEL_PY.exists():
        raise FileNotFoundError(f"model.py not found: {MODEL_PY}")

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=12 * mm,
        rightMargin=10 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title="baseline/model.py — Flight Reading Key Sections",
        author="TAAC 2026 offline prep",
    )

    styles = getSampleStyleSheet()
    base_font = CJK_FONT_NAME or "Helvetica"
    title_style = ParagraphStyle(
        "TitleBig",
        parent=styles["Heading1"],
        fontName=base_font,
        fontSize=16,
        leading=20,
        spaceAfter=6,
    )
    section_style = ParagraphStyle(
        "SectionHead",
        parent=styles["Heading2"],
        fontName=base_font,
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#0b5cad"),
        spaceBefore=4,
        spaceAfter=4,
    )
    meta_style = ParagraphStyle(
        "Meta",
        parent=styles["BodyText"],
        fontName=base_font,
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#555555"),
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        parent=styles["BodyText"],
        fontName=base_font,
        fontSize=9,
        leading=11,
        leftIndent=12,
        bulletIndent=2,
    )
    code_style = ParagraphStyle(
        "Code",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=7.2,
        leading=8.4,
    )

    story: list = []

    story.append(
        Paragraph("baseline/model.py — Flight Reading Key Sections", title_style)
    )
    story.append(
        Paragraph(
            "配套文档: docs/10-research/flight-reading-list-2026-05-06.md "
            "+ docs/00-context/model-py-section-map.md<br/>"
            "⭐ 重点段落；⭐⭐ 核心必读；⭐⭐⭐ EXP-013 关键差异源头",
            meta_style,
        )
    )
    story.append(Spacer(1, 8 * mm))

    toc_items = []
    for idx, (title, start, end, paper, _focus) in enumerate(SECTIONS, start=1):
        toc_items.append(
            f"{idx:02d}. {title}  (L{start}-L{end})  ← {paper}"
        )
    story.append(Paragraph("Table of Contents", section_style))
    for item in toc_items:
        story.append(Paragraph(item, bullet_style))
    story.append(PageBreak())

    for idx, (title, start, end, paper, focus) in enumerate(SECTIONS, start=1):
        story.append(Paragraph(f"{idx:02d}. {title}", section_style))
        story.append(
            Paragraph(
                f"Lines {start}-{end} ({end - start + 1} lines) &nbsp;·&nbsp; "
                f"Paper ref: {paper}",
                meta_style,
            )
        )
        story.append(Paragraph("<b>飞机上读的时候重点关注：</b>", meta_style))
        for bullet in focus:
            story.append(Paragraph(f"• {bullet}", bullet_style))
        story.append(Spacer(1, 2 * mm))

        code = read_lines(MODEL_PY, start, end)
        story.append(Preformatted(code, code_style))
        story.append(PageBreak())

    doc.build(story)
    size_kb = OUTPUT_PDF.stat().st_size / 1024
    print(f"Generated: {OUTPUT_PDF}")
    print(f"Size: {size_kb:.1f} KB")


if __name__ == "__main__":
    build_pdf()
