// ============================================================================
// Flight Reading PDF Template — iPad Pro optimized
//
// Designed for reading technical docs on iPad Pro (A5 size, ~8-11" screens),
// following conventions from popular open-source projects:
//   - Typst official docs (A4 default, serif body, generous spacing)
//   - rust-lang/book (1.6 line-height, comfortable measure ~65-75 chars)
//   - 阮一峰博客 (warm gray body color, 黑体 title + 宋体/英文衬线 body)
//
// Font stack (macOS system fonts, no installation needed):
//   - Body CN:   PingFang SC    (iOS/iPadOS native screen-optimized sans)
//   - Body EN:   Charter        (Apple's iBooks default serif, static TTF)
//   - Heading:   Helvetica Neue + PingFang SC (clean sans for hierarchy)
//   - Emphasis:  Songti SC      (serif for pull-quotes / block comments)
//   - Code:      Menlo          (Apple's default programming font)
//
// NOTE: We avoid variable fonts like "New York" because Typst 0.14 does
// not fully support them yet (renders with warnings).
// ============================================================================

#let flight-doc(
  title: none,
  subtitle: none,
  author: "TAAC 2026 offline prep",
  date: none,
  // Page settings — A5 is the sweet spot for iPad Pro portrait reading
  paper: "a5",
  // Font sizing — slightly larger than typical, for long-form reading
  body-font-size: 10pt,
  body-leading: 0.8em,           // line-height ≈ 1.7
  // Colors — warm dark gray instead of pitch black (easier on eyes)
  text-color: rgb("#2d2d2d"),
  meta-color: rgb("#666666"),
  link-color: rgb("#0b5cad"),
  quote-bg: rgb("#f5f1e8"),       // warm cream for quote blocks
  code-bg: rgb("#f4f4f6"),        // subtle cold gray for code blocks
  accent-color: rgb("#b8463a"),   // rust-red for emphasis (matches Songti warmth)
  body,
) = {
  // ---------- Document metadata ----------
  set document(title: title, author: author)

  // ---------- Page layout ----------
  // A5 = 148 x 210 mm. Generous outer margin for thumb-holding iPad.
  set page(
    paper: paper,
    margin: (
      top: 15mm,
      bottom: 15mm,
      inside: 14mm,
      outside: 14mm,
    ),
    header: context {
      let page-num = counter(page).get().first()
      if page-num > 1 {
        set text(size: 7.5pt, fill: meta-color, font: ("Helvetica Neue", "PingFang SC"))
        grid(
          columns: (1fr, auto),
          align(left)[#title],
          align(right)[
            #counter(page).display() / #context counter(page).final().first()
          ],
        )
        v(2pt)
        line(length: 100%, stroke: 0.4pt + meta-color.lighten(40%))
      }
    },
  )

  // ---------- Global text defaults ----------
  // Font stack (macOS system fonts, no installation needed):
  //   - Body CN:   PingFang SC    (iOS/iPadOS native screen-optimized sans)
  //   - Body EN:   Charter        (Apple's iBooks default serif, static TTF)
  //   - Emoji:     Apple Color Emoji (for 📭 💤 ✨ etc in tables)
  set text(
    font: ("Charter", "PingFang SC", "Apple Color Emoji"),
    size: body-font-size,
    fill: text-color,
    lang: "zh",
    region: "cn",
    cjk-latin-spacing: auto,
  )
  set par(
    leading: body-leading,
    justify: true,
    first-line-indent: 0pt,
    spacing: 0.9em,
  )

  // ---------- Headings ----------
  // Body-level H1 is common in markdown sources (one per "paper" / "section").
  // Keep it clearly smaller than the cover title so the hierarchy reads.
  show heading: set block(above: 1.2em, below: 0.55em)
  show heading.where(level: 1): it => {
    set text(
      font: ("Helvetica Neue", "PingFang SC"),
      size: 15pt,
      weight: "bold",
      fill: text-color,
    )
    block(it.body)
    v(-0.45em)
    line(length: 100%, stroke: 1pt + accent-color)
    v(0.1em)
  }
  show heading.where(level: 2): it => {
    set text(
      font: ("Helvetica Neue", "PingFang SC"),
      size: 12.5pt,
      weight: "bold",
      fill: accent-color,
    )
    it
  }
  show heading.where(level: 3): it => {
    set text(
      font: ("Helvetica Neue", "PingFang SC"),
      size: 11pt,
      weight: "bold",
      fill: rgb("#0b5cad"),
    )
    it
  }
  show heading.where(level: 4): it => {
    set text(
      font: ("Helvetica Neue", "PingFang SC"),
      size: 10pt,
      weight: "bold",
      fill: text-color,
    )
    it
  }

  // ---------- Links ----------
  show link: it => text(fill: link-color, it)

  // ---------- Inline code ----------
  // Allow inline code to break mid-identifier when it overflows a narrow
  // column (e.g. inside a table cell on A5). Without this, long symbols
  // like `MultiSeqHyFormerBlock` punch through table borders.
  show raw.where(block: false): it => box(
    fill: code-bg,
    inset: (x: 3pt, y: 1pt),
    outset: (y: 2pt),
    radius: 2pt,
    text(
      font: ("Menlo", "PingFang SC"),
      size: 0.85em,
      fill: rgb("#b8463a"),
      it,
    ),
  )

  // ---------- Code blocks ----------
  // On A5, Python lines often exceed the measure. We shrink to 7.5pt and
  // keep word-wrap on (Typst default) so lines that overflow break at
  // natural token boundaries rather than being clipped. The line number
  // prefix ("42 | ...") from our snippet builder stays visually stable
  // because we avoid justification inside code blocks.
  show raw.where(block: true): it => block(
    fill: code-bg,
    inset: (x: 9pt, y: 7pt),
    radius: 4pt,
    width: 100%,
    stroke: (left: 2pt + accent-color.lighten(20%)),
    text(
      font: ("Menlo", "PingFang SC"),
      size: 7.5pt,
      fill: rgb("#1a1a1a"),
      {
        set par(leading: 0.5em, justify: false, first-line-indent: 0pt)
        it
      },
    ),
  )

  // ---------- Block quotes ----------
  show quote.where(block: true): it => block(
    fill: quote-bg,
    inset: (x: 12pt, y: 10pt),
    radius: 3pt,
    width: 100%,
    stroke: (left: 3pt + accent-color),
    text(
      font: ("Songti SC", "Charter"),
      style: "italic",
      fill: text-color,
      it.body,
    ),
  )

  // ---------- Lists ----------
  set list(
    indent: 4pt,
    body-indent: 6pt,
    marker: ([•], [◦], [‣]),
    spacing: 0.65em,
  )
  set enum(
    indent: 4pt,
    body-indent: 6pt,
    spacing: 0.65em,
  )

  // ---------- Tables ----------
  // On A5 default equal-width columns squeeze long text. We use a slightly
  // smaller font size for tables so more text fits per column before wrap.
  set table(
    stroke: 0.5pt + rgb("#cccccc"),
    fill: (_, row) => if row == 0 { rgb("#efefef") } else { none },
    inset: (x: 5pt, y: 4pt),
  )
  show table: set text(size: 8.5pt)
  show table.cell.where(y: 0): set text(weight: "bold", size: 9pt)

  // Markdown renders tables using `figure(table(...))`. Disable the auto
  // caption so tables sit tight against surrounding text.
  show figure.where(kind: table): set figure(numbering: none)

  // ---------- Emphasis ----------
  show strong: it => text(fill: accent-color, weight: "bold", it.body)

  // ---------- Cover page ----------
  if title != none {
    align(left + top)[
      #v(12mm)
      #text(
        font: ("Helvetica Neue", "PingFang SC"),
        size: 26pt,
        weight: "bold",
        fill: text-color,
      )[#title]

      #if subtitle != none [
        #v(3mm)
        #text(
          font: ("Songti SC", "Charter"),
          size: 13pt,
          style: "italic",
          fill: meta-color,
        )[#subtitle]
      ]

      #v(4mm)
      #line(length: 40%, stroke: 1.5pt + accent-color)
      #v(3mm)

      #text(font: ("Helvetica Neue", "PingFang SC"), size: 9pt, fill: meta-color)[
        #author
        #if date != none [ · #date ]
      ]
    ]
    v(10mm)
  }

  // ---------- Document body ----------
  body
}
