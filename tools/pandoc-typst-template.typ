// Pandoc typst template — injects our iPad-friendly style.
//
// Pandoc calls this file with document metadata bound to standard variables.
// We wrap the pandoc-generated `body` inside our flight-doc function.

#import "flight-reading-template.typ": flight-doc

// Pandoc emits `#horizontalrule` for markdown `---`. Typst has no built-in
// for this, so define a thin rule that matches our accent color.
#let horizontalrule = align(center, block(
  spacing: 1.2em,
  line(length: 30%, stroke: 0.6pt + rgb("#999999")),
))

// Pandoc's typst writer may also emit these helpers that aren't built in.
#let blockquote(body) = quote(block: true, body)

$if(title)$
#let doc-title = [$title$]
$else$
#let doc-title = none
$endif$

$if(subtitle)$
#let doc-subtitle = [$subtitle$]
$else$
#let doc-subtitle = none
$endif$

$if(author)$
#let doc-author = "$for(author)$$author$$sep$, $endfor$"
$else$
#let doc-author = "TAAC 2026 offline prep"
$endif$

$if(date)$
#let doc-date = "$date$"
$else$
#let doc-date = none
$endif$

#show: flight-doc.with(
  title: doc-title,
  subtitle: doc-subtitle,
  author: doc-author,
  date: doc-date,
)

$body$
