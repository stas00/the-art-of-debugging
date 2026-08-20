# Book Building

This document assumes you're working from the root of the repo.

## Installation requirements

1. Install python packages used during book build
```
pip install -r build/requirements.txt
```

2. Download the free version of [Prince XML](https://www.princexml.com/download/). It's used to build the pdf version of this book.

3. Install the system tools used by the ebook targets:
   - `pdftk` - assembles the final PDF (bookmarks, cover, metadata) in `make pdf`
   - `pandoc` - converts the HTML to EPUB in `make epub`


## Build html

```
make html
```

## Build pdf

```
make pdf
```

It will first build the html target and then will use it to build the pdf version.

## Build epub

```
make epub
```

It will first build the html target and then will use it to build the epub version.


## Check links and anchors

Fast scan of local links and anchors (no HTML build, no `markdown_it`):
```
make check-links-local-fast
```

Full HTML build then `linkchecker` on local links:
```
make check-links-local
```

To additionally also check external links:
```
make check-links-all
```
use the latter sparingly to avoid being banned for hammering servers.

Deliberate redirect sweep (external URLs that have moved; needs network, slow):
```
make check-redirects
```

Batch-check newly added external URLs only:
```
build/check-new-links.sh URL [URL ...]
```

## Style and programs

Report hard-wrapped prose (the book is one line per paragraph):
```
make check-style
```

Re-pad Markdown tables after an edit:
```
make fix-tables
```

Compile-check reader-facing programs (does not run them):
```
make check-programs
```


## Move md files/dirs and adjust relative links


e.g. `slurm` => `orchestration/slurm`
```
src=slurm
dst=orchestration/slurm

mkdir -p orchestration
git mv $src $dst
perl -pi -e "s|$src|$dst|" chapters-md.txt
python build/mdbook/mv-links.py $src $dst
git checkout $dst
make check-links-local

```

## Resize images

When included images are too large, make them smaller a bit:

```
mogrify -format png -resize 1024x1024\> *png
```

Commit figures as PNG rather than SVG, even when the tool emits SVG. Verified
2026-08-18 across all three targets: GitHub sanitizes SVG and strips the
click-to-zoom and text search that are its only advantage, and Prince runs a
wide vector graph off the page edge and clips its labels. Convert with
`rsvg-convert -w 1024` and, where the interactive view matters, tell readers to
generate their own SVG.
