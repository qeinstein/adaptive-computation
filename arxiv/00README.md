arXiv submission package. Self-contained: compiles with pdflatex alone, no bibtex
run required.

Upload `arxiv-submission.tar.gz` from the repository root, or the contents of this
directory (main.tex plus figures/). arXiv's AutoTeX does not run BibTeX, so the
bibliography is embedded in main.tex as a thebibliography environment instead of
being generated from a .bib file. refs.bib is kept here for provenance only and is
not read by the build; it is excluded from the upload archive.

This copy is generated from paper/main.tex and differs from it in exactly one
respect: the two lines

    \bibliographystyle{plainnat}
    \bibliography{refs}

are replaced by the equivalent thebibliography block. Every other character is
identical. Regenerate after any change to paper/main.tex rather than editing this
file directly, or the two will drift.

Suggested arXiv categories: cs.CL primary, cs.LG secondary.
