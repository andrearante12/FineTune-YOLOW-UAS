# Paper (offline LaTeX)

Files are laid out flat, exactly like the Overleaf project, so they can be copied back and forth.

## Compile
- **VS Code** (LaTeX Workshop extension): open `paper/main.tex`, save to rebuild, `Ctrl+Alt+V` to view the PDF.
- **Terminal:** `make` (build once), `make watch` (rebuild on save), `make clean` (remove build files).

Needs: `texlive-latex-extra texlive-publishers texlive-science texlive-fonts-extra latexmk`.

Note: `new-aiaa.cls` already sets `\bibliographystyle{new-aiaa}`, so the extra
`\bibliographystyle{aiaa}` in `main.tex` is commented out (a second style makes BibTeX error).
