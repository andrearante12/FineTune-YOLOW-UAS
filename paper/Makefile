# Build the paper offline with latexmk (pdflatex + bibtex, reruns as needed).
#   make        build main.pdf once
#   make watch  rebuild automatically every time you save
#   make clean  remove build artifacts (keeps main.pdf)

MAIN := main

.PHONY: all watch clean distclean

all:
	latexmk -pdf -interaction=nonstopmode -file-line-error $(MAIN).tex

watch:
	latexmk -pdf -pvc -interaction=nonstopmode -file-line-error $(MAIN).tex

clean:
	latexmk -c $(MAIN).tex

distclean:
	latexmk -C $(MAIN).tex
