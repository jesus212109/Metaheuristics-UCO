all: pdf

pdf:
	pdflatex -interaction=nonstopmode report_en.tex
	pdflatex -interaction=nonstopmode report_en.tex

clean:
	rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk

.PHONY: all pdf clean
