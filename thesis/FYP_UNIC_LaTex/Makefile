all: thesis.tex # thesis_ref.bib
	pdflatex -shell-escape thesis.tex
	bibtex thesis.aux
	pdflatex -shell-escape thesis.tex
	pdflatex -shell-escape thesis.tex

clean:
	rm -f *.aux *.log *.out *.pyg *.toc *.bbl *.blg thesis.pdf
