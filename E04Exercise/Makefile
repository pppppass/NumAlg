.PHONY: all
all: Answer.pdf

Answer.pdf: Answer.tex | Answer.aux
	lualatex Answer.tex

Answer.aux: Answer.tex
	lualatex Answer.tex

.PHONY: report
report: Answer.pdf
