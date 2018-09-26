TBLOBJS = Table1.tbl

.PHONY: all
all: Answer.pdf

Answer.pdf: Answer.tex $(TBLOBJS)
	lualatex Answer.tex

$(TBLOBJS): $(wildcard Result.dir) Plot.py
	python Plot.py

Result.dir: Problem7.py
	python Problem7.py

%.py: %.ipynb
	jupyter nbconvert --to script $*

.PHONY: report
report: Answer.pdf

.PHONY: run
run: Result.dir
