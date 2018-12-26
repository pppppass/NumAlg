PGFOBJS = Figure1.pgf Figure2.pgf
RUNOBJS = Problem.run

.PHONY: all
all: Answer.pdf

Answer.pdf: Answer.tex $(PGFOBJS) | Answer.aux
	lualatex Answer.tex

Answer.aux: Answer.tex
	lualatex Answer.tex

Figure1.pgf Figure2.pgf: $(wildcard Problem.run)
	touch $@

.PRECIOUS: %.py
%.py: %.ipynb
	jupyter nbconvert --to script $*

%.run: %.py utils/symeig.py utils/ortho.py utils/geneig.py utils/models.py
	python $<
	touch $@

.PHONY: report
report: Answer.pdf

.PHONY: run
run: $(RUNOBJS)
