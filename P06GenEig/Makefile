PGFOBJS = Figure1.pgf Figure2.pgf Figure3.pgf
TBLOBJS = Table1.tbl Table2.tbl
RUNOBJS = Problem1.run Problem2.run

.PHONY: all
all: Answer.pdf

Answer.pdf: Answer.tex $(PGFOBJS) $(TBLOBJS) | Answer.aux
	lualatex Answer.tex

Answer.aux: Answer.tex
	lualatex Answer.tex

Figure1.pgf Table1.tbl: $(wildcard Problem1.run)
	touch $@

Figure2.pgf Figure3.pgf Table2.tbl: $(wildcard Problem2.run)
	touch $@

.PRECIOUS: %.py
%.py: %.ipynb
	jupyter nbconvert --to script $*

%.run: %.py
	python $<
	touch $@

.PHONY: report
report: Answer.pdf

.PHONY: run
run: $(RUNOBJS)
