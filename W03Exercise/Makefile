PGFOBJS = Figure1.pgf
TBLOBJS = Table1.tbl Table2.tbl Table3.tbl Table4.tbl Table5.tbl
RUNOBJS = Problem4.run Problem5.run

.PHONY: all
all: Answer.pdf

Answer.pdf: Answer.tex $(PGFOBJS) $(TBLOBJS) | Answer.aux
	lualatex Answer.tex

Answer.aux: Answer.tex
	lualatex Answer.tex

.SECONDEXPANSION:
$(PGFOBJS) $(TBLOBJS): $(wildcard $(RUNOBJS)) Plot.py
	python Plot.py
	sed -i 's/<LABEL1~*>/\\verb"spsolve"/' Figure1.pgf
	sed -i 's/<LABEL2~*>/\\verb"solve"/' Figure1.pgf
	sed -i 's/<LABEL3~*>/Time (\\Si{s})/' Figure1.pgf

.PRECIOUS: %.py
%.py: %.ipynb
	jupyter nbconvert --to script $*

%.run: %.py utils.py
	python $<
	touch $@

.PHONY: report
report: Answer.pdf

.PHONY: run
run: $(RUNOBJS)
