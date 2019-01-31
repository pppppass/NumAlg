PGFOBJS = Figure1.pgf Figure2.pgf Figure3.pgf Figure4.pgf Figure5.pgf Figure6.pgf Figure7.pgf Figure8.pgf Figure9.pgf
TBLOBJS = Table01.tbl Table02.tbl Table031.tbl Table032.tbl Table03.tbl Table041.tbl Table042.tbl Table05.tbl Table06.tbl Table07.tbl Table08.tbl Table091.tbl Table092.tbl Table101.tbl Table102.tbl Table11.tbl Table12.tbl Table13.tbl Table141.tbl Table14.tbl Table151.tbl Table15.tbl Table161.tbl Table16.tbl Table171.tbl Table172.tbl Table181.tbl Table182.tbl Table19.tbl Table20.tbl Table21.tbl Table2211.tbl Table221.tbl Table2221.tbl Table222.tbl Table22.tbl Table2311.tbl Table231.tbl Table2321.tbl Table232.tbl Table23.tbl Table241.tbl Table242.tbl Table251.tbl Table252.tbl
RUNOBJS = Problem0.run Problem1.run Problem2.run Problem3.run Problem4.run ProblemCorr.run
PYDEPS = dgs.py drivers.py gs.py mat.py models.py pro.py res.py spec.py
CDEPS = op/cg.c op/dgs.c op/gs.c op/mat.c op/mg.c op/op.h op/pcg.c op/pro.c op/res.c op/spec.c op/wrappers.c

.PHONY: all
all: Report.pdf

Report.pdf: Report.tex $(PGFOBJS) $(TBLOBJS) | pnote.cls pdef.sty Report.aux
	lualatex Report.tex

Report.aux: Report.tex | pnote.cls pdef.sty
	lualatex Report.tex

pnote.cls pdef.sty:
	ln -s ptmpls/$@ .

$(TBLOBJS): $(wildcard $(RUNOBJS))
	touch $@

$(PGFOBJS): $(wildcard $(RUNOBJS))
	sed -i 's/−/-/g' $@
	touch $@

setup.cfg:
	if [ -z "$$CONDA_PREFIX" ];\
	then\
		echo "Anaconda not found";\
		exit 1;\
	fi
	echo "[build_ext]" > setup.cfg
	echo "include_dirs=${CONDA_PREFIX}/include" >> setup.cfg
	echo "library_dirs=${CONDA_PREFIX}/lib" >> setup.cfg

op.so: setup.py setup.cfg $(CDEPS)
	python setup.py build_ext --inplace
	FILENAMES=`ls | grep 'op.*\.so' | grep -v 'op.so'`;\
	for FILE in $${FILENAMES};\
	do\
		mv $${FILE} op.so;\
	done

.PRECIOUS: %.py
%.py: %.ipynb
	jupyter nbconvert --to script $*

%.run: %.py op.so $(PYDEPS)
	python $<
	touch $@

.PHONY: run
run: $(RUNOBJS)
