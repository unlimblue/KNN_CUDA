.PHONY : build reqs install clean
NINJA := $(-v ninja > /NUL)


build : reqs
	python setup.py bdist_wheel

reqs :
ifndef NINJA 
	copy %cd%\ninja C:\Windows\System32\bin
endif 
	pip3 install -r requirements.txt

install :
	pip3 install --upgrade --find-links=dist KNN_CUDA

clean :
	-rm -rf build %cd%\dist\* *.egg-info
