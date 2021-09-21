Two utilities "interpolateData" and "textMatrix2Matrix" are needed for predicting and rendering the surface results. We provided the two executable files for Linux in the /bin/ folder. But if you need to run the code on other platforms, please do as follow. 

Hereby we use the public library Vega FEM for compiling our two utility files.
1. Download the Vega FEM library from http://barbic.usc.edu/vega/
2. Copy the "libraries" and "Makefile-headers" folders from Vega FEM into folder "deep_emulator/code/vega_FEM/".
3. Compile the utilities "interpolateData" and "textMatrix2Matrix" via Makefile.
