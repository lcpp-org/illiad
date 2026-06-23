Some inputs are on top some on bottom
Maybe point at some parameters that affect the runtime the most, like if I want to do a quick run, what parameter specifically loads the speed
how many times does the particle goes around when doing Poincare
With Poincare it is expected for the user to know that the output will be used in the later steps which is reasonable. However, I do not really know what is inside the file, units, array shapes, etc. Since the code saves everything an a numpy file, maybe it is a good idea to also save a summary. Basically a metadata for humans to read.
runPoincare creates an empty Poincare folder in data and does not store data there
When typing non existent anlys_subdir name it will create a folder with this name and will spit out an error