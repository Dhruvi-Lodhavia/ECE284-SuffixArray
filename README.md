<TO DO: How to run GPU-parallel, sequential codes>

## Run libdivsufsort
1. Go to the extract folder.
```
cd ~
cd project_ece284/extract/
```
2. Copy the dataset (e.g. Homo_sapiens_60M.fa).
```
cp -rf ../../GPU/data/Homo_sapiens_60M.fa .
```
3. Extract relevant information and store it in another text file.
```
./kseq_test Homo_sapiens_60M.fa > pattern.txt
```
4. Go to the libdivsufsort folder.
```
cd ../libdivsufsort/
```
5. Compile the code.
```
cd build/examples/
make
```
6. Copy the extracted dataset.
```
cp -rf ../../../extract/pattern.txt .
```
7. Run libdivsufsort algorithm and store the suffix array in a text file named 'out_baseline.txt'.
```
./mksary pattern.txt out_baseline.txt
```
\
\

## Check functionality of the proposed implementation against state-of-the-art (libdivsufsort)
1. Go to the compare folder.
```
cd ~
cd project_ece284/compare/
```
2. Run the code to compare the text file results obtained from the proposed implementation and libdivsufsort.
```
./kseq_test ../../GPU/build/out_GPU.txt ../libdivsufsort/build/examples/out_baseline.txt
```

If functionality matches, it flashes the following message-
**PASS: Sequential vs Parallel implementation functionality MATCHING!**
\
*Note: you can use this script to compare any two text files. Just run the executable two text files to be compared as arguments.*
