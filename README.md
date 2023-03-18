## Setting Up GPU and Sequential codes

1. SSH into the DSMLP server (dsmlp-login.ucsd.edu) using the AD account. MacOS and Linux users can SSH into the server using the following command (replace `dlodhavia` with your username)

```
ssh dlodhavia@dsmlp-login.ucsd.edu
```
2. Next, clone the project repository in your HOME directory using the following command
```
cd ~
git clone https://github.com/Dhruvi-Lodhavia/ECE284-SuffixArray.git
```

3. Download a copy of the TBB version 2019_U9 into your HOME directory:

```
wget https://github.com/oneapi-src/oneTBB/archive/2019_U9.tar.gz
tar -xvzf 2019_U9.tar.gz
```

4. For running the sequential baseline code, 
```
cd ECE284-SuffixArray
cd Sequential  
```
orelse for running the GPU parallelized code,
```
cd ECE284-SuffixArray
cd GPU
```
5. Before executing the `run-commands.sh` file, run
```
chmod u+x run-commands.sh
```
in order to make the .sh file an executable. 
This would have to be done in both sequential and gpu folders.

6. The datasets are given in the `GPU/data/Datasets.zip`  
Unzip the folder by
```
unzip Datasets.zip
```



7. Modify the `run-commands.sh` file according to the dataset being used eg.

For sequential - 

```
./suffix -r ../../GPU/data/Datasets/Homo_sapiens_60M.fa -T 8
```

For GPU - 

```
./suffix -r ../data/Datasets/Homo_sapiens_60M.fa -T 8
```

8. We will be using a Docker container, namely `yatisht/ece284-wi23:latest`, for submitting a job on the cluster containing the right virtual environment to build and test the code. To submit a job that executes `run-commands.sh` script located inside the `Sequential or GPU` directory on a VM instance with 8 CPU cores, 16 GB RAM and 1 GPU device (this is the maxmimum allowed request on the DSMLP platform), the following command can be executed from the VS Code or DSMLP Shell Terminal (replace the username and directory names below appropriately):

```
ssh dlodhavia@dsmlp-login.ucsd.edu /opt/launch-sh/bin/launch.sh -c 8 -g 1 -m 16 -i yatisht/ece284-wi23:latest -f ${HOME}/ECE284-SuffixArray/Sequential/run-commands.sh
```
or
```
ssh dlodhavia@dsmlp-login.ucsd.edu /opt/launch-sh/bin/launch.sh -c 8 -g 1 -m 16 -i yatisht/ece284-wi23:latest -f ${HOME}/ECE284-SuffixArray/GPU/run-commands.sh
```

9. steps for comparing results are given at the end of README




## Run libdivsufsort
1. Go to the extract folder.
```
cd ~
cd ECE284-SuffixArray/extract/
```
2. Copy the dataset (e.g. Homo_sapiens_60M.fa).
```
cp -rf ../GPU/data/Datasets/Homo_sapiens_60M.fa .
```
3. Extract relevant information and store it in another text file.
```
chmod u+x kseq_test
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
chmod u+x mksary
./mksary pattern.txt out_baseline.txt
```


## Check functionality: GPU-parallel vs libdivsufsort
1. Go to the compare folder.
```
cd ~
cd ECE284-SuffixArray/compare/
```
2. Run the code to compare the text file results obtained from the proposed implementation (GPU-parallel) and libdivsufsort.
```
chmod u+x kseq_test
./kseq_test ../GPU/build/out.txt ../libdivsufsort/build/examples/out_baseline.txt
```

3. To compare the text file results obtained from the proposed implementation (Sequential) and libdivsufsort, run the following.
```
chmod u+x kseq_test
./kseq_test ../Sequential/build/out_suffix.txt ../libdivsufsort/build/examples/out_baseline.txt
```

If functionality matches, it flashes the following message -
**PASS: Sequential vs Parallel implementation functionality MATCHING!**
\
*Note: you can use this script to compare any two text files. Just run the executable two text files to be compared as arguments.*
