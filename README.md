## Random forest 
This project contains an implementation of the Random Forest algorithm and also some testing done with it.

#### Results found
The results obtained and description of the algorithm can be found in the file: `random\_forest.pdf`

#### How to build
run `uv run setup.py build_ext --inplace` then `uv run main.py` or just run the `full_run.sh` script

#### Features
- Built with cython and with multithreading (disable GIL for large speedups)

#### Some comparisons to sklearn
- The training time is comparable to the sklearn implementation (≈ 1.1-1.2 the time it takes for sklearn to converge)
- Test data metrics such as accuracy approximately the same as the sklearn implementation
- Training data metrics slightly better than testing data metrics but not as good as the sklearn implementation

#### Tested on 
https://archive.ics.uci.edu/dataset/183/communities+and+crime
