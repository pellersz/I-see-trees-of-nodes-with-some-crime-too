### Random forest implemenation 
Not much else to say about it

#### How to build
run `uv run setup.py build_ext --inplace` then `uv run main.py` (or i'll just make a shellscript so it will be easier)

#### Features
- The tree is implemented with Cython so have fun reading reference counting and memory allocations
- It should work faster if ran with a python version which supports free-threading (once I implement it :P)

#### Tested on 
https://archive.ics.uci.edu/dataset/183/communities+and+crime
