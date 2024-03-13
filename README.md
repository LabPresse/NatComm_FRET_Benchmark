# NatComm_FRET_Benchmark
Simulation Datasets and HMM Analysis Code for Nature Communication Matters Arising

## Julia Installation

All the codes are written in Julia language for high performance/speed (similar to C and Fortran) and its open-source/free availability. Julia also allows easy parallelization of all the codes. To install julia, please download and install julia language installer from their official website (see below) for your operating system or use your package manager. The current version of the code has been successfully tested on linux (Ubuntu 22.04), macOS 12, and Windows.

https://julialang.org/

Like python, Julia also has an interactive environment (commonly known as REPL) which can be used to add packages and perform simple tests as shown in the picture below.


![Screenshot from 2023-11-08 14-36-41](https://github.com/ayushsaurabh/B-SIM/assets/87823118/05bdffb9-6857-4209-9d8d-97cedd3a3578)


In Windows, this interactive environment can be started by clicking on the Julia icon on Desktop that is created upon installation or by going into the programs menu directly. On Linux or macOS machines, julia REPL can be accessed by simply typing julia in the terminal. We use this environment to install some essential julia packages that help simplify linear algebra and statistical calculations, and plotting. To add these packages via julia REPL, **first enter the julia package manager by executing `]` command in the REPL**. Then simply execute the following command to install all these packages at the same time. 

```add Distributed, Random, SpecialFunctions, Distributions, LinearAlgebra, Statistics, Plots, HDF5, TiffImages, FixedPointNumbers, FFTW```


![Screenshot from 2023-11-08 14-40-31](https://github.com/ayushsaurabh/B-SIM/assets/87823118/27ffde07-7eb8-40a5-871b-cc4ea0e34859)

**To get out of the package manager, simply hit the backspace key.**

### Environment Creation
**This is for advanced users who already have Julia installed.**
If you already have Julia and do not want to alter your default environment, you can go to the directory where this software is, then 
1. Run Julia then type `]` and `activate .`;
2. Or run Julia in terminal via `julia --project`.
   
These two ways are equivalent. Both of them create a new Julia environment the first time you run it, or otherwise switch to this environment.
