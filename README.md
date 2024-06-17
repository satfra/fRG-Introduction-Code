# fRG Introduction Code

## Getting the Repo

To download the codes, first you need to install git. This depends on your system, but e.g. on Ubuntu you simply do
```bash
$ sudo apt install git
```
Then you get this repo by invoking
```bash
$ git clone https://github.com/satfra/fRG-Introduction-Code.git
```

## Mathematica Code

You need the entirety of the folder, as the notebook `Yang-Mills.nb` needs the `.m` files lying around. 
You can also see some result plots (almost) at scaling in the folder.

## Julia Code

Of course, you need Julia installed. I recommend using the vscode julia extension, which is excellent.
Then, to install all necessary julia packages, just open a REPL (in vscode by opening the command palette and searching for repl), press `]` and type
```julia
  add OrdinaryDiffEq, LinearAlgebra, SparseArrays, FastBroadcast, PreallocationTools, Plots, SpecialFunctions, ShiftedArrays, BenchmarkTools
```
