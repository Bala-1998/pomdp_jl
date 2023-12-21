# pomdp_jl

The Repo contains two folders - folder 1:  discrete_drive_pomdp consisits of the files I am currently working on, which include the python implementation of the simple lane collision scenario and the julia notebook version of it. 
Folder 2: consists of example notebooks in julia which use the POMDPs.jl library for solving some scenarios.

# Installation for running the julia notebooks

1. Install julia REPL using this: https://github.com/JuliaLang/juliaup

2. POMDPs.jl and associated solver packages can be installed using Julia's package manager. For example, to install POMDPs.jl and the QMDP solver package, type the following in the Julia REPL:

  using Pkg; Pkg.add("POMDPs"); Pkg.add("QMDP")

3. All the notebooks can be run normally in vscode using the installed julia kernel

# For More information on Julia Tutorials:

https://www.youtube.com/playlist?list=PLhQ2JMBcfAsi76O13sJzk4LXA_mu5sd9E

