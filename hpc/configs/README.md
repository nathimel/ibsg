# README

This directory contains files containing hdyra overrides for array job batching for hpc.

Each file contains one list of overrides per line, as the array batch job will use the array job ID together with awk to index a line and sed the overrides as commandline args to our main script. 

This main script is currently for curve estimation, because curve estimation can take multiple hours but only differ by a few parameters, thus it is well suited for hpc array jobs.

For estimating curves, recall that Dirac delta priors will break embo / i.e. the IB curve estimation. 
- UPDATE: I am trying adding some epsilon PRECISION to see if this helps.