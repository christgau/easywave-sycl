# SYCL port of easyWave

This is a SYCL port of the easyWave tsunami simulation software created by members of the Zuse Institute Berlin.
Originally, the application was developed at the German Research Center for Geosciences, Potsdam.
Please find the original version at https://git.gfz-potsdam.de/id2/geoperil/easyWave.
The repository at GFZ also contains data to use with easyWave.
The source code here has been obtained from the original CUDA code using the Intel DPC++ Compatibility Tool and was subsequently developed further.

## Important 

The primary objective of this source repository is to explore the usage of the SYCL standard and the performance of its implementations on different accelerators.
While care is taken to ensure computational correctness compared to the original source code, no guarantees can be made on this.
Therefore, no warranty is provided by the authors.
For usage in its original use-case, i.e. early warnings on tsunamis, refer the repository of GFZ.

## Usage

### Compilation

You need a SYCL compiler to build the code.
The Intel oneAPI DPC++ Compiler allows to compile the code for both Intel CPUs and GPUs as well as FPGAs.
Refer to Intel oneAPI documentation for details on the DPC++ compiler.
The open source Intel LLVM compiler (available at https://github.com/intel/llvm) can be used to compile versions for Nvidia (CUDA) and AMD GPUs as well, if the compiler is built accordingly.
According `make(1)` include files are part of this repository. 

If an appropriate SYCL compiler is available, create a link or copy the according include file and name it `make.inc`.
Running GNU make afterwards should successfully build easyWave.

### Running easyWave

To run easyWave, obtain input data from the GFZ repository first.
An execution using SYCL devices (inluding CPUs) requires the `-gpu` command line switch.
A typical run can be started as follows:

```sh
data_path="/path/to/gfz/data"
timestep=5
simulation_time=1440
propagation_time=1500 # disable intermediate outputs
grid="e2r4Pacific.grd"
fault="uz.Tohoku11.grd"

./easywave -gpu -verbose -propagate ${propagation_time} -step ${timestep} -grid "${data_path}/grids/${grid}" -source "${data_path}/faults/${fault}" -time ${simulation_time}
```

This will compute the propagation of the tsunami caused by the Tohuku earth quake in March 2011 near Japan.
