
Algorithm: SemiDIR_OPT references from the literature "An efficient GPU algorithm for lattice Boltzmann method on sparse complex geometries"

*****************************************************************************************************************************************************
Many fluid flow problems, such as the porous media and arterial blood flow, contain sparse complex geometries. However, when the lattice Boltzmann method (LBM) based on graphics processing unit (GPU) is used to simulate these problems, it tends to suffer from low computational performance and high memory consumption. In this paper, a GPU-based LBM algorithm (SemiDIR_OPT) for sparse complex geometries with high computational performance and low memory consumption is proposed. The collected structure of arrays (CSOA) storage layout and AA access pattern are adopted to enhance memory access efficiency and reduce memory consumption, respectively. A semi-direct addressing scheme is employed to further reduce memory consumption, and then it is improved by using an address index array to increase the utilization of GPU threads and a node classification coding scheme to reduce access to global memory. The validated numerical results indicate that the present algorithm has an optimal computational performance and its highest computational performance can be improved several times relative to the other algorithms compared. Furthermore, the memory consumption of the algorithm is significantly lower than that of the other two algorithms based on direct or indirect addressing schemes.

****************************************************************************************************************************************************
The algorithm source code compilation instructions:
(1) Prepare a Linux system (e.g. Ubuntu) configured with NVIDIA's CUDA compilation environment (Version 10.0 or higher);
(2) Put the source files “SemiDIR_OPT.cu” and “common.cuh”together with the porous media flow geometry file“Porous_Media.dat”(This file needs to be extracted from "Porous_Media.zip") in the specified directory；
(3) In a command line window on the Linux system, switch to the directory where the source files are located and execute the command “nvcc -o SemiDIR_OPT kernel.cu” to compile them;
(4) In the current directory, run the generated program file“SemiDIR_OPT”;
(5) To simulate porous media flow at other porosities, just modify the supplied “Porous_Media X.dat” file (This file needs to be extracted from "Porous_Media_X.zip") to “Porous_Media.dat” and place it in the current directory.

In addition, we also provide the compiled program “SemiDIR_OPT” for this algorithm, which can be run on a Linux system equipped with a GPU by putting it into the specified directory together with the file “Porous_Media.dat”.
