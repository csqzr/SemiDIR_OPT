



//----------------------------------------------------------------------------------------------------------------------------------------------
//	Algorithm: SemiDIR_OPT	references from the literature "An efficient GPU algorithm for lattice Boltzmann method on sparse complex geometries"			         
//  Algorithm design and program implementation by Zhangrong Qin 
//  At runtime, the flow field geometry information "Porous_Media.dat" file needs to be placed in the same directory as the program
//  Simulation results "Porous_Media_Flow.dat" can be viewed using Tecplot 360
// 
//-----------------------------------------------------------------------------------------------------------------------------------------------

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


//Flow field size
#define DX  128
#define DY  128
#define DZ  128

#define P   i*DY*DZ+j*DZ+k 
#define PP  ii*DY*DZ+jj*DZ+kk

#define Pf  ((cnn)/(STRIDE)*DQ+f)*STRIDE+ (cnn)%(STRIDE) //Index of distribution functions in global memory using CSOA storage layout
#define PPf  ((PP)/(STRIDE)*DQ+f)*STRIDE+ (PP)%(STRIDE)

#define Re_Pf ((P)/(STRIDE)*DQ+d_Re[f])*STRIDE+ (P)%(STRIDE)
#define Re_PPf ((PP)/(STRIDE)*DQ+d_Re[f])*STRIDE+ (PP)%(STRIDE)


#define FOR_iDX_jDY_kDZ	 for(int i=0; i<DX;++i) for(int j=0;j<DY;++j) for(int k=0; k<DZ;++k)
#define FOR_iDX_jDY_kDZ_Fluid  for(int i=0; i<DX;++i) for(int j=0;j<DY;++j) for(int k=0; k<DZ;++k) if(Type[P] == FLUID)

#define ThreadX 128 /*64/128/192/256*/ //Thread block size

#define INLET_DEN 1.0005  //Outlet Density
#define OUTLET_DEN 1.0    //Inlet Density

#define STRIDE 64   //the grouping length of the CSOA storage layout

#define GPUID 0    // the GPU used


// the number range of each node group
unsigned int NFluid= 0;
unsigned int NWall = 0;
unsigned int NInlet = 0;
unsigned int NOutlet = 0;
unsigned int NSoild = 0;
unsigned int FLUID_NUM = 0;

//Definition of node type
#define FLUID  1  //01B
#define WALL   2  //10B
#define INLET  0  //00B
#define OUTLET 3  //11B
#define SOILD  4  //
#define UNKNOWN  97


#define DQ  19  //D3Q19 LB Model

//The projections of the discrete velocity ei on the X, Yand Z axes, respectively
const int Ex[19] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0 };
const int Ey[19] = { 0, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0, 1,-1, 1,-1 };
const int Ez[19] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1 };

__constant__ int d_Ex[19] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0 };
__constant__ int d_Ey[19] = { 0, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0, 1,-1, 1,-1 };
__constant__ int d_Ez[19] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1 };


const int Re[DQ] = { 0, 2, 1, 4, 3, 6, 5, 8, 7, 10,9,12,11,14,13,16,15,18,17};
__constant__ int d_Re[DQ] = { 0, 2, 1, 4, 3, 6, 5, 8, 7, 10,9,12,11,14,13,16,15,18,17 };


const double w0 = 1.0 / 3, w1 = 1.0 / 18, w2 = 1.0 / 36;
const double Alpha[DQ] = { w0, w1, w1, w1, w1, w1, w1, w2, w2, w2, w2, w2, w2, w2, w2, w2, w2, w2, w2 };
__constant__ double d_Alpha[DQ] = { 1.0 / 3, 1.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 18,
1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36 };

//Equilibrium distribution function
inline double feq(int f, double den, double Vx, double Vy, double Vz)
{
	double dotMet = Vx * Ex[f] + Vy * Ey[f] + Vz * Ez[f];
	return den * Alpha[f] * (1.0 + 3.0*dotMet + 4.5*dotMet*dotMet - 1.5*(Vx*Vx + Vy*Vy + Vz*Vz));
}

__inline__ __device__  double d_Feq(int f, double Den, double Vx, double Vy, double Vz)
{
	double DotMet = Vx*d_Ex[f] + Vy*d_Ey[f] + Vz*d_Ez[f];
	return Den*d_Alpha[f] * (1.0 + 3.0*DotMet + 4.5*DotMet*DotMet - 1.5*(Vx*Vx + Vy*Vy + Vz*Vz));
}


