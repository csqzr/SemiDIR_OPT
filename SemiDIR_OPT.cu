


//----------------------------------------------------------------------------------------------------------------------------------------------
//	Algorithm: SemiDIR_OPT	references from the literature "An efficient GPU algorithm for lattice Boltzmann method on sparse complex geometries"			         
//  Algorithm design and program implementation by Zhangrong Qin 
//  At runtime, the flow field geometry information "Porous_Media.dat" file needs to be placed in the same directory as the program
//  Simulation results "Porous_Media_Flow.dat" can be viewed using Tecplot 360
// 
//-----------------------------------------------------------------------------------------------------------------------------------------------

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <string>
#include <sstream> 
#include <fstream> 
#include <iomanip>
#include <iostream> 
#include <time.h>

#include "common.cuh"
#include <stdlib.h>

using namespace std;

//Quantity used on the host
int *Type;
double *Den, *Vx, *Vy, *Vz;

double *FA; //The fluid array used to store the distribution functions of the nodes
unsigned int *AIA; //The address index array
int *FIA; //The fluid index array

long long *NTA; //The node type array

double Mass, Density, Tau, ReTau, Viscosity;
int    NowStep, AllStep, ShowStep, SaveStep;

cudaEvent_t StartTime, EndTime;

bool InitCUDA()
{
	int count = 0;
	int i = 0;

	cudaGetDeviceCount(&count);
	if (count == 0) {
		printf("NOT found GPU!\n");
		return false;
	}
	cudaDeviceProp prop;
	for (i = 0; i < count; i++) {
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if (prop.major >= 1) {
				break;
			}
		}
	}
	if (i == count) {
		printf("NOT found GPU!\n");
		return false;
	}
	cudaGetDeviceProperties(&prop, GPUID);
	printf("GPU is: %s\n", prop.name);
	cudaSetDevice(GPUID);
	printf("CUDA initialized success.\n");
	return true;
}

void MallocHostMemory()
{
	Type= new int[DX*DY*DZ];
	Den = new double[DX*DY*DZ];
	Vx = new double[DX*DY*DZ];
	Vy = new double[DX*DY*DZ];
	Vz = new  double[DX*DY*DZ];
}

void FreeData()
{
	cudaFree(AIA);
	cudaFree(FIA);
	cudaFree(NTA);
	cudaFree(FA);
	
	delete Type;
	delete Den;
	delete Vx;
	delete Vy;
	delete Vz;
}

void Initialize()
{
	Density = 1.0;
	Tau = 1.0;
	
	ReTau =1.0 / Tau;
	
	AllStep = 10000;
	SaveStep = 10000;
	ShowStep = 10000;
	NowStep = 1;
	
	//GPU Initialization------------------------------------------------------
	if (!InitCUDA())
	{
		cout << "No CUDA Device." << endl;
	}
	MallocHostMemory();//Allocate memory space on the host
}

//Allocate space for FIA, FA, AIA and NTA in GPU global memory
void MallocGPUMemory()
{
	int n0 = 0;
	FOR_iDX_jDY_kDZ
	{
		n0++;
	    int p = i*DY*DZ + j*DZ + k;
		if (Type[p] == FLUID)
		{
			NFluid++;
		}
		if (Type[p] == WALL)
		{
			NWall++;
		}
		if (Type[p] == INLET)
		{
			NInlet++;
		}
		if (Type[p] == OUTLET)
		{
			NOutlet++;
		}
		if (Type[p] == SOILD)
		{
			NSoild++;
		}
	}
	printf("Total number of nodes in the flow field :  ");	cout << n0 << " ---------------------" << endl;
	printf("Number of fluid nodes:  "); cout << NFluid << " -----------------" << endl;
	printf("Number of wall nodes:  "); cout << NWall << " -----------------" << endl;
	printf("Number of soild nodes: "); cout << NSoild << " -----------------" << endl;
	printf("Number of inlet nodes: "); cout << NInlet << " ----------------" << endl;
	printf("Number of outlet nodes: "); cout << NOutlet << " ---------------" << endl;


	FLUID_NUM = NFluid + NInlet + NOutlet; //Total number of nodes to be calculated in the flow field

	//Allocate space for FIA, FA, AIA and NTA in GPU global memory
	cudaMallocManaged(&NTA, NFluid * sizeof(long long));
	cudaMallocManaged(&FA, (ceil(NFluid / ((double)STRIDE)) * STRIDE * DQ) * sizeof(double));
	cudaMallocManaged(&AIA, NFluid * sizeof(unsigned int));
	cudaMallocManaged(&FIA, DX*DY*DZ* sizeof(int));
	printf("**************************************************************************************\n");
}
void SetFlowField()
{
	printf("Porous media flow: DX=%d   DY=%d  DZ=%d  Tau=%f \n", DX, DY, DZ, Tau);
	//Reading flow field geometric data from a file-------------------------------------------
	ifstream File;
	File.open("Porous_Media.dat");
	if (!File.is_open())
	{
		cout << "Porous_Media.dat file not found!" << endl;
	}
	char   Buffer[64];
	istringstream Iss;
	int cnn = 0;
	while (!File.eof())
	{
		int i, j, k, flg;
		File.getline(Buffer, 64);
		if (cnn < 1)
			cout << Buffer << endl;
		else
		{
			Iss.clear();
			Iss.str(Buffer);
			Iss >> i >> j >> k >> flg;
			Type[P] = flg;
		}
		cnn++;
	}
	File.close();
	
	//Allocate space for FIA, FA, AIA and NTA in GPU global memory
	MallocGPUMemory();
	cnn = 0;
	int d_Fluid = 0, d_Wall = 0, d_Inlet = 0, d_Outlet = 0;
	//Implementation of the group coding scheme
	FOR_iDX_jDY_kDZ
	{
	   Vx[P] = Vy[P] = Vz[P] = 0;
	   FIA[P] = -1;
	   if (Type[P] == FLUID)
		{
			cnn = d_Fluid;
			FIA[P] = cnn;
			AIA[cnn] = P;
			Den[P] = 1.0;

			for (int f = 0; f < DQ; ++f)
				FA[((cnn) / (STRIDE)*DQ + f)*STRIDE + (cnn) % (STRIDE)] = feq(f, Den[P], Vx[P], Vy[P], Vz[P]);
			d_Fluid++;
		}
		else if (Type[P] == WALL)
		{
			cnn = NFluid + d_Wall;
			FIA[P] = cnn;
			d_Wall++;
		}
		else if (Type[P] == INLET)
		{
			cnn = NFluid + NWall + d_Inlet;
			FIA[P] = cnn;
			d_Inlet++;
		}
		else if (Type[P] == OUTLET)
		{
			cnn = NFluid + NWall + NInlet + d_Outlet;
			FIA[P] = cnn;
			d_Outlet++;
		}
	}

    //The number range of each node group
	NWall = NFluid + NWall;
	NInlet = NFluid + d_Wall + d_Inlet;
	NOutlet = NFluid + NWall + d_Inlet + d_Outlet;

	cnn = 0;
	//Implementation of the node type binary encoding scheme----------------------------------------
	long long p = 0;
	FOR_iDX_jDY_kDZ
	{
		if (Type[P] == FLUID)
		{
			cnn = FIA[P];
			int qq = 1;
			int flg = 0;
			NTA[cnn] = 1LL;
			for (int f = 1; f < 19; f++)
			{
				flg = 0;
				int ii = i + Ex[f]; if (ii < 0 || ii >= DX) { flg = 1; }
				int jj = j + Ey[f]; if (jj < 0 || jj >= DY) { flg = 1; }
				int kk = k + Ez[f]; if (kk < 0 || kk >= DZ) { flg = 1; }
				if (flg == 1)
				{
					qq = qq + 2; continue;
				}
				int _type = Type[PP];
				//01B fluid
				if ((_type >> 1 & 1) == 0 && (_type & 1) == 1)
				{
					p = 1LL << (qq + 1);
					NTA[cnn] |= p;
					p = 0;
					NTA[cnn] &= ~(p&(1LL << (qq + 2)));
				}
				//11B  OULET
				if ((_type >> 1 & 1) == 1 && (_type & 1) == 1)
				{
					p = 1LL << (qq + 1); 
					NTA[cnn] |= p;
					p = 1LL << (qq + 2);
					NTA[cnn] |= p;
				}
				//10B  WALL
				if ((_type >> 1 & 1) == 1 && (_type & 1) == 0)
				{
					p = 0;
					NTA[cnn] &= ~(p&(1LL << (qq + 1)));
					p = 1LL << (qq + 2);
					NTA[cnn] |= p;
				}
				//00B INLET
				if ((_type >> 1 & 1) == 0 && (_type & 1) == 0)
				{
					p = 0;
					NTA[cnn] &= ~(p&(1LL << (qq + 1)));
					NTA[cnn] &= ~(p&(1LL << (qq + 2)));
				}
				qq = qq + 2;
			}
		}
	}
}

void ShowData(int step)
{
	Mass = 0;
	
	for (int i = 0; i < DX; ++i) for (int j = 0; j < DY; ++j) for (int k = 0; k < DZ; ++k)
	{
		if (Type[P]== FLUID)
			Mass += Den[P];
	}
	printf("Steps= %d\t Mass= %.13f\n", step, Mass);
}

void SaveData()
{
	char filename[100];
	FILE* fp;

	sprintf(filename, "Porous_Media_Flow.dat");
	if ((fp = fopen(filename, "w")) == NULL)  
		printf("cannot open the %s file\n", filename);
	fputs("Title=\"LBM Porous_Media_Flow\"\n", fp);

	fputs("VARIABLES=\"X\",\"Y\",\"Z\",\"T\",\"U\",\"V\",\"W\"\n", fp);
	fprintf(fp, "ZONE T=\"BOX\",I=%d,J=%d,K=%d,F=POINT\n", DX-2, DY-2,DZ-2);

	for (int i = 1;i <DX-1;i++)
	 for (int j = 1;j < DY-1;j++)
		for (int k = 1;k < DZ - 1;k++)
		{
			int pp = i * DY * DZ + j * DZ + k;
			if (i < 10 ||i> DY - 10)
				Type[pp] = 0LL;
			fprintf(fp, "%d,%d,%d,%d,%.12f,%.12f,%.12f\n", i, j ,k, Type[pp],Vx[pp], Vy[pp], Vz[pp]);
		}

	fclose(fp);
}

__global__ void SemiDIR_OPT_Odd(long long *NTA, double *FA, unsigned int NFluid, double ReTau)
{
	unsigned int n = blockIdx.x*blockDim.x + threadIdx.x;
	if (n >= NFluid)	return;

	double reg_F[DQ-1]; //The GPU register

	double mVx = 0, mVy = 0, mVz = 0;
	unsigned int f,p0,nt;

	long long types_ = NTA[n];

	double D0 = FA[(n / (STRIDE)*DQ) * STRIDE + n % (STRIDE)]; //f0 distribution function
	double mDen = D0;

	//Calculation of macroscopic quantities
    #pragma unroll
	for (f = 1; f < DQ; f++)
	{
		reg_F[f-1] = FA[(n / (STRIDE)*DQ + f)*STRIDE + n % (STRIDE)];
		mDen += reg_F[f - 1];
		mVx += reg_F[f - 1] * d_Ex[f];
		mVy += reg_F[f - 1] * d_Ey[f];
		mVz += reg_F[f - 1] * d_Ez[f];
	}
	mVx /= mDen;
	mVy /= mDen;
	mVz /= mDen;

	
	FA[(n / (STRIDE)*DQ) * STRIDE + n % (STRIDE)] = D0 - ReTau*(D0 - d_Feq(0, mDen, mVx, mVy, mVz)); // f0 collision

	//f1-f18  
    #pragma unroll
	for (f = 1; f < DQ; f++)
	{	
		nt = (types_ >> (2 * f)) & 3; //Extracts the type of the node expressed using binary bits
		if (nt == FLUID || nt == WALL)
		  {
			 FA[(n/ (STRIDE)*DQ + d_Re[f])*STRIDE +n % (STRIDE)] = reg_F[f-1] - ReTau*(reg_F[f-1] - d_Feq(f, mDen, mVx, mVy, mVz)); //f1-f18 collision  and  the half-way bounce boundary condition is employed on the solid walls
		  }
		else
		{
			p0 = (f % 2 == 0) ? f - 1 : f + 1; //Obtain the opposite direction of the f-direction
			FA[(n / (STRIDE)*DQ + p0)*STRIDE + n % (STRIDE)] = d_Feq(p0,(nt == INLET)*INLET_DEN + (nt == OUTLET)*OUTLET_DEN, mVx, mVy, mVz) + (1.0 - ReTau)*(reg_F[p0 - 1] - d_Feq(p0, mDen, mVx, mVy, mVz));
			//The nonequilibrium extrapolation boundary condition is applied at the outlet and inlet
		}
	}
}

__global__ void SemiDIR_OPT_Even(int *FIA, unsigned int *AIA, double *FA, unsigned int NFluid, unsigned int NWall, unsigned int NInlet, unsigned int NOutlet, double ReTau)
{
	unsigned int n = blockIdx.x*blockDim.x + threadIdx.x;
	if (n >= NFluid)	return;

	double reg_F[DQ - 1]; //The GPU register
	__shared__ unsigned int sh_nei_index[ThreadX*DQ - 1];//the GPU shared memory

	unsigned int f, p, pp, ppp,in,p0;
	double mVx = 0, mVy = 0, mVz = 0;

	p = AIA[n];                 // To get direct address index p in the full flow field
	//To get the coordinate information (i,j,k) of the node in the full flow field
	int i = p / DY / DZ;
	int j = (p / DZ) % DY;
	int k = p % DZ;

	double D0 = FA[(n / (STRIDE)*DQ)*STRIDE + n % (STRIDE)]; //f0
	double mDen = D0;

#pragma unroll
	for (f = 1; f < DQ; f++)  
	{
		//To  get the direct address index pp of the neighboring node
		pp = (i + d_Ex[f]) * DY * DZ + (j + d_Ey[f]) * DZ + k + d_Ez[f];
		//To  get the index number of the data for the neighboring node in the FA
		in = FIA[pp];
		sh_nei_index[threadIdx.x + (f - 1)* ThreadX] = in; // Save  to  the shared  memory

		p0 = (f % 2 == 0) ? f - 1 : f + 1; //Obtain the opposite direction of the f-direction
	
		ppp = (in < NFluid) ? (in / (STRIDE)*DQ + f)*STRIDE + in % (STRIDE) : (n / (STRIDE)*DQ + d_Re[f])*STRIDE + n % (STRIDE);

		reg_F[p0 - 1] = FA[ppp]; //Streaming

		// Calculating macroscopic quantities
		mDen += reg_F[p0 - 1];
		mVx += reg_F[p0 - 1] * d_Ex[p0];
		mVy += reg_F[p0 - 1] * d_Ey[p0];
		mVz += reg_F[p0 - 1] * d_Ez[p0];
	}
	mVx /= mDen;
	mVy /= mDen;
	mVz /= mDen;

	
	//f0  collision and streaming
	FA[(n / (STRIDE)*DQ)*STRIDE + n % (STRIDE)] = D0 - ReTau*(D0 - d_Feq(0, mDen, mVx, mVy, mVz));

#pragma unroll
	for (f = 1; f < DQ; f++)
	{
		in = sh_nei_index[threadIdx.x + (f - 1)* ThreadX];

		// Streaming
		if (in >= NWall && in < NOutlet)//INLET & OUTLET type
		{
			p0 = (f % 2 == 0) ? f - 1 : f + 1;
			ppp = (in >= NWall && in < NInlet);
			FA[(n / (STRIDE)*DQ + p0)*STRIDE + n % (STRIDE)] = d_Feq(p0, ppp*INLET_DEN + (1 - ppp)*OUTLET_DEN, mVx, mVy, mVz) + (1.0 - ReTau)*(reg_F[p0 - 1] - d_Feq(p0, mDen, mVx, mVy, mVz));
			//The nonequilibrium extrapolation boundary condition   
		}
		else
		{
			ppp = (in < NFluid) ? (in / (STRIDE)*DQ + f)*STRIDE + in % (STRIDE) : (n / (STRIDE)*DQ + d_Re[f])*STRIDE + n % (STRIDE);
			FA[ppp] = reg_F[f - 1] - ReTau*(reg_F[f - 1] - d_Feq(f, mDen, mVx, mVy, mVz)); 
			// Collision and Streaming,and at the same time, the halfway bounce boundary condition is applied
		}
	}
}

void CalculateMacVariable()
{
	int  nn = 0;
	for (int i = 0; i < DX; ++i) for (int j = 0; j < DY; ++j) for (int k = 0; k < DZ; ++k)
	{
		if (Type[P] == FLUID)
		{
			double mdf, mDen = 0, mVx = 0, mVy = 0, mVz = 0;
			for (unsigned int f = 0; f < DQ; f++)
			{
				mdf = FA[((nn) / (STRIDE)*DQ + f)*STRIDE + (nn) % (STRIDE)];
				mDen += mdf;
				mVx += mdf*Ex[f];
				mVy += mdf*Ey[f];
				mVz += mdf*Ez[f];
			}
			Den[P] = mDen;
			Vx[P] = mVx / mDen;
			Vy[P] = mVy / mDen;
			Vz[P] = mVz / mDen;
			nn++;
		}
	}
}

int main(void){

	Initialize();
	SetFlowField(); //Set flow field information
	ShowData(NowStep);


	dim3 threads(ThreadX);  //Number of threads per thread block
	dim3 blocks((NFluid + ThreadX - 1) / ThreadX);   //Number of thread blocks per grid

	cudaEventCreate(&StartTime);
	cudaEventRecord(StartTime, 0);

	for (NowStep = 1; NowStep <= AllStep; NowStep++)
	{
		if (NowStep % 2 != 0)  
			SemiDIR_OPT_Odd << <blocks, threads >> >(NTA,FA, NFluid,ReTau);  //Calling kernel function
		else
		{
			SemiDIR_OPT_Even << <blocks, threads >> > (FIA, AIA, FA, NFluid, NWall, NInlet, NOutlet, ReTau); ////Calling kernel function
		}
		//if (NowStep%ShowStep == 0) { cudaDeviceSynchronize();  ShowData(); }
	}
	
	cudaEventCreate(&EndTime);
	cudaEventRecord(EndTime, 0);
	cudaEventSynchronize(EndTime);

	cudaDeviceSynchronize();
	CalculateMacVariable();
	ShowData(NowStep-1);

	float Time;
	cudaEventElapsedTime(&Time, StartTime, EndTime);
	Time /= 1000;
	double MFUPS = (double(AllStep) / 1e6) * (FLUID_NUM / ((double)Time));

	printf("**************************************************************************************\n");

	printf("The computational performance: MFUPS = %.2f,     Runningtime = %.3f Seconds\n", MFUPS, Time);

	SaveData();
	FreeData();
	cudaThreadExit();
	return 0;
}

