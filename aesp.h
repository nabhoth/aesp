/***************************************************************************
*   Copyright (C) 2006 by martin lukac   				  *
*   lukacm@ece.pdx.edu   						  *
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
*   This program is distributed in the hope that it will be useful,       *
*   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
*   GNU General Public License for more details.                          *
*                                                                         *
*   You should have received a copy of the GNU General Public License     *
*   along with this program; if not, write to the                         *
*   Free Software Foundation, Inc.,                                       *
*   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
***************************************************************************/
//Uncomment to allow multi-threaded computation for smaller circuits 8 < qubits
//#define __SSIZE__
//Uncoment to allow QMDD representation for circuits up to 7 qubits
//#define __QMDD__
//Uncoment to unleash the standard things
//#define __STDR__
//#define __TIMING__

#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <time.h>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <pthread.h>
#include <complex>
#include <sstream>
#include "./lock.h"

//#include <cutil.h>
//#include <cutil_inline.h>
#include "cublas.h"
#include "cuda.h"
#include "curand_kernel.h"

// Thread block size
#define BLOCK_SIZE 32 


//#include <gsl_cblas.h>
//#include <gsl_complex_math.h>
//#include <gsl_permutation.h>
//#include <gsl_blas.h>
//#include <gsl_linalg.h>

using namespace std;


//CUDA params 
//BLOCKS and MAXSIZE should be related. 
//Simplest case is PBLOCKS*PBLOCKS=MAXSIZE.
const int PBLOCKS = 32;
// BLOCKS for evolving Cells
// Simplest case is CBLOCKS*CBLOCKS =CELLX*CELLY
const int CBLOCKS = 32;

//Simulation Parameters
//Emisivity - default is particles/square/second
//this amounts to emit particles once every 300000m (speed of litght unit move)
const double EMS = 50;
// max number of of particles per cell
const int MAXP = 100;
// max length of time interval 1s10e-MAXT before resampling direction
const int MAXT = 10000;	
//maximal number of total particles in simulator
//const int MAXPSIZE = 1048576;	
const int MAXPSIZE = 1024;	
//speed of particle
const double V = 300000;
// dt
const double ST = 0.0001;
//distance of single step
const int S = 30;
//X and Y size of the whole 2D plane of simulation
const float DIMX = 1024;
const float DIMY = 1024;
//number of cells in the plane, each smae geometry and dimensions
const float CELLX = 32;
const float CELLY = 32;
#define PI 3.14159265

//Boundary permeability
#define PERMEABILITY 1

//Should Particles annhilate
#define ANNHILATION 0

//Should there be refraction while entering new cells
#define INTERACTION 0

//Source numbers
#define RADIATE 1

//coordinates of the radiation source
int radiate_x = 16;
int radiate_y = 16;

//int base_functions;
// input file stream
static ifstream in_stream;	
// output file stream
static ofstream out_stream;	

typedef struct particle
{
//  coordinates within the whole space
//  whole space is given by CELLX*CELLS*CELLY*CELLS
	int x;		
	int y;
// id of the cell in the array
	int id;
// final coordinates of this particles in the whole field
	int xf;
	int yf;
// direction vector of the particle - angle from origin
	double dir;		
// speed of the particle
	int speed;
// energy of the particle
	int energy;	
	double s;
	int alive;
} particle;

//the cell structure 
typedef struct cell {
	int x0;
	int y0;
	int id;
	int pcount;
	double temperature;
	double ann;
	double cre;
	double emissivity;
	int busy;
	int radiant;
} cell;


//host and device array of cells 
cell *h_cells;
cell *d_cells;

//host and device array of particles
particle *d_elements;
particle *h_elements;

//host and device counter of available particle
int h_index;
int *d_index;

//host and device array of indexes with available particles
int *h_states;
int *d_states;

//system energy counters
int h_total;
int h_active;
int h_residual;
int *d_total;
int *d_active;
int *d_residual;

//device state of random generator
curandState *globalstate;

//CUDA mutex
Lock lock;

__global__ void propagate(Lock, particle*, cell*, curandState*, int*, int*, int*);
__global__ void setup_kernel(particle*, unsigned long);
__global__ void evolve_p_state(particle*, cell*, curandState*, int*, int*, int*);
__global__ void evolve_c_state(particle*, cell*, curandState*, int*, int*, int*);

