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
#include <time.h>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <pthread.h>
#include <complex>
#include <sstream>

#ifdef __MPI__
   #include "mpi.h"
#endif
#ifdef __CUDA_SDK__
#include <cuComplex.h>
//#include <cutil.h>
//#include <cutil_inline.h>
#include "cublas.h"
#include "cuda.h"
#endif
//#ifdef __GSLBLAS__
#include <gsl_cblas.h>
#include <gsl_complex_math.h>
#include <gsl_blas.h>
//#endif
using namespace std;

// Thread block size
#define BLOCK_SIZE 2

// max number of individuals
const int  MAXNUMBER = 100;
//size of the group for group fitnes evaluation
const int GROUPSIZE = 15;	
// max number generations allowed
const int MAXGEN = 1000000;	
//const float CONSTRAINT = 0.5; // 50% of the wheel has fitness of 1
// max number of gates used
const int MAXNUMOFGATES = 50;	
// 2 to the n        er, here set n = 10, so 2^10*2^10. also max number of individual matirx size
const int MAXGATEDEM = 1048576;	
// max number of wires for input of an individual is MAXINDIWIRENUM
const int MAXINDVWIRENUM = 10;	
// max number of segments per individual is MAXSEGNUM
const int MAXSEGNUM = 200;	
// max number of chars in an individual string
const int MAXSTRING = 150;	
// max number of input for any single gate
const int MAXGATEINPUT = 10;	
// max number of measurement gates
const int MAXMEASUREGATE = 50;	
// we are not looking for a single gate solution (before minimization)
const int MINCIRCUITSIZE = 2;	
const int CHARMAX = 126;
const int CHARMIN = 33;
typedef struct d_list group;

/* global mutex for our program. assignment initializes it */
static pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;


/***********************
* the translation structure - between the GA and the Exahustive searcher
* or other modular extensions
************************/
struct nameT
{
	string normal;
	string nonorm;
};
// temporary structures used only in incialization
struct tempT
{
// number of gates in this catagory
	int numberGates;	
// for a particular number of input and output, the list of the gates can be used
	char nameGate[MAXNUMOFGATES];	
};

struct qGate
{
// number of input/output wires
	int numIO;		
// cost of the gate
	int Cost;		
// representation
	char representation, parentA, parentB;	
// matrix representing this gate
	cuComplex *gateMatrix1;	
//The String Name of this gate
	string my_string;	
	int valuedness;
	string rev_lib_string;
	string rev_lib_char;
};
struct stringqGate
{
	int wires[MAXGATEINPUT];
//The String Name of this gate
	string my_string;	
	char representation;
	int valuedness;
};
struct qState
{
	string initstate;
	string currentstate;
	int currentpos;
	int initpos;
	int valuedness;

};
/************************
*Holds the definition of the result function/behavior
************************/
struct qDataGate
{
// number of input/output wires
	int numIO;		
	int valuedness;
	float trace;
// matrix representing this gate
	  cuComplex *densityErrorMatrix;	
	string my_string;
};

struct result
{
	float avFitness [100000];
	float avError[100000];
	float avCost[100000];
	int counter;
	int step;
};
// structure representing an individual in the GA population
struct Individual
{
// number of wires of this individual
	int ioNumber;		
// number of segments at the creation, used only for intialization
	int segmentNumber;	
// circuit representation as string
	string my_string;	
// circuit representation as string
	string rev_lib_string;	
//each circuit is now generated with phase
	cuComplex phase;	
	cuComplex phases[MAXNUMOFGATES];
// fitness value
	float fitness;		
// Cost of the circuit
	float Cost;		
// Error of the circuit
	float Error;		
// for share fitness calcualtion
	int Groups;		
	int Rank;
	//to avoid races copy the gate arrays to every single individual
// all agtes are stored in here
	qGate *gateArray[MAXNUMOFGATES];	
	int valuedness;

};

struct Crc
{
// number of input/output wires
	int ioNumber;		
// counter of all representation so far found
	int Counter;		
// representation -- initial
	char representation;	
// matrix representing this circuit
	cuComplex *gateMatrix1;	

};

struct Solution
{
	int ioNumber;
	float error;
	float fitness;
	float cost;
	string my_string;
};

struct d_list
{
	int V;
	struct d_list *next;
};

// compare two gates
bool compareGate (qGate, qGate);	
bool compareRep (char*, char*);	
void copyRep (char*, char*);	
void incRep(char*);
// makes a copy oif a quantum gate
qGate copyGate (qGate *);	
qGate *cGate (qGate *);
qGate *ccGate(qGate*, qGate*);
void densityErrorMatrix (qGate *, qGate *);
cuComplex generatePhase ();
void vectorInitZero(int, cuComplex *);
void vectorScalarMult(int, cuComplex, cuComplex *, cuComplex *);
void generatePhases (Individual *);
// tensor product of gate A and gate B
qGate tensorProduct (qGate *, qGate *);	
// tensor product of gate A and gate B, returning A
qGate *tensorProduct2 (qGate *, qGate *);	
//generates an Identity Matrix
void destroyGate(qGate *); 
//generates an Identity Matrix
void initGate(qGate *, int, int, int); 
//generates an Identity Matrix
void initMatrix(qGate *, int); 
//generates empty Matrix
void zeroMatrix(qGate *, int); 
//generates matrix filled with don't cares
void dcMatrix(qGate *, int);
//return the number of proper gates in  the String
int getGatesfromString (string);	
//subtracts two matrices returns the result in the first one
qGate* matrixSubstr(qGate*, qGate*); 
//matrix multiplication using BLAS -> CBLAS
void cblasMatrixProduct(qGate*, qGate*, qGate*);
//matrix multiplication using CUDA BLAS -> CUBLAS
void cublasMatrixProduct (qGate *, qGate *, qGate *, int); 
// matrix product 'matrixProduct3 (qGate *R, qGate *A, qGate *B)' of gate A and gate B into R
void matrixProduct3 (qGate *, qGate *, qGate *);	
// tensor product 'tensorProduct3 (qGate *R, qGate *A, qGate *B)' of gate A and gate B into R
void tensorProduct3 (qGate *, qGate *, qGate *);	
double cuFloatComplex_arg (cuComplex);
double cuFloatComplex_abs (cuComplex);
cuComplex cuFloatComplex_inverse (cuComplex);
double cuFloatComplex_logabs (cuComplex);
cuComplex cuFloatComplex_pow (cuComplex, cuComplex);
cuComplex cuFloatComplex_sqrt (cuComplex);
cuComplex cuFloatComplex_div (cuComplex, cuComplex);
cuComplex cuFloatComplex_sub (cuComplex, cuComplex);

#ifdef __QMDD__
void decodeGateToRevLib(qGate *);
void buildQMDDFromqString(string);
void doQMDDFitness(Individual*, Individual*, bool);
#endif
short sign(float);
void dec2bin(long, char*, int);
#ifdef __CUDA__
//multiplies a Matrix by a Vector and returns a Vector, Matrix is of size w x w 
void cpuMulv(cuComplex*, cuComplex*, cuComplex*, int); 
void cleanCUDA(cuComplex*, cuComplex*,cuComplex*,cuComplex*,cuComplex*,cuComplex*);
__global__ void matrixMul(cuComplex* A, cuComplex* B, int wA, int wB, cuComplex* C);
__global__ void Muldv(cuComplex*, cuComplex*, int, cuComplex*);
__global__ void vvMul(cuComplex*, cuComplex*, int, cuComplex*, int);
__global__ void Divdv(cuComplex*, cuComplex*, int, cuComplex*);
__global__ void initState(int, cuComplex, cuComplex*, int);
__global__ void nullState(int, cuComplex*);
//requires two host matrices and a host vector, two device matrices and a device vector
void Mulv(cuComplex*, cuComplex*, int,  cuComplex*, cuComplex*, cuComplex*, cuComplex*);
//requires three host matrices and three device matrices pointers 
void Mulm(cuComplex*, cuComplex*, int, cuComplex*, cuComplex*, cuComplex*, cuComplex*);
//matrix multiplication using device matrices pointers
void Mulmnoset(int, cuComplex*, cuComplex*, cuComplex*); 
//matrix multiplication using device matrices pointers
void Mulvnoset(int, cuComplex*, cuComplex*, cuComplex*); 
//matrix multiplication using device matrices pointers
void Mulvvnoset(int, cuComplex*, cuComplex*, cuComplex*); 
void Divvnoset(int, cuComplex*, cuComplex*, cuComplex*);
#endif

class QE
{
      public:
	QE ()
	{
		SolutionCount = 5;
	}
	 ~QE ()
	{
	};

/////////////////////////////////////////////////////////////////////////////
//pointer to the synthetized circuit
	qGate *INDV;
//pointer to teh synthesized circuit properties
	qDataGate *dataINDV;
//field of pointers to all gates read from file
	qGate *gateArray[MAXNUMOFGATES];
//the best gate -- the result we are looking for
	qGate bestOne;
//field with all circuits we are searching for
	Crc *resultCircuits[MAXNUMBER];
//field with names of input gates
	string nameArray[MAXNUMOFGATES];
//field of gates to be used only for specified gates
//and their permutations
	qGate *restricArray[MAXNUMOFGATES];
//field of gates for restricted synthesis
//only pointers to array indexes
	int *restriGate[MAXNUMOFGATES];
//solutions in the form of strings
	string *solutionArray[MAXNUMOFGATES];
//solution
	Solution bestlocal;
//a character
	char end;
//number fo gates from file
	int numofgates;
//terinating condition
	int generationCondition;
//number of desired segment
	int Segments;
//some global help variable
	int segPointer;
//gates organized b number of wires
	int gateCounter[MAXSEGNUM];
//number of resulting gates
	int population;
//the switch enabling/disabling the use of phase -

//memory for the Segment manipulator
	qState *segmentstate;

//the maximum parametrized allowed of solutions
	int SolutionCount;
//sometimes we need a general gradient value
	float error;
//streams
	ifstream in_stream;
	ofstream out_stream;

	bool compareIndividual (qGate *);
	group *insertB (group * A, group * m_List);
	void insertA (group * A, group * m_List);
	group *removeFirst (group * m_List);
	group *moveL (group * m_List);
	int sizeL (group * m_List);
	int getGate (char b);
	bool setTerminatingCondition (int type);
	cuComplex assignVal (float x);
	void addGateSet (qGate * array[]);
	void setBestIndividual ();
	void closeStream ();
	void synthSegment (bool begin);
	bool synthPermutations (bool begin);
	bool synthAlterations (bool begin);
	bool synthJumps (bool begin);
	void synthRestric (bool begin);
	group *genPerms (group * head, bool perm);
	void setSegments (int begin);
	void restricSegment ();
	qGate *matrixProduct (qGate * A, qGate * B);
	void matrixProduct2 (qGate * R, qGate * A, qGate * B);
	qGate *densityErrorMatrix (qGate * A, qGate * B);
	bool compareGate (qGate * A);
	void Matrix (bool display);
	void subInit ();
	string **setgateNormArray (qGate **, qGate, int, float);

};

// holds GA functions
class GA
{
      public:


//The counter of individual in the Threaded mode
	static int populationCounter;	
//the method used to Thread the GA
	static void *startSub (void *arg);	
//measurement switch 0 - turns of measurement, >0 number of measured qubits so as the order of measurement
	static int measurement;	

	void run ();

	  GA ()
	{
	}
	 ~GA ()
	{
#ifdef __CUDA__
		cleanCUDA(d_M1, d_M2, d_M3, d_MV, d_VI, d_VO);
#endif
	};

	 /////////////
	 //GA settings
	 /////////////
	void setpopulationNumber(int);
	int getpopulationNumber();
	 void setsegments(int);
	 int getsegments();
	 void setmincircuitsize(int);
	 int getmincircuitsize();
	 void setalterations(int);
	 int getalterations();
	 void setproba_Mutation(float);
	 float getproba_Mutation();
	 void setproba_Crossover(float);
	 float getproba_Crossover();
	 void setalpha(float);
	 float getalpha();
	 void setbeta(float);
	 float getbeta();
	 void setalpha1(float);
	 float getalpha1();
	 void setbeta1(float);
	 float getbeta1();
	 void setdivider(int);
	 int getdivider();
	 void setphase(int);
	 int getphase();
	 void setdisplaylevel(int);
	 int getdisplaylevel();
	 void setGa(int);
	 int getGa();
	 void setmutation(int);
	 int getmutation();
	 void setcrossover(int);
	 int getcrossover();
	 void setreplicator(int);
	 int getreplicator();
	 void setfitness(int);
	 int getfitness();
	 void setgrouped(int);
	 int getgrouped();
	 void setpareto(int);
	 int getpareto();
	 void setthreshold(int);
	 int getthreshold();
	 void setresultnum(int);
	 int getresultnum();
	 void setmeasurement(int);
	 int getmeasurement();

// initialize the population
	void initializePop (int, string, string);	
// With a 1 point crossover probability recombine 2 parents into 2 new child
	void apply1Crossover (int, int);	
// With a 2 point crossover probability recombine 2 parents into 2 new child
	void apply2Crossover (int, int);	
// A CO operation that preserves the structure a bit more - With a 1 point crossover probability recombine 2 parents into 2 new child
	void apply1CrossoverP (int, int);	
// A CO operation that preserves the structure a bit more - With a 2 point crossover probability recombine 2 parents into 2 new child
	void apply2CrossoverP (int, int);	
	void applyMutation(int, bool);
	void applyCrossover(int, int);
// Replication of teh selected individuals either SUS or RW
	void applyReplication ();	
	void *doFitness();
	void *doMatrixFitness(Individual*,bool);
	void *doMatrixFitnessPoly(Individual*,bool);
	void *doMeasureFitness(Individual*,bool);
	void *doMultiMeasureFitness(Individual*,bool);


	void applyRMutation (int, bool);
//A mutation operator that does preserbe the current circuit structure completely
	void applyRMutationP(int, bool);		
	void applyIMutation (int, bool);
//A mutation operator that does preserbe the current circuit structure completely
	void applyIMutationP(int, bool);		
	void evaluatePopulation ();
// somemethod setting final criteria
	void setTerminatingCondition (int);	
// some method returning if the final conditions were reached
	bool terminatingCondition ();	
// calculation of the individual matrix representation for each individual
	void individualMatrix (Individual * i);	
// clear everything in one individual
	void clearAll (int);	
// method minimizing circuit ... not used always
	int minimizeCirc (Individual *);	
	void repairForceCircuit (int);
// method close streams to files
	void closeStream ();	
// writes down the result of the run
	void output (int);	
// writes down the best individual from one generation and/or displays the statistics of the overall run
	void outputBest (bool);	
// copy one Individual and output it
	Individual setIndv (Individual);	
//copy two indidual by their addresses
	void setIndv (Individual *, Individual *);	
	void injectNormalCircuits (string **);
	void injectNormalSegments (string **);
//generates all gates from the at the NumIO of the whole circuit
	qGate **getPermutativeCircuits ();	
//translates all parallel blocks to individual NumIO sized circuits
	qGate **getPermutativeSegments ();	
	void calcAverage();

	// size of the population < MAXNUMOFGATES
	int populationNumber;
//the set of probabilistic expectations generated as the result of measurement of ht eindividual states
	cuComplex expectationsAllState[MAXMEASUREGATE*2][MAXGATEINPUT*4];   
//the set of probabilistic expectations generated as the result of measurement of ht eindividual states
	// both here bellow fields will
	// be in the future replaced by
	// array of pointers
// population (array of object Individual)
	Individual *population[MAXNUMBER];	
// population of individuals used for replication ... non optimal to be improved
	Individual *new_population[MAXNUMBER];	
// best Individual from the run
	Individual *bestIndv;	
	qDataGate bestData;
	//method of fitness application
	//0 - normal, 1 - groupped shared
	bool grouped;
	//indicator of pareto rplication
	//0 - no, 1 - yes
	bool pareto;
// looks for the best individual with fitness 1
	bool getFitness ();	
	nameT *normNames[MAXNUMOFGATES];
//the set of meaurement operators defined by parameters from file
	qGate *measurements[MAXMEASUREGATE];	
//the set of meaurement operators defined by parameters from file
	 qGate *measurementsAllState[MAXMEASUREGATE];    
//two fields per input by Max size of the input-output mapping
	cuComplex (*measureexpected)[MAXGATEINPUT*2]; 
	int measurementQBits[MAXGATEINPUT];
	int measured_desired_output_records;
	int *measured_desired_output_records_idx;
	qGate *permutativeArray[MAXNUMOFGATES];
// all agtes are stored in here
//	qGate **gateArray;
	qGate *gateArray[MAXNUMOFGATES];

// retunrs teh char-representation of the gate at teh int index from gateArray
	int getGate (char);	
// number of gates to be use < MAXNUMOFGATES
	int numofgates;		
	int numofnormalgates;
// gate/circuit we are looking for
	qGate *finalGate;	
// gate/circuit we are looking for
	Individual *finalIndividual;	
//how many generations does the GA have to do before call a module
	int alterations;	
//display switch
	bool display;
	int progress;
	result finalResult;
	//the level of output details
	//0 - th lowest
	//1 - middle
	//3 - full;
	int displaylevel;

      private:

	ostringstream convert_stream;
// input file stream
	ifstream in_stream;	
// output file stream
	ofstream out_stream;	

	//device matrices for Matrix Matrix multiplication
#ifdef __CUDA__
	cuComplex* d_M1;
	cuComplex* d_M2;
	cuComplex* d_M3;
	//device Matrix and Vector for Matrix Vector multiplication
	cuComplex* d_MV;
	cuComplex* d_VI;
	cuComplex* d_VO;

	cuComplex* d_ME_00;
	cuComplex* d_ME_01;
	cuComplex* d_ME_10;
	cuComplex* d_ME_11;

	cuComplex* d_VV_M;
	cuComplex* d_Value;
#endif
	qGate *measures_o[10], *measures_i[10];
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////

	int avrghst[1000];
	int besthst[1000];
	// probabilities of mutation and crossover
	float proba_Mutation;
	float proba_Crossover;
	// scaling for the complex fitness function
	// alpha + beta = 1
	float alpha;
	float beta;
	//scaling fo the pareto replication
	//default alpha1 = 1, beta1 = 1
	float alpha1;
	float beta1;
	//number of segments estimated for solution
	int segments;
	//minimum number of segments - size pressure
	int mincircuitsize;
	// estimated cost divider for the solution circuit
	int divider;
	// type of GA Baldwinian or normal
	//0 - nomral, 1 - baldwinian
	int Ga;
	// use elitism
	//0 - yes, 1 - no
	int elitism;
	//setting different types of mutation
	//0 - normal, 1 - bitwise
	int mutation;
	//setting different types of crossover
	//0 - 1point, 1 - 2point
	int crossover;
	//different types of selection
	//0 - RW, 1- SUS
	int replicator;
	// type of fitness function
	//0 - simple0, 1 - simple1, 2 - complex0, 3 - complex1
	int fitness;
	// specific parameter to adjust correctness of entanglement formula 
	double error_adjust;
	//Threshold for replication
	//0 - no threshold otherwise must be smaller than 1
	float threshold;
	//more general it is used to allow synthesis from pure gates or rotations
	int phase;
	//number of user specified generations
	int generations;

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////

// nullGate - wire
	qGate nullGate;		
// sorted by input wires in asending order
	tempT stringArray[MAXINDVWIRENUM];	
// the number of wire for input and output of the result gate
	int resultnum;		
// termination criterian
	int generationCondition;	
// some boolean condition
	bool condition;		
	float best;
	float (*history)[4];

//enable tje sequence detection search
	int seq_detect_enabled;	
//enable the sequence generatiion search
	int seq_gener_enabled;	
//the array holding the resulting desired sequence
	cuComplex (*sequence_desired)[2];	
	void applySingleObjReplication ();
// Replication for pareto optimal GA
	void applyParetoReplication ();	
//allocates the array of flots for statistics
	void initiateStorage(); 

//matrix manipulators --simple
// matrix product of gate A and gate B
	void matrixProduct2 (qGate *, qGate *);	
// reduce teh total length of the circuit
	void reduceSegNumber (int);	
// reads teh gates from a file and set them in an array
	void initializeGates ();	
// creates string representation for each individual
	void makeIndvString (Individual*);	
// set an array used only for intialization
	void setStringArray ();	
// method repairing circuit after mutation automatically called from mutation
	void repairCircuit (int);	
#ifdef __CUDA__
	qGate *computeCUDAMatrix (Individual *);
	qGate *computeCUDANATIVEMatrix (Individual *);
#else
	qGate *computeMatrix (Individual *);
#endif
	void* doMeasureFASeqFitness(Individual*, bool);
	void*  doEntanglState(qGate*, cuComplex*, bool);
	void* computeMeasurement(Individual *);
	//for minimizer of quantum circuits
// compares one quantum gate to all available
	int findGate (qGate *);	
// remove redundantsegments in the circuit
	void removeRedund (Individual *);	
// creates a new segment of gates in string representation
	basic_string < char >synthSegment (int, int);	
// merge two equivalent or equal segments
	void mergeSegs (Individual *, int, int);	
	void *generateMeasureOps ();
	void *generateMultiMeasureOps ();
	void generateMeasureExpect ();
	void generateExtendMatrix(char);
	int getqGate(string);
	void decodeIndvStrToRevLibStr(Individual *);
	void StrToIntArray(string, int*);
	void IntArrayToStr(int* , string);
	char loadGateArray(int, char);

};





