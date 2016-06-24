/***************************************************************************
 *   Copyright (C) 2006 by martin lukac   *
 *   lukacm@ece.pdx.edu   *
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
#ifdef __MPI__
#include "EpiG_mpi.h"
#else
#include "EpiG.h"
#endif

/*global static field for thread counting */
int GA::populationCounter;

/* global static switch for measurement/nonmeaurement */
int GA::measurement;
/****************************************
 * generates the population order in fitness value:
 * (inc = 0) decreasing order
 * (inc = 1) increasing order
 ****************************************/
void GA::evaluatePopulation() {
	int c = 0;
	float totfit = 0.0;
	float toterr = 0.0;
	float totcost = 0.0;
	for (int t = 0; t < populationNumber; t++) {
		totfit += population[t]->fitness;
		toterr += population[t]->Error;
		totcost += population[t]->Cost;
		if (population[t]->fitness > population[c]->fitness) {
			c = t;
		}
	}
	totfit /= populationNumber;
	totcost /= populationNumber;
	toterr /= populationNumber;
	history[generationCondition][0] = float(totfit);
	history[generationCondition][1] = float(population[c]->fitness);
	history[generationCondition][2] = float(toterr);
	history[generationCondition][3] = float(totcost);
	progress = -1;
	cout << "Best current generation fitness: " << population[c]->fitness
	<< endl;
	if (bestIndv->fitness >= 1) {
		progress = 1;
	} else if (population[c]->fitness > bestIndv->fitness) {
		progress = 1;
		cout << "Best current fitness: " << bestIndv->fitness << endl;
		cout << "New best fitness: " << c << " :: " << population[c]->fitness<< endl;
		setIndv(bestIndv, population[c]);
		outputBest(false);
		bestIndv->fitness = 0 + population[c]->fitness;
		cout << "Best new fitness: " << bestIndv->fitness << endl;
		cout << "Best new cost: " << bestIndv->Cost << endl;
	}
	cout << "Average current fitness: " << totfit << endl;
}

/****************************************
 * static - the function called to invoke the program as a thread
 ****************************************/
void* GA::startSub(void *arg) {
	GA* obj = (GA*) arg;
	obj->doFitness();
	return 0;
}


/****************************************
 * computes the fitness
 * ****************************************/
void* GA::doFitness() {
	int rc = pthread_mutex_lock(&mtx);
	int ind = GA::populationCounter;
	GA::populationCounter = (GA::populationCounter + 1) % populationNumber;
	rc = pthread_mutex_unlock(&mtx);
	
#ifdef __QMDD__

	decodeIndvStrToRevLibStr(population[ind]);
	doQMDDFitness(population[ind], finalIndividual, false);
#else
	if (measurement > 0) {
		if (seq_detect_enabled > 0)
			doMeasureFASeqFitness(population[ind], false);
		else 
			doMeasureFitness(population[ind], false);
	} else if (measurement == 0){
		doMatrixFitness(population[ind], false);
	} else if (measurement == -1){
		doMatrixFitnessPoly(population[ind], false);
	}
#endif
}
/****************************************
 * generate the fitness value of the Individual and displays it if desired
 ****************************************/
void* GA::doMatrixFitness(Individual *ind, bool display) {
	//two dummy gates and one pointer for operations
	Individual *indi = ind;
	qGate* myfinalG;
	int rowcount, rowcounter,maxcount;
	int errorcounter = 0;
	float val = indi->valuedness;
	float numIO, error, equal;

	indi->Error = 0;
	indi->fitness = 0;

	//threaded code
	int rc = pthread_mutex_lock(&mtx);
	if (rc) {
		cout << "Matrix locked: " << ind << endl;
	}

	long time = clock();
#ifdef __CUDA__
	myfinalG = computeCUDAMatrix(indi);
#else
	myfinalG = computeMatrix(indi);
#endif
	rc = pthread_mutex_unlock(&mtx);

	//set some more parameters
	myfinalG->numIO = indi->ioNumber;
	numIO = myfinalG->numIO;
	if (phase == 1) {
		maxcount = (int(pow(pow(val, numIO),2)));
		if (cuCrealf(indi->phase) != 0 && cuCimagf(indi->phase) != 0) {
			for (int k = 0; k < maxcount; k++)
					myfinalG->gateMatrix1[k] = cuCmulf(myfinalG->gateMatrix1[k], indi->phase);

		}
	}

	//direct correspondence between number of wires in the goal and current individual
	error = 0;
	errorcounter = 0;
	if (myfinalG->numIO != finalGate->numIO) {
		indi->fitness = 0;
		error = 1;
	} else {
		equal = 0;
		rowcount = (int(pow(val, numIO)));
		for (int a = 0; a < rowcount; a++)
			equal += cuCabsf(cuCmulf(myfinalG->gateMatrix1[a*rowcount+a], cuConjf(myfinalG->gateMatrix1[a*rowcount+a])));
		if (equal == pow(val, numIO)) {
			error = 1;
		} else {
			maxcount = (int(pow(pow(val, numIO),2)));
			for (int a = 0; a < maxcount; a++) {
					//simple comparison -- allowing don't cares as value "10"
					if (cuCrealf(finalGate->gateMatrix1[a]) != 0 && cuCimagf(finalGate->gateMatrix1[a]) != -1) {
						errorcounter++;
						//error += abs(abs(finalGate->gateMatrix1[a][b]) - abs(myfinalG->gateMatrix1[a][b]));
						error += cuCabsf(cuCsubf(cuCmulf(finalGate->gateMatrix1[a], cuConjf(finalGate->gateMatrix1[a])),cuCmulf(myfinalG->gateMatrix1[a], cuConjf(myfinalG->gateMatrix1[a]))));
				}
			}
		}
	}

	//normalizing error
	if (errorcounter != 0)
		indi->Error = error / errorcounter;
	else
		indi->Error = error;
	//generate fitness value
	if (indi->Error != 0) {
		switch (replicator) {
		case 0:
			//simple fitness1
			indi->fitness = (1 - indi->Error);
			break;
		case 1:
			//simple fitness1
			indi->fitness = (1 / (indi->Error + 1));
			break;
		case 2:
			//scaled complex fitness1
			indi->fitness = (alpha * (1 - indi->Error) + beta * (exp(-pow(abs(
					divider - indi->Cost), 2))));
			break;
		case 3:
			//scaled complex fitness2
			indi->fitness = (alpha * (1 / (indi->Error + 1)) + beta * (exp(
					-pow(abs(divider - indi->Cost), 2))));
			break;
		case 4:
			indi->fitness = (1 / ((1 + exp(-indi->Error) / 2)));
			break;
		case 5:
			indi->fitness = (exp(-indi->Error));
			break;

		}
	} else {
		indi->Error = 0;
		indi->fitness = 1;//we found a good individual;
	}

	if (display) {
		out_stream << "Matrix: " << endl;
		maxcount = (int(pow(pow(val, myfinalG->numIO),2)));
		rowcount = (int(pow(val, myfinalG->numIO)));
		rowcounter = 0;
		for (int a = 0; a < rowcount; a++) {
		for (int b = 0; b < rowcount; b++) {
				out_stream << "(" << cuCrealf(myfinalG->gateMatrix1[a*rowcount+b])<< "," << cuCimagf(myfinalG->gateMatrix1[a*rowcount+b]) << ")"<< "";

			}
		out_stream << endl;
		}
		out_stream << " -------- " << endl;
	}
	time = clock() - time;
//	cout<<"CUBLAS fitness done: "<<(float)(time/CLOCKS_PER_SEC)<<", for "<<indi->ioNumber<<" qubits "<<endl;
	//destroyGate(myfinalG);
	delete (myfinalG);
}


/****************************************
 * generate the fitness value of the Individual and displays it if desired
 ****************************************/
void* GA::doMatrixFitnessPoly(Individual *ind, bool display) {
	//two dummy gates and one pointer for operations
	Individual *indi = ind;
	qGate* myfinalG;
	int rowcount, rowcounter,maxcount;
	int errorcounter = 0;
	float val = indi->valuedness;
	float numIO, error, equal;
	cuComplex entangl[7];

	indi->Error = 0;
	indi->fitness = 0;

	//threaded code
	int rc = pthread_mutex_lock(&mtx);
	if (rc) {
		cout << "Matrix locked: " << ind << endl;
	}

	long time = clock();
#ifdef __CUDA__
	myfinalG = computeCUDAMatrix(indi);
#else
	myfinalG = computeMatrix(indi);
#endif
	rc = pthread_mutex_unlock(&mtx);

/* Calculate d1, d2, d3 */
	
	rowcount = (int(pow(val, numIO)));

		doEntanglState(myfinalG, entangl, false);
		error = 4*cuCabsf(cuCaddf(entangl[0],cuCsubf(cuCmulf(make_cuComplex(4,0),entangl[2]),cuCmulf(make_cuComplex(2,0),entangl[1]))));
//	error = 4*cuCabsf(cuCaddf(entangl[0],cuCsubf(cuCmulf(make_cuComplex(4,0),entangl[2]),cuCmulf(make_cuComplex(2,0),entangl[1]))));
	//set some more parameters
		error -= error_adjust;
		if (error < 0 ) error = -error;
		indi->Error = error;

	if (indi->Error != 0) {
		switch (replicator) {
		case 0:
			//simple fitness1
			indi->fitness = (1 - indi->Error);
			break;
		case 1:
			//simple fitness1
			indi->fitness = (1 / (indi->Error + 1));
			break;
		case 2:
			//scaled complex fitness1
			indi->fitness = (alpha * (1 - indi->Error) + beta * (exp(-pow(abs(
					divider - indi->Cost), 2))));
			break;
		case 3:
			//scaled complex fitness2
			indi->fitness = (alpha * (1 / (indi->Error + 1)) + beta * (exp(
					-pow(abs(divider - indi->Cost), 2))));
			break;
		case 4:
			indi->fitness = (1 / ((1 + exp(-indi->Error) / 2)));
			break;
		case 5:
			indi->fitness = (exp(-indi->Error));
			break;

		}
	} else {
		indi->Error = 0;
		indi->fitness = 1;//we found a good individual;
	}

	if (display) {
		out_stream << "Matrix: " << endl;
		maxcount = (int(pow(pow(val, myfinalG->numIO),2)));
		rowcount = (int(pow(val, myfinalG->numIO)));
		rowcounter = 0;
		for (int a = 0; a < rowcount; a++) {
		for (int b = 0; b < rowcount; b++) {
				out_stream << "(" << cuCrealf(myfinalG->gateMatrix1[a*rowcount+b])<< "," << cuCimagf(myfinalG->gateMatrix1[a*rowcount+b]) << ")"<< "";

			}
		out_stream << endl;
		}
		out_stream << " -------- " << endl;
	}
	time = clock() - time;
//	cout<<"CUBLAS fitness done: "<<(float)(time/CLOCKS_PER_SEC)<<", for "<<indi->ioNumber<<" qubits "<<endl;
	//destroyGate(myfinalG);
	delete (myfinalG);
}

/****************************************
 * the String array represents the gates by the number of wires
 * and present a list of gates for each amount of wires
 ****************************************/
void GA::setStringArray() {
	qGate *Q;
	int count[MAXNUMOFGATES];

	for (int j = 0; j < MAXGATEINPUT; j++) {
		count[j] = 0;
		for (int i = 0; i < numofgates; i++) {
			Q = gateArray[i];
			if (Q->numIO == (j + 1)) {
				stringArray[j].nameGate[count[j]] = Q->representation;
				count[j]++;
			}
			stringArray[j].numberGates = count[j];
		}
	}
}
/****************************************
 * this generates the String repressenting every individual.
 *it is called upon the structure of the initial population
 ****************************************/
void GA::makeIndvString(Individual *I) {
	int tempNumber, tempNumber2, tempGate, counter, temporyhold;
	char gate;
	//Individual *I = population[i];
	counter = 0;
	I->my_string = string("");
	for (int j = 0; j < I->segmentNumber; j++) // for each segment of this individual
	{
		if (I->ioNumber > MAXGATEINPUT) // if the input has more than MAXGATEINPUT (10)
			tempGate = MAXGATEINPUT; // max gate to use
		else
			tempGate = I->ioNumber;
		tempNumber = I->ioNumber; // the number of inputs
		do {
			tempNumber2 = rand() % tempGate; // which gate to use
		} while (stringArray[tempNumber2].numberGates == 0);

		if (I->ioNumber > 1) // if not only one input
		{
			I->my_string += 'p';
			counter++;
		}

		while ((tempNumber - (tempNumber2 + 1)) > 0) // while more gate could be parallel
		{
			temporyhold = rand() % stringArray[tempNumber2].numberGates;
			gate = stringArray[tempNumber2].nameGate[temporyhold];

			I->my_string += gate;
			counter++;

			tempNumber = tempNumber - (tempNumber2 + 1);
			if (tempNumber > tempGate) // use the smaller one of the two
			{
				temporyhold = rand() % stringArray[tempNumber2].numberGates;
				gate = stringArray[tempNumber2].nameGate[temporyhold];
				do {
					tempNumber2 = rand() % (tempGate);
				} while (stringArray[tempNumber2].numberGates == 0);
			} else {
				do {
					tempNumber2 = rand() % (tempNumber);
				} while (stringArray[tempNumber2].numberGates == 0);
			}
		}
		temporyhold = rand() % stringArray[tempNumber2].numberGates;
		gate = stringArray[tempNumber2].nameGate[temporyhold]; // when only can choose one input gate

		I->my_string += gate;
		counter++;

		if (I->ioNumber > 1) {
			I->my_string += 'p';
			counter++;
		}
	}
}
/****************************************
 * this is the main ENd output function
 * when the parameter individual = -1;the whole Input/Output Settings,
 * GA settings are dumped into file
 * when the parameter individual != -1; The given individual is printed out
 ****************************************/
void GA::output(int individual) {
	qGate *qG;
	int columns;

	if (individual != -1) {

		out_stream << "---------------------------" << endl;
		out_stream << " End of run at " << generationCondition << endl;
		out_stream << "---------------------------" << endl;
		out_stream << "----- Individual #" << individual << "-----" << endl;

		if (measurement > 0) {
			//		//this->display = true;
			//		makeMeasureFitness(population[individual], true);
			//	} else {
			//	  makeFitness(population[individual], true);
			//	}
		}

	} else {
		//		if (!(measurement > 0)) {
		out_stream << "There are " << numofgates
		<< " gates from the input file and synthesized.\n";
		float val = (float) gateArray[0]->valuedness;
		out_stream << "The radix is: " << val << "\n";
		for (int i = 0; i < numofgates; i++) {
			qG = gateArray[i];
			out_stream << "The gate number " << i << " is: " << endl;
			out_stream << "And the gate representation is: "
			<< gateArray[i]->representation << endl;
			out_stream << "Number of inputs is: " << qG->numIO << endl;
			out_stream << "Parent A is: " << qG->parentA << endl;
			out_stream << "Parent B is: " << qG->parentB << endl;

			float numIO = (float) gateArray[0]->numIO;
			columns = (int(pow(val, numIO)));
			for (int j = 0; j < columns; j++) {
				for (int k = 0; k < columns; k++) {
					//out_stream << qG->gateMatrix1[j][k] << " ";
					out_stream << "  (" << cuCrealf(qG->gateMatrix1[j+k*columns])<< "," << cuCimagf(qG->gateMatrix1[j+k*columns]) << ")";

				}
				out_stream << endl;
			}
		}
		out_stream << "The Target gate(s): \n";
		out_stream << "Number of inputs is: " << finalGate->numIO << endl;
		columns = (int(pow(val, resultnum)));
		for (int m = 0; m < columns; m++) {
			for (int n = 0; n < columns; n++) {
				//out_stream << finalGate->gateMatrix1[n][m] << " ";
				out_stream << "  (" << cuCrealf(finalGate->gateMatrix1[n+m*columns])<< "," << cuCimagf(finalGate->gateMatrix1[n+m*columns]) << ")";
			}
			out_stream << endl;
		}
	}
	this->display = false;
}
/****************************************
 * outputs the Best Invididual
 ****************************************/
void GA::outputBest(bool end) {
	if (bestIndv->fitness >= 1 || progress > 0) {
		out_stream << "---------------------------" << endl;
		out_stream << " Generation at " << generationCondition << endl;
		out_stream << "---------------------------" << endl;
		out_stream << "----- Best Individual -----" << endl;
		out_stream << bestIndv->ioNumber << " <- number of inputs " << endl;
		out_stream << bestIndv->my_string << " <- Representation " << endl;
		out_stream << bestIndv->fitness << " <- Fitness " << endl;
		out_stream << bestIndv->Error << " <- Error " << endl;
		out_stream << bestIndv->Cost << " <- Cost " << endl;
		out_stream << bestIndv->valuedness << " <- Radix " << endl;
		if (displaylevel >= 1) {

#ifdef __QMDD__

			decodeIndvStrToRevLibStr(bestIndv);
			doQMDDFitness(bestIndv, finalIndividual, true);
#else
			if (measurement > 0) {
				if (seq_detect_enabled > 0)
					doMeasureFASeqFitness(bestIndv, true);
				else 
					doMeasureFitness(bestIndv, true);
			} else if (measurement == 0){
				doMatrixFitness(bestIndv, true);
			} else if (measurement == -1){
				doMatrixFitnessPoly(bestIndv, true);
			}

#endif
		}
		out_stream << "---------------------------" << endl;
	}
	if (end && displaylevel > 0) {
		out_stream << "-----Results------" << endl;
		out_stream << "Avg. Fitness, Best Fit.,  Avg. Error,    Avg. Cost  "
		<< endl;
		for (int i = 0; i < generations; i++) {
			for (int j = 0; j < 4; j++) {
				out_stream << history[i][j] << "  ";
			}
			out_stream << endl;
		}
		out_stream << "-----Results------" << endl;
	}
}
/****************************************
 * Restrictions:
 * 1. no fan out - ex, 5 inputs has to have 5 outputs
 * 2. maximum 10 I/O (geometric increase of complexity with more I/Os)
 ****************************************/
void GA::initializePop(int level, string filename, string out_prefix) {

	string inFile = filename;
	string outFile;
	string post = (".out");
	string oFile = "";
	time_t rawtime;
	struct tm * timeinfo;
	int columns;
	condition = false;

	// Open input and output files.
	in_stream.open(inFile.c_str());
	if (in_stream.fail()) {
		cout << "\"Daa set \" file opening failed.\n";
		exit(1);
	}

	outFile = inFile;

	time(&rawtime);
	timeinfo = localtime(&rawtime);
	oFile = asctime(timeinfo);

	unsigned int t = 0;
	while (1) {
		//remove spaces
		if (t < oFile.length()) {
			if (oFile[t] >= '0' && oFile[t] <= 'z') {
				outFile += oFile[t];
			} else {
				if (oFile[t] == ' ') {
				} else {
					break;
				}
			}
		} else
			break;
		t++;
	}

	outFile += post;
	outFile = out_prefix + outFile;
	cout << "\nOutput file will be: " << outFile << endl << endl;
	out_stream.open(outFile.c_str());
	if (out_stream.fail()) {
		cout << "Output file opening failed." << outFile.c_str() << "\n";
		exit(1);
	}

	//initialize the gates
	cout << "Gates Initialization begins." << endl;
	initializeGates();
	if (measurement > 0){
		cout<<"Measurement operators generation begins:"<<endl;
		generateMeasureExpect();
		cout<<"Measurement operators generation succesful."<<endl;
	}
	int w = (int(pow((float) gateArray[0]->valuedness, (float) resultnum)));
	cout << "Gates Initialized " <<endl;

	//only single qubit based sequence detection
	for (int r = 0; r < gateArray[0]->valuedness; r++) {
		measures_o[r] = measurements[r];
		measures_i[r] = measurements[(int) gateArray[0]->valuedness + r];
	}

	//cudaGetErrorString(
#ifdef __CUDA__
	int m_size = w * w * sizeof(cuComplex);
	int v_size = w * sizeof(cuComplex);
	printf("CUDA error: %s\n", cudaGetErrorString(cudaMalloc((void**)&d_M1, m_size)));
	cudaMalloc((void**)&d_M2, m_size);
	cudaMalloc((void**)&d_M3, m_size);
	cudaMalloc((void**)&d_MV, m_size);
	cudaMalloc((void**)&d_VI, v_size);
	cudaMalloc((void**)&d_VO, v_size);

	cudaMalloc((void**)&d_ME_00, m_size);
	cudaMalloc((void**)&d_ME_01, m_size);
	cudaMalloc((void**)&d_ME_10, m_size);
	cudaMalloc((void**)&d_ME_11, m_size);
	cudaMalloc((void**)&d_VV_M, v_size);
	cudaMalloc((void**)&d_Value, sizeof(cuComplex));

	if (measurement > 0){
	(cudaMemcpy(d_ME_00, measures_o[0]->gateMatrix1, m_size, cudaMemcpyHostToDevice));
	(cudaMemcpy(d_ME_01, measures_o[1]->gateMatrix1, m_size, cudaMemcpyHostToDevice));
	(cudaMemcpy(d_ME_10, measures_i[0]->gateMatrix1, m_size, cudaMemcpyHostToDevice));
	(cudaMemcpy(d_ME_11, measures_i[1]->gateMatrix1, m_size, cudaMemcpyHostToDevice));
	}
#endif
	cout << "CUDA Initialized " <<endl;

	//initialize the String array representing the gates
	setStringArray();
	cout << "Generating intial population of circuits" << endl;
	GA::populationCounter = 0;
	//generate the initial population/new_population of individuals
	for (int i = 0; i < populationNumber; i++) {
		population[i] = new Individual;
		new_population[i] = new Individual;
		//all individuals have the width of the output
		population[i]->ioNumber = finalGate->numIO;
		population[i]->valuedness = finalGate->valuedness;

		float val = (float) gateArray[0]->valuedness;
#ifdef __STDR__
		for (int a = 0; a < numofgates; a++) {
			population[i]->gateArray[a] = new qGate;
			//cout<<gateArray[a]<<endl;

			population[i]->gateArray[a]->representation
			= gateArray[a]->representation; // starting from A
			population[i]->gateArray[a]->numIO = gateArray[a]->numIO; // starting from A
			population[i]->gateArray[a]->valuedness = gateArray[a]->valuedness; // starting from A
			population[i]->gateArray[a]->Cost = gateArray[a]->Cost; // starting from A
			columns = (int(pow(val, (float) gateArray[a]->numIO)));
			population[i]->gateArray[a]->gateMatrix1 = new cuComplex[columns*columns];
			for (int j = 0; j < columns; j++) {
				for (int k = 0; k < columns; k++) {
					population[i]->gateArray[a]->gateMatrix1[j+k*columns]= make_cuFloatComplex(cuCrealf(gateArray[a]->gateMatrix1[j+k*columns]), cuCimagf(gateArray[a]->gateMatrix1[j+k*columns]));
				}
			}
		}
#endif

		//population[i]->segmentNumber = rand()%(2*segments)+5;
		population[i]->segmentNumber = segments;
		//cout<<population[i]->segmentNumber<<endl;
		if (phase == 1) {
			population[i]->phase = generatePhase();
		}
		//population[i]->segmentNumber = rand()%MAXSEGNUM+1;
		//generate the String representing this circuit
		makeIndvString(population[i]);
		//population[i]->my_string = "p!!!p";
		//generate phases for every single gate
		if (phase == 1) {
			generatePhases(population[i]);
		}
		//dump to the output file the intialized individual
//	cout << "Generating intial population of circuits3" << endl;
#ifdef __STDR__
		if (measurement > 0){
//	cout << "Generating intial population of circuits4" << endl;
			if (seq_detect_enabled > 0)
				doMeasureFASeqFitness(population[i], false);
			else 
				doMeasureFitness(population[i], false);
			} else if (measurement == 0){
//	cout << "Generating intial population of circuits40" << endl;
				doMatrixFitness(population[i], false);
//	cout << "Generating intial population of circuits41" << endl;
			} else if (measurement == -1){
				doMatrixFitnessPoly(population[i], false);
			}
#endif
#ifdef __QMDD__
//	cout << "Generating intial population of circuits5" << endl;
		decodeIndvStrToRevLibStr(population[i]);
//	cout << "Generating intial population of circuits6" << endl;
		doQMDDFitness(population[i], finalIndividual, false);
//	cout << "Generating intial population of circuits7" << endl;
#endif
	}
	//init the best individual
	cout << "Creating Best Individual Storage" << endl;
	bestIndv = new Individual();
	bestIndv->segmentNumber = segments;
	bestIndv->ioNumber = finalGate->numIO;
	bestIndv->valuedness = finalGate->valuedness;
	int val = bestIndv->valuedness;
#ifdef __STDR__
	for (int a = 0; a < numofgates; a++) {
		bestIndv->gateArray[a] = new qGate;
		initGate(bestIndv->gateArray[a], gateArray[a]->numIO, bestIndv->valuedness,1);
		bestIndv->gateArray[a]->representation = gateArray[a]->representation; // starting from A
		columns = (int(pow(val, (float) gateArray[a]->numIO)));
		for (int j = 0; j < columns; j++) {
			for (int k = 0; k < columns; k++) {
				bestIndv->gateArray[a]->gateMatrix1[j+k*columns]= make_cuFloatComplex(cuCrealf(gateArray[a]->gateMatrix1[j+k*columns]), cuCimagf(gateArray[a]->gateMatrix1[j+k*columns]));
			}
		}
	}
#endif
	makeIndvString(bestIndv);
	//bestIndv->my_string = "p'pp(p";
	if (measurement > 0){
		if (seq_detect_enabled > 0)
			doMeasureFASeqFitness(bestIndv, false);
		else 
			doMeasureFitness(bestIndv, false);
			} else if (measurement == 0){
//	cout << "Generating intial population of circuits40" << endl;
				doMatrixFitness(bestIndv, false);
//	cout << "Generating intial population of circuits41" << endl;
			} else if (measurement == -1){
				doMatrixFitnessPoly(bestIndv, false);
			}
#ifdef __QMDD__
		decodeIndvStrToRevLibStr(bestIndv);
		doQMDDFitness(bestIndv, finalIndividual, false);
#endif


	outputBest(false);
	initiateStorage();
	cout << "Initialization done" << endl;
	out_stream << "\nInitialization done" << endl;
}


/****************************************
 * Reads the parameters from the open input streams and generates all gates
 * as well as initiates the measurement operator generation if requested
 ****************************************/
void GA::initializeGates() {
        complex<float> in = complex<float> (0., 0.);

	float x, y;
	int counter = 0;
	int seq_type = 0;
	int dc = 0;
	int ycounter;
	int *measurement_indexes;
	int valuedness = 0;
	int resnum, outnum, a, b, columns;
	char chcounter = char('!');
	std::string temp;
	
	char unknown;
	char tempchar;
	char buff[100];
	char binary[100];

	//reading the input file
	cout << "Reading input parameters" << endl;
	in_stream >> populationNumber; //int
	in_stream >> segments; //int
	in_stream >> mincircuitsize; //int
	in_stream >> alterations; //int
	in_stream >> generations; //int
	in_stream >> proba_Mutation; //float
	in_stream >> proba_Crossover; //float
	in_stream >> alpha; //float
	in_stream >> beta; //float
	in_stream >> alpha1; //float
	in_stream >> beta1; //float
	in_stream >> divider; //int
	in_stream >> phase; //int
	in_stream >> displaylevel; //int
	//reading file divider
	in_stream >> temp;
	in_stream >> Ga; //int
	in_stream >> elitism; //int
	in_stream >> mutation; //int
	in_stream >> crossover; //int
	in_stream >> replicator; //int
	in_stream >> fitness; //int
	in_stream >> grouped; //bool
	in_stream >> pareto; //bool
	in_stream >> threshold; //float
	in_stream >> resultnum;
	in_stream >> valuedness;
	in_stream >> error_adjust;
	in_stream >> seq_detect_enabled;
//	in_stream >> seq_gener_enabled;
	in_stream >> measurement; //

	if (measurement > 0) {

		int hh = (int) (pow((float) valuedness,(float) resultnum));
		int g = (int) measurement * valuedness;
		measureexpected = new cuComplex[hh][MAXGATEINPUT*2];

		int i = 0;
		int h = 0;
		for (h = 0; h < measurement; h++)
			in_stream >> measurementQBits[i++];
		cout << "Reading Measurements for " << i << " qubits" << endl;
		in_stream >> measured_desired_output_records;
		cout << measured_desired_output_records << " <- defined outputs" << endl;
		measurement_indexes = new int[h];

		if (seq_detect_enabled > 0 || seq_gener_enabled > 0){
			//read the sequence in as a set of 0 or 1 or MV values with appropriate probabilities
			sequence_desired = new cuComplex[measured_desired_output_records][2];
			for (int m = 0; m < measured_desired_output_records; m++) {
				in_stream >> unknown;
				if (!(unknown == '+' || unknown == '*'))
					a = unknown;

				in_stream >> tempchar;
				seq_type == 0;
				if (tempchar == '+'){
					seq_type == 1; //one or more
					in_stream >> tempchar;
				}
				else if (tempchar == '*'){
					seq_type == 2; //zero or more
					in_stream >> tempchar;
				}
				in_stream >> in;
				sequence_desired[m][0] =  make_cuFloatComplex(int(a-48),0);
				sequence_desired[m][1] =  make_cuFloatComplex(real(in),imag(in));
				cout<<m<<": (" << cuCrealf(sequence_desired[m][0])<<") -> ("<< cuCrealf(sequence_desired[m][1]) <<", "<<cuCimagf(sequence_desired[m][1]) << ")"<<endl;
			}
			in_stream >> temp;
		} else {

			for (int b = 0; b < MAXGATEINPUT*2; b++) {
				for (int c = 0; c < hh; c++) {
					measureexpected[c][b] = make_cuFloatComplex(0, -1);
				}
			}
			//the input of measured qubits is a sequence of numbers separated by space
			i = 0;
			for (int b = 0; b < measured_desired_output_records; b++) {
				in_stream >> outnum;
				measurement_indexes[i++] = outnum;
				cout << outnum << "  ";
				in_stream >> tempchar;
				cout << tempchar << "  ";
				for (int h = 0; h < measurement; h++) {
					if (h > 0 ){
						in_stream >> tempchar;
					cout << tempchar;
					}
					for (int g = 0; g < valuedness; g++) {
						in_stream >> in;
						measureexpected[outnum][h * valuedness + g]= make_cuFloatComplex(real(in), imag(in));
					cout <<"("<<cuCrealf(measureexpected[outnum][h * valuedness + g])<<", "<<cuCimagf(measureexpected[outnum][h * valuedness + g]) << ")  ";
					}
				}
			cout << endl;
			}


			in_stream >> temp;

			measured_desired_output_records_idx = new int[i];
			for (int j = 0; j < i ; j++)
				measured_desired_output_records_idx[j] = measurement_indexes[j];

		}
	} else {
		do {
			in_stream >> temp;
		} while (temp.at(0) != '-');

	}


/*		int g = (int) (pow((float) valuedness,(float) resultnum));
		for (int b = 0; b < g; b++) {
			for (int c = 0; c < MAXGATEINPUT*2; c++) {
				cout << c<<", "<<b<<", ("<<cuCrealf(measureexpected[b][c])<<", "<<cuCimagf(measureexpected[b][c]) << ")  ";
			}
			cout << endl;
		}
*/

	//reading file divider
	chcounter = loadGateArray(valuedness, chcounter);
        cout <<"Gates read in and initialized succesfully "<<endl;
        cout <<"Extended Gates generation begins:"<<endl;
	generateExtendMatrix(chcounter);
        cout <<"All gates initialized succesfully. "<<endl;
        cout << " Writting Output " << endl;


	//generate the header of the output file
	out_stream << " -------------------------------------------------------- "
	<< endl;
	out_stream << "         output file of synthesis: settings               "
	<< endl;
	out_stream << " -------------------------------------------------------- "
	<< endl;
	out_stream << " Size of the population:                                  "
	<< populationNumber << endl;
	out_stream << " Number of segments in each circuit (approx):             "
	<< segments << endl;
	out_stream << " Minimal Number of segments in each circuit (approx):     "
	<< mincircuitsize << endl;
	out_stream << " Number of GA generations before EX search starts:        "
	<< alterations << endl;
	out_stream << " Total number of GA generations:                          "
	<< generations << endl;
	out_stream << " Mutation probability is:                                 "
	<< proba_Mutation << endl;
	out_stream << " Crossover probability is:                                "
	<< proba_Crossover << endl;
	out_stream << " Factor alpha is (for complex fitness):                   "
	<< alpha << endl;
	out_stream << " Factor beta is (for complex fitness):                    "
	<< beta << endl;
	out_stream << " Factor alpha1 is (for pareto replication):               "
	<< alpha1 << endl;
	out_stream << " Factor beta1 is  (for pareto replication):               "
	<< beta1 << endl;
	out_stream << " Estimated minimum cost of the goal circuit:              "
	<< divider << endl;
	out_stream << " Phase switch (0 - off, 1 - on):                          "
	<< phase << endl;
	out_stream << " Display level:                                           "
	<< displaylevel << endl;
	out_stream << " -------------------------------------------------------- "
	<< endl;
	out_stream << " Type of GA (0 - normal, 1 - Darwinian):                  "
	<< Ga << endl;
	out_stream << " Use Elitism (0 - no, 1 - yes):                           "
	<< elitism << endl;
	out_stream << " Type of mutation (0 - normal, 1 - bitwise):              "
	<< mutation << endl;
	out_stream << " Type of crossover (0 - 1point, 1 - 2point):              "
	<< crossover << endl;
	out_stream << " Type of replication (0 - RW, 1 - SUS):                   "
	<< replicator << endl;
	out_stream << " Type of fitness (0 - simplest, 3 - complex):             "
	<< fitness << endl;
	out_stream << " Fitness calculation (0 - individual, 1 - grouped):       "
	<< grouped << endl;
	out_stream << " Pareto optimization (0 - no, 1 - yes):                   "
	<< pareto << endl;
	out_stream << " Max-Fitness Threshold for Replication (0 - no, 1< >0):   "
	<< threshold << endl;
	out_stream << " The number of wires the final circuit has:               "
	<< resultnum << endl;
	out_stream << " Valuedness of designed logic:                            "
	<< valuedness << endl;
	out_stream << " Error adjustment for Entanglement design:                            "
	<< error_adjust << endl;
	out_stream << " Design of Sequential Logic/ Sequence Detector:           "
	<< seq_detect_enabled <<endl;
	out_stream << " Measurement used:                                        "
	<< measurement << endl;



	if (measurement > 0) {
		out_stream
		<< " Measured bits are:                                       ";
		for (int a = 0; a < measurement; a++)
			out_stream << measurementQBits[a] << " ";
		out_stream << endl;

		
		if (seq_detect_enabled > 0 || seq_gener_enabled > 0){
			for (int c = 0; c < measured_desired_output_records; c++) {
					out_stream << cuCrealf(sequence_desired[c][0])<<": ";
					out_stream << " (" << cuCrealf(sequence_desired[c][1]) << ","<< cuCimagf(sequence_desired[c][1]) << ")";
					out_stream << endl;
				}
				out_stream << endl;
		} else {
			for (int c = 0; c < measured_desired_output_records; c++) {
				out_stream <<c<<": "<<measured_desired_output_records_idx[c]<< ": ";
				for (int b = 0; b < measurement * valuedness; b++){
					out_stream << " (" << cuCrealf(measureexpected[measured_desired_output_records_idx[c]][b]) << ","<< cuCimagf(measureexpected[measured_desired_output_records_idx[c]][b]) << ")";
				}
				out_stream << endl;
			}
		}
/*

			temp = "";
			temp += ".version Target Individual\n.numvars ";
			sprintf(buff, "%d", resultnum);
			counter = 0;
			while(buff[counter] > 0)
				temp += buff[counter++];	
			temp += "\n.variables ";
			for (ycounter = 0; ycounter < resnum; ycounter++){
				sprintf(buff, "%d", ycounter);
				counter = 0;
				temp += 'w';	
				while(buff[counter] > 0)
					temp += buff[counter++];	
				temp += ' ';	
			}
			temp += "\n.inputs ";
			for (ycounter = 0; ycounter < resnum; ycounter++){
				sprintf(buff, "%d", ycounter);
				counter = 0;
				temp += 'w';	
				while(buff[counter] > 0)
					temp += buff[counter++];	
				temp += ' ';	
			}
			temp += "\n.outputs ";
			for (ycounter = 0; ycounter < resnum; ycounter++){
				sprintf(buff, "%d", ycounter);
				counter = 0;
				temp += 'w';	
				while(buff[counter] > 0)
					temp += buff[counter++];	
				temp += ' ';	
			}
			temp += "\n.begin\n";

			dcMatrix(finalGate, resnum);
			columns = (int(pow((float) valuedness, (float) resnum)));
			for (int m = 0; m < measured_desired_output_records; m++) {
				s = "";
				in_stream >> a;
				in_stream >> tempchar;
				in_stream >> b;
				dec2bin(b, binary, resnum);
				cout <<binary<<";; "<< a << " " << tempchar << " " << b;
				in_stream >> tempchar;
				in_stream >> in;
				cout << " " << tempchar << " " << in << endl;
				finalGate->gateMatrix1[a+b*columns] = make_cuFloatComplex(real(in),
						imag(in));
				counter = 0;
				while(binary[counter] > 0)
					temp += binary[counter++];
				temp += "\n";
			}

			temp += ".end\n";
			counter = 0;
			while (counter < temp.size())
				finalIndividual->rev_lib_string += temp.at(counter++);	


*/

	}

/*		for (int c = 0; c < measured_desired_output_records; c++) {
				cout <<c<<": "<<measured_desired_output_records_idx[c]<< ": ";
			for (int b = 0; b < measurement * valuedness; b++){
				cout << " (" << cuCrealf(measureexpected[measured_desired_output_records_idx[c]][b]) << ","<< cuCimagf(measureexpected[measured_desired_output_records_idx[c]][b]) << ")";
			}
			cout << endl;
		}
		cout<<endl;

		for (int c = 0; c < measured_desired_output_records; c++) 
			 cout <<c<<": "<<measured_desired_output_records_idx[c]<< endl;;

		int g = (pow((float) valuedness,(float) resultnum));
		for (int b = 0; b < g; b++) {
			for (int c = 0; c < resultnum*2; c++) {
				cout << c<<", "<<b<<", ("<<cuCrealf(measureexpected[b][c])<<", "<<cuCimagf(measureexpected[b][c]) << ")  ";
			}
			cout << endl;
		}
exit(0);
*/
	
#ifdef __QMDD__
        cout<<"Generating RevLib Gate Representation:"<<endl;
	for (int a = 0; a < numofgates; a++) 
		decodeGateToRevLib(gateArray[a]);
        cout<<"RevLib Gate Representation succesful."<<endl;
#endif
	cout << " Writting Output " << endl;
	out_stream << " -------------------------------------------------------- "
	<< endl;
	out_stream << "            The radix of this logic is:                   "
	<< valuedness << endl;
	out_stream << " -------------------------------------------------------- "
	<< endl;
	out_stream << "            Number of Input Gates:                        "
	<< numofgates << endl;
	for (int a = 0; a < numofgates; a++) {
		out_stream << "Representation: " << gateArray[a]->representation
		<< endl;
		out_stream << " Cost of the gate:  " << gateArray[a]->Cost << endl;
		out_stream << " IO of the gate:  " << gateArray[a]->numIO << endl;
		out_stream << " The name of this gate is: " << gateArray[a]->my_string<< endl;
		out_stream << " The RevLib representation of this gate is: " << gateArray[a]->rev_lib_string<< endl;
		columns = (int(pow((float) valuedness,(float) gateArray[a]->numIO)));
		for (int j = 0; j < columns; j++) {
			for (int k = 0; k < columns; k++) {
				out_stream << "  ("
				<< cuCrealf(gateArray[a]->gateMatrix1[j+k*columns]) << ","
				<< cuCimagf(gateArray[a]->gateMatrix1[j+k*columns]) << ")";
			}
			out_stream << endl;
		}
	}
	out_stream << " -------- Generating the desired output evaluation ------ "
	<< endl;
	if (measurement == 0) {

		out_stream << " Desired Gate: " << endl << endl;
		out_stream << " IO of the gate:  " << finalGate->numIO << endl;
		//cout<<endl;
		columns = (int(pow((float) valuedness, (float) resultnum)));
		for (int m = 0; m < columns; m++) {
			for (int n = 0; n < columns; n++) {
				out_stream << "  (" << cuCrealf(finalGate->gateMatrix1[m+n*columns])
				<< "," << cuCimagf(finalGate->gateMatrix1[m+n*columns]) << ")";
			}
			out_stream << endl;
		}
	} else if (measurement == 1) {

		out_stream << " Observables:  " << endl;
		out_stream << " Desirded Qubits of the whole system: "
		<< finalGate->numIO << endl;
		;
		out_stream << " Desirded Valuedness: " << finalGate->valuedness << endl;
		out_stream << endl;
		generateMeasureOps();
		delete [] measurement_indexes;
	} else if (measurement == -1) {
		out_stream<< " Entanglement complexity measure is given by :"<<endl;
		out_stream<< "|d1 - 2d2 + 4d3|" <<endl;
	}

	out_stream << " -------------------------------------------------------- "
	<< endl;
	out_stream << "                   Output of synthesis:                   "
	<< endl;
	out_stream << " -------------------------------------------------------- "
	<< endl;
}
/*******************************************
 * Create the measurement oerators for Single Qubits
 * for all orthonormal states in a given valuedness
 *******************************************/
void* GA::generateMeasureOps() {

	float numIO = (float) finalGate->numIO;
	float val = (float) finalGate->valuedness;
	int val_count = 0;
	bool zeros = true;
	int a = 0;
	int m = 0;
	int columns;
	int qubitindex = 0;
	string srank = "";
	int current_step = (int) (pow(val, (float) (qubitindex + 1)));
	int step_length = (int) (pow(val, (float) qubitindex));

	//generate measurement opertors for every single qubit
	cout << "generating measurement circuits: " << MAXMEASUREGATE << "   "
	<< numIO << endl;
	while (a < resultnum * val) {
		convert_stream.str("");
		convert_stream << qubitindex;
		srank = convert_stream.str();

		measurements[a] = new qGate;
		measurements[a]->valuedness = (int) val;
		measurements[a]->numIO = (int) numIO;
		measurements[a]->representation = char(val_count);
		measurements[a]->my_string += "th qubit";
		measurements[a]->my_string = srank + "  " + measurements[a]->my_string;

		columns = (int) (pow(val, numIO));
		measurements[a]->gateMatrix1 = new cuComplex[columns*columns];
		for (int x = 0; x < columns; x++) {
			for (int y = 0; y < columns; y++) {
				measurements[a]->gateMatrix1[x+y*columns] = make_cuFloatComplex(0, 0);
				if (x == y) {
					if (step_length > 0) {
						measurements[a]->gateMatrix1[x+y*columns]= make_cuFloatComplex(1, 0);
						step_length--;
					}
					current_step--;
					if (current_step <= 0) {
						current_step = (int) (pow((float) val,
								(float) (qubitindex + 1)));
						step_length = (int) (pow((float) val,
								(float) qubitindex));
					}
				}
			}
		}
		val_count++;
		if (val_count >= val) {
			val_count = 0;
			qubitindex++;
			step_length = (int) (pow((float) val, (float) qubitindex));
			current_step = (int) (pow((float) val, (float) (qubitindex + 1)));
		} else {
			//change the way how it starts for the next observable of this qubit
			step_length = 0;
			//for each observable shift the matrices index
			current_step = val_count * (int) (pow((float) val,
					(float) qubitindex));
		}
		a++;
	}
	//output the measurement operators
	for (int a = 0; a < val * resultnum; a++) {
		out_stream << "Measurement Gates:" << endl;
		//	out_stream<< measurements[a]->representation <<endl;
		out_stream << measurements[a]->my_string << endl;
		out_stream << "  IO of the gate  " << measurements[a]->numIO << endl;
		columns = (int(pow(val, (float) measurements[a]->numIO)));
		for (int j = 0; j < columns; j++) {
			for (int k = 0; k < columns; k++) {
				out_stream << "  (" << cuCrealf(measurements[a]->gateMatrix1[j+k*columns]) << "," << cuCimagf(measurements[a]->gateMatrix1[j+k*columns]) << ")";
				//out_stream<< measurements[a]->gateMatrix1[j][k]<<' ';
			}
			out_stream << endl;
		}
	}
	cout << "Done" << endl;

}
/*******************************************
 * Initialize the expectations probabilities for measured global states
 *******************************************/
void GA::generateMeasureExpect() {

	int l = finalGate->numIO;
	int p = (int(pow((float) 2, (float) l)));
	for (int g = 0; g < p; g++) {
		expectationsAllState[g][0] = make_cuFloatComplex(1, 0);
		expectationsAllState[g][1] = make_cuFloatComplex(1, 0);
	}

	for (int k = 0; k < p; k++) {
		//	cout << " For input state: "<<k <<endl;
		for (int m = 0; m < measurement; m++) {
			//check if entanglement is desired
			if (cuCrealf(measureexpected[k][2* m ]) == cuCrealf(
					measureexpected[k][2* m + 1])) {
				if (cuCrealf(measureexpected[k][2* m ]) != 1 && cuCrealf(
						measureexpected[k][2* m ]) != 0) {
					expectationsAllState[k][0] = cuCmulf(
							expectationsAllState[k][0],
							measureexpected[k][2* m ]);
					expectationsAllState[k][1] = cuCmulf(
							expectationsAllState[k][1], measureexpected[k][2* m
							                                               + 1]);
				}
			} else {
				if (cuCrealf(measureexpected[k][2* m ]) > cuCrealf(
						measureexpected[k][2* m + 1])) {
					expectationsAllState[k][0] = cuCmulf(
							expectationsAllState[k][0],
							measureexpected[k][2* m ]);
					expectationsAllState[k][1] = cuCmulf(
							expectationsAllState[k][1], measureexpected[k][2* m
							                                               + 1]);
				} else if (cuCrealf(measureexpected[k][2* m ]) <= cuCrealf(
						measureexpected[k][2* m + 1])) {
					expectationsAllState[k][0] = cuCmulf(
							expectationsAllState[k][0], measureexpected[k][2* m
							                                               + 1]);
					expectationsAllState[k][1] = cuCmulf(
							expectationsAllState[k][1],
							measureexpected[k][2* m ]);
				}
			}
		}
	}
}
/*******************************************
 * Create all mullti state measurements for desired output states
 * This is designed so as we can distinguish  between the most wanted state
 * and all other ones, thus creating E_0 and I-E_0
 *******************************************/
void* GA::generateMultiMeasureOps() {
	int l = finalGate->numIO;
	float numIO = (float) finalGate->numIO;
	//the counter for measurement matrices is implemented as follows
	//2^(abs) index - numNBGR-1 (to get to the index array values)
	//int k = (int)pow(2.0, (float)(abs(index - (tRule->numNGBR-1))));
	int n, m, columns = 0;
	bool zeros = true;
	int val = (float) finalGate->valuedness;
	int a = (int) (resultnum * val);
	string srank;
	qGate *measure, *measure0, *measure1, *measureM0, *measureM1, *temp;
	int p = (int(pow((float) val, (float) l)));

	//generate measurement opertors for every single qubit
	//cout<< "generating measurement circuits: "<< MAXMEASUREGATE;
	while (a < l * 2) {
		n = 0;

		convert_stream.flush();
		measure0 = new qGate();
		initGate(measure0, resultnum, val, 1);
		measure1 = new qGate();
		initGate(measure1, resultnum, val, 1);
		measureM0 = new qGate();
		initGate(measureM0, resultnum, val, 1);
		measureM1 = new qGate();
		initGate(measureM1, resultnum, val, 1);
		initMatrix(measureM1, resultnum);
		temp = new qGate();
		initGate(temp, resultnum, val, 1);
		//for every orthonormal state on the input
		//generate two measurement operators in the form E_0 and I-E_0
		cout << "Generating Measurements For all desired states" << endl;
		for (int k = 0; k < p; k++) {

			initMatrix(measureM0, resultnum);
			//initMatrix(&measureM1, resultnum);
			for (int m = 0; m < (measurement * val); m++) {
				measure0 = ccGate(measurements[(measurementQBits[m] * val)],
						measure0);
				//check if entanglement is desired
				if (cuCrealf(measureexpected[k][val * m]) == cuCrealf(
						measureexpected[k][2* m + 1])) {
					if (cuCrealf(measureexpected[k][2* m ]) == 1) {
						initMatrix(temp, resultnum);
						cblasMatrixProduct(measureM0, measure1, temp);
						measureM0 = ccGate(temp, measureM0);
					} else if (cuCrealf(measureexpected[k][2* m ]) == 0) {
						if (!(cuCimagf(measureexpected[k][2* m ]) == 1
								&& cuCimagf(measureexpected[k][2* m + 1]) == 1)) {
							initMatrix(temp, resultnum);
							cblasMatrixProduct(measureM0, measure0, temp);
							measureM0 = ccGate(temp, measureM0);
						}
					} else {
						if (!(cuCimagf(measureexpected[k][2* m ]) == 1
								&& cuCimagf(measureexpected[k][2* m + 1]) == 1)) {
							initMatrix(temp, resultnum);
							cblasMatrixProduct(measureM0, measure0, temp);
							measureM0 = ccGate(temp, measureM0);
						}
					}
				} else {
					if (cuCrealf(measureexpected[k][2* m ]) > cuCrealf(
							measureexpected[k][2* m + 1])) {
							initMatrix(temp, resultnum);
							cblasMatrixProduct(measureM0, measure0, temp);
							measureM0 = ccGate(temp, measureM0);
					} else if (cuCrealf(measureexpected[k][2* m ]) <= cuCrealf(
							measureexpected[k][2* m + 1])) {
							initMatrix(temp, resultnum);
							cblasMatrixProduct(measureM0, measure1, temp);
							measureM0 = ccGate(temp, measureM0);
					}
				}
			}

			measureM1 = matrixSubstr(measureM1, measureM0);
			convert_stream << k;
			srank = convert_stream.str();
			measurementsAllState[2* k ] = new qGate();
			initGate(measurementsAllState[2* k ], resultnum, val, 1);
			measurementsAllState[2* k ] = ccGate(measureM0,	measurementsAllState[2* k ]);
			measurementsAllState[2* k ]->my_string = srank
			+ " Desired input state";
			//	cout<<k<<": measurement operator :"<<measurementsAllState[2*k]<<endl;

			//measurementsAllState[2*k+1] = cGate(&measureM1);
			measurementsAllState[2* k + 1] = new qGate();
			initGate(measurementsAllState[2* k +1], resultnum, val, 1);
			measurementsAllState[2* k + 1] = ccGate(measureM1, measurementsAllState[2* k + 1]);
			measurementsAllState[2* k + 1]->my_string = srank
			+ " Undesired state ";
			//	cout<<k<<": measurement operator :"<<measurementsAllState[2*k+1]<<endl;
		}
	}

	//output the measurement operators
	for (int a = 0; a < p; a++) {
		//	cout<<a<<": measurement operator :"<<measurementsAllState[2*a]<<endl;
		out_stream << "Whole state Measurement Gates:" << endl;
		out_stream << measurementsAllState[2* a ]->my_string << endl;
		out_stream << measurementsAllState[2* a ]->representation << endl;
		out_stream << "  IO of the gate  "
		<< measurementsAllState[2* a ]->numIO << endl;
		columns =  (int(pow((float) 2, (float) measurementsAllState[2*a ]->numIO)));
		for (int j = 0; j < columns; j++) {
			for (int k = 0; k < columns; k++) {
				//out_stream<< measurementsAllState[2*a]->gateMatrix1[j][k]<<' ';
				out_stream << "  (" << cuCrealf(measurementsAllState[2* a ]->gateMatrix1[j+k*columns]) << ","<< cuCimagf(measurementsAllState[2* a ]->gateMatrix1[j+k*columns])<< ")";
			}
			out_stream << endl;
		}

		//	cout<<a<<": measurement operator :"<<measurementsAllState[2*a+1]<<endl;
		out_stream << measurementsAllState[2* a + 1]->my_string << endl;
		out_stream << measurementsAllState[2* a + 1]->representation << endl;
		out_stream << "  IO of the gate  "
		<< measurementsAllState[2* a + 1]->numIO << endl;
		columns = (int(pow((float) 2, (float) measurementsAllState[2*a + 1]->numIO)));
		for (int j = 0; j < columns; j++) {
			for (int k = 0; k < columns; k++) {
				out_stream << "  (" << cuCrealf(measurementsAllState[2* a + 1]->gateMatrix1[j+k*columns])<< "," << cuCimagf(measurementsAllState[2* a + 1]->gateMatrix1[j+k*columns])<< ")";
				//out_stream<< measurementsAllState[2*a+1]->gateMatrix1[j][k]<<' ';
			}
			out_stream << endl;
		}

	}
	delete (measure0, measure1);

}
/*******************************************
 * Verifies the input gates and generates all missing ones
 * for multiple qubits
 *******************************************/
void GA::generateExtendMatrix(char chcounter) {

	int counter = 0;
	string name;
	qGate *gate, *extgate, *tempgate, *sgate;

	int controls[MAXINDVWIRENUM];
	int i_placement[MAXINDVWIRENUM];
	int columns, initial, wire_counter;
	int c_count = 0;
	int topwire = 2;
	int function = 0;
	bool found;

	gate = new qGate;
					cout<<"tensor: "<<gateArray[0]->my_string<<"  "<<gate->my_string<<endl;
	initGate(gate, resultnum, gateArray[0]->valuedness, 1);
					cout<<"tensor: "<<gate->my_string<<"  "<<gate->my_string<<endl;
	tempgate = new qGate;
	initGate(tempgate, resultnum, gateArray[0]->valuedness, 1);
	extgate = new qGate;
	initGate(extgate, resultnum, gateArray[0]->valuedness, 1);
	sgate = new qGate;
	initGate(sgate, 2, gateArray[0]->valuedness, 1);
	initial = 1;
	initMatrix(sgate, 2);
	columns = (int(pow((float)sgate->valuedness,(float)sgate->numIO)));

					cout<<"tensor: "<<gate->my_string<<"  "<<gate->my_string<<endl;
if (gateArray[0]->valuedness == 2){
	sgate->gateMatrix1[1+1*columns] = make_cuFloatComplex(0, 0);
	sgate->gateMatrix1[1+2*columns] = make_cuFloatComplex(1, 0);
	sgate->gateMatrix1[2+1*columns] = make_cuFloatComplex(1, 0);
	sgate->gateMatrix1[2+2*columns] = make_cuFloatComplex(0, 0);
} else if (gateArray[0]->valuedness == 3){
	zeroMatrix(sgate, 2);
        sgate->gateMatrix1[0] = make_cuFloatComplex(1, 0);
        sgate->gateMatrix1[12] = make_cuFloatComplex(1, 0);
        sgate->gateMatrix1[24] = make_cuFloatComplex(1, 0);
        sgate->gateMatrix1[28] = make_cuFloatComplex(1, 0);
        sgate->gateMatrix1[40] = make_cuFloatComplex(1, 0);
        sgate->gateMatrix1[52] = make_cuFloatComplex(1, 0);
        sgate->gateMatrix1[56] = make_cuFloatComplex(1, 0);
        sgate->gateMatrix1[68] = make_cuFloatComplex(1, 0);
        sgate->gateMatrix1[80] = make_cuFloatComplex(1, 0);
}
					cout<<"tensor: "<<gate->my_string<<"  "<<gate->my_string<<endl;
	while (true)
		if (topwire < resultnum) {
			topwire++;
			zeroMatrix(tempgate, extgate->numIO);

			counter = 0;
			for (int a = numofgates - 1; a >= 0; a--) {
				if (gateArray[a]->numIO == topwire - 1) {
					c_count = 0;
					wire_counter = 0;
					gate->valuedness = gateArray[a]->valuedness;
					initMatrix(gate, gateArray[a]->numIO);
					gate = ccGate(gateArray[a], gate);

					cout<<"tensor: "<<gate->my_string<<"  "<<gate->my_string<<endl;
					for (int h = 0 ; h < MAXINDVWIRENUM; h++)
						controls[h] = 0;
					
					for (int m = 0; m < resultnum; m++){
						controls[m] = -10;i_placement[m] = -10;
					}
					name = "";
					function = -10;
					for (int m = 0; m < gate->my_string.length(); m++){
						name += gate->my_string.at(m);
						if (gate->my_string.at(m) == 'C'){
							i_placement[c_count] = m;
							controls[c_count] = wire_counter;
							c_count++;
							wire_counter++;
						} else if (gate->my_string.at(m) == 'I'){
							wire_counter++;
						} else {
							if (function == -10){
								function =  wire_counter;
								wire_counter++;
							}
						}
					}
					for (int m = 0; m < resultnum; m++)
						cout << controls[m] <<", ";
					cout <<endl;
					for (int m = 0; m < resultnum; m++)
						cout << i_placement[m] <<", ";
					cout <<endl;

/*					cout<<"tensor: "<<gate->my_string<<"  "<<gate->my_string<<endl;
					 columns = (int(pow((float)gate->valuedness,(float)gate->numIO)));
					 for (int j = 0; j < columns; j++) {
						 for (int k = 0; k < columns; k++) {
							 cout << "  (" << cuCrealf(gate->gateMatrix1[j+k*columns]) << "," << cuCimagf(gate->gateMatrix1[j+k*columns])<< ")";
						 }
						 cout << endl;
					 }
					 cout << endl;

					 cout<<gateArray[0]->my_string<<"  before tensor product"<<endl;

					cout<<"tensor: "<<gate->my_string<<"  "<<gateArray[0]->my_string<<endl;
					 columns = (int(pow((float)gate->valuedness,(float)gate->numIO)));
					 for (int j = 0; j < columns; j++) {
						 for (int k = 0; k < columns; k++) {
							 cout << "  (" << cuCrealf(gate->gateMatrix1[j+k*columns]) << "," << cuCimagf(gate->gateMatrix1[j+k*columns])<< ")";
						 }
						 cout << endl;
					 }
					 cout << endl;
*/
					tensorProduct3(tempgate, gateArray[0] ,gate);
					gate = ccGate(tempgate, gate);

/*					 cout<<gate->my_string<<"  after tensor product"<<endl;
					 columns = (int(pow((float)gate->valuedness,(float)gate->numIO)));
					 for (int j = 0; j < columns; j++) {
						 for (int k = 0; k < columns; k++) {
							 cout << "  (" << cuCrealf(gate->gateMatrix1[j+k*columns]) << "," << cuCimagf(gate->gateMatrix1[j+k*columns])<< ")";
						 }
						 cout << endl;
					 }
					 cout << endl;

*/
					for (int u = 0; u < c_count; u++){
						initMatrix(extgate, resultnum);
						for (int l = 0; l < topwire; l++) {
							if (u == 0 && l == 0 ){
								extgate = ccGate(sgate, extgate);
								l++;
/*					 cout<<"l == u &&  == 0, initiated the extgate"<<endl;
					 columns = (int(pow((float)extgate->valuedness,(float)extgate->numIO)));
					 for (int j = 0; j < columns; j++) {
						 for (int k = 0; k < columns; k++) {
							cout << "  (" << cuCrealf(extgate->gateMatrix1[j+k*columns]) << "," << cuCimagf(extgate->gateMatrix1[j+k*columns]) << ")";
						 }
						 cout << endl;
					 }
*/
							}else if (l == 0){
								extgate = ccGate(gateArray[0], extgate);
/*					cout<<"2tensor: "<<gate->my_string<<"  "<<gateArray[0]->my_string<<endl;
					 cout<<"l == 0, initiated the extgate"<<endl;
					 columns = (int(pow((float)extgate->valuedness,(float)extgate->numIO)));
					 for (int j = 0; j < columns; j++) {
						 for (int k = 0; k < columns; k++) {
							cout << "  (" << cuCrealf(extgate->gateMatrix1[j+k*columns]) << "," << cuCimagf(extgate->gateMatrix1[j+k*columns]) << ")";
						 }
						 cout << endl;
					 }
*/
							}else if( u == l){
								tensorProduct3(tempgate, extgate, sgate);
								extgate = ccGate(tempgate, extgate);
								l++;
/*					 cout<<"l == u, initiated the extgate"<<endl;
					 columns = (int(pow((float)extgate->valuedness,(float)extgate->numIO)));
					 for (int j = 0; j < columns; j++) {
						 for (int k = 0; k < columns; k++) {
							cout << "  (" << cuCrealf(extgate->gateMatrix1[j+k*columns]) << "," << cuCimagf(extgate->gateMatrix1[j+k*columns]) << ")";
						 }
						 cout << endl;
					 }
*/
							} else {
								tensorProduct3(tempgate, extgate, gateArray[0]);
								extgate = ccGate(tempgate, extgate);
/*					 cout<<"tensor extgate"<<endl;
					 columns = (int(pow((float)extgate->valuedness,(float)extgate->numIO)));
					 for (int j = 0; j < columns; j++) {
						 for (int k = 0; k < columns; k++) {
							cout << "  (" << cuCrealf(extgate->gateMatrix1[j+k*columns]) << "," << cuCimagf(extgate->gateMatrix1[j+k*columns]) << ")";
						 }
						 cout << endl;
					 }
*/
							}

						}
						extgate->numIO = topwire;

/*					
					  

					 cout<<extgate->my_string<<"before matrix product"<<endl;
					 columns = (int(pow((float)extgate->valuedness,(float)extgate->numIO)));
					 for (int j = 0; j < columns; j++) {
						 for (int k = 0; k < columns; k++) {
							cout << "  (" << cuCrealf(extgate->gateMatrix1[j+k*columns]) << "," << cuCimagf(extgate->gateMatrix1[j+k*columns]) << ")";
						 }
						 cout << endl;
					 }
					 cout << endl;
					 
					 for (int j = 0; j < columns; j++) {
						 for (int k = 0; k < columns; k++) {
							cout << "  (" << cuCrealf(gate->gateMatrix1[j+k*columns]) << "," << cuCimagf(gate->gateMatrix1[j+k*columns]) << ")";
						 }
						 cout << endl;
					 }
					 cout << endl;

*/						zeroMatrix(tempgate, extgate->numIO);
						//int w = (int(pow(pow((float) gate->valuedness,(float) gate->numIO),2)));
						cblasMatrixProduct(extgate, gate, tempgate);
						//cblasMatrixProduct(gateArray[0], gateArray[0], tempgate);
						
						//cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasTrans, columns, columns, columns, &alpha, extgate->gateMatrix1, columns, extgate->gateMatrix1, columns, &beta, tempgate->gateMatrix1, columns);

/*						 cout<<tempgate->my_string<<"  after matrix product"<<endl;
					 columns = (int(pow((float)tempgate->valuedness,(float)tempgate->numIO)));
					 for (int j = 0; j < columns; j++) {
						 for (int k = 0; k < columns; k++) {
							cout << "  (" << cuCrealf(tempgate->gateMatrix1[j+k*columns]) << "," << cuCimagf(tempgate->gateMatrix1[j+k*columns]) << ")";
						 }
						 cout << endl;
					 }
					 cout << endl;
*/				
						gate = ccGate(tempgate, gate);
						zeroMatrix(tempgate, extgate->numIO);

/*					 for (int j = 0; j < columns; j++) {
						 for (int k = 0; k < columns; k++) {
							cout << "  (" << cuCrealf(gate->gateMatrix1[j+k*columns]) << "," << cuCimagf(gate->gateMatrix1[j+k*columns]) << ")";
						 }
						 cout << endl;
					 }
					 cout << endl;
*/
						cblasMatrixProduct(gate, extgate, tempgate);
						gate = ccGate(tempgate, gate);
	
						gate->numIO = topwire;
						gate->my_string = "";
/*					 columns = (int(pow((float)gate->valuedness,(float)gate->numIO)));
					 for (int j = 0; j < columns; j++) {
						 for (int k = 0; k < columns; k++) {
							cout << "  (" << cuCrealf(gate->gateMatrix1[j+k*columns]) << "," << cuCimagf(gate->gateMatrix1[j+k*columns]) << ")";
						 }
						 cout << endl;
					 }
					 cout << endl;
*/
						for (int m = 0; m < name.length(); m++)
							gate->my_string += name.at(m);
						if (i_placement[u] > function)
							gate->my_string.insert(i_placement[u], "I");
						else 
							gate->my_string.insert(i_placement[u]+1, "I");
/*						if (i_placement[u] == 0)
							gate->my_string.insert(1, "I");
						else if (i_placement[u] == (gate->my_string.length())-1)
							gate->my_string.insert(i_placement[u], "I");
						else if (gate->my_string.at(i_placement[u]+1) != 'C' && gate->my_string.at(i_placement[u]+1) != 'C' )
							gate->my_string.insert(i_placement[u]+1, "I");
						else
							gate->my_string.insert(i_placement[u], "I");


*/
						found = false;
						for (int g = 0 ; g < (numofgates + counter); g++)
							if (gateArray[g]->my_string == gate->my_string)
								found = true;
						if (!found){
						gateArray[numofgates + counter] = new qGate;
						initGate(gateArray[numofgates + counter], gate->numIO, gate->valuedness, 1);
						gateArray[numofgates + counter] = ccGate(gate,
								gateArray[numofgates + counter]);
						gateArray[numofgates + counter]->Cost = 2;
						chcounter++;
						if (char(chcounter) == char('I') || chcounter == 112 || chcounter == 127)
							gateArray[numofgates + counter]->representation
							= char(++chcounter);
						else
							gateArray[numofgates + counter]->representation
							= char(chcounter);
/*

						 cout<<gateArray[numofgates + counter]->my_string<<"final result  after tensor product"<<endl;
					 columns = (int(pow((float)gateArray[numofgates + counter]->valuedness,(float)gateArray[numofgates + counter]->numIO)));
					 for (int j = 0; j < columns; j++) {
						 for (int k = 0; k < columns; k++) {
							 cout << "  (" << cuCrealf(gateArray[numofgates + counter]->gateMatrix1[j+k*columns]) << "," << cuCimagf(gateArray[numofgates + counter]->gateMatrix1[j+k*columns])<< ")";
						 }
						 cout << endl;
					 }
					 cout << endl;
*/
						counter++;
						}
//exit(0);
					}
				}
			}
			numofgates += counter;
		} else
			break;
	destroyGate(extgate);
	destroyGate(tempgate);
	destroyGate(gate);
	destroyGate(sgate);
//	cout << "done"<<endl;
	delete extgate, tempgate, gate, sgate;

}
/***********************************
 * helper functionnn
 *********************************/
char GA::loadGateArray(int valuedness, char chcounter){

	string s;
	int columns,counter,resnum,a,b, ycounter;
	char tempchar;
	string temp;
	char binary[100];
	char buff[100];
        complex<float> in = complex<float> (0., 0.);
	//bedining reading the gates
	in_stream >> numofgates;
	cout<<"Reading in input gates set: "<<numofgates<<endl;
	//char chcounter = charcounter;
	// get first gate
	for (int a = 0; a < numofgates; a++) {
	cout << " Writing Gates " << numofgates<<" "<< gateArray<<endl;
		gateArray[a] = new qGate;
	cout << " Writing Output " << numofgates<<endl;
		gateArray[a]->valuedness = valuedness;
		gateArray[a]->representation = char(chcounter++);
	cout << " Writing Output " << endl;
		in_stream >> gateArray[a]->numIO;
		in_stream >> gateArray[a]->Cost;
		initGate(gateArray[a], gateArray[a]->numIO,  gateArray[a]->valuedness, gateArray[a]->Cost);

		in_stream >> gateArray[a]->my_string;
		columns = (int(pow((float) valuedness,(float) gateArray[a]->numIO)));
		gateArray[a]->gateMatrix1 = new cuComplex[columns*columns];

		for (int j = 0; j < columns; j++) {
			for (int k = 0; k < columns; k++) {
				in_stream >> in;
				gateArray[a]->gateMatrix1[j+k*columns] = make_cuFloatComplex(real(in),imag(in));
			}
		}
		counter++;
	}


	finalGate = new qGate;
	finalIndividual = new Individual;
	initGate(finalGate, resultnum, valuedness, 1);
	if (measurement == 0) {
		// get the expected result matrix from the file
		in_stream >> resnum;
		in_stream >> measured_desired_output_records;
		if (measured_desired_output_records == 0) {
			columns = (int(pow((float) valuedness, (float) resnum)));
			for (int m = 0; m < columns; m++) {
				for (int n = 0; n < columns; n++) {
					in_stream >> in;
					finalGate->gateMatrix1[m+n*columns] = make_cuFloatComplex(real(in), imag(in));
				}
			}
		} else {
			
#ifdef __QMDD__
				temp = "";
				temp += ".version Target Individual\n.numvars ";
				sprintf(buff, "%d", resultnum);
				counter = 0;
				while(buff[counter] > 0)
					temp += buff[counter++];	
				temp += "\n.variables ";
				for (ycounter = 0; ycounter < resnum; ycounter++){
					sprintf(buff, "%d", ycounter);
					counter = 0;
					temp += 'w';	
					while(buff[counter] > 0)
						temp += buff[counter++];	
					temp += ' ';	
				}
				temp += "\n.inputs ";
				for (ycounter = 0; ycounter < resnum; ycounter++){
					sprintf(buff, "%d", ycounter);
					counter = 0;
					temp += 'w';	
					while(buff[counter] > 0)
						temp += buff[counter++];	
					temp += ' ';	
				}
				temp += "\n.outputs ";
				for (ycounter = 0; ycounter < resnum; ycounter++){
					sprintf(buff, "%d", ycounter);
					counter = 0;
					temp += 'w';	
					while(buff[counter] > 0)
						temp += buff[counter++];	
					temp += ' ';	
				}
				temp += "\n.begin\n";
#endif
				dcMatrix(finalGate, resnum);
				columns = (int(pow((float) valuedness, (float) resnum)));
				for (int m = 0; m < measured_desired_output_records; m++) {
					s = "";
					in_stream >> a;
					in_stream >> tempchar;
					in_stream >> b;
					dec2bin(b, binary, resnum);
					cout <<binary<<";; "<< a << " " << tempchar << " " << b;
					in_stream >> tempchar;
					in_stream >> in;
					cout << " " << tempchar << " " << in << endl;
					finalGate->gateMatrix1[a+b*columns] = make_cuFloatComplex(real(in),imag(in));
					counter = 0;
#ifdef __QMDD__
					while(binary[counter] > 0)
						temp += binary[counter++];
					temp += "\n";
#endif
				}

				counter = 0;
#ifdef __QMDD__
				temp += ".end\n";
				while (counter < temp.size())
					finalIndividual->rev_lib_string += temp.at(counter++);	
#endif
			}
		cout << temp<<endl;
	}

}
/***********************************
 * helper functionnn
 *********************************/
int GA::getGate(char b) {
	for (int a = 0; a < numofgates; a++)
		if (gateArray[a]->representation == b)
			return a;
	return '0';
}
/****************************************
 * helper function to choose mutation
 ***************************************/
void GA::applyMutation(int indv, bool repair) {
	if (mutation != 0)
		applyIMutationP(indv, repair);
	else
		applyRMutationP(indv, repair);
}
/****************************************
 * helper function to choose mutation
 ***************************************/
void GA::applyCrossover(int indv1, int indv2) {
	if (crossover == 0)
		apply1CrossoverP(indv1,indv2);
	else
		apply2CrossoverP(indv1,indv2);
}

/*******************************************
 * helper function
 *******************************************/
int GA::getqGate(string b) {
	for (int a = 0; a < numofgates; a++) {
		if (b.length() == gateArray[a]->my_string.length()) {
			//cout<<a<<"; "<<b<<", "<<gateArray[a]->my_string<<"   "<<(gateArray[a]->my_string).compare(0, b.length(), b) <<endl;
			if ((gateArray[a]->my_string).compare(0,
					gateArray[a]->my_string.length(), b) >= 0)
				return a;
		}
	}
	return -1;
}
/***********************************
 * apply the mutation on the individual of the current population
 * take one gate and change it to something else
 *********************************/
void GA::applyRMutationP(int indv, bool repair) {

	Individual *I = population[indv];

	unsigned int proba, proba2, ios;
	char gate;
	qGate temp;
	proba = (int) (1 - (float(1.0 * rand() / ((float) RAND_MAX + 1.0))));

	if (proba <= proba_Mutation) {
		proba = rand() % 1;
		//change only the phase of the circuit
		if (phase == 1) {
			if (proba == 0) {
				I->phase = generatePhase();
				return;
			}
		}
//			cout<<I->my_string.length()<<", "<<I->my_string<<endl;

		do {
			proba2 = (rand() % (I->my_string.length() - 1));
			if (proba2 < 0)
				proba2 = 0;
			if (proba2 >= I->my_string.length() - 1)
				proba2 = I->my_string.length() - 1;
			gate = I->my_string.at(proba2);
		} while (gate == 'p');

		ios = gateArray[getGate(gate)]->numIO;
		
                do {
                        proba = (rand() % (stringArray[ios-1].numberGates));
                        if (proba < 0)
                                proba = 0;
                        gate = stringArray[ios-1].nameGate[proba];
                } while (gate == 'p');
	
		I->my_string.replace(proba2, 1, 1, gate);

	}
	if (repair)
		repairCircuit(indv);
}
/***********************************
 *apply the mutation on the individual of the current population
 *andfor every segment apply the mutation operation
 *********************************/
void GA::applyIMutationP(int indv, bool repair) {

	Individual *I = population[indv];
	int proba, proba2, temp0, iterations, ios;
	char gate;
	qGate temp;
	complex<float> tempc;

	temp0 = 0;
	for (unsigned int m = 0; m < I->my_string.size(); m++) {
		if (I->my_string.at(m) == 'p')
			temp0++;
	}

	I->segmentNumber = temp0;
	iterations = getGatesfromString(I->my_string);
	//gate wise mutation
	for (int a = 0; a < iterations; a++) {

		//generate probabilty for the next index
		proba = (int) (1 - (float(1.0 * rand() / ((float) RAND_MAX + 1.0))));

		//try iterator
		if (proba <= proba_Mutation) {
			//three parameters to possibly mutate
			proba = rand() % 3;
			//change only the phase of the circuit
			switch (proba) {
			case 3:
				//modify the global phase
				I->phase = generatePhase();
				break;
			case 2:
				//mutate the n phases of the circuit
				proba2 = rand() % iterations;
				temp0 = 0;
				while (1) {
					if (temp0 < I->my_string.length()) {
						if (I->my_string[temp0] != 'p')
							proba2--;
					} else
						break;

					if (proba2 == 0 && I->my_string[temp0] != 'p') {
						I->phases[proba2] = generatePhase();
						break;
					}
					temp0++;
				}
				break;
			case 1:

				do {
					proba2 = (rand() % (I->my_string.length() - 1));
					if (proba2 < 0)
						proba2 = 0;
					gate = I->my_string.at(proba2);
				} while (gate == 'p');

				ios = gateArray[getGate(gate)]->numIO;
		                do {
                		        proba = (rand() % (stringArray[ios-1].numberGates));
		                        if (proba < 0)
		                                proba = 0;
					gate = stringArray[ios-1].nameGate[proba];
		                } while (gate == 'p');
	
				I->my_string.replace(proba2, 1, 1, gate);
				break;
			}
		}

	}
	if (repair)
		repairCircuit(indv);
}/***********************************
 * apply the mutation on the individual of the current population
 * take one gate and change it to something else
 *********************************/
void GA::applyRMutation(int indv, bool repair) {

	Individual *I = population[indv];

	unsigned int proba, proba2;
	char gate;
	qGate temp;
	proba = (int) (1 - (float(1.0 * rand() / ((float) RAND_MAX + 1.0))));

	if (proba <= proba_Mutation) {
		proba = rand() % 1;
		//change only the phase of the circuit
		if (phase == 1) {
			if (proba == 0) {
				I->phase = generatePhase();
				return;
			}
		}

		do {
			proba2 = (rand() % (I->my_string.length() - 1));
			if (proba2 < 0)
				proba2 = 0;
			gate = I->my_string.at(proba2);
		} while (gate != 'p' || proba2 > (I->my_string.length() - 3));

		if (I->my_string.at(proba2 + 1) == 'p')
			proba2++;
		int proba3 = I->my_string.find("p", proba2 + 1);

		I->my_string.erase(proba2, (proba3 - proba2 + 1));

		//tricky part is to keep the width of the circuit constant
		basic_string<char> tempS;
		int V = 0;
		do {
			temp = *gateArray[rand() % numofgates];
			if (temp.numIO <= (I->ioNumber - V) && temp.numIO > 0) {
				V += temp.numIO;
				tempS.insert(tempS.length(), 1, temp.representation);
			}
		} while (V != I->ioNumber);
		tempS.insert(0, "p");
		tempS.insert(tempS.length(), "p");

		I->my_string.insert(proba2, tempS);

		if (repair)
			repairCircuit(indv);
	}//add here parts moving gate location in the circuit -- this must be done in order to do circuits with controls such
	//such as I41_zz do max for circuits with 5 variables.
	//cout <<"mutated string: "<<I->my_string<<endl;
}
/***********************************
 *apply the mutation on the individual of the current population
 *andfor every segment apply the mutation operation
 *********************************/
void GA::applyIMutation(int indv, bool repair) {

	Individual *I = population[indv];
	int proba, proba2, temp0, iterations;
	char gate;
	qGate temp;
	complex<float> tempc;

	temp0 = 0;
	for (unsigned int m = 0; m < I->my_string.size(); m++) {
		if (I->my_string.at(m) == 'p')
			temp0++;
	}

	I->segmentNumber = temp0;
	iterations = getGatesfromString(I->my_string);
	//gate wise mutation
	for (int a = 0; a < iterations; a++) {

		//generate probabilty for the next index
		proba = (int) (1 - (float(1.0 * rand() / ((float) RAND_MAX + 1.0))));

		//try iterator
		if (proba <= proba_Mutation) {
			//three parameters to possibly mutate
			proba = rand() % 3;
			//change only the phase of the circuit
			switch (proba) {
			case 3:
				//modify the global phase
				I->phase = generatePhase();
				break;
			case 2:
				//mutate the n phases of the circuit
				proba2 = rand() % iterations;
				temp0 = 0;
				while (1) {
					if (temp0 < I->my_string.length()) {
						if (I->my_string[temp0] != 'p')
							proba2--;
					} else
						break;

					if (proba2 == 0 && I->my_string[temp0] != 'p') {
						I->phases[proba2] = generatePhase();
						break;
					}
					temp0++;
				}
				break;
			case 1:
				//mutate the string - single gate
				do {
					proba2 = (rand() % I->my_string.length()) - 1;
					if (proba2 < 0)
						proba2 = 0;
					gate = I->my_string.at(proba2);
				} while (gate != 'p' || proba2 > (I->my_string.length() - 3));

				if (I->my_string.at(proba2 + 1) == 'p')
					proba2++;
				int proba3 = I->my_string.find("p", proba2 + 1);

				I->my_string.erase(proba2, (proba3 - proba2 + 1));

				//tricky part is to keep the width of the circuit constant
				basic_string<char> tempS;
				int V = 0;
				do {
					temp = *gateArray[rand() % numofgates];
					if (temp.numIO <= (I->ioNumber - V) && temp.numIO > 0) {
						V += temp.numIO;
						tempS.insert(tempS.length(), 1, temp.representation);
					}
				} while (V != I->ioNumber);
				tempS.insert(0, "p");
				tempS.insert(tempS.length(), "p");
				I->my_string.insert(proba2, tempS);
				break;
			}
		}

	}
	if (repair)
		repairCircuit(indv);
}
/**************************
 * check the length
 ***************************/
void GA::repairCircuit(int a) {

	int counter = 0;
	Individual *I = population[a];

	if ((int) I->my_string.length() > 0) {
		for (int A = 0; A < (int) I->my_string.length(); A++)
			if (I->my_string.at(A) == 'p')
				counter++;
		//compare cicuits to the minimum required size and twice the maximum size
		if (counter < mincircuitsize  || counter > (segments+mincircuitsize) ) {
			//reinitialize the circuit
			I->segmentNumber = segments;
			makeIndvString(I);
			generatePhases(I);
		}
	} else {
		I->segmentNumber = segments;
		makeIndvString(I);
		generatePhases(I);
	}

}
/*******************************************
 *repair the circuit
 *check the width
 ********************************************/
void GA::repairForceCircuit(int a) {
	int begin, end, counter = 0;
	Individual *I = population[a];
	qGate temp;
	//one pointer for the beginning of the segment
	begin = 0;
	//onepointer for the current position
	end = 1;

	//we put separators on the end and on the beginning
	if (I->my_string.at(begin) != 'p')
		I->my_string.insert(I->my_string.at(begin), 1, 'p');
	if (I->my_string.at(I->my_string.length() - 1) != 'p')
		I->my_string.insert(I->my_string.at(I->my_string.length() - 1), 1, 'p');
	//detect wrongly placed separators
	cout << "repair 0 in: " << I->my_string << endl;
	return;
	if (I->my_string.size() > segments * I->ioNumber) {
		end = I->my_string.find('p', (segments * I->ioNumber) / 2);
		if (end < I->my_string.size()) {
			if (I->my_string.at(end + 1) == 'p')
				end++;
			I->my_string.erase(end, I->my_string.length() - end + 1);
			return;
		}
		return;
	} else
		return;
	counter = I->my_string.length() - 2;
	//detect wrongly placed separators
	cout << "repair 1 in: " << I->my_string << endl;
	while (true) {
		//if we are between 1 and before-last element
		if (counter > 0 && counter < I->my_string.length() - 1) {
			if (I->my_string.at(counter) == 'p')
				//if this, previous and next are all three 'p' we can erase it
				if (I->my_string.at(counter + 1) == 'p') {
					if (I->my_string.at(counter - 1) == 'p')
						I->my_string.erase(counter, 1);
				} else if (I->my_string.at(counter + 1) != 'p')
					//if this is 'p', but previous and next are all gates, add one
					if (I->my_string.at(counter - 1) != 'p')
						I->my_string.insert(counter, 1, 'A');
		} else
			break;

		counter--;
	}
	//one pointer for the beginning of the segment
	cout << "repair 1 out: " << I->my_string << endl;
	begin = 0;
	//onepointer for the current position
	end = 1;
	//counter of the wires in a segment
	counter = 0;

	//check every segment of the string circuit
	while (true) {

		if (I->my_string.at(begin) != 'p')
			I->my_string.erase(begin, 1);
		else if (I->my_string.at(end) == 'p') {
			//if the current end pointer is NOT on gate
			if ((counter != resultnum)) {
				//if the resulting wirecount is bigger than desired
				if (counter > resultnum)
					//erase if the segment is too fat
					I->my_string.erase(begin, end - begin + 1);
				else {
					//add wires if the given segment is too short on wires
					do {
						temp = *gateArray[0];
						//					if (temp.numIO <= (resultnum - counter))
						counter += temp.numIO;
						I->my_string.insert(I->my_string.at(end - 1), 1, 'A');
					} while (counter < I->ioNumber);
				}
				//if the end is pointing to p and we got correct count -- increment
			} else {
				begin = end + 1;
				end = I->my_string.find('p', begin + 1);
				counter = 0;
				cout << begin << " " << end << endl;

			}
			//if the current pointer - end - is pointing on a gate
		} else {
			counter += gateArray[getGate(I->my_string.at(end))]->numIO;
			end++;
		}
	}
	cout << "repair 2 out: " << I->my_string << endl;

}
/************************************************
 *Restrictions:
 *1. replace gate with same number of I/O
 *2. replace one gate with n I/O with any combination of smaller gates with
 *same number of I/O
 *3. conserve strict correspondence of connections
 *void operator=(const class bla & other) { mymember=other.mymember; }
 *************************************************/
void GA::apply1CrossoverP(int ind1, int ind2) {
	//	apply rules and make crossover
	int proba1, proba2, segments_a, segments_b, o_e;
	cuComplex temp[MAXNUMOFGATES];
	Individual *I = new_population[ind1];
	Individual *J = new_population[ind2];
	basic_string<char> temp1;
	basic_string<char> temp2;
	int index1, index2, iterations1, iterations2;

	//individual 1
	do {
		proba1 = (rand() % (I->my_string.size()));
	} while (proba1 >= I->my_string.length() - 1 || I->my_string.at(proba1)
			!= 'p' || proba1 <= 0);

	//cout<<proba1;

	if ((I->my_string.at(proba1)) == 'p')
		if ((I->my_string.at(proba1 + 1)) == 'p')
			proba1 += 1;

	temp1 = I->my_string.substr(proba1);

	segments_a = 0;
	o_e = 0;
	for (int k = 0; k < I->my_string.size(); k++){
		if (k > proba1)
			break;
		if (I->my_string.at(k) == 'p'){
			if (o_e / 2 == 0 && o_e > 0)
				segments_a++;
			o_e++;
		}
	}
	//individual 2
	for (int k = 0; k < J->my_string.size(); k++){
		if (I->my_string.at(k) == 'p'){
			if (o_e / 2 == 0 && o_e > 0)
				segments_b++;
			o_e++;
		}
		if (segments_b == segments_a){
			break;
			proba2 = k;
		}
	}

	if ((J->my_string.at(proba2)) == 'p')
		if ((J->my_string.at(proba2 + 1)) == 'p')
			proba2 += 1;

	temp2 = J->my_string.substr(proba2);

	iterations1 = getGatesfromString(I->my_string);
	iterations2 = getGatesfromString(J->my_string);

	//get the phase size
	//we need both indexes in the array of phases
	index1 = 0;
	index2 = 1;
	for (int a = 0; a < I->my_string.length(); a++)
		if (a < proba1) {
			if (I->my_string[a] != 'p') {
				index1 += 1;
			}
		} else {
			if (I->my_string[a] != 'p') {
				temp[a - proba1] = make_cuFloatComplex(cuCrealf(I->phases[a]),cuCimagf(I->phases[a]));
			}
		}

	for (int a = 0; a < proba2; a++)
		if (J->my_string[a] != 'p')
			index2 += 1;

	//	swap the phase sequences
	for (int a = index2; a < (iterations2 - index2); a++)
		I->phases[index1 + a - index2] = J->phases[a];

	for (int a = index1; a < (iterations1 - index1); a++)
		J->phases[index2 + a - index1] = make_cuFloatComplex(cuCrealf(temp[a- index1]), cuCimagf(temp[a - index1]));

	if (temp1.length() >= 3 && temp2.length() >= 3) {
		I->my_string.replace(proba1, temp1.length(), temp2, 0, temp2.length());
		J->my_string.replace(proba2, temp2.length(), temp1, 0, temp1.length());
	}

}

/************************************************
 *Restrictions:
 *1. replace gate with same number of I/O
 *2. replace one gate with n I/O with any combination of smaller gates with
 *same number of I/O
 *3. conserve strict correspondence of connections
 *void operator=(const class bla & other) { mymember=other.mymember; }
 *************************************************/
void GA::apply1Crossover(int ind1, int ind2) {
	//	apply rules and make crossover
	int proba1, proba2;
	cuComplex temp[MAXNUMOFGATES];
	Individual *I = new_population[ind1];
	Individual *J = new_population[ind2];
	basic_string<char> temp1;
	basic_string<char> temp2;
	int index1, index2, iterations1, iterations2;

	//individual 1
	do {
		proba1 = (rand() % (I->my_string.size()));
	} while (proba1 >= I->my_string.length() - 1 || I->my_string.at(proba1)
			!= 'p');

	//cout<<proba1;

	if (proba1 <= 0)
		temp1 = I->my_string.substr(proba1);
	else {
		if ((I->my_string.at(proba1)) == 'p')
			if ((I->my_string.at(proba1 + 1)) == 'p')
				proba1 += 1;

		temp1 = I->my_string.substr(proba1);
	}

	//individual 2

	do {
		proba2 = (rand() % (J->my_string.size()));
	} while (proba2 >= J->my_string.length() - 1 || J->my_string.at(proba2)
			!= 'p');

	//cout<<" "<<proba2<<endl;
	if (proba2 <= 0)
		temp2 = J->my_string.substr(proba2);
	else {
		if ((J->my_string.at(proba2)) == 'p')
			if ((J->my_string.at(proba2 + 1)) == 'p')
				proba2 += 1;

		temp2 = J->my_string.substr(proba2);
	}

	iterations1 = getGatesfromString(I->my_string);
	iterations2 = getGatesfromString(J->my_string);

	//get the phase size
	//we need both indexes in the array of phases
	index1 = 0;
	index2 = 1;
	for (int a = 0; a < I->my_string.length(); a++)
		if (a < proba1) {
			if (I->my_string[a] != 'p') {
				index1 += 1;
			}
		} else {
			if (I->my_string[a] != 'p') {
				temp[a - proba1] = make_cuFloatComplex(cuCrealf(I->phases[a]),cuCimagf(I->phases[a]));
			}
		}

	for (int a = 0; a < proba2; a++)
		if (J->my_string[a] != 'p')
			index2 += 1;

	//	swap the phase sequences
	for (int a = index2; a < (iterations2 - index2); a++)
		I->phases[index1 + a - index2] = J->phases[a];

	for (int a = index1; a < (iterations1 - index1); a++)
		J->phases[index2 + a - index1] = make_cuFloatComplex(cuCrealf(temp[a - index1]), cuCimagf(temp[a - index1]));

	if (temp1.length() >= 3 && temp2.length() >= 3) {
		I->my_string.replace(proba1, temp1.length(), temp2, 0, temp2.length());
		J->my_string.replace(proba2, temp2.length(), temp1, 0, temp1.length());
	}

}
/*******************************************
 * two pint crossover but no phases swap yet
 ********************************************/
void GA::apply2CrossoverP(int ind1, int ind2) {
	//apply rules and make crossover
	//first it is used to store segment number
	int proba[4];
	basic_string<char> temp1;
	basic_string<char> temp2;
	Individual *I = new_population[ind1];
	Individual *J = new_population[ind2];
	int a, b, c, d, n, count;
	int pos1, pos2;

	if (I->ioNumber == J->ioNumber) {
		 //get string cuting points
		do {
			//the number of segments to swap on individual 1
			proba[0] = (rand() % (I->segmentNumber / 2));
		} while (proba[0] <= 0 && proba[0] > J->segmentNumber / 2);
		a = proba[0];

		do {
			//the location of the cut
			proba[1] = (rand() % (I->segmentNumber));
		} while ((proba[1] + proba[0]) >= I->segmentNumber-1 || (proba[1] + proba[0]) >= J->segmentNumber-1);
		b = proba[1];

		n = 0;
		count = 0;
		temp1 = "";
		while (count < (b + a)) {

			if (I->my_string.at(n) == 'p')
				count++;
			//the string starts between the indexes of the segments
			if (count > b) {
				temp1 += I->my_string.at(n);
			}
			n++;
		}
		pos1 = I->my_string.find(temp1, 0);

		n = 0;
		count = 0;
		temp2 = "";
		while (count < (a + b)) {

			if (J->my_string.at(n) == 'p')
				count++;
			//the string starts between the indexes of the segments
			if (count > b) {
				temp2 += J->my_string.at(n);
			}
			n++;
		}
		pos2 = J->my_string.find(temp2, 0);

		I->my_string.replace(pos1, temp1.length(), temp2, 0, temp2.length());
		J->my_string.replace(pos2, temp2.length(), temp1, 0, temp1.length());
		

	}
}
/*******************************************
 * two pint crossover but no phases swap yet
 ********************************************/
void GA::apply2Crossover(int ind1, int ind2) {
	//apply rules and make crossover
	//first it is used to store segment number
	int proba[4];
	basic_string<char> temp1;
	basic_string<char> temp2;
	Individual *I = new_population[ind1];
	Individual *J = new_population[ind2];
	int a, b, c, d, n, count;
	int pos1, pos2;

	if (I->ioNumber == J->ioNumber) {
		 //get string cuting points
		do {
			//the number of segments to swap on individual 1
			proba[0] = (rand() % (I->segmentNumber / 2));
		} while (proba[0] <= 0);
		a = proba[0];

		do {
			//the location of the cut
			proba[1] = (rand() % (I->segmentNumber));
		} while ((proba[1] + proba[0]) >= I->segmentNumber-1);
		b = proba[1];

		do {
			//the number of segments to swap on individual 2
			proba[2] = (rand() % (J->segmentNumber / 2));
		} while (proba[2] <= 0);
		c = proba[2];

		do {
			//the location of the cut
			proba[3] = (rand() % (J->segmentNumber));
		} while ((proba[3] + proba[2]) >= J->segmentNumber-1);
		d = proba[3];
		n = 0;
		count = 0;
		temp1 = "";
		while (count < (b + a)) {

			if (I->my_string.at(n) == 'p')
				count++;
			//the string starts between the indexes of the segments
			if (count > b) {
				temp1 += I->my_string.at(n);
			}
			n++;
		}
		pos1 = I->my_string.find(temp1, 0);

		n = 0;
		count = 0;
		temp2 = "";
		while (count < (c + d)) {

			if (J->my_string.at(n) == 'p')
				count++;
			//the string starts between the indexes of the segments
			if (count > d) {
				temp2 += J->my_string.at(n);
			}
			n++;
		}
		pos2 = J->my_string.find(temp2, 0);

		I->my_string.replace(pos1, temp1.length(), temp2, 0, temp2.length());
		J->my_string.replace(pos2, temp2.length(), temp1, 0, temp1.length());
		

	}
}
/************************************************
 * replicate the population according to the settings
 ************************************************/
void GA::applyReplication() {
	if (pareto > 0)
		applyParetoReplication();
	else
		applySingleObjReplication();
}

void GA::applySingleObjReplication() {
	float fitnessSum = 0;
	float fitnessavg = 0;
	float first, second, tempsum, proba, best;
	int indv1, indv2;

	for (int i = 0; i < populationNumber; i++)
		fitnessSum += population[i]->fitness;

	fitnessavg /= populationNumber;
	int counter = 0;
	bool condition = false;
	while (!condition) {
		if (replicator == 0) {
			//roulette wheel
			//select only individuals with fitness > threshold
			if (threshold > 0)
				do {
					indv1 = rand() % (populationNumber - 1);
					indv2 = rand() % (populationNumber - 1);
				} while (population[indv1]->fitness < threshold
						&& population[indv2]->fitness < threshold);
			else {
				first = (1 - (float(1.0 * rand() / ((float) RAND_MAX + 1.0))))
				* fitnessSum;
				second = (1 - (float(1.0 * rand() / ((float) RAND_MAX + 1.0))))
				* fitnessSum;
			}
		} else {
			//stochastic universal sampling
			//select only individuals with fitness > threshold
			if (threshold > 0)
				do {
					indv1 = rand() % (populationNumber - 1);
					indv2 = rand() % (populationNumber - 1);
				} while (population[indv1]->fitness < threshold
						&& population[indv2]->fitness < threshold);
			else {
				first = ((1 - (float(1.0 * rand() / ((float) RAND_MAX + 1.0))))
						* fitnessSum) / 2;
				second = fitnessSum - first;
			}
		}

		if (threshold == 0) {
			tempsum = 0;
			for (int c = 0; c < populationNumber; c++) {
				tempsum += population[c]->fitness;
				if (tempsum >= first) {
					indv1 = c;
					break;
				}
			}

			tempsum = 0;
			for (int d = 0; d < populationNumber; d++) {
				tempsum += population[d]->fitness;
				if (tempsum >= second) {
					indv2 = d;
					break;
				}
			}
		}

		if (counter == populationNumber - 1) {
			setIndv(new_population[counter], population[indv1]);
			counter++;
		} else {
			setIndv(new_population[counter], population[indv1]);
			setIndv(new_population[counter + 1], population[indv2]);

			proba = 1 - (float(1.0 * rand() / ((float) RAND_MAX + 1.0)));
			if (proba < proba_Crossover && indv1 != indv2) {
				applyCrossover(counter, counter+1);
			}

			counter += 2;
		}
		if (counter >= populationNumber - 1)
			condition = true;


	}

	if (elitism > 0){

		best = population[0]->fitness;
		indv1 = 0;
		for (int t = 1; t < populationNumber; t++) {
			if (population[t]->fitness > best) {
				best = population[t]->fitness;
				indv1 = t;
			}
		}
		setIndv(population[0], population[indv1]);
		for (int i = 0; i < populationNumber-1; i++) {
			setIndv(population[i+1], new_population[i]);
		}

	} else {
		for (int i = 0; i < populationNumber; i++) {
			setIndv(population[i], new_population[i]);
		}
	}

}
/*************************************************
 * Pareto replication with initial settings
 *************************************************/
void GA::applyParetoReplication() {
	float proba;
	int indv1, indv2, tempsum, fitnessSum, first, second;
	int counter = 0;
	bool condition = false;

	for (int i = 0; i < populationNumber; i++)
		population[i]->Rank = -1;
	int temp_rank;

	for (int i = 0; i < populationNumber; i++) {
		temp_rank = 0;
		for (int j = 0; j < populationNumber; j++) {
			if (population[i]->Rank <= -1) {
				if ((alpha1 * population[i]->Error) <= population[j]->Error
						&& (beta1 * population[i]->Cost) <= population[j]->Cost)
					temp_rank++;
			}
		}
		population[i]->Rank = temp_rank;
	}

	fitnessSum = 0;
	for (int i = 0; i < populationNumber; i++) {
		fitnessSum += population[i]->Rank;
	}

	while (!condition) {
		if (replicator == 0) {
			//roulette wheel
			if (threshold > 0)
				do {
					indv1 = rand() % (populationNumber - 1);
					indv2 = rand() % (populationNumber - 1);
				} while (population[indv1]->fitness < threshold
						&& population[indv2]->fitness < threshold);
			else {
				first = (int) (1 - (float(1.0 * rand() / ((float) RAND_MAX
						+ 1.0)))) * fitnessSum;
				second = (int) (1 - (float(1.0 * rand() / ((float) RAND_MAX
						+ 1.0)))) * fitnessSum;
			}
		} else {
			//stochastic uniersal sampling
			if (threshold > 0)
				do {
					indv1 = rand() % (populationNumber - 1);
					indv2 = rand() % (populationNumber - 1);
				} while (population[indv1]->fitness < threshold
						&& population[indv2]->fitness < threshold);
			else {
				first = (int) ((1 - (float(1.0 * rand() / ((float) RAND_MAX
						+ 1.0)))) * fitnessSum) / 2;
				second = fitnessSum - first;
			}
		}

		if (threshold == 0) {
			tempsum = 0;
			for (int c = 0; c < populationNumber; c++) {
				tempsum += population[c]->Rank;
				if (tempsum >= first) {
					indv1 = c;
					break;
				}
			}

			tempsum = 0;
			for (int d = 0; d < populationNumber; d++) {
				tempsum += population[d]->Rank;
				if (tempsum >= second) {
					indv2 = d;
					break;
				}
			}
		}

		if (counter == populationNumber - 1) {
			setIndv(new_population[counter], population[indv1]);
			counter++;
		} else {

			setIndv(new_population[counter], population[indv1]);
			setIndv(new_population[counter + 1], population[indv2]);

			proba = 1 - (float(1.0 * rand() / ((float) RAND_MAX + 1.0)));
			if (proba < proba_Crossover && indv1 != indv2)
				applyCrossover(counter, counter+1);
			counter += 2;
		}
		if (counter >= populationNumber - 1)
			condition = true;
	}

	if (elitism > 0){

		best = population[0]->fitness;
		first = 0;
		for (int t = 1; t < populationNumber; t++) {
			if (population[t]->fitness > best) {
				best = population[t]->fitness;
				first = t;
			}
		}
		setIndv(population[0], population[first]);
		for (int i = 0; i < populationNumber-1; i++) {
			setIndv(population[i+1], new_population[i]);
		}

	} else {
		for (int i = 0; i < populationNumber; i++) {
			setIndv(population[i], new_population[i]);
		}
	}

}
/*************************************************
 * copy one individual to another one
 *************************************************/
void GA::setIndv(Individual *indv1, Individual *indv2) {
	indv1->fitness = indv2->fitness;
	indv1->ioNumber = (int) indv2->ioNumber;
	indv1->segmentNumber = (int) indv2->segmentNumber;
	indv1->Cost = (int) indv2->Cost;
	indv1->Error = (float) indv2->Error;
	indv1->Groups = (int) indv2->Groups; // for share fitness calcualtion
	indv1->Rank = (int) indv2->Rank;
	indv1->valuedness = indv2->valuedness;
	indv1->my_string.erase();
	indv1->my_string = string("");
	indv1->rev_lib_string = string("");
	indv1->phase = indv2->phase;
	for (int a = 0; a < getGatesfromString(indv2->my_string); a++)
		indv1->phases[a] = indv2->phases[a];
	for (unsigned int a = 0; a < indv2->my_string.size(); a++)
		indv1->my_string += indv2->my_string[a];
	for (unsigned int a = 0; a < indv2->rev_lib_string.size(); a++)
		indv1->rev_lib_string += indv2->rev_lib_string[a];
}
/*************************************************
 * this cuts the amount of segments to 200
 *************************************************/
void GA::reduceSegNumber(int a) {
	int counter, n;

	counter = 0;
	n = 0;

	if (population[a]->ioNumber > 1) {
		while (n < (int) population[a]->my_string.length()) {
			if (population[a]->my_string.at(n) == 'p')
				counter++;
			if (counter >= 200) {
				population[a]->my_string.erase(n,
						population[a]->my_string.length());
				break;
			}
			n++;
		}
	}
}
/**************************
 * this works only if the circuits are normalized
 * we generate the Individual from gates and replace them s the
 * n last one in the population.
 * the gates are translated to normal circuit using the
 * permutativeArray parents
 ***************************/
void GA::injectNormalCircuits(string **refs) {
	int counter = 0;
	int count = 5;
	int tcounter = 0;

	if (refs == NULL)
		return;
	cout << "injecting circuits back to GA" << endl;
	while (true) {
		if (counter >= count - 1)
			break;
		else if (refs[counter] == NULL || refs[counter]->size() <= 2)
			break;
		counter++;
	}
	for (int a = 0; a < counter; a++)
		cout << "* " << refs[a] << endl;

	for (int a = counter - 1; a >= 0; a--) {
		cout << "* " << a << endl;
		cout << "* " << refs[a] << endl;
		delete (population[populationNumber - a]);
		population[populationNumber - a] = new Individual();
		population[populationNumber - a]->my_string = string("");
		cout << "* " << refs[a] << "  " << refs[a]->size() << endl;
		for (unsigned int b = 0; b < refs[a]->size(); b++) {
			tcounter = 0;
			//fetch the tranlsated string from tbe permutative array
			while (permutativeArray[tcounter]->representation != refs[a]->at(b))
				tcounter++;
			cout << "* " << permutativeArray[tcounter]->my_string << endl;
			population[populationNumber - a]->my_string += 'p'
				+ permutativeArray[tcounter]->my_string + 'p';
		}
		//trivial phase
		population[populationNumber - a]->phase = make_cuFloatComplex(0, 1);
		population[populationNumber - a]->ioNumber = finalGate->numIO;
		cout << "<*> " << population[populationNumber - a]->my_string << endl;
		if (measurement > 0)
			doMeasureFitness(population[populationNumber - a], true);
		else
			doMatrixFitness(population[populationNumber - a], true);
		cout << "<> " << population[populationNumber - a]->my_string << endl;
	}
}
/*******************************************************
 *this works only if the circuits are normalized
 *we generate the Individual from gates and replace them s the
 *n last one in the population.
 *the gates aretranslated to normal circuit using the
 *permutativeArray parents
 *******************************************************/
void GA::injectNormalSegments(string **refs) {
	int counter = 0;
	int count = 5;
	int tcounter = 0;

	if (refs == NULL)
		return;
	cout << "injecting circuits back to GA" << endl;
	//check if there are valid solutions
	while (true) {
		if (counter >= count - 1)
			break;
		else if (refs[counter] == NULL || refs[counter]->size() <= 2)
			break;
		counter++;
	}
	//print obtained circuits
	for (int a = 0; a < counter; a++)
		cout << "* " << refs[a] << endl;

	//add all solutions into new indivduals and replace them in the GA
	for (int a = counter - 1; a >= 0; a--) {
		cout << "* " << a << endl;
		cout << "* " << refs[a] << endl;
		//erase the first n that are the smallest ones.
		delete (population[populationNumber - a]);
		population[populationNumber - a] = new Individual();
		population[populationNumber - a]->my_string = string("");
		cout << "* " << refs[a] << "  " << refs[a]->size() << endl;
		for (unsigned int b = 0; b < refs[a]->size(); b++) {
			tcounter = 0;
			//fetch the tranlsated string from tbe permutative array
			while (permutativeArray[tcounter]->representation != refs[a]->at(b))
				tcounter++;
			cout << "* " << permutativeArray[tcounter]->my_string << endl;
			//generate teh circuit string in the individual
			population[populationNumber - a]->my_string += 'p'
				+ permutativeArray[tcounter]->my_string + 'p';
		}
		population[populationNumber - a]->ioNumber = finalGate->numIO;
		cout << "<*> " << population[populationNumber - a]->my_string << endl;
		//evaluate the added circuit
		if (measurement > 0)
			doMeasureFitness(population[populationNumber - a], true);
		else
			doMatrixFitness(population[populationNumber - a], true);
		cout << "<> " << population[populationNumber - a]->my_string << endl;
	}
}
/*************************************************
 * Returns the aray of gates generated as single gate per parallel segments
 * All gates have thus size CircuitSize and can be simply swapped
 *************************************************/
qGate** GA::getPermutativeCircuits() {

	qGate *workgate, *multigate, *multigate2, *tempgate;
	string names = "";
	int gatecounter = 0;
	int c = 0;
	int rep = 0;
	int columns;
	multigate = new qGate;
	initGate(multigate, resultnum, gateArray[0]->valuedness, 1);
	tempgate = new qGate;
	initGate(tempgate, resultnum, gateArray[0]->valuedness, 1);
	string s = bestIndv->my_string;
	for (int m = 0; m < MAXNUMOFGATES; m++)
		permutativeArray[m] = NULL;
	//cout<<"starting generating: "<<s<<"  "<<endl;
	for (unsigned int a = 0; a < s.length(); a++) {
		//cout<<"starting generating "<<s.at(a)<<"  "<<endl;
		rep = 0;
		//we found a real gate
		if (s.at(a) != 'p' && s.at(a) != 'A') {
			//cout<<"starting generating 0"<<endl;
			for (int b = 0; b < numofgates; b++) {
				//remove the separators and the wire gates
				if (gateArray[b]->representation == s.at(a)) {
					//check wether it was already added
					for (unsigned int m = 0; m < names.size(); m++)
						if (names.at(m) == s.at(a))
							rep = 100;
					workgate = gateArray[b];
				}
			}
			//cout<<"starting generating 1"<<endl;
			//if yes do not add this gate again
			if (rep == 0) {
				//add the gate
				names += workgate->representation;
				//here comes the transformation for non normal gate to normal ones
				//if the given gate is smaller than the final one
				//cout<<"starting generating 2"<<endl;
				if (finalGate->numIO > workgate->numIO) {
					//generate as many new gates -- normal ones
					//cout<<"starting generating 3"<<endl;
					for (int t = 0; t <= finalGate->numIO - workgate->numIO; t++) {
						if (permutativeArray[gatecounter] != NULL)
							delete (permutativeArray[gatecounter]);
						permutativeArray[gatecounter] = new qGate;
						initGate(permutativeArray[gatecounter], resultnum, finalGate->valuedness, 1);
						permutativeArray[gatecounter]->my_string = string("");
						//how many W's we have to add to create a shifted one
						for (int j = 0; j <= (finalGate->numIO
								- workgate->numIO); j++) {
							if (j == t)
								permutativeArray[gatecounter]->my_string
								+= workgate->representation;
							else
								permutativeArray[gatecounter]->my_string += 'A';
						}
						permutativeArray[gatecounter]->representation
						= char('A' + gatecounter);
						if (t == 0)
							multigate = ccGate(workgate, multigate);
						else
							multigate = ccGate(gateArray[0], multigate);
						//calculate the matrix of the segment
						c = 0;
						while (true) {
							if (c != t)
								multigate2 = gateArray[0];
							else
								multigate2 = workgate;
							//using Kronecker multiplication
							tensorProduct3(multigate, multigate2, tempgate);
							multigate = ccGate(tempgate, multigate);
							c++;
							if (c >= (finalGate->numIO - workgate->numIO))
								break;
						}
						//cout<<"starting generating 4"<<endl;
						//fill the matrix of the gate
						columns = (int(pow(float(2), finalGate->numIO)));
						for (int e = 0; e < columns; e++)
							for (int f = 0; f < columns; f++)
								permutativeArray[gatecounter]->gateMatrix1[e+f*columns]
								                                              = multigate->gateMatrix1[e+f*columns];
						permutativeArray[gatecounter]->numIO = finalGate->numIO;
						gatecounter++;
					}
				} else {
					if (permutativeArray[gatecounter] != NULL)
						delete (permutativeArray[gatecounter]);
					permutativeArray[gatecounter] = NULL;
					permutativeArray[gatecounter] = new qGate;
					initGate(permutativeArray[gatecounter], finalGate->numIO, finalGate->valuedness, 1);
					permutativeArray[gatecounter]->representation = char('A'
							+ gatecounter);
					columns = (int(pow(float(2), finalGate->numIO)));
					for (int e = 0; e < columns; e++)
						for (int f = 0; f < columns; f++)
							permutativeArray[gatecounter]->gateMatrix1[e+f*columns]
							                                              = workgate->gateMatrix1[e+f*columns];
					gatecounter++;
				}
			}
		}
	}
	//cout<<"starting generating 10"<<endl;
	numofnormalgates = gatecounter;

	for (int n = 0; n < numofnormalgates; n++)
		cout << permutativeArray[n]->representation << " : "
		<< permutativeArray[n]->my_string << endl;
	cout << endl;
	destroyGate(multigate);
	destroyGate(tempgate);
	delete multigate;
	delete tempgate;
	return permutativeArray;
}
/*************************************************
 * Returns a set of gates representing single segments of the circuit
 * this can be used if circuits with big size are generated
 *************************************************/
qGate** GA::getPermutativeSegments() {

	qGate *workgate, *multigate, *multigate2, *tempgate;
	string names = "";
	int gatecounter = 0;
	int columns;
	string s = bestIndv->my_string;
	bool processing = false;
	multigate = new qGate;
	initGate(multigate, resultnum, finalGate->valuedness, 1);
	tempgate = new qGate;
	initGate(tempgate, resultnum, finalGate->valuedness, 1);
	for (int m = 0; m < MAXNUMOFGATES; m++) {
		permutativeArray[m] = NULL;
	}
	cout << "starting generating: " << s << "  " << endl;
	for (unsigned int a = 0; a < s.length(); a++) {
		//cout<<"starting generating "<<s.at(a)<<"  "<<endl;
		//we found a real gate
		if (s.at(a) == 'p') {
			if (a + 1 < s.length()) {
				if (s.at(a + 1) == 'p') {
					cout << "Name generated : " << names << endl;
					processing = false;
					gatecounter++;
				} else {
					processing = true;
					names = string("");
				}
			} else {
				cout << "Name generated : " << names << endl;
				processing = false;
			}
		} else {
			workgate = gateArray[getGate(s.at(a))];
			names += workgate->representation;
			//do the multiplication when the last gate is registered
			if (a + 1 < s.length() && s.at(a + 1) == 'p') {
				cout << "starting generating 0" << endl;
				//do the final gate multiplication
				for (int n = 0; n < names.length(); n++) {
					if (n == 0) {//set the 0'th gate
						workgate = gateArray[getGate(names.at(n))];
						if (permutativeArray[gatecounter] != NULL)
							delete (permutativeArray[gatecounter]);
						permutativeArray[gatecounter] = new qGate();
						permutativeArray[gatecounter]->representation
						= char('A' + gatecounter);
						multigate = ccGate(workgate, multigate);
						permutativeArray[gatecounter]->my_string
						= string(names);
						initGate(permutativeArray[gatecounter], finalGate->numIO, finalGate->valuedness, 1);
						cout << "starting generating 10: "
						<< permutativeArray[gatecounter]->my_string
						<< endl;
					} else {//add the next gate using the tensor product
						multigate2 = gateArray[getGate(names.at(n))];
						initMatrix(tempgate, multigate2->numIO);
						tensorProduct3(multigate, multigate2, tempgate);
						multigate = ccGate(tempgate, multigate);
					}
				}
				columns = (int(pow(float(2), finalGate->numIO)));
				for (int e = 0; e < columns; e++)
					for (int f = 0; f < columns; f++)
						permutativeArray[gatecounter]->gateMatrix1[e+f*columns]
						                                              = multigate->gateMatrix1[e+f*columns];

			}
		}
	}
	gatecounter++; //addlast gate
	//cout<<"starting generating 10"<<endl;
	numofnormalgates = gatecounter;

	for (int n = 0; n < numofnormalgates; n++) {
		cout << permutativeArray[n]->representation << " : "
		<< permutativeArray[n]->my_string << endl;
	}
	cout << bestIndv->my_string << endl;
	cout << endl;
	destroyGate(multigate);
	destroyGate(tempgate);
	delete multigate;
	delete tempgate;

	return permutativeArray;
}
/*************************************************
 *
 *************************************************/
bool GA::terminatingCondition() {
	return condition;
}
/*************************************************
 * the way of stopping things
 *************************************************/
void GA::setTerminatingCondition(int counter) {
	generationCondition = counter;
	if (counter >= MAXGEN || counter >= generations) {
		out_stream << " generation max reached " << endl << endl << endl;
		condition = true;
	} else if (getFitness()) {
		out_stream << " Solution found " << endl << endl << endl;
		condition = true;
	} else
		condition = false;
}
/*************************************************
 * check wether or not the correct solution was found
 * and if not it refresh the bestIndividual to the newest best value
 *************************************************/
bool GA::getFitness() {
	float sigma = 0.0005;
	if ((bestIndv->fitness + sigma) >= 1)
		return true;
	else
		return false;
}

/*******************************************
 *
 *******************************************/
void GA::closeStream() {
	in_stream.close();
	out_stream.close();
}
/*******************************************
 * compare two gates
 *******************************************/
int GA::findGate(qGate *A) {
	int a = 0;
	bool matrixTest;
	int result = -1;
	int columns;
	float val = (float) A->valuedness;
	while (a < numofgates) {
		out_stream << a << " " << numofgates << endl;
		matrixTest = true;

		if (A->numIO != gateArray[a]->numIO)
			result = -1;
		else {
			columns  = (int(pow(val, (float) A->numIO)));
			for (int i = 0; i < columns; i++)
				for (int j = 0; j < columns; j++)
					if (cuCrealf(A->gateMatrix1[i+j*columns]) != cuCrealf(gateArray[a]->gateMatrix1[i+j*columns]) || cuCimagf(A->gateMatrix1[i+j*columns]) != cuCimagf(gateArray[a]->gateMatrix1[i+j*columns])) matrixTest = false;
			if (matrixTest)
				result = a;
			else
				result = -1;

		}

		if (result != -1)
			break;
		else
			a++;
	}

	return result;

}
/*******************************************
 * the super-method intializing the circuit minimization
 * now the minimzation is done here only as Darwinian learning
 *******************************************/
int GA::minimizeCirc(Individual *qC) {
	int begin0, begin1, end0, end1;
	bool equal;
	int a, b;
	int result = 0;

	if (qC->ioNumber > 1) {
		begin0 = 0;
		end0 = qC->my_string.find("p", begin0 + 1) + 1;
		begin1 = end0;
		end1 = qC->my_string.find("p", begin1 + 1) + 1;
		b = 0;

		while (end1 > 0) {
			equal = false;
			a = 0;
			if ((qC->my_string.substr(begin0, (end0 - begin0))).length()
					== (qC->my_string.substr(begin1, (end1 - begin1))).length()) {
				while (a <= (end0 - begin0 - 2)) {
					if ((qC->my_string.at(begin0 + a)) != 'p') {
						if (gateArray[getGate(qC->my_string.at(begin0 + a))]->numIO
								!= gateArray[getGate(qC->my_string.at(begin1
										+ a))]->numIO) {
							equal = false;
							break;
						} else {
							if (a >= end0 - begin0 - 2) {
								equal = true;
							}
						}
					}
					a++;
				}
			}
			if (!equal) {
				for (int a = 1; a < (end0 - begin0 - 1); a++) {
					result
					+= gateArray[getGate(qC->my_string.at(begin0 + a))]->Cost;
				}
			}
			begin0 = begin1;
			end0 = qC->my_string.find("p", begin0 + 1) + 1;
			begin1 = end0;
			end1 = qC->my_string.find("p", begin1 + 1) + 1;
		}
	}
	for (a = 1; a < (end0 - begin0 - 1); a++) {
		result += gateArray[getGate(qC->my_string.at(begin0 + a))]->Cost;
	}
	qC->Cost = result;

	return result;
}
/*******************************************
 * removes physically redundant segments in the concenred circuit
 *******************************************/
void GA::removeRedund(Individual *C) {
	int begin0, begin1, end0, end1;
	bool equal, erased;
	int a, b;

	if (C->ioNumber > 1) {
		begin0 = 0;
		end0 = C->my_string.find("p", begin0 + 1) + 1;
		begin1 = end0;
		end1 = C->my_string.find("p", begin1 + 1) + 1;
		b = 0;

		while (end0 < (int) C->my_string.length() - 3 || end0 < 0) {
			equal = false;
			erased = false;
			a = 0;
			if (C->my_string.substr(begin0, (end0 - begin0))
					== C->my_string.substr(begin1, (end1 - begin1))) {
				C->my_string.erase(begin0, (end1 - begin0));
				erased = true;
			}

			if (erased) {
				if (begin0 != 0) {
					begin0 = 0;
					end0 = C->my_string.find("p", begin0 + 1) + 1;

					begin1 = end0;
					end1 = C->my_string.find("p", begin1 + 1) + 1;
				} else {
					begin1 = end0;
					end1 = C->my_string.find("p", begin1 + 1) + 1;
					erased = false;
				}
			} else {
				begin0 = begin1;
				end0 = C->my_string.find("p", begin0 + 1) + 1;
				if (end0 < (int) C->my_string.length() - 3) {
					begin1 = end0;
					end1 = C->my_string.find("p", begin1 + 1) + 1;
				}
			}
		}
	}
}
/*******************************************
 * physically modyfies the circuit for lamarckian learning
 *******************************************/
void GA::mergeSegs(Individual *Circ, int first, int second) {
	basic_string<char> tempStr;
	qGate *temp, *fGate, *sGate;
	int gateCount;
	fGate = new qGate;
	initGate(fGate, resultnum, finalGate->valuedness, 1);
	sGate = new qGate;
	initGate(sGate, resultnum, finalGate->valuedness, 1);

	int endfrsSg = Circ->my_string.find("p", first + 1) + 1;
	int endscdSg = Circ->my_string.find("p", second + 1) + 1;
	tempStr.append(1, 'p');

	for (int a = 0; a < (endfrsSg - first - 1); a++) {
		if (Circ->my_string.at(first + a) != 'p') {

			fGate = ccGate(gateArray[getGate(Circ->my_string.at(first + a))], fGate);
			sGate = ccGate(gateArray[getGate(Circ->my_string.at(second + a))],sGate);
			initMatrix(temp, fGate->numIO);
			cblasMatrixProduct(fGate, sGate, temp);
			gateCount = findGate(temp);

			if (gateCount != -1) {
				tempStr.append(1, gateArray[gateCount]->representation);
			} else {
				gateArray[numofgates] = new qGate;
				initGate(gateArray[numofgates], temp->numIO, temp->valuedness, 1);
				gateArray[numofgates] = ccGate(temp, gateArray[numofgates]);

				if (gateArray[numofgates - 1]->representation == 'o')
					gateArray[numofgates]->representation
					= gateArray[numofgates - 1]->representation
					+ char(2);
				else if (gateArray[numofgates - 1]->representation == char(264))
					gateArray[numofgates]->representation
					= gateArray[numofgates - 1]->representation
					+ char(6);
				else
					gateArray[numofgates]->representation
					= gateArray[numofgates - 1]->representation
					+ char(1);

				gateArray[numofgates]->Cost = fGate->Cost;
				gateArray[numofgates]->parentA = fGate->representation;
				gateArray[numofgates]->parentB = fGate->representation;
				tempStr.append(1, gateArray[numofgates]->representation);
				numofgates++;
			}
		}
	}
	tempStr.append(1, 'p');
	Circ->my_string.erase(first, (endscdSg - first));
	Circ->my_string.insert(first, tempStr);

	destroyGate(fGate);
	destroyGate(sGate);
	delete fGate, sGate;
}
/* calculates the average of fitness and error */
void GA::calcAverage() {

	int counter = finalResult.counter;
	float avfitness = 0;
	float averror = 0;
	float avcost = 0;
	for (int t = 0; t < populationNumber; t++) {
		avfitness += population[t]->fitness;
		averror += population[t]->Error;
		avcost += population[t]->Cost;
	}
	finalResult.avFitness[counter] = avfitness / populationNumber;
	finalResult.avError[counter] = averror / populationNumber;
	finalResult.avCost[counter] = avcost / populationNumber;
	finalResult.counter = counter + 1;
}
/*******************************************
 * Initiate the length of the arrays for history records
 *******************************************/
void GA::initiateStorage() {
	cout << "Initiating Storage for " << generations << " generations." << endl;
	history = new float[generations][4];
}

/**********************************
* decodes the name of the gate into a RevLib compatible format
**********************************/
void GA::decodeIndvStrToRevLibStr(Individual *I){
	string rev_lib_circuit;
	int c_count = 0;
	string u;
	int i_gate,i,j,k;
	int wire_counter = 0;
	char buff[10];
	int y, x;
	int indexes[I->ioNumber];
	rev_lib_circuit = ".version Curr Individual\n";
	
	rev_lib_circuit += ".numvars ";
	sprintf(buff, "%d", I->ioNumber);
	x = 0;
	while(buff[x] > 0)
		rev_lib_circuit += buff[x++];	
	rev_lib_circuit += "\n.variables ";
	for (y = 0; y < I->ioNumber; y++){
		sprintf(buff, "%d", y);
		x = 0;
		rev_lib_circuit += 'w';	
		while(buff[x] > 0)
			rev_lib_circuit += buff[x++];	
		rev_lib_circuit += ' ';	
	}
	rev_lib_circuit += "\n.inputs ";
	for (y = 0; y < I->ioNumber; y++){
		sprintf(buff, "%d", y);
		x = 0;
		rev_lib_circuit += 'w';	
		while(buff[x] > 0)
			rev_lib_circuit += buff[x++];	
		rev_lib_circuit += ' ';	
	}
	rev_lib_circuit += "\n.outputs ";
	for (y = 0; y < I->ioNumber; y++){
		sprintf(buff, "%d", y);
		x = 0;
		rev_lib_circuit += 'w';	
		while(buff[x] > 0)
			rev_lib_circuit += buff[x++];	
		rev_lib_circuit += ' ';	
	}
	rev_lib_circuit += "\n.begin\n";
	for (int m = 0; m < I->my_string.length(); m++){
		if (I->my_string.at(m) != 'p'){
			i_gate = getGate(I->my_string.at(m));	
			if (i_gate != 0){
				for (i = 0; i < gateArray[i_gate]->rev_lib_char.length(); i++)
					rev_lib_circuit += gateArray[i_gate]->rev_lib_char.at(i);
				 rev_lib_circuit += ' ';
			if (wire_counter == 0){
				for (i = 0; i < gateArray[i_gate]->rev_lib_string.length(); i++){
					if (i == 0)
						rev_lib_circuit += 'w';
					rev_lib_circuit += gateArray[i_gate]->rev_lib_string.at(i);
					if (gateArray[i_gate]->rev_lib_string.at(i) == ' ' && i !=  0)
						rev_lib_circuit += 'w';
				}
				rev_lib_circuit += '\n';
				wire_counter += gateArray[i_gate]->numIO;
			}else {
				for (i = 0; i < I->ioNumber; i++)
					indexes[i] = -1;
				StrToIntArray(gateArray[i_gate]->rev_lib_string, indexes);
				for (i = 0; i < I->ioNumber; i++){
					if (indexes[i] == -1)
						break; 					
					indexes[i] += wire_counter;
				}


					y = 0;
					do {
						sprintf(buff, "%d", indexes[y]);
						x = 0;
						rev_lib_circuit += 'w';	
						while(buff[x] > 0)
							rev_lib_circuit += buff[x++];
						rev_lib_circuit += ' ';
					}while(indexes[++y] > -1);

				rev_lib_circuit += '\n';
				wire_counter += gateArray[i_gate]->numIO;
			}
			} else wire_counter++;
		} else wire_counter = 0;
	}
	rev_lib_circuit += ".end";
	I->rev_lib_string = "";
	for (y = 0; y < rev_lib_circuit.length(); y++)
		I->rev_lib_string += rev_lib_circuit.at(y);
}

void GA::StrToIntArray(string s, int *data){
	string buff;
	int begin, data_count, y, x;
	begin = 0;
	data_count = 0;
	y = 0;
	do {
		if (s.at(y) == ' '){
			buff = "";
			for (x = begin; x < y; x++)
				buff += s.at(x);
			begin = y;
			data[data_count++] = atoi(buff.c_str()); 
		} else if (y == s.length()-1){
                        buff = "";
                        for (x = begin; x <= y; x++)
                                buff += s.at(x);
                        begin = y;
                        data[data_count++] = atoi(buff.c_str());
                }
	}while(++y <s.length());

}

void GA::IntArrayToStr(int *data, string str){
	char buff[10];
	int y, x;
	y = 0;
	do {
		sprintf(buff, "%d", data[y]);
		x = 0;
		while(buff[x] > 0)
		str += buff[x++];
		str += ' ';

	}while(data[++y] > -1);

	cout<<"in: "<< str.length()<<":out"<<endl;
}
