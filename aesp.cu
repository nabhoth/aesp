#include "aesp.h"

__global__ void setup_kernel(curandState *globalstate, unsigned long seed)
{
	    int id = threadIdx.x + blockIdx.x*blockDim.x;
	        if (id<MAXSIZE)
			    curand_init(seed, id, 0, &globalstate[id]);
}

/**************************************************************
* For each particle determine if it should die (be absorbed)
**************************************************************/
__global__ void evolve_p_state(particle *d_elements, cell *d_cell, curandState *globalState, int *d_states, int *d_index, int *d_active)
{
    	int id =  blockIdx.x+blockDim.x*threadIdx.x;
    	if (id<MAXSIZE){
    	particle *p = &d_elements[id];
	if (atomicCAS(&p->alive,1,1)> 0){
		int xcell = p->x >> (int)log2(CELLS);
		int ycell = p->y >> (int)log2(CELLS);
		curandState state = globalState[id];
		double d = curand_uniform_double(&state);
	printf("Rand:  %f, Particle id %i, Cell Ann: %f\n", d, p->id, (d_cell[(int)(xcell+ycell*CELLS)]).ann);
		if (d > (d_cell[(int)(xcell+ycell*CELLS)]).ann){
	printf("Rand:  %f, PArticle Id: %i, \n", d, p->id);
			atomicSub(&(d_cell[(int)(xcell+ycell*CELLS)]).pcount,1); 
    			    atomicSub(&p->alive,1);
    			    int r = atomicSub(d_active,1);
		   	    //  __syncthreads();
    			    int j = atomicAdd(d_index,1);
		  	    // __syncthreads();
    			    d_states[j] = p->id;

			p->x = 0;
			p->y = 0;
			p->speed = 0;
	    	//	printf("Particle %i from Block %i with x=%i, y=%i died, Bloc counter is now %i\n",p->id, (int)(xcell+ycell*CELLS), d_cell[(int)(xcell+ycell*CELLS)].x0, d_cell[(int)(xcell+ycell*CELLS)].y0, d_cell[(int)(xcell+ycell*CELLS)].pcount);
		}
		}
	}

}

/**************************************************************
* For each cell determine if new particles should be created
**************************************************************/
__global__ void evolve_c_state(particle *d_elements, cell *d_cell, curandState *globalState, int *d_states, int *x)
{
    int id =  blockIdx.x+blockDim.x*threadIdx.x;
    cell *c = &d_cell[id];
    curandState state = globalState[id];
    double d = curand_uniform_double(&state);
    if (d < c->cre){
	    int j = atomicSub(x,1);
	    int id = d_states[j];
	    d_states[j+1] = -1;
	    particle *p = &d_elements[id];
	    p->alive = 1;
    	d = curand_uniform_double(&state);
	p->x = d*CELLX;
    	d = curand_uniform_double(&state);
	p->y = d*CELLY;
	p->speed = 0;
	int xcell = p->x >> (int)log2(CELLS);
	int ycell = p->y >> (int)log2(CELLS);
//	printf("%i %i %i\n",xcell,ycell,(int)(xcell+ycell*CELLS));
	(d_cell[(int)(xcell+ycell*CELLS)]).pcount++; 
//	printf("%i %i %i\n",xcell,ycell,cells[(int)(xcell+ycell*CELLS)]->pcount);

    }

}
/**************************************************************
* Moving the particle in some direction, blah blah blah
**************************************************************/
__global__ void propagate(Lock lock, particle *d_elements, cell *d_cell, curandState *globalState, int *d_states, int *d_index, int *d_active)
{
    int id =  blockIdx.x+blockDim.x*threadIdx.x;
    if (id<MAXSIZE){
	    particle *p = &d_elements[id];
	    int xcell0 = p->x >> (int)log2(CELLS);
	    int ycell0 = p->y >> (int)log2(CELLS);
	    int xcell1;
	    int ycell1;
	    if (atomicCAS(&p->alive,1,1)> 0){
		    if (p->speed == 0){
			    curandState state = globalState[id];
			    double d = curand_uniform_double(&state);
			    p->speed = V;
			    p->dir = (d*360.0);///(double)RAND_MAX
    			    p->s =p->speed*ST;
    			    p->xf = (int)(cos(p->dir)*(double)S)+p->x;
    			    p->yf = (int)(sin(p->dir)*(double)S)+p->y;
    			    p->x = (int)(cos(p->dir)*p->s)+p->x;
    			    p->y = (int)(sin(p->dir)*p->s)+p->y;
    			    xcell1 = p->x >> (int)log2(CELLS);
    			    ycell1 = p->y >> (int)log2(CELLS);
    		    } else {
    			    p->x = (int)(cos(p->dir)*p->s)+p->x;
    			    p->y = (int)(sin(p->dir)*p->s)+p->y;
    			    xcell1 = p->x >> (int)log2(CELLS);
    			    ycell1 = p->y >> (int)log2(CELLS);
    			    if (p->x > p->xf || p->y > p->yf)
    				    p->speed = 0;
    		    }
//		    printf("Random: %f, Angle: %f\n", d,p->dir);
    		    if (p->x >CELLX || p->x < 0 || p->y > CELLY || p->y < 0){
    			    atomicSub(&p->alive,1);
    			    int r = atomicSub(d_active,1);
		   	    //  __syncthreads();
    			    int j = atomicAdd(d_index,1);
		  	    // __syncthreads();
    			    d_states[j] = p->id;
	    		    //	printf("Particle %i is dead, Remaining particles: %i, at %i\n",p->id,r-1, j);
    		    }else if (xcell0 != xcell1 || ycell0!=ycell1){
    			    atomicSub(&d_cell[(int)(xcell0+ycell0*CELLS)].pcount,1);
    			    atomicAdd(&d_cell[(int)(xcell1+ycell1*CELLS)].pcount,1);
	    		    printf("Particle %i crossed from Block %i with x=%i, y=%i to block %i with x=%i, y=%i\n",p->id, (int)(xcell0+ycell0*CELLS), d_cell[(int)(xcell0+ycell0*CELLS)].x0, d_cell[(int)(xcell0+ycell0*CELLS)].y0, (int)(xcell1+ycell1*CELLS), d_cell[(int)(xcell1+ycell1*CELLS)].x0, d_cell[(int)(xcell1+ycell1*CELLS)].y0);
    		    }
	    }
    }
}



int init_particle(particle *p, int  x, int y, int id){

	p->x = x;
	p->y = y;
	p->id = id;
	p->speed = 0;
	p->alive = 1;
	return 0;
}



int init_cell(cell *c, int a , int b, int *h_states, particle *p){

	c->x0 = a;
	c->y0 = b;
	c->pcount = 0;
	c->emissivity = EMS;
	c->cre = ((double)rand()/(double)RAND_MAX)/c->emissivity;
	c->ann = ((double)rand()/(double)RAND_MAX)/c->emissivity;
	for (int a = 0; a < c->emissivity;a++){
		int x = rand()%CELLX;
		int y = rand()%CELLY;
		if (h_index >=1){
		h_index--;
		h_states[h_index] = -1;
		init_particle(&p[h_index],x,y,h_index);
		}
	}
	return 0;
}

int destroy_particle(particle *p){
	free(p);
	return 0;

}

int main(){

	h_index = MAXSIZE;
	int a,b = 0;
	int memSize = MAXSIZE*sizeof(curandState);
	cudaMalloc((void**) &globalstate, memSize);
	printf("%i ", time(NULL));

	//initialize an array of pointers to particles
	memSize = MAXSIZE*sizeof(particle);
	cudaMalloc((void**) &d_elements, memSize);

	//initialize an array of inactive particles indices
	memSize = MAXSIZE*sizeof(int);
	cudaMalloc((void**) &d_states, memSize);

	//initialize an semaphore for counting inactive particles
	memSize = sizeof(int);
	cudaMalloc(&d_index, memSize);

	//initialize an semaphore for counting inactive particles
	memSize = sizeof(int);
	cudaMalloc(&d_total, memSize);
	cudaMalloc(&d_active, memSize);
	cudaMalloc(&d_residual, memSize);


	//initialize an array of spatial cells
	memSize = CELLS*CELLS*sizeof(cell);
	cudaMalloc((void**) &d_cells, memSize);

	//initialize h_states
	h_states = new int[MAXSIZE];
	for (int b = 0; b < MAXSIZE;b++)
		h_states[b] = MAXSIZE-b-1;

		printf("States Done\n");
	//initialize all cells - space locations
	srand((unsigned)time(NULL));
	h_cells = new cell[(int)(CELLS*CELLS)];
	h_elements = new particle[MAXSIZE];
	for (int d = 0; d < (int)(CELLS*CELLS);d++){
		init_cell(&h_cells[d],a,b,h_states,h_elements);
		a++;
		if (a >= CELLS){a=0; b++;}
	}
		printf("Cells Done %i, %i \n", h_index, MAXSIZE);
	for (int a = MAXSIZE-1; a >= MAXSIZE-10;a--){
		printf("Particle %i at coordinates %i, %i is alive %d\n", h_elements[a].id,h_elements[a].x, h_elements[a].y, h_elements[a].alive);
		//printf("Particle %i is at x coordinate %i and y coordinate %i\n",a,h_elements[a].x,h_elements[a].y);
	}

	h_total = MAXSIZE;
	h_active = MAXSIZE;
	h_residual = 0.0;


	//allocate space for cells and particles on cuda
	memSize = sizeof(int);
	printf("CUDA error: %s\n", cudaGetErrorString(cudaMemcpy(d_total, &h_total, memSize, cudaMemcpyHostToDevice)));
	printf("CUDA error: %s\n", cudaGetErrorString(cudaMemcpy(d_active, &h_active, memSize, cudaMemcpyHostToDevice)));
	printf("CUDA error: %s\n", cudaGetErrorString(cudaMemcpy(d_residual, &h_residual, memSize, cudaMemcpyHostToDevice)));
	printf("CUDA error: %s\n", cudaGetErrorString(cudaMemcpy(d_index, &h_index, memSize, cudaMemcpyHostToDevice)));
	memSize = MAXSIZE*sizeof(int);
	printf("CUDA error: %s\n", cudaGetErrorString(cudaMemcpy(d_states, h_states, memSize, cudaMemcpyHostToDevice)));
	memSize = CELLS*CELLS*sizeof(cell);
	printf("CUDA error: %s\n", cudaGetErrorString(cudaMemcpy(d_cells, h_cells, memSize, cudaMemcpyHostToDevice)));
	memSize = MAXSIZE*sizeof(particle);
	printf("CUDA error: %s\n", cudaGetErrorString(cudaMemcpy(d_elements, h_elements, memSize, cudaMemcpyHostToDevice)));
	//cudaDeviceSynchronize();

//	for (int a = MAXSIZE-1; a >= MAXSIZE-10;a--){
//		printf("Particle %i is at x coordinate %i and y coordinate %i\n",a,h_elements[a].x,h_elements[a].y);
//	}

	//interesctPropagate(d_elements, globalstate);
	for (int p = 0; p< 10000; p++){
//		propagate<<<BLOCKS, BLOCKS>>>(lock, d_elements, d_cells, globalstate, d_states, d_index, d_active);
		setup_kernel <<<BLOCKS, 512 >>>(globalstate, time(NULL));
		evolve_p_state<<<BLOCKS, 512>>>(d_elements, d_cells, globalstate, d_states, d_index, d_active);
		evolve_c_state<<<BLOCKS, 1>>>(d_elements, d_cells, globalstate, d_states, d_index);
	}

	memSize = MAXSIZE*sizeof(particle);
	printf("CUDA error: %s\n", cudaGetErrorString(cudaMemcpy(h_elements, d_elements, memSize, cudaMemcpyDeviceToHost)));
	memSize = sizeof(int);
	printf("CUDA error: %s\n", cudaGetErrorString(cudaMemcpy(&h_active, d_active, memSize, cudaMemcpyDeviceToHost)));
	memSize = MAXSIZE*sizeof(int);
	printf("CUDA error: %s\n", cudaGetErrorString(cudaMemcpy(h_states, d_states, memSize, cudaMemcpyDeviceToHost)));

	printf("Final particle count is: %i\n",h_active);

	for (int a = MAXSIZE-1; a >= MAXSIZE-10;a--){
		printf("Particle %i at coordinates %i, %i is alive %d\n", h_elements[a].id,h_elements[a].x, h_elements[a].y, h_elements[a].alive);
		//printf("Particle %i is at x coordinate %i and y coordinate %i\n",a,h_elements[a].x,h_elements[a].y);
	}

/*	for (int a = MAXSIZE-1; a >= 0;a--){
		int j = h_states[a];
		int count = 0;
		for (int b = MAXSIZE-1; b >= 0;b--){
			if (h_states[b] == j) count++;
		}
		printf("Particle index at %i with id %i has been dealocated %i times\n", a, h_states[a], count);
	}
*/
return 0;
}