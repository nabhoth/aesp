#include "aesp.h"

__global__ void setup_kernel(curandState *globalstate, unsigned long seed)
{
	    int id = threadIdx.x + blockIdx.x*blockDim.x;
	        if (id<MAXPSIZE)
			    curand_init(seed, id, 0, &globalstate[id]);
}


/**************************************************************
* For each cell determine if new particles should be created
**************************************************************/
__global__ void evolve_c_state(particle *d_elements, cell *d_cell, curandState *globalState, int *d_states, int *d_index, int *d_active)
{
    int id =  threadIdx.x+blockDim.x*blockIdx.x;
    if (id<CELLX*CELLY){
	    cell *c = &d_cell[id];
	    if (atomicCAS(&c->busy,0,1)> 0){
		    if (RADIATE == 0 && c->radiant == 0){
			    curandState state = globalState[id];
			    double d = curand_uniform_double(&state);
			    if (atomicCAS(d_index,0,*d_index)>0){
				    if (d < c->cre){
					    int j = atomicSub(d_index,1);
					    int r = atomicAdd(d_active,1);
					    int idc = d_states[j+1];
					    d_states[j+1] = -1;
					    particle *p = &d_elements[idc];
					    p->alive = 1;
					    d = curand_uniform_double(&state);
					    p->x = d*DIMX/CELLX+c->x0*DIMX/CELLX;
					    d = curand_uniform_double(&state);
					    p->y = d*DIMY/CELLY+c->y0*DIMY/CELLY;
//					    p->speed = 0;
					    atomicAdd(&c->pcount,1);
					    //				    int recind = p->x/(int)(DIMX/CELLX)+ (p->y/(int)(DIMY/CELLY)*CELLY);
					    //    				    printf("EC Particle %i from Block %i (has now %i particles) with x=%i, y=%i is reactivated, reconstructed cell id is %i dead index is %i\n",idc, id, c->pcount,p->x, p->y, recind, *d_index );
				    }
				    atomicCAS(&c->busy,1,0);
			    }
		    } else {
			    for (int m = 0; m < c->emissivity; m++){
				    curandState state = globalState[id];
				    double d = curand_uniform_double(&state);
				    if (atomicCAS(d_index,0,*d_index)>0){
					    int j = atomicSub(d_index,1);
					    int r = atomicAdd(d_active,1);
					    int idc = d_states[j+1];
					    d_states[j+1] = -1;
					    particle *p = &d_elements[idc];
					    p->alive = 1;
					    d = curand_uniform_double(&state);
					    p->x = d*DIMX/CELLX+c->x0*DIMX/CELLX;
					    d = curand_uniform_double(&state);
					    p->y = d*DIMY/CELLY+c->y0*DIMY/CELLY;
//					    p->speed = 0;
					    atomicAdd(&c->pcount,1);
				    }
			    }
    		    }
	    }
    }
}

/**************************************************************
* For each particle determine if it should die (be absorbed)
**************************************************************/
__global__ void evolve_p_state(particle *d_elements, cell *d_cell, curandState *globalState, int *d_states, int *d_index, int *d_active)
{
    	int id =  threadIdx.x+blockDim.x*blockIdx.x;
    	if (id<MAXPSIZE){
		particle *p = &d_elements[id];
		if (atomicCAS(&p->alive,1,p->alive)> 0){
			int xcell = p->x/(int)(DIMX/CELLX);
			int ycell = p->y/(int)(DIMY/CELLY);
			int cellid = xcell+ycell*CELLY;

//			printf("EP Particle %i at %i and %i from Block %i with x=%i, y=%i d index is %i\n",p->id, p->x, p->y, cellid, d_cell[cellid].x0, d_cell[cellid].y0, *d_index);
			curandState state = globalState[id];
			double d = curand_uniform_double(&state);
			if (d > (d_cell[cellid]).ann){
				int pc = atomicSub(&(d_cell[cellid]).pcount,1); 
    				atomicSub(&p->alive,1);
    				int r = atomicSub(d_active,1);
    				int j = atomicAdd(d_index,1);
    				d_states[j] = p->id;
				p->x = 0;
				p->y = 0;
//				p->speed = 0;
//				printf("EP Particle %i from Block %i with x=%i, y=%i died,  d index is %i, counter is now %i\n",p->id, cellid, d_cell[cellid].x0, d_cell[cellid].y0, *d_index, pc);
			}
		}
	}

}

/**************************************************************
* Moving the particle in some direction, blah blah blah
**************************************************************/
__global__ void propagate(Lock lock, particle *d_elements, cell *d_cell, curandState *globalState, int *d_states, int *d_index, int *d_active)
{
    	int id =  threadIdx.x+blockDim.x*blockIdx.x;
    	if (id<MAXPSIZE){
    		particle *p = &d_elements[id];
		int xcell0 = p->x/(int)(DIMX/CELLX);
		int ycell0 = p->y/(int)(DIMY/CELLY);
		int cellid0 = xcell0+ycell0*CELLY;

    		int xcell1;
    		int ycell1;
		int cellid1;
//		double tempdir = 0;
    		if (atomicCAS(&p->alive,1,p->alive)> 0){
			printf("Index %i of PRP Particle %i at %i and %i with angle %f from Block %i with x=%i, y=%i and id=%i d index is %i\n",id,p->id, p->x, p->y, p->dir, cellid0, d_cell[cellid0].x0, d_cell[cellid0].y0, d_cell[cellid0].id, *d_index);
//    			if (p->speed == 0){
//  			} else {
    				p->x = (int)(cos(p->dir)*p->s)+p->x;
    				p->y = (int)(sin(p->dir)*p->s)+p->y;
				xcell1 = p->x/(int)(DIMX/CELLX);
				ycell1 = p->y/(int)(DIMY/CELLY);
//    				if (p->x > p->xf || p->y > p->yf)
//    					p->speed = 0;
//    			}
			printf("PRP Particle %i at %i and %i with angle %f\n",p->id, p->x, p->y, p->dir);
			cellid1 = xcell1+ycell1*CELLY;
    			if (p->x >DIMX || p->x < 0 || p->y > DIMY || p->y < 0){
				if (PERMEABILITY < 1){
					if (p->x >DIMX){
						if (p->dir > ((double)(4/3)/(double)(PI))){
							p->dir = (4/3*PI)-(p->dir-((4/3)*PI));	
							p->x = DIMX-(p->x-DIMX);
						} else if (p->dir <((double)PI/(double)2)){
							p->dir = PI-p->dir;	
							p->x = DIMX-(p->x-DIMX);
						}
					} else if (p->x < 0){
						if (p->dir > PI){
							p->dir = 2*PI-(p->dir-PI);
							p->x = -p->x;
						} else if (p->dir > ((double)PI/(double)(2))){
							p->dir = PI-p->dir;
							p->x = -p->x;
						}
					}
					if (p->y > DIMY){
						if (p->dir < ((double)PI/(double)(2))){
							p->dir = (2*PI)-p->dir;
							p->y = DIMY-(p->y-DIMY);
						} else if (p->dir < PI){
							p->dir = PI+(PI-p->dir);
							p->y = DIMY-(p->y-DIMY);
						}
					} else if (p->y < 0){
						if (p->dir < ((double)(4/3)/(double)(PI))){
							p->dir = PI+(p->dir-PI);
							p->y = -p->y;
						} else if (p->dir < ((double)2*PI)){
							p->dir = (PI/2)+(p->dir-((4/3)*PI));
							p->y = -p->y;
						}
					}
//					printf("Reflected Particle %i at %i and %i with angle %f from Block %i with x=%i, y=%i and id=%i d index is %i\n",p->id, p->x, p->y, p->dir,cellid0, d_cell[cellid0].x0, d_cell[cellid0].y0, d_cell[cellid0].id, *d_index);
				} else {
					atomicSub(&p->alive,1);
					int r = atomicSub(d_active,1);
					int j = atomicAdd(d_index,1);
					atomicSub(&d_cell[cellid0].pcount,1);
					d_states[j] = p->id;
					printf("PRP Particle %i at %i and %i from Block %i with x=%i, y=%i dnd id=%i  index is %i died by exit from field\n",p->id, p->x, p->y, cellid1, d_cell[cellid1].x0, d_cell[cellid1].y0, d_cell[cellid1].id, *d_index);
				}
			}else if (xcell0 != xcell1 || ycell0!=ycell1){
				atomicSub(&d_cell[cellid0].pcount,1);
				atomicAdd(&d_cell[cellid1].pcount,1);
				if (INTERACTION > 0){
	    				curandState state = globalState[id];
    					double d = curand_uniform_double(&state);
    					p->dir = (d*2*PI);
				}
//    				p->s =p->speed*ST;
  //  				p->xf = (int)(cos(p->dir)*(double)S)+p->x;
    //				p->yf = (int)(sin(p->dir)*(double)S)+p->y;
    //				p->x = (int)(cos(p->dir)*p->s)+p->x;
    //				p->y = (int)(sin(p->dir)*p->s)+p->y;
//				xcell1 = p->x/(int)(DIMX/CELLX);
//				ycell1 = p->y/(int)(DIMY/CELLY);

				printf("PRP Particle %i crossed from Block %i with x=%i, y=%i to block %i with x=%i, y=%i\n",p->id, cellid0, d_cell[cellid0].x0, d_cell[cellid0].y0, cellid1, d_cell[cellid1].x0, d_cell[cellid1].y0);
			}
    		}
    	}
}



int init_particle(particle *p, int  x, int y, int id){

	p->x = x;
	p->y = y;
	p->id = id;
	p->dir = ((double)rand()/(double)RAND_MAX)*360;
	p->speed = 0;
	p->s = V*ST;
	p->alive = 1;
	return 0;
}



int init_cell(cell *c, int a , int b, int *h_states, particle *p){

	c->x0 = a;
	c->y0 = b;
	c->id = a+b*CELLY;
	c->pcount = 0;
	c->emissivity = EMS;
	c->cre = ((double)rand()/(double)RAND_MAX)/c->emissivity;
	c->ann = ((double)rand()/(double)RAND_MAX)/c->emissivity;
	c->radiant = 0;
	if (radiate_x == a && radiate_y == b && RADIATE > 0){
		c->radiant = 1;
		for (int v = 0; v < c->emissivity;v++){
			int x = a*DIMX/CELLX+rand()%(int)(DIMX/CELLX);
			int y = b*DIMY/CELLY+rand()%(int)(DIMY/CELLY);
			if (h_index >=1){
			h_states[h_index] = -1;
			h_index--;
			c->pcount++;
			init_particle(&p[h_index],x,y,h_index);
//		int x0 = p[h_index].x/(int)(DIMX/CELLX);
//		int y0 = p[h_index].y/(int)(DIMY/CELLY);
//		int cellid = x0+y0*CELLY;
//		printf("Particle %i from Block %i (is a=%i and b=%i and has now %i particles) was created at x=%i, y=%i, extracted cell id=%i\n",h_index, c->id, a,b, c->pcount,x, y, cellid);
			}
		}
	} else if (RADIATE < 1){
		for (int v = 0; v < c->emissivity;v++){
			int x = a*DIMX/CELLX+rand()%(int)(DIMX/CELLX);
			int y = b*DIMY/CELLY+rand()%(int)(DIMY/CELLY);
			if (h_index >=1){
			h_states[h_index] = -1;
			h_index--;
			c->pcount++;
			init_particle(&p[h_index],x,y,h_index);
//		int x0 = p[h_index].x/(int)(DIMX/CELLX);
//		int y0 = p[h_index].y/(int)(DIMY/CELLY);
//		int cellid = x0+y0*CELLY;
//		printf("Particle %i from Block %i (is a=%i and b=%i and has now %i particles) was created at x=%i, y=%i, extracted cell id=%i\n",h_index, c->id, a,b, c->pcount,x, y, cellid);
			}
		}
	}
	return 0;
}

int destroy_particle(particle *p){
	free(p);
	return 0;

}

int main(){



	h_index = MAXPSIZE;
	int a,b = 0;
	int memSize = MAXPSIZE*sizeof(curandState);
	cudaMalloc((void**) &globalstate, memSize);

	//initialize an array of pointers to particles
	memSize = MAXPSIZE*sizeof(particle);
	cudaMalloc((void**) &d_elements, memSize);

	//initialize an array of inactive particles indices
	memSize = MAXPSIZE*sizeof(int);
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
	memSize = CELLX*CELLY*sizeof(cell);
	cudaMalloc((void**) &d_cells, memSize);

	//initialize h_states
	h_states = new int[MAXPSIZE];
	for (int b = 0; b < MAXPSIZE;b++)
		h_states[b] = MAXPSIZE-b-1;

		printf("States Done\n");
	//initialize all cells - space locations
	srand((unsigned)time(NULL));
	h_cells = new cell[(int)(CELLX*CELLY)];
	h_elements = new particle[MAXPSIZE];
	srand((unsigned)time(NULL));
//	double angle = 0;
//	int xx = 0;
//	int yy = 0;
	for (int d = 0; d < (int)(CELLX*CELLY);d++){
		init_cell(&h_cells[d],a,b,h_states,h_elements);
		a++;
		if (a >= CELLX){a=0; b++;}
//		printf("Angle %f x0=%i and y0=%i",angle, 0,0);
//	xx = (int)(cos(angle)*(double)S);
  //  	yy = (int)(sin(angle)*(double)S);
	
//		printf(" and xx=%i yy=%i\n",xx,yy);
//		angle += (double)(2*PI)/(double)(CELLX*CELLY);
//		printf("Cell %i (is a=%i and b=%i and has now %i particles and %i index) \n",d, a,b, h_cells[d].pcount,h_cells[d].id);
	}
		//printf("Cells Done %i, %i \n", h_index, MAXPSIZE);
//	for (int a = MAXPSIZE-1; a >= MAXPSIZE-10;a--){
//		printf("Particle %i at coordinates %i, %i is alive %d\n", h_elements[a].id,h_elements[a].x, h_elements[a].y, h_elements[a].alive);
//	}

	h_total = MAXPSIZE;
	h_active = MAXPSIZE-h_index;
	h_residual = 0.0;


	//allocate space for cells and particles on cuda
	memSize = sizeof(int);
	printf("Initial particle count is: %i\n",h_active);
	printf("Initial particle count is: %i\n",h_index);
	printf("CUDA error: %s\n", cudaGetErrorString(cudaMemcpy(d_total, &h_total, memSize, cudaMemcpyHostToDevice)));
	printf("CUDA error: %s\n", cudaGetErrorString(cudaMemcpy(d_active, &h_active, memSize, cudaMemcpyHostToDevice)));
	printf("CUDA error: %s\n", cudaGetErrorString(cudaMemcpy(d_residual, &h_residual, memSize, cudaMemcpyHostToDevice)));
	printf("CUDA error: %s\n", cudaGetErrorString(cudaMemcpy(d_index, &h_index, memSize, cudaMemcpyHostToDevice)));
	memSize = MAXPSIZE*sizeof(int);
	printf("CUDA error: %s\n", cudaGetErrorString(cudaMemcpy(d_states, h_states, memSize, cudaMemcpyHostToDevice)));
	memSize = CELLX*CELLY*sizeof(cell);
	printf("CUDA error: %s\n", cudaGetErrorString(cudaMemcpy(d_cells, h_cells, memSize, cudaMemcpyHostToDevice)));
	memSize = MAXPSIZE*sizeof(particle);
	printf("CUDA error: %s\n", cudaGetErrorString(cudaMemcpy(d_elements, h_elements, memSize, cudaMemcpyHostToDevice)));
	//cudaDeviceSynchronize();

	for (int p = 0; p< 1000; p++){
		propagate<<<PBLOCKS, PBLOCKS>>>(lock, d_elements, d_cells, globalstate, d_states, d_index, d_active);
		setup_kernel <<<PBLOCKS, PBLOCKS >>>(globalstate, time(NULL));
		if  (ANNHILATION > 0){
			evolve_p_state<<<PBLOCKS, PBLOCKS>>>(d_elements, d_cells, globalstate, d_states, d_index, d_active);
		}
		setup_kernel <<<PBLOCKS, PBLOCKS >>>(globalstate, time(NULL));
	//	evolve_c_state<<<CBLOCKS, CBLOCKS>>>(d_elements, d_cells, globalstate, d_states, d_index, d_active);
	}

	memSize = MAXPSIZE*sizeof(particle);
	printf("CUDA error: %s\n", cudaGetErrorString(cudaMemcpy(h_elements, d_elements, memSize, cudaMemcpyDeviceToHost)));
	memSize = sizeof(int);
	printf("CUDA error: %s\n", cudaGetErrorString(cudaMemcpy(&h_active, d_active, memSize, cudaMemcpyDeviceToHost)));
	memSize = MAXPSIZE*sizeof(int);
	printf("CUDA error: %s\n", cudaGetErrorString(cudaMemcpy(h_states, d_states, memSize, cudaMemcpyDeviceToHost)));

	printf("Final particle count is: %i\n",h_active);

	for (int a = MAXPSIZE-1; a >= MAXPSIZE-10;a--){
		printf("Particle %i at coordinates %i, %i is alive %d\n", h_elements[a].id,h_elements[a].x, h_elements[a].y, h_elements[a].alive);
	}

return 0;
}
