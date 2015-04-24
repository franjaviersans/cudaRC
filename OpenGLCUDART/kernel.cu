#include "kernel.cuh"



struct stackNode
{
	int index;
	int actualChild;
};

__device__ bool ray_triangle( const float4 V1,  // Triangle vertices
                           const float4 V2,
                           const float4 V3,
                           const float4 O,  //Ray origin
                           const float4 D,  //Ray direction
							float *t,
						   float &u, 
						   float &v)
{

	float4 e1, e2;  //Edge1, Edge2
	float4 P, Q, T;
	float det, inv_det;
	
	//Find vectors for two edges sharing V1
	SUB(e1, V2, V1);
	SUB(e2, V3, V1);
	//Begin calculating determinant - also used to calculate u parameter
	CROSS(P, D, e2);
	//if determinant is near zero, ray lies in plane of triangle
	det = DOT(e1, P);
	//NOT CULLING
	if(det > -EPSILON && det < EPSILON) return false;
	inv_det = 1.f / det;
 
	//calculate distance from V1 to ray origin
	SUB(T, O, V1);
 
	//Calculate u parameter and test bound
	u = DOT(T, P) * inv_det;
	//The intersection lies outside of the triangle
	if(u < 0.f - EPSILON || u > 1.f + EPSILON) return false;
 
	//Prepare to test v parameter
	CROSS(Q, T, e1);
 
	//Calculate V parameter and test bound
	v = DOT(D, Q) * inv_det;
	//The intersection lies outside of the triangle
	if(v < 0.f - EPSILON  || u + v  > 1.f + EPSILON) return false;
 
	*t = DOT(e2, Q) * inv_det;
 
	return *t > EPSILON; //ray intersection
}

__device__ bool ray_box(const float4 & O,  //Ray origin
						const float4 & D,  //Ray direction
						float * t,
						const float3 & amin,
						const float3 & amax)
{

	float md;
	float tmin = -1.0f *FLT_MAX ;
	float tmax = FLT_MAX;
	*t = FLT_MAX;
	float4 p;
	p.x = (amin.x + amax.x)/2.0f - O.x;
	p.y = (amin.y + amax.y)/2.0f - O.y;
	p.z = (amin.z + amax.z)/2.0f - O.z;

	float e,f,t1,t2,aux;

	for(int i=0;i<3;++i){
		md = (i==0)?abs(amax.x - amin.x)/2.0f:((i==1)?abs(amax.y - amin.y)/2.0f:abs(amax.z - amin.z)/2.0f);
		e = (i==0)?p.x:((i==1)?p.y:p.z);
		f = (i==0)?D.x:((i==1)?D.y:D.z);

		if ( abs(f) > 0.000001f ){
			t1 = (e + md)/f;
			t2 = (e - md)/f;

			if(t1 > t2){ aux = t1; t1 = t2; t2 = aux;}
			if(t1 > tmin) tmin = t1;
			if(t2 < tmax) tmax = t2;
			if(tmin > tmax) return false;
			if(tmax < 0) return false;	
		}else if(-e - md > 0 || -e + md < 0) return false;
	}

	*t = tmin;

	return true;
}

__device__ int octreeRayIntersection(	const float4 & O,  //Ray origin
										const float4 & D,  //Ray direction
										const Cell * octree,
										const uint3 * const id,
										const float4 * const pos,
										float *dist,
										float &u,
										float &v)
{
	int actual = 0;
	int init, num;
	uint3 idtri;
	stackNode Stack[MAXDEPTH + 2];
	float4 V0, V1, V2;
	float3 amin, amax;
	stackNode *Node;
	int idInter = -1, child_index;
	float t;
	float v1, u1;

	Stack[actual].index = 0;
	Stack[actual].actualChild = 0;
	actual++;

	while(actual > 0)
	{

		Node = &Stack[actual - 1];


		if(Node->actualChild >= octree[Node->index].numChilds)
		{
			--actual;
		}
		else if(octree[Node->index].type == LEAF)
		{

			init = octree[Node->index].firstChild;
			num = octree[Node->index].numChilds;

			for(int i = 0; i < num; ++i)
			{
				idtri = id[octree[init + i].firstChild];
				V0 = pos[idtri.x];
				V1 = pos[idtri.y];
				V2 = pos[idtri.z];

				t = FLT_MAX;
				if(ray_triangle(V0, V1, V2, O, D, &t, u1, v1) && t < *dist)
				{
					*dist = t;
					u = u1;
					v = v1;
					idInter = octree[init + i].firstChild; 
					//printf("Alguien aca");
				}
			}

			--actual;
		}
		else if(octree[Node->index].type == INTERNAL)
		{
			child_index = octree[Node->index].firstChild + Node->actualChild; 

			amin.x = octree[child_index].minBox.x;
			amin.y = octree[child_index].minBox.y;
			amin.z = octree[child_index].minBox.z;
				
			amax.x = octree[child_index].maxBox.x;
			amax.y = octree[child_index].maxBox.y;
			amax.z = octree[child_index].maxBox.z;

			if(ray_box(O, D, &t, amin, amax) 
				&& t < *dist){
				//Insert the new child in the stack
				Stack[actual].index = child_index;
				Stack[actual].actualChild = 0;
				++actual;
				if(Stack[actual].index == 9){ 
					actual = 3;
				}
			}

			++Node->actualChild;
		}
		else
		{
		/*	printf("AJA \n");
			for(unsigned int i =0; i< actual;++i){
				printf("%d %d\n", Stack[i].index, Stack[i].actualChild );
			}
			for(int k = 0;k<= Node->index; ++k)
				printf("%d %d %d\n",octree[k].type, INTERNAL, LEAF);*/
			printf("SHOULD NEVER COME HERE %d %d\n", Node->index, octree[Node->index].type);
		}
	}

	return idInter;
}



__global__ void kernelRC(uchar4 *buffer, const unsigned int width, const unsigned int height, 
						 const uint3 * const id, const float4 * const pos, const float4  * const normal, const float2 * const tex,  
						 const unsigned int num_vert, const unsigned int num_tri, const Options options, const Cell * const octree)
{
	/*unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int tpos = y * width + x;

	

	if(x < width && y < height)
	{

		buffer[tpos].x = 0;
		buffer[tpos].y = 0;
		buffer[tpos].z = 0;
		buffer[tpos].w = 0;

		float4 origin;
		float4 dir;
		float4 vaux;

		//dir(options.priX + x * options.incX, options.priY + y * options.incY, -1.0f ,0.0f)
		origin.x = 0.0f; origin.y = 0.0f; origin.z = 0; origin.w = 1;
		dir.x = options.priX + x * options.incX; dir.y = options.priY + y * options.incY; dir.z = -1; dir.w = 1;

		SUB(dir, dir, origin);

		vaux = origin;
		MULT(origin, options.modelView, vaux);

		vaux = dir;
		MULT(dir, options.modelView, vaux);


		unsigned int i=0;
		uint3 idtri;
		for(; i < num_tri; ++i)
		{
			//if(x>=300 && x<=600 && y >= 300 && y <= 600)
			//{
			idtri = id[i];
			float4 V0, V1, V2;

			V0 = pos[idtri.x];
			V1 = pos[idtri.y];
			V2 = pos[idtri.z];
			
			float t = FLT_MAX;
			if(ray_triangle( V0, V1, V2, origin, dir, &t))
			{
				buffer[tpos].x = 255;
				buffer[tpos].y = 255;
				buffer[tpos].z = 255;
				buffer[tpos].w = 255;
			}
			//}
		}
	}*/
	
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int tpos = y * width + x;
	float u, v;

	if(x < width && y < height)
	{

		int intersect = -1;
		float D = FLT_MAX;

		buffer[tpos].x = 0;
		buffer[tpos].y = 0;
		buffer[tpos].z = 0;
		buffer[tpos].w = 0;

		float4 origin;
		float4 dir;
		float4 vaux;
		//dir(options.priX + x * options.incX, options.priY + y * options.incY, -1.0f ,0.0f)
		origin.x = 0.0f; origin.y = 0.0f; origin.z = 0.0f; origin.w = 1.0f;
		dir.x = options.priX + x * options.incX; dir.y = options.priY + y * options.incY; dir.z = -1.0f; dir.w = 1.0f;

		SUB(dir, dir, origin);

		vaux = origin;
		MULT(origin, options.modelView, vaux);

		vaux = dir;
		MULT(dir, options.modelView, vaux);

		intersect = octreeRayIntersection(origin, dir, octree, id, pos, &D, u, v);
		if(intersect != -1)
		{

			uint3 idtri = id[intersect];
			float4 V0 = normal[idtri.x];
			float4 V1 = normal[idtri.y];
			float4 V2 = normal[idtri.z];
			if(u < 0.0f - EPSILON || u  > 1.0f + EPSILON  || v <0.0f - EPSILON  || v > 1.0f + EPSILON  ) printf("no deberia ocurrir %f %f %f \n",u, v, 1.0f - (u+v));
			BARI(buffer[tpos].x, 255.0f * V0.x, 255.0f * V1.x, 255.0f * V2.x, u, v);
			BARI(buffer[tpos].y, 255.0f * V0.y, 255.0f * V1.y, 255.0f * V2.y, u, v);
			BARI(buffer[tpos].z, 255.0f * V0.z, 255.0f * V1.z, 255.0f * V2.z, u, v);
			buffer[tpos].w = 255;
		}
	}
}



CUDAClass::CUDAClass()
{
	d_pos = NULL;
	d_normal = NULL;
	d_tex = NULL;
	d_id = NULL;
}

CUDAClass::~CUDAClass()
{
	checkCudaErrors(cudaFree(d_pos));
	checkCudaErrors(cudaFree(d_normal));
	checkCudaErrors(cudaFree(d_tex));
	checkCudaErrors(cudaFree(d_id));
	checkCudaErrors(cudaFree(d_octree));

	cudaDeviceReset();
}


void CUDAClass::cudaSetObject(const std::vector<CVertex> *ptr_puntos,const std::vector<CTriangle> *ptr_caras, const vector<Cell> *ptr_octree)
{
	float4 *h_pos = new float4[(*ptr_puntos).size()];
	float4 *h_normal = new float4[(*ptr_puntos).size()];
	float2 *h_tex = new float2[(*ptr_puntos).size()];
	uint3 *h_id = new uint3[(*ptr_caras).size()];

	num_vert = (*ptr_puntos).size();
	num_tri = (*ptr_caras).size();

	checkCudaErrors(cudaMalloc((void**)&d_id,sizeof(uint3) * (*ptr_caras).size()));
	checkCudaErrors(cudaMalloc((void**)&d_pos,sizeof(float4) * (*ptr_puntos).size()));
	checkCudaErrors(cudaMalloc((void**)&d_normal,sizeof(float4) * (*ptr_puntos).size()));
	checkCudaErrors(cudaMalloc((void**)&d_tex,sizeof(float2) * (*ptr_puntos).size()));
	checkCudaErrors(cudaMalloc((void**)&d_octree, sizeof(Cell) * (*ptr_octree).size()));
	


	for(unsigned int i=0;i<(*ptr_caras).size();++i)
	{
		h_id[i].x = (*ptr_caras)[i].V0;
		h_id[i].y = (*ptr_caras)[i].V1;
		h_id[i].z = (*ptr_caras)[i].V2;
	}


	for(unsigned int i=0;i<(*ptr_puntos).size();++i)
	{
		h_pos[i].x = (*ptr_puntos)[i].v.x;
		h_pos[i].y = (*ptr_puntos)[i].v.y;
		h_pos[i].z = (*ptr_puntos)[i].v.z;
		h_pos[i].w = 1.0f;


		h_normal[i].x = (*ptr_puntos)[i].normal.x;
		h_normal[i].y = (*ptr_puntos)[i].normal.y;
		h_normal[i].z = (*ptr_puntos)[i].normal.z;
		h_normal[i].w = 0.0f;

		h_tex[i].x = (*ptr_puntos)[i].texture.x;
		h_tex[i].y = (*ptr_puntos)[i].texture.y;
	}

	checkCudaErrors(cudaMemcpy(d_id,h_id, sizeof(uint3) * (*ptr_caras).size(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_pos,h_pos, sizeof(float4) * (*ptr_puntos).size(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_normal,h_normal, sizeof(float4) * (*ptr_puntos).size(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_tex,h_tex, sizeof(float2) * (*ptr_puntos).size(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_octree, (*ptr_octree).data(), sizeof(Cell) * (*ptr_octree).size(), cudaMemcpyHostToDevice));


	delete [] h_pos;
	delete [] h_normal;
	delete [] h_tex;
	delete [] h_id;

}


// Helper function for using CUDA to add vectors in parallel.
void CUDAClass::cudaRC(uchar4 *d_buffer, unsigned int width, unsigned int height, Options options)
{
	dim3 blockDim(16, 16, 1);
	dim3 gridDim((width + blockDim.x)/blockDim.x, (height + blockDim.y)/blockDim.y, 1);

	GpuTimer timer;
	timer.Start();
	kernelRC<<<gridDim, blockDim>>>(d_buffer, width, height, d_id, d_pos, d_normal, d_tex, num_vert, num_tri, options, d_octree);
	timer.Stop();

	printf("%f \n", timer.Elapsed());

	// Check for any errors launching the kernel
    checkCudaErrors(cudaGetLastError());
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
	checkCudaErrors(cudaDeviceSynchronize());

	
}

