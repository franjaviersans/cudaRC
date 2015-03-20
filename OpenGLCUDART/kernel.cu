#include "kernel.cuh"

#define EPSILON 0.000001
#define CROSS(dest, v1, v2) \
	dest.x = v1.y*v2.z - v1.z*v2.y; \
	dest.y = v1.z*v2.x - v1.x*v2.z; \
	dest.z = v1.x*v2.y - v1.y*v2.x; \
	dest.w = 0;
#define DOT(v1, v2) (v1.x*v2.x+v1.y*v2.y+v1.z*v2.z + v1.w * v2.w)
#define SUB(dest, v1, v2) \
	dest.x = v1.x - v2.x; \
	dest.y = v1.y - v2.y; \
	dest.z = v1.z - v2.z; \
	dest.w = v1.w - v2.w;

#define MULT(dest, mat,p) \
	dest.x = mat[0] * p.x + mat[4] * p.y + mat[8] * p.z + mat[12] * p.w; \
	dest.y = mat[1] * p.x + mat[5] * p.y + mat[9] * p.z + mat[13] * p.w; \
	dest.z = mat[2] * p.x + mat[6] * p.y + mat[10] * p.z + mat[14] * p.w;\
	dest.w = mat[3] * p.x + mat[7] * p.y + mat[11] * p.z + mat[15] * p.w; 


__device__ bool ray_triangle( const float4 V1,  // Triangle vertices
                           const float4 V2,
                           const float4 V3,
                           const float4 O,  //Ray origin
                           const float4 D  //Ray direction
						   )
{

	float4 e1, e2;  //Edge1, Edge2
	float4 P, Q, T;
	float det, inv_det, u, v;
	float t;
 
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
	if(u < 0.f || u > 1.f) return false;
 
	//Prepare to test v parameter
	CROSS(Q, T, e1);
 
	//Calculate V parameter and test bound
	v = DOT(D, Q) * inv_det;
	//The intersection lies outside of the triangle
	if(v < 0.f || u + v  > 1.f) return false;
 
	t = DOT(e2, Q) * inv_det;
 
	return t > EPSILON; //ray intersection
}

__global__ void kernelRC(uchar4 *buffer, unsigned int width, unsigned int height, 
						 uint3 * id, float4 * pos, float4  * normal, float2 * tex, 
						 unsigned int num_vert, unsigned int num_tri, Options options)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
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

		//dir(options.priX + x * options.incX, options.priY + y * options.incY, -1.0f ,0.0f)
		origin.x = 0.0f; origin.y = 0.0f; origin.z = 0; origin.w = 1;
		dir.x = options.priX + x * options.incX; dir.y = options.priY + y * options.incY; dir.z = -1; dir.w = 1;

		SUB(dir, dir, origin);


		unsigned int i=0;
		uint3 idtri;
		for(; i < num_tri; ++i)
		{
			//if(x>=300 && x<=600 && y >= 300 && y <= 600)
			//{
			idtri = id[i];
			float4 vaux, V0, V1, V2;

			vaux = pos[idtri.x];
			MULT(V0, options.modelView, vaux);

			vaux = pos[idtri.y];
			MULT(V1, options.modelView, vaux);

			vaux = pos[idtri.z];
			MULT(V2, options.modelView, vaux);
			
			
			if(ray_triangle( V0, V1, V2, origin, dir))
			{
				buffer[tpos].x = 255;
				buffer[tpos].y = 255;
				buffer[tpos].z = 255;
				buffer[tpos].w = 255;
			}
			//}
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
}


void CUDAClass::cudaSetObject(const std::vector<CVertex> *ptr_puntos,const std::vector<CTriangle> *ptr_caras)
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


	delete [] h_pos;
	delete [] h_normal;
	delete [] h_tex;
	delete [] h_id;

}


// Helper function for using CUDA to add vectors in parallel.
void CUDAClass::cudaRC(uchar4 *d_buffer, unsigned int width, unsigned int height, Options & options)
{
	dim3 blockDim(16, 16, 1);
	dim3 gridDim((width + blockDim.x)/blockDim.x, (height + blockDim.y)/blockDim.y, 1);

	GpuTimer timer;
	timer.Start();
	kernelRC<<<gridDim, blockDim>>>(d_buffer, width, height, d_id, d_pos, d_normal, d_tex, num_vert, num_tri, options);
	timer.Stop();

	printf("%f \n", timer.Elapsed());

	// Check for any errors launching the kernel
    checkCudaErrors(cudaGetLastError());
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
	checkCudaErrors(cudaDeviceSynchronize());

	
}

