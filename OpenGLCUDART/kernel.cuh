#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "include/glm/glm.hpp"
#include "include/glm/gtc/matrix_transform.hpp"
#include "timer.h"
#include "utils.h"
#include "Vertex.h"
#include "Triangle.h"
#include "Octree.h"
#include <vector>
#include <stdio.h>

using std::vector;

struct Options{
	float priX;
	float priY;
	float incX;
	float incY;
	float modelView[16];
};

class CUDAClass
{
public:
	CUDAClass();
	~CUDAClass();
	float4 *d_pos;
	float4 *d_normal;
	float2 *d_tex;
	uint3 *d_id;
	Cell * d_octree;
	unsigned int num_vert, num_tri;

	void cudaRC(uchar4 *, unsigned int, unsigned int, Options &);
	void cudaSetObject(const vector<CVertex> *ptr_puntos,const vector<CTriangle> *ptr_caras, const vector<Cell> *ptr_octree);

private:

};




__global__ void kernelRC(uchar4 *buffer, const unsigned int width, const unsigned int height, 
						 const uint3 * const id, const float4 * const pos, const float4  * const normal, const float2 * const tex,  
						 const unsigned int num_vert, const unsigned int num_tri, const Options options, const Cell * const octree);
