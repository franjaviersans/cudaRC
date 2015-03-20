#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "include/glm/glm.hpp"
#include "include/glm/gtc/matrix_transform.hpp"
#include "timer.h"
#include "utils.h"
#include "Vertex.h"
#include "Triangle.h"
#include <vector>
#include <stdio.h>


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
	unsigned int num_vert, num_tri;

	void cudaRC(uchar4 *, unsigned int, unsigned int, Options &);
	void cudaSetObject(const std::vector<CVertex> *ptr_puntos,const std::vector<CTriangle> *ptr_caras);

private:

};




__global__ void kernelRC(uchar4 *, unsigned int, unsigned int, uint3, float4, float4, float2, Options);
