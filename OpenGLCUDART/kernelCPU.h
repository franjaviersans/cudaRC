#ifndef KERNELCPU_H
#define KERNELCPU_H

#include "Vertex.h"
#include "Triangle.h"
#include "Octree.h"
#include "kernel.cuh"
#include <vector>
#include <stdio.h>


using std::vector;

class CPURCClass
{
public:
	CPURCClass();
	~CPURCClass();
	float4 *h_pos;
	float4 *h_normal;
	float2 *h_tex;
	uint3 *h_id;
	Cell *h_octree;
	unsigned int num_vert, num_tri;

	void RC(uchar4 *, unsigned int, unsigned int, Options);
	void SetObject(const vector<CVertex> *ptr_puntos,const vector<CTriangle> *ptr_caras, const vector<Cell> *ptr_octree);

private:

};

#endif