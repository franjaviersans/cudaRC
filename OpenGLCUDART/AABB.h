#ifndef AABB_H
#define AABB_H

#include "cuda_runtime.h"

#include "Vector4D.h"
#include "Definitions.h"
#include "Triangle.h"
#include <math.h>


class AABB{
	public:
		CVector4D amin;
		CVector4D amax;

	public:
		AABB(){};
		AABB(CVector4D min, CVector4D max):amin(min), amax(max){}
		bool Interseccion(AABB A);
		int triBoxOverlap(CVector4D vV0, CVector4D vV1, CVector4D vV2);
		int planeBoxOverlap(CVector4D normal, CVector4D vert, CVector4D maxbox);
		//bool Interseccion(float4 ray,float &D, int &intersec);
};

#endif 