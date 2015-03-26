#ifndef AABB_H
#define AABB_H

#include "cuda_runtime.h"

#include "Vector4D.h"

class AABB{
	public:
		CVector4D amin;
		CVector4D amax;

	public:
		AABB(){};
		AABB(CVector4D min, CVector4D max):amin(min), amax(max){}
		bool Interseccion(AABB A);
		//bool Interseccion(float4 ray,float &D, int &intersec);
};

#endif 