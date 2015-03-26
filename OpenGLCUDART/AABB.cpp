#include "AABB.h"
#include <float.h>
#include <algorithm>

bool AABB::Interseccion(AABB A){
	for(int i=0;i<3;++i){
		if(amin[i] > A.amax[i] || A.amin[i] > amax[i])
			return false;
	}
	return true;
}

/*
bool AABB::Interseccion(float4 ray,float &D, int &intersec){

	float md;
	float tmin = -1*FLT_MAX ;
	float tmax = FLT_MAX ;
	punto p = (amin + amax)/2.0f - ray.origen;
	float e,f,t1,t2;

	for(int i=0;i<3;++i){
		md = abs(amax[i] - amin[i])/2.0f;
		e = p[i];
		f = ray.direccion[i];
		if ( abs(f) > 0.000001f ){
			t1 = (e + md)/f;
			t2 = (e - md)/f;

			if(t1 > t2) std::swap(t1,t2);
			if(t1 > tmin) tmin = t1;
			if(t2 < tmax) tmax = t2;
			if(tmin > tmax) return false;
			if(tmax < 0) return false;	
		}else if(-e - md > 0 || -e + md < 0) return false;
	}

	return true;
}
*/