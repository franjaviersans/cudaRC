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


/********************************************************/

/* AABB-triangle overlap test code                      */

/* by Tomas Akenine-Möller                              */

/* Function: int triBoxOverlap(float boxcenter[3],      */

/*          float boxhalfsize[3],float triverts[3][3]); */

/* History:                                             */

/*   2001-03-05: released the code in its first version */

/*   2001-06-18: changed the order of the tests, faster */

/*                                                      */

/* Acknowledgement: Many thanks to Pierre Terdiman for  */

/* suggestions and discussions on how to optimize code. */

/* Thanks to David Hunt for finding a ">="-bug!         */

/********************************************************/




int AABB::planeBoxOverlap(CVector4D normal, CVector4D vert, CVector4D maxbox)	// -NJMP-
{
	int q;

	float v, sign;
	CVector4D vmin, vmax;

	for(q=X;q<=Z;q++)
	{
		v=vert[q];					// -NJMP-
		sign = (normal[q]>0.0f)? -1:1;

		if(q == X )
		{
			vmin.x= sign * maxbox[q] - v;	// -NJMP-
			vmax.x= -sign * maxbox[q] - v;	// -NJMP-
		}
		else if(q == Y )
		{
			vmin.y= sign * maxbox[q] - v;	// -NJMP-
			vmax.y= -sign * maxbox[q] - v;	// -NJMP-
		}
		else if(q == Z )
		{
			vmin.z= sign * maxbox[q] - v;	// -NJMP-
			vmax.z= -sign * maxbox[q] - v;	// -NJMP-
		}
	}

	if(DOT(normal,vmin) >0.0f) return 0;	// -NJMP-
	if(DOT(normal,vmax) >=0.0f) return 1;	// -NJMP-

	return 0;
}







int AABB::triBoxOverlap(CVector4D vV0, CVector4D vV1, CVector4D vV2)
{
  /*    use separating axis theorem to test overlap between triangle and box */

  /*    need to test for overlap in these directions: */

  /*    1) the {x,y,z}-directions (actually, since we use the AABB of the triangle */

  /*       we do not even need to test these) */

  /*    2) normal of the triangle */

  /*    3) crossproduct(edge from tri, {x,y,z}-directin) */

  /*       this gives 3x3=9 more tests */

   CVector4D v0,v1,v2;

   CVector4D boxcenter, boxhalfsize;

   boxcenter = (amin + amax) / 2.0f;
   boxhalfsize = (amax - amin) / 2.0f;
//   float axis[3];

   float min,max,p0,p1,p2,rad,fex,fey,fez;		// -NJMP- "d" local variable removed
   CVector4D normal,e0,e1,e2;



   /* This is the fastest branch on Sun */

   /* move everything so that the boxcenter is in (0,0,0) */

   SUB(v0,vV0,boxcenter);
   SUB(v1,vV1,boxcenter);
   SUB(v2,vV2,boxcenter);



   /* compute triangle edges */

   SUB(e0,v1,v0);      /* tri edge 0 */
   SUB(e1,v2,v1);      /* tri edge 1 */
   SUB(e2,v0,v2);      /* tri edge 2 */



   /* Bullet 3:  */

   /*  test the 9 tests first (this was faster) */

   fex = fabsf(e0[X]);
   fey = fabsf(e0[Y]);
   fez = fabsf(e0[Z]);

   AXISTEST_X01(e0[Z], e0[Y], fez, fey);
   AXISTEST_Y02(e0[Z], e0[X], fez, fex);
   AXISTEST_Z12(e0[Y], e0[X], fey, fex);



   fex = fabsf(e1[X]);
   fey = fabsf(e1[Y]);
   fez = fabsf(e1[Z]);

   AXISTEST_X01(e1[Z], e1[Y], fez, fey);
   AXISTEST_Y02(e1[Z], e1[X], fez, fex);
   AXISTEST_Z0(e1[Y], e1[X], fey, fex);



   fex = fabsf(e2[X]);
   fey = fabsf(e2[Y]);
   fez = fabsf(e2[Z]);

   AXISTEST_X2(e2[Z], e2[Y], fez, fey);
   AXISTEST_Y1(e2[Z], e2[X], fez, fex);
   AXISTEST_Z12(e2[Y], e2[X], fey, fex);



   /* Bullet 1: */

   /*  first test overlap in the {x,y,z}-directions */

   /*  find min, max of the triangle each direction, and test for overlap in */

   /*  that direction -- this is equivalent to testing a minimal AABB around */

   /*  the triangle against the AABB */



   /* test in X-direction */

   FINDMINMAX(v0[X],v1[X],v2[X],min,max);
   if(min>boxhalfsize[X] || max<-boxhalfsize[X]) return 0;



   /* test in Y-direction */

   FINDMINMAX(v0[Y],v1[Y],v2[Y],min,max);
   if(min>boxhalfsize[Y] || max<-boxhalfsize[Y]) return 0;



   /* test in Z-direction */

   FINDMINMAX(v0[Z],v1[Z],v2[Z],min,max);
   if(min>boxhalfsize[Z] || max<-boxhalfsize[Z]) return 0;



   /* Bullet 2: */

   /*  test if the box intersects the plane of the triangle */

   /*  compute plane equation of triangle: normal*x+d=0 */

   CROSS(normal,e0,e1);

   // -NJMP- (line removed here)

   if(!planeBoxOverlap(normal,v0,boxhalfsize)) return 0;	// -NJMP-



   return 1;   /* box and triangle overlaps */

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