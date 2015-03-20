#ifndef Triangle_H
#define Triangle_H

#include "Vector4D.h"
#include "Vertex.h"
#include <iterator>

class CWriteTriangle;
class CTriangle{
public:
	unsigned int V0;
	unsigned int V1;
	unsigned int V2;
	CVector4D normal;

	CTriangle(){};
	unsigned int operator [](unsigned int &index){
		if(index == 0) return V0;
		if(index == 1) return V1;
		if(index == 2) return V2;

		return 0;
	}

	unsigned int operator [](int &index){
		if(index == 0) return V0;
		if(index == 1) return V1;
		if(index == 2) return V2;

		return 0;
	}

	unsigned int operator [](int index){
		if(index == 0) return V0;
		if(index == 1) return V1;
		if(index == 2) return V2;

		return 0;
	}

	void setVertexIndex(int index, unsigned int value){
		if(index == 0) V0 = value;
		if(index == 1) V1 = value;
		if(index == 2) V2 = value;
	}
};


#endif