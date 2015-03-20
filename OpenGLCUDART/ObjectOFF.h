#ifndef objetooff_H
#define objetooff_H

#include <string>
#include <iterator>
#include <vector>
#include <fstream>
#include <stdlib.h>
#include <algorithm>
#include "Object.h"


class CObjectOFF: public CMyObject {

	private:
		std::fstream entrada;
		unsigned int num, i, j, tam, aux;
		float pointx, pointy, pointz;
		float maxx,minx,maxy,miny,maxz,minz;

	public:
		std::vector<CVertex> vertex;
		std::vector<CTriangle> faces;

		CObjectOFF();
		CObjectOFF(float arr[16],const std::vector<CVertex> &ptr_puntos,const std::vector<CTriangle> &ptr_caras);
		~CObjectOFF();
		void norm();
		void center();
		void normalize();
		bool openFile(const std::string& pFile);
		void Draw();
		int size();
		void print();
		std::vector<CVertex> * getVertex(){return &vertex;};
		std::vector<CTriangle> * getFaces(){return &faces;};
};

#endif 