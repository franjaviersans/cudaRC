#ifndef OBJECT_H
#define OBJECT_H

#include <string>
#include <vector>
#include "Vertex.h"
#include "Triangle.h"

class CMyObject{

	public:
		CMyObject(){};
		virtual bool openFile(const std::string& pFile) = 0;
		virtual std::vector<CVertex> * getVertex() = 0;
		virtual std::vector<CTriangle> * getFaces() = 0;
};

#endif 