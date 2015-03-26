#ifndef Octree_H
#define Octree_H
#include "AABB.h"
#include "Vertex.h"
#include "Triangle.h"
#include <vector>
using namespace std;

#define MAXINLEAF 4

class Octree{
	public:
		AABB Caja;
		vector<unsigned int> primitivas;
		bool Hoja;
		unsigned int nivel;
		Octree *hijos[8];

	public:
		Octree(const std::vector<CVertex> * const vertex, const std::vector<CTriangle> * const faces, AABB objectAABB);
		Octree(AABB a, int nive, const std::vector<CVertex> * const vertex, const std::vector<CTriangle> * const faces);
		void Subdividir();
		~Octree();
		/*Objeto* Recorrer(Rayo ray,float &D, int &intersec);
		Objeto* Recorrer_Ocluder(Rayo ray,float fLightDist);*/

	private:
		const std::vector<CVertex> *m_vertex;
		const std::vector<CTriangle> *m_faces;
		
};

#endif