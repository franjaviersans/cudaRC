#include "Octree.h"
#include <float.h>
#include <iostream>


Octree::Octree(const std::vector<CVertex> * const vertex, const std::vector<CTriangle> * const faces, AABB objectAABB){
	m_vertex = vertex;
	m_faces = faces;
	Hoja = true;
	nivel = 0;
	Caja = objectAABB;
	
	for(unsigned int i=0;i< (*faces).size();++i)
	{
		primitivas.push_back(i);
	}

	if(primitivas.size() > MAXINLEAF)
	{
		Subdividir();
	}
	
}

Octree::Octree(AABB a, int nive, const std::vector<CVertex> * const vertex, const std::vector<CTriangle> * const faces)
{
	Caja = AABB(a.amin,a.amax); 
	Hoja = true; 
	nivel = nive+1;
	m_vertex = vertex;
	m_faces = faces;

	cout<<nivel<<endl;
}

Octree::~Octree(){

}

AABB CalcularCaja(CVector4D V0, CVector4D V1, CVector4D V2){
	CVector4D mini, maxi;
	mini.x = min(V0.x, min(V1.x, V2.x));
	mini.y = min(V0.y, min(V1.y, V2.y));
	mini.z = min(V0.z, min(V1.z, V2.z));

	maxi.x = max(V0.x, max(V1.x, V2.x));
	maxi.y = max(V0.y, max(V1.y, V2.y));
	maxi.z = max(V0.z, max(V1.z, V2.z));

	AABB caja(mini, maxi);

	return caja;
}

void Octree::Subdividir(){
	Hoja =  false;

	CVector4D centro = (Caja.amin+Caja.amax)/2.0f; 
	hijos[0] = new Octree(AABB(Caja.amin,centro),nivel, m_vertex, m_faces);
	hijos[1] = new Octree(AABB(CVector4D(centro.x,Caja.amin.y,Caja.amin.z, 1.0f),CVector4D(Caja.amax.x,centro.y,centro.z, 1.0f)),nivel, m_vertex, m_faces);
	hijos[2] = new Octree(AABB(CVector4D(centro.x,Caja.amin.y,centro.z, 1.0f),CVector4D(Caja.amax.x,centro.y,Caja.amax.z, 1.0f)),nivel, m_vertex, m_faces);
	hijos[3] = new Octree(AABB(CVector4D(Caja.amin.x,Caja.amin.y,centro.z, 1.0f),CVector4D(centro.x,centro.y,Caja.amax.z, 1.0f)),nivel, m_vertex, m_faces);
	hijos[4] = new Octree(AABB(CVector4D(Caja.amin.x,centro.y,Caja.amin.z, 1.0f),CVector4D(centro.x,Caja.amax.y,centro.z, 1.0f)),nivel, m_vertex, m_faces);
	hijos[5] = new Octree(AABB(CVector4D(centro.x,centro.y,Caja.amin.z, 1.0f),CVector4D(Caja.amax.x,Caja.amax.y,centro.z, 1.0f)),nivel, m_vertex, m_faces);
	hijos[6] = new Octree(AABB(centro,Caja.amax),nivel, m_vertex, m_faces);
	hijos[7] = new Octree(AABB(CVector4D(Caja.amin.x,centro.y,centro.z, 1.0f),CVector4D(centro.x,Caja.amax.y,Caja.amax.z, 1.0f)),nivel, m_vertex, m_faces);

	for(int j=0;j<8;++j)
	{
		for each (unsigned int tri_index in primitivas)
		{

			AABB caja = CalcularCaja((*m_vertex)[(*m_faces)[tri_index].V0].v, (*m_vertex)[(*m_faces)[tri_index].V1].v, (*m_vertex)[(*m_faces)[tri_index].V2].v);

			if(hijos[j]->Caja.Interseccion(caja))
			{
				hijos[j]->primitivas.push_back(tri_index);
			}
		}

		/*if(hijos[j]->primitivas.size() > MAXINLEAF)
		{
			cout<< hijos[j]->primitivas.size()<<endl;
			hijos[j]->Subdividir();
		}*/
	}
}



/*Objeto* Octree::Recorrer(Rayo ray,float &D, int &intersec){

	float Dist = FLT_MAX;
	int intersec_aux;
	Objeto* id = NULL;

	if(Caja.Interseccion(ray,Dist,intersec_aux)){

		if(!Hoja){
			Objeto* id_aux = NULL;

			for(int j=0;j<8;++j){
				id_aux = hijos[j]->Recorrer(ray,Dist,intersec_aux);
				if(id_aux != NULL && Dist < D){
					D = Dist;
					id = id_aux;
					intersec = intersec_aux;
				}
			}		
			return id;

		}else{
			for(unsigned int i=0;i<primitivas.size();++i){
				if(primitivas[i]->Interseccion(ray, Dist,intersec_aux)){
					if(id == NULL || Dist < D){
						D = Dist;
						id = primitivas[i];
						intersec = intersec_aux;
					}
				}
			}

			return id;
		}
	}

	return NULL;
}

Objeto* Octree::Recorrer_Ocluder(Rayo ray,float fLightDist){

	float Dist = FLT_MAX;
	int intersec_aux;
	Objeto* id=NULL;

	if(Caja.Interseccion(ray,Dist,intersec_aux)){

		if(!Hoja){
			Objeto* id_aux = NULL;

			for(int j=0;j<8;++j){
				id_aux = hijos[j]->Recorrer_Ocluder(ray,Dist);
				if(id_aux != NULL){
					return id_aux;
				}
			}		

		}else{
			for(unsigned int i=0;i<primitivas.size();++i){
				if(primitivas[i]->Interseccion(ray, Dist,intersec_aux)){
					if(Dist < fLightDist){
						return primitivas[i];
					}
				}
			}
		}
	}

	return NULL;
}*/