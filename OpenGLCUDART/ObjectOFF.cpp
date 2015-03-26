#include "ObjectOFF.h"
#include <limits>       // std::numeric_limits

CObjectOFF::CObjectOFF(){
	//Incializa todas las variables 

	maxi.x = -9999999;
	mini.x = 9999999;
	maxi.y = -9999999;
	mini.y = 9999999;
	maxi.z = -9999999;
	mini.z = 9999999;
}

CObjectOFF::CObjectOFF(float arr[16],const std::vector<CVertex> &ptr_puntos,const std::vector<CTriangle> &ptr_caras){
	CObjectOFF();
	unsigned int i,j;

	for(i = 0;i<ptr_puntos.size();++i){
		CVertex vert;
		vert.v.init(ptr_puntos[i].v.x*arr[0]+ptr_puntos[i].v.y*arr[4]+ptr_puntos[i].v.z*arr[8]+arr[12],
			ptr_puntos[i].v.x*arr[1]+ptr_puntos[i].v.y*arr[5]+ptr_puntos[i].v.z*arr[9]+arr[13],
			ptr_puntos[i].v.x*arr[2]+ptr_puntos[i].v.y*arr[6]+ptr_puntos[i].v.z*arr[10]+arr[14],1.0f);

		vertex.push_back(vert);

		maxi.x = std::max(maxi.x,vert.v.x);
		mini.x = std::min(mini.x,vert.v.x);
		maxi.y = std::max(maxi.y,vert.v.y);
		mini.y = std::min(mini.y,vert.v.y);
		maxi.z = std::max(maxi.z,vert.v.z);
		mini.z = std::min(mini.z,vert.v.z);
	}

	for(std::vector<CTriangle>::iterator f = faces.begin();f != faces.end();++f){
		CTriangle tri;

		for(j = 0;j<3;++j){
			tri.setVertexIndex(j, (*f)[j]);
		}

		faces.push_back(tri);
	}

	norm();
}

CObjectOFF::~CObjectOFF(){
	if(!vertex.empty())	vertex.clear();
	if(!faces.empty())	faces.clear();
}

void CObjectOFF::norm(){
	//Saca los vectores normales de las caras
	CVector4D a,b,c,au1,au2;
	
	for(std::vector<CTriangle>::iterator f = faces.begin();f != faces.end();++f){
		//Calculate normal per face
		//Se agarran 3 puntos del plano para poder sacar el vector normal
		CVector4D a(vertex[(*f)[0]].v);
		CVector4D b(vertex[(*f)[1]].v);
		CVector4D c(vertex[(*f)[2]].v);
		//Se hace la resta entre b-c y entre a-c
		CVector4D au1=a-b;
		CVector4D au2=b-c;
		//Se hace el producto cruz entre los vectores resultantes au1xau2
		(*f).normal= producto_cruz(au1 , au2);

		(*f).normal.normalizar();

		for(j=0; j < 3 ;++j)
			vertex[(*f)[j]].normal += (*f).normal;
	}

	//Se normaliza cada uno de los vectores normales de cada vertice
	for(i=0;i<vertex.size();++i)
		vertex[i].normal.normalizar();
}

void CObjectOFF::center(){
	//Busca los centros de cada eje para centrar
	float cx=(maxi.x+mini.x)/2, cy=(maxi.y+mini.y)/2, cz=(maxi.z+mini.z)/2;

	//Centra los puntos del bbox
	mini.x = mini.x-cx;
	maxi.x = maxi.x-cx;
	mini.y = mini.y-cy;
	maxi.y = maxi.y-cy;
	mini.z = mini.z-cz;
	maxi.z = maxi.z-cz;

	//Centra cada uno de los puntos del objetooff
	for(i=0;i<vertex.size();++i){
		vertex[i].v.x = vertex[i].v.x-cx;
		vertex[i].v.y = vertex[i].v.y-cy;
		vertex[i].v.z = vertex[i].v.z-cz;
	}
}

void CObjectOFF::normalize(){
	float norm;
			
	//Se busca el mayor de las dimensiones para normalizar con esa dimension
	norm=std::max(maxi.x-mini.x,std::max(maxi.y-mini.y,maxi.z-mini.z)); 

	mini.x/=norm;
	maxi.x/=norm;
	mini.y/=norm;
	maxi.y/=norm;
	mini.z/=norm;
	maxi.z/=norm;

	//Aplicar la normalizacion a cada punto
	for(i=0;i<vertex.size();++i){
		vertex[i].v.x /= norm;
		vertex[i].v.y /= norm;
		vertex[i].v.z /= norm;
	}
}

bool CObjectOFF::openFile(const std::string& pFile){
	int i;

	CVector4D auxiliar;
	int num_points, num_caras, v0, v1, v2;

	entrada.open(pFile, std::ios::in);
	char auxstring[4];

	if (!entrada.is_open())
		return false;

	entrada>>auxstring;
	entrada>>num_points>>num_caras>>num;
	if(!vertex.empty()) vertex.clear();
	for(i=0;i<num_points;++i){
		CVertex punto_auxi;

		entrada>>pointx>>pointy>>pointz;
		
		//Busca los puntos maximos y minimos de cada eje para poder crear la caja envolvente
		if(pointx<mini.x) mini.x=pointx;
		if(pointx>maxi.x) maxi.x=pointx;
		if(pointy<mini.y) mini.y=pointy;
		if(pointy>maxi.y) maxi.y=pointy;
		if(pointz<mini.z) mini.z=pointz;
		if(pointz>maxi.z) maxi.z=pointz;

		//Se guarda cada punto en un vector
		punto_auxi.v.init(pointx,pointy,pointz,1.0f);

		vertex.push_back(punto_auxi);
	}

	//Se guarda cada cara en un vector
	for(i=0;i<num_caras;++i){
		CTriangle auxi_tri;
		
		entrada>>tam;
		if(tam == 3){

			for(j=0;j<tam;++j){
				entrada>>aux;
				auxi_tri.setVertexIndex(j, aux);
			}

			faces.push_back(auxi_tri);
		}else{
			//Triangularize object
			entrada>>v0>>v1;			

			for(j=2;j<tam;++j){
				entrada>>v2;

				auxi_tri.setVertexIndex(0, v0);
				auxi_tri.setVertexIndex(1, v1);
				auxi_tri.setVertexIndex(2, v2);
				faces.push_back(auxi_tri);

				v1 = v2;
			}

		}
	}

	

	entrada.close();

	return true;
}

void CObjectOFF::Draw(){
	/*if(!puntos.empty() && !caras.empty()){
		//Dibuja cada uno de los poligonos
		for(i=0;i<caras.size();++i){
			for(j=0;j<(int)caras[i]->v.size();++j){
				//Asigna las normales, las coordenadas de texturas y los puntos a dibujar
				glNormal3f(caras[i]->v[j]->normal.x,caras[i]->v[j]->normal.y,caras[i]->v[j]->normal.z);
				glVertex3f(caras[i]->v[j]->v.x,caras[i]->v[j]->v.y,caras[i]->v[j]->v.z);
			}
			glEnd();
		}	
	}*/
}




int CObjectOFF::size(){
	return this->vertex.size();
}

void CObjectOFF::print(){

	/*std::ofstream off("out.txt");
    std::cout.rdbuf(off.rdbuf()); //redirect std::cout to out.txt!

	std::list<CTriangle*>::iterator f;

	if(!vertex.empty() && !faces.empty()){
		i=0;
		//Dibuja cada uno de los poligonos
		for(f = faces.begin();f != faces.end();++f,++i){
			std::cout<<">>>>>>>>>>>>>>Triangulo<<<<<<<<<<<<<"<<i<<std::endl;
			for(j=0;j<( *f)->v.size();++j){
			//Asigna las normales, las coordenadas de texturas y los puntos a dibujar
				
				std::cout<<"NORMAL "<<j<<std::endl;
				std::cout<<(*f)->v[j]->normal;
				std::cout<<"PUNTO "<<j<<std::endl;
				std::cout<<(*f)->v[j]->v;
			}
		}	
	}*/
}