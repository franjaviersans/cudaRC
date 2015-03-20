#include "Vector4D.h"



void CVector4D::init(const float &xx, const float &yy, const float &zz, const float &ww){
	x = xx;
	y = yy;
	z = zz;
	w = ww;
}

float CVector4D::longitud(){
	//Funcion para retornar la magnitud del vector
	return sqrtf(this->x*this->x+this->y*this->y+this->z*this->z+this->w*this->w);
}

void CVector4D::normalizar(){
	//Funcion de normalizacion del vector
	float longi=longitud();
	if(longi==0.0) longi=1;
	x/=longi;
	y/=longi;
	z/=longi;
	w/=longi;
}

void CVector4D::operator +=(const CVector4D & a){
	x+=a.x;
	y+=a.y;
	z+=a.z;
	w+=a.w;
}

float CVector4D::operator [](const int &id){
	if(id == 0) return x;
	if(id == 1) return y;
	if(id == 2) return z;
	if(id == 3) return w;

	return 0.0f;
}

float distancia(const CVector4D &a, const CVector4D &b){
	//Funcion que realiza el producto cruz entre dos vectores
	float x = a.x - b.x;
	float y = a.y - b.y;
	float z = a.z - b.z;
	float w = a.w - b.w;
	return sqrtf(x*x + y*y + z*z + w*w); 
}

CVector4D producto_cruz(const CVector4D &a, const CVector4D &b){
	//Funcion que realiza el producto cruz entre dos vectores
	return CVector4D(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x, 0.0); 
}

float producto_punto(const CVector4D &a, const CVector4D &b){
	//Funcion que realiza el producto punto entre dos vectores
	return a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w;
}	


CVector4D operator +(const CVector4D & a, const CVector4D & b){
	//Sobrecarga para la suma de vectores
	return CVector4D(a.x+b.x,a.y+b.y,a.z+b.z,a.w+b.w);
}	

CVector4D operator -(const CVector4D & a, const CVector4D & b){
	//Sobrecarga para la resta de vectores
	return CVector4D(a.x-b.x,a.y-b.y,a.z-b.z,a.w-b.w);
}

CVector4D operator *(const CVector4D & a, float aux){
	//Sobrecarga para la resta de vectores
	return CVector4D(a.x*aux,a.y*aux,a.z*aux,a.w*aux);
}

CVector4D operator *(float aux, const CVector4D & a){
	return CVector4D(a.x*aux,a.y*aux,a.z*aux,a.w*aux);
}

CVector4D operator /(const CVector4D & a, float aux){
	//Sobrecarga para la resta de vectores
	return CVector4D(a.x/aux,a.y/aux,a.z/aux,a.w/aux);
}

CVector4D operator *(const CVector4D & a, const CVector4D & b){
	return CVector4D(a.x*b.x,a.y*b.y,a.z*b.z,a.w*b.w);
}

std::ostream& operator <<(std::ostream &o,const CVector4D &p)
{
	o<<p.x<<" "<<p.y<<" "<<p.z<<" "<<p.w<<std::endl;

	return o;
}