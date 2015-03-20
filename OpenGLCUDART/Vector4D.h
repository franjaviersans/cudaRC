#ifndef Vector4Df_H
#define Vector4Df_H

#include <iostream>

class CVector4D
{
public:
	float x;
	float y;
	float z;
	float w;

	CVector4D(): x(0), y(0), z(0), w(0){};
	CVector4D(const int &xx, const int &yy, const int &zz, const int &ww): x((float)xx), y((float)yy), z((float)zz), w((float)ww){};
	CVector4D(const float &xx, const float &yy, const float &zz, const float &ww): x(xx), y(yy), z(zz), w(ww){};
	void init(const float &, const float &, const float &, const float &);
	float operator [](const int&);
	void operator +=(const CVector4D &);
	float longitud();
	void normalizar();

	friend std::ostream& operator <<(std::ostream &,const CVector4D &);
	friend CVector4D producto_cruz(const CVector4D &, const CVector4D &);
	friend float producto_punto(const CVector4D &, const CVector4D &);	
	friend CVector4D operator +(const CVector4D &, const CVector4D &);	
	friend CVector4D operator -(const CVector4D &, const CVector4D &);
	friend CVector4D operator *(const CVector4D &, float);
	friend CVector4D operator *(float, const CVector4D &);
	friend CVector4D operator /(const CVector4D &, float);
	friend CVector4D operator *(const CVector4D &);
	friend float distancia(const CVector4D &a, const CVector4D &b);
	


/*	CVector4D(POINT p): x((float)p.x), y((float)p.y){};
	
	CVector4D(float xx, float yy): x(xx), y(yy){};*/
};

#endif