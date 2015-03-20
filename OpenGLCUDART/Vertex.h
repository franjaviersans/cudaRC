#ifndef VERTEX_H
#define VERTEX_H

#include "Vector4D.h"
	
class CVertex{
public:
	CVector4D v;
	CVector4D normal;
	CVector4D texture;

	CVertex():v(), normal(), texture(){};
	CVertex(CVector4D _v, CVector4D _normal, CVector4D _texture):v(_v), normal(_normal), texture(_texture){};
	CVertex(int id){
		normal.init(0.0f,0.0f,0.0f,0.0f);
		v.init(0.0f,0.0f,0.0f,1.0f);
		texture.init(0.0f,0.0f,0.0f,0.0f);
	}
};

class CFinalVert
{
public:
	CVector4D m_ProyectedCoord;
	CVertex m_Vertex;

	CFinalVert(CVector4D p, CVertex v): m_ProyectedCoord(p), m_Vertex(v){};
	CFinalVert(){};
};

#endif 