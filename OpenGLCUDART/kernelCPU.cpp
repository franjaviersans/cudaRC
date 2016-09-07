#include "kernelCPU.h"

struct stackNodeCPU
{
	int index;
	int actualChild;
};

/*
__device__ uchar4 ComputeLight(const float4 & vert, const float4 & normal, const float4 &light, const float4 &eye, const Options &Options){

	uchar4 color;
	color.x = 255.0f;
	color.y = 255.0f;
	color.z = 255.0f;
	color.w = 255.0f;

	if(Options.bLight){
		CVector4D L = light - vert.v, normal;
		uchar4 spec;
		spec.x = 0.0f;
		spec.y = 0.0f;
		spec.z = 0.0f;
		spec.w = 0.0f;
		normal = vert.normal;
		normal.normalizar();
		L.normalizar();

		//m_bTexture

		float difintesity = max(producto_punto(normal, L),0.0f);

		if(difintesity > 0.0f){
			//Calculate specular contribution
			float intSpec;
			CVector4D V = eye - vert.v, H;
			V.normalizar();

			H = L + V;
			H.normalizar();
	
			intSpec = max(producto_punto(H, normal), 0.0f);
			spec = Options.specular * pow(intSpec, Options.shininess);
		}

		//Specular + diffuse
		color = Options.ambient + Options.diffuse * difintesity + spec;

		color.r  = CUDACode::CLAMP(max(Options.ambient.r, color.r), 0.0f, 255.0f);
		color.g  = CUDACode::CLAMP(max(Options.ambient.g, color.g), 0.0f, 255.0f);
		color.b  = CUDACode::CLAMP(max(Options.ambient.b, color.b), 0.0f, 255.0f);
	}
	

	return color;
}*/

bool ray_triangleCPU( const float4 V1,  // Triangle vertices
                           const float4 V2,
                           const float4 V3,
                           const float4 O,  //Ray origin
                           const float4 D,  //Ray direction
							float *t,
						   float &u, 
						   float &v
						   )
{

	float4 e1, e2;  //Edge1, Edge2
	float4 P, Q, T;
	float det, inv_det;
	
	//Find vectors for two edges sharing V1
	SUB(e1, V2, V1);
	SUB(e2, V3, V1);
	//Begin calculating determinant - also used to calculate u parameter
	CROSS(P, D, e2);
	//if determinant is near zero, ray lies in plane of triangle
	det = DOT(e1, P);
	//NOT CULLING
	if(det > -EPSILON && det < EPSILON) return false;
	inv_det = 1.f / det;
 
	//calculate distance from V1 to ray origin
	SUB(T, O, V1);
 
	//Calculate u parameter and test bound
	u = DOT(T, P) * inv_det;
	//The intersection lies outside of the triangle
	if(u < 0.f - EPSILON || u > 1.f + EPSILON) return false;
 
	//Prepare to test v parameter
	CROSS(Q, T, e1);
 
	//Calculate V parameter and test bound
	v = DOT(D, Q) * inv_det;
	//The intersection lies outside of the triangle
	if(v < 0.f - EPSILON  || u + v  > 1.f + EPSILON) return false;
 
	*t = DOT(e2, Q) * inv_det;
 
	return *t > EPSILON; //ray intersection
}

bool ray_boxCPU(const float4 & O,  //Ray origin
						const float4 & D,  //Ray direction
						float * t,
						const float3 & amin,
						const float3 & amax){

	float md;
	float tmin = -1.0f *FLT_MAX ;
	float tmax = FLT_MAX;
	*t = FLT_MAX;
	float4 p;
	p.x = (amin.x + amax.x)/2.0f - O.x;
	p.y = (amin.y + amax.y)/2.0f - O.y;
	p.z = (amin.z + amax.z)/2.0f - O.z;

	float e,f,t1,t2,aux;

	for(int i=0;i<3;++i){
		md = (i==0)?abs(amax.x - amin.x)/2.0f:((i==1)?abs(amax.y - amin.y)/2.0f:abs(amax.z - amin.z)/2.0f);
		e = (i==0)?p.x:((i==1)?p.y:p.z);
		f = (i==0)?D.x:((i==1)?D.y:D.z);

		if ( abs(f) > 0.000001f ){
			t1 = (e + md)/f;
			t2 = (e - md)/f;

			if(t1 > t2){ aux = t1; t1 = t2; t2 = aux;}
			if(t1 > tmin) tmin = t1;
			if(t2 < tmax) tmax = t2;
			if(tmin > tmax) return false;
			if(tmax < 0) return false;	
		}else if(-e - md > 0 || -e + md < 0) return false;
	}

	*t = tmin;

	return true;
}

int octreeRayIntersectionCPU(	const float4 & O,  //Ray origin
										const float4 & D,  //Ray direction
										const Cell * const octree,
										const uint3 * const id,
										const float4 * const pos,
										float *dist,
										float &u,
										float &v)
{
	int actual = 0;
	int init, num;
	uint3 idtri;
	stackNodeCPU Stack[MAXDEPTH + 2];
	float4 V0, V1, V2;
	float3 amin, amax;
	stackNodeCPU *Node;
	int idInter = -1, child_index;
	float t;
	float v1, u1;

	Stack[actual].index = 0;
	Stack[actual].actualChild = 0;
	actual++;

	while(actual > 0)
	{

		Node = &Stack[actual - 1];


		if(Node->actualChild >= octree[Node->index].numChilds)
		{
			--actual;
		}
		else if(octree[Node->index].type == LEAF)
		{

			init = octree[Node->index].firstChild;
			num = octree[Node->index].numChilds;

			for(int i = 0; i < num; ++i)
			{
				idtri = id[octree[init + i].firstChild];
				V0 = pos[idtri.x];
				V1 = pos[idtri.y];
				V2 = pos[idtri.z];

				t = FLT_MAX;
				ray_triangleCPU(V0, V1, V2, O, D, &t, v1, u1);
				if(t < *dist)
				{
					*dist = t;
					u = u1;
					v = v1;
					idInter = octree[init + i].firstChild; 
					//printf("Alguien aca");
				}
			}

			--actual;
		}
		else if(octree[Node->index].type == INTERNAL)
		{
			child_index = octree[Node->index].firstChild + Node->actualChild; 

			amin.x = octree[child_index].minBox.x;
			amin.y = octree[child_index].minBox.y;
			amin.z = octree[child_index].minBox.z;
				
			amax.x = octree[child_index].maxBox.x;
			amax.y = octree[child_index].maxBox.y;
			amax.z = octree[child_index].maxBox.z;

			if(ray_boxCPU(O, D, &t, amin, amax) 
				&& t < *dist){
				//Insert the new child in the stack
				Stack[actual].index = child_index;
				Stack[actual].actualChild = 0;
				++actual;
				if(Stack[actual].index == 9){ 
					actual = 3;
				}
			}

			++Node->actualChild;
		}
		else
		{
		/*	printf("AJA \n");
			for(unsigned int i =0; i< actual;++i){
				printf("%d %d\n", Stack[i].index, Stack[i].actualChild );
			}
			for(int k = 0;k<= Node->index; ++k)
				printf("%d %d %d\n",octree[k].type, INTERNAL, LEAF);*/
			printf("SHOULD NEVER COME HERE %d %d\n", Node->index, octree[Node->index].type);
		}
	}

	return idInter;
}

void CPURC(uchar4 *buffer, const unsigned int width, const unsigned int height, 
						 const uint3 * const id, const float4 * const pos, const float4  * const normal, const float2 * const tex,  
						 const unsigned int num_vert, const unsigned int num_tri, const Options options, const Cell * const octree, unsigned int x, unsigned int y)
{

	unsigned int tpos = y * width + x;
	float u, v;

	if(x < width && y < height)
	{

		int intersect = -1;
		float D = FLT_MAX;

		buffer[tpos].x = 0;
		buffer[tpos].y = 0;
		buffer[tpos].z = 0;
		buffer[tpos].w = 0;

		float4 origin;
		float4 dir;
		float4 vaux;
		//dir(options.priX + x * options.incX, options.priY + y * options.incY, -1.0f ,0.0f)
		origin.x = 0.0f; origin.y = 0.0f; origin.z = 0.0f; origin.w = 1;
		dir.x = options.priX + x * options.incX; dir.y = options.priY + y * options.incY; dir.z = -1.0f; dir.w = 1.0f;

		SUB(dir, dir, origin);

		vaux = origin;
		MULT(origin, options.modelView, vaux);

		vaux = dir;
		MULT(dir, options.modelView, vaux);

		intersect = octreeRayIntersectionCPU(origin, dir, octree, id, pos, &D, u, v);
		if(intersect != -1)
		{
			uint3 idtri = id[intersect];
			float4 V0 = normal[idtri.x];
			float4 V1 = normal[idtri.y];
			float4 V2 = normal[idtri.z];

			BARI(buffer[tpos].x, 255.0f * V0.x, 255.0f * V1.x, 255.0f * V2.x, u, v);
			BARI(buffer[tpos].y, 255.0f * V0.y, 255.0f * V1.y, 255.0f * V2.y, u, v);
			BARI(buffer[tpos].z, 255.0f * V0.z, 255.0f * V1.z, 255.0f * V2.z, u, v);
			buffer[tpos].w = 255;
		}
	}
}



CPURCClass::CPURCClass()
{
	h_pos = NULL;
	h_normal = NULL;
	h_tex = NULL;
	h_id = NULL;
}

CPURCClass::~CPURCClass()
{
	delete h_pos;
	delete h_normal;
	delete h_tex;
	delete h_id;
	delete h_octree;
}


void CPURCClass::SetObject(const std::vector<CVertex> *ptr_puntos,const std::vector<CTriangle> *ptr_caras, const vector<Cell> *ptr_octree)
{
	h_pos = new float4[(*ptr_puntos).size()];
	h_normal = new float4[(*ptr_puntos).size()];
	h_tex = new float2[(*ptr_puntos).size()];
	h_id = new uint3[(*ptr_caras).size()];
	h_octree = new Cell[(*ptr_octree).size()];

	num_vert = (*ptr_puntos).size();
	num_tri = (*ptr_caras).size();

	for(unsigned int i=0;i<(*ptr_caras).size();++i)
	{
		h_id[i].x = (*ptr_caras)[i].V0;
		h_id[i].y = (*ptr_caras)[i].V1;
		h_id[i].z = (*ptr_caras)[i].V2;
	}


	for(unsigned int i=0;i<(*ptr_puntos).size();++i)
	{
		h_pos[i].x = (*ptr_puntos)[i].v.x;
		h_pos[i].y = (*ptr_puntos)[i].v.y;
		h_pos[i].z = (*ptr_puntos)[i].v.z;
		h_pos[i].w = 1.0f;


		h_normal[i].x = (*ptr_puntos)[i].normal.x;
		h_normal[i].y = (*ptr_puntos)[i].normal.y;
		h_normal[i].z = (*ptr_puntos)[i].normal.z;
		h_normal[i].w = 0.0f;

		h_tex[i].x = (*ptr_puntos)[i].texture.x;
		h_tex[i].y = (*ptr_puntos)[i].texture.y;
	}

	memcpy(h_octree, (*ptr_octree).data(), sizeof(Cell) * (*ptr_octree).size());

}


// Helper function for using CUDA to add vectors in parallel.
void CPURCClass::RC(uchar4 *d_buffer, unsigned int width, unsigned int height, Options options)
{

	for(unsigned int i=0;i<width;++i)
	{
		for(unsigned int j=0;j<height;++j)
		{
			
			CPURC(d_buffer, width, height, h_id, h_pos, h_normal, h_tex, num_vert, num_tri, options, h_octree, i, j);
		}
	}
	cout<<"AQUI"<<endl;
}

