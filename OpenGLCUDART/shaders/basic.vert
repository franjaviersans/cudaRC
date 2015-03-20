#version 440

uniform mat4 mProjection, mModelView;

layout(location = 0) in vec2 vVertex;
layout(location = 1) in vec3 vColor;
layout(location = 2) in vec2 vTex;

out vec3 vVertexColor;
out vec2 vVertexTex;

void main()
{
	vVertexTex = vTex;
	vVertexColor = vColor;
	gl_Position = mProjection * mModelView * vec4(vVertex,0.0f,1.0f);
}