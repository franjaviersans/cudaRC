#version 440

in vec3 vVertexColor;
in vec2 vVertexTex;

uniform sampler2D tex;
uniform uint width, height;

layout(location = 0) out vec4 vFragColor;

void main(void)
{
	vFragColor = texelFetch(tex, ivec2(vVertexTex.s * width, vVertexTex.t * height),0);
	//vFragColor = vec4(vVertexTex, 0.0f,1.0f);
}

