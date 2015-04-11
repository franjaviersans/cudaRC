#define GLFW_DLL
#include "include/GL/glew.h"
#include "include/GLFW/glfw3.h"
#include "include/glm/glm.hpp"
#include "include/glm/gtc/matrix_transform.hpp"
#include "include/glm/gtc/type_ptr.hpp"
#include "include/glm/gtx/quaternion.hpp"
#include "GLSLProgram.h"
#include "ObjectOFF.h"
#include "Octree.h"
#include <stdlib.h>
#include <string>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"
#include "kernel.cuh"
#include "kernelCPU.h"

#pragma comment(lib, "lib/glfw3dll.lib")
#pragma comment(lib, "lib/glew32.lib")
#pragma comment(lib, "opengl32.lib")

#define BUFFER_OFFSET(i) ((char *)NULL + (i))
#define MYPI 3.14159265

#define GPU

using namespace std;


///< Only wrapping the glfw functions
namespace glfwFunc
{
	CObjectOFF * off;
	#ifdef GPU
		CUDAClass cuda;
	#else	
		CPURCClass cpuclass;
	#endif
	
	
	Options m_Options;
	GLFWwindow* glfwWindow;
	const unsigned int WINDOW_WIDTH = 1280;
	const unsigned int WINDOW_HEIGHT = 768;
	const float NCP = 0.5f;
	const float FCP = 5.0f;
	const float fAngle = 45.f;
	double lastx, lasty;
	float s = 1.0f, tx = 0.0f, ty = 0.0f, tz = 0.0f;
	int pres = -1;
	//Variables to do rotation
	glm::quat quater, q2;
	glm::mat4x4 RotationMat = glm::mat4x4();
	string strNameWindow = "Hello GLFW";

	cudaGraphicsResource * cudaPboResource;
	GLuint gl_texturePtr, gl_pixelBufferObject;

	CGLSLProgram m_program;
	glm::mat4x4 mProjMatrix, mModelViewMatrix;
	GLuint m_idVAO;

	///< Function to build a simple triangle
	void initiateQuad()
	{
		//Figure
		float vfVertexT [] = {-1.f, -1.f, 1.f, -1.f, 1.f, 1.f, -1.f, 1.f};
		float vfColorT [] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 1.0f, 1.0f, 1.0f};
		float vfTextT [] = {0.0f, 1.0f, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f};
		GLuint idVBO;

		//Generate the Vertex Array
		glGenVertexArrays(1, &m_idVAO);
		glBindVertexArray(m_idVAO);		

			//Generate the Buffer Object
			glGenBuffers(1, &idVBO);
			glBindBuffer(GL_ARRAY_BUFFER, idVBO);

				//Load the data
				glBufferData(GL_ARRAY_BUFFER, sizeof(vfVertexT) + sizeof(vfColorT) + sizeof(vfTextT), NULL, GL_STATIC_DRAW);
				glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vfVertexT), vfVertexT);
				glBufferSubData(GL_ARRAY_BUFFER, sizeof(vfVertexT), sizeof(vfColorT), vfColorT);
				glBufferSubData(GL_ARRAY_BUFFER, sizeof(vfVertexT) + sizeof(vfColorT), sizeof(vfTextT), vfTextT);
		
				//Map the atribute array to an atibute location in the shader
				glEnableVertexAttribArray(m_program.getLocation("vVertex"));
				glVertexAttribPointer(m_program.getLocation("vVertex"), 2, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0)); //Vertex
				glEnableVertexAttribArray(m_program.getLocation("vColor"));
				glVertexAttribPointer(m_program.getLocation("vColor"), 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(sizeof(vfVertexT))); //Colors
				glEnableVertexAttribArray(m_program.getLocation("vTex"));
				glVertexAttribPointer(m_program.getLocation("vTex"), 2, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(sizeof(vfVertexT) + sizeof(vfColorT))); //Texture
		
			//Unbind Buffer Object
			glBindBuffer(GL_ARRAY_BUFFER, 0);
		//Unbind Vertex Array
		glBindVertexArray(0);
	}
	

	int sumar(Octree oc)
	{
		if(oc.Hoja)
		{
			return oc.primitivas.size();
		}
		else
		{
			int acum = 0;
			for (int i = 0; i < 8; i++)
			{
				acum += sumar(*(oc.hijos[i]));
			}

			return acum;
		}
	}

	int nivel(Octree oc)
	{
		if(oc.Hoja)
		{
			return 0;
		}
		else
		{
			int actual = 0;
			for (int i = 0; i < 8; i++)
			{
				actual = max(actual, nivel((*(oc.hijos[i]))));
			}

			return actual + 1;
		}
	}

	///
	/// Init all data and variables.
	/// @return true if everything is ok, false otherwise
	///
	bool initialize()
	{
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);

		//Glew INIT
		glewExperimental = GL_TRUE;
		if(glewInit() != GLEW_OK) 
		{
			cout << "- glew Init failed :(" << endl;
			return false;
		}

		std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
		std::cout << "GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
		std::cout << "Vendor: " << glGetString(GL_VENDOR) << std::endl;
		std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;

		//Load the shaders
		m_program.loadShader("shaders/basic.vert", CGLSLProgram::VERTEX);
		m_program.loadShader("shaders/basic.frag", CGLSLProgram::FRAGMENT);
		//Link the shaders in a program
		m_program.create_link();
		//Enable the program
		m_program.enable();
				//Link the attributes and the uniforms
				m_program.addAttribute("vVertex");
				m_program.addAttribute("vColor");
				m_program.addAttribute("vTex");
				m_program.addUniform("mProjection");
				m_program.addUniform("mModelView");
				m_program.addUniform("width");
				m_program.addUniform("height");
		//Disable the program
		m_program.disable();

		//Function to initiate a triangle
		initiateQuad();

		//CUDA code
		// Free any previously allocated buffers
		// ... code skipped
 
		// Allocate new buffers.
		glEnable(GL_TEXTURE_2D);
		glGenTextures(1, &gl_texturePtr);
		glBindTexture(GL_TEXTURE_2D, gl_texturePtr);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
 
		glGenBuffers(1, &gl_pixelBufferObject);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl_pixelBufferObject);
		glBufferData(GL_PIXEL_UNPACK_BUFFER, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);
 
		//Bind the PBO to a cuda resource
		#ifdef GPU
			checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cudaPboResource, gl_pixelBufferObject, cudaGraphicsMapFlagsWriteDiscard));
		#endif

		quater = glm::quat(0.0f,1.0f,0.0f,0.0f);


		//Load an object
		off = new CObjectOFF();
			
		//Set the image to the  class
		if(off->openFile("E:/Users/franjav/Desktop/Modelos/off/space_station.off")){
			/*m_translate.x = 0.0;
			m_translate.y = 0.0;
			m_translate.z = 0.0;
			m_translate.w = 0.0;
			m_scale = 1.0f;
			m_rotatemat = Eyes();*/
			((CObjectOFF *)off)->center();
			((CObjectOFF *)off)->normalize();
			((CObjectOFF *)off)->norm();


			//Set the octree with the object
			Octree oc(off->getVertex(), off->getFaces(), AABB(CVector4D(off->minBox().x, off->minBox().y, off->minBox().z, 1.0f), 
																CVector4D(off->maxBox().x, off->maxBox().y, off->maxBox().z, 1.0f)));

			vector<Cell> vec;
			oc.toLinear(&vec);

			/*
			cout<<"SUMAR "<<sumar(oc)<<endl;
			cout<<"NIVEL "<<nivel(oc)<<endl;
			
			for(int i=0;i<vec.size();++i)
			{
				if(vec[i].type == TRIANGLE){
					cout<<i<<"  "<<((vec[i].type == LEAF)?"Hoja":((vec[i].type == INTERNAL)?"Internal":"Tri"))<<"  "<<vec[i].numChilds<<"  "<<
						vec[i].firstChild<<"  "<<
						" ("<<(*off->getVertex())[(*off->getFaces())[vec[i].firstChild].V0].v.x<<","<<(*off->getVertex())[(*off->getFaces())[vec[i].firstChild].V0].v.y<<","<<(*off->getVertex())[(*off->getFaces())[vec[i].firstChild].V0].v.z<<")  "<<
						" ("<<(*off->getVertex())[(*off->getFaces())[vec[i].firstChild].V1].v.x<<","<<(*off->getVertex())[(*off->getFaces())[vec[i].firstChild].V1].v.y<<","<<(*off->getVertex())[(*off->getFaces())[vec[i].firstChild].V1].v.z<<")  "<<
						" ("<<(*off->getVertex())[(*off->getFaces())[vec[i].firstChild].V2].v.x<<","<<(*off->getVertex())[(*off->getFaces())[vec[i].firstChild].V2].v.y<<","<<(*off->getVertex())[(*off->getFaces())[vec[i].firstChild].V2].v.z<<")  "
						;
				}else{
					cout<<i<<"  "<<((vec[i].type == LEAF)?"Hoja":((vec[i].type == INTERNAL)?"Internal":"Tri"))<<"  "<<vec[i].numChilds<<"  "<<
						vec[i].firstChild<<
						" ("<<vec[i].minBox.x<<","<<vec[i].minBox.y<<","<<vec[i].minBox.z<<
						")  "<<" ("<<vec[i].maxBox.x<<","<<vec[i].maxBox.y<<","<<vec[i].maxBox.z<<")";
				}
				cout<<endl;
			}*/

			#ifdef GPU
				cuda.cudaSetObject(off->getVertex(), off->getFaces(), &vec);
			#else	
				cpuclass.SetObject(off->getVertex(), off->getFaces(), &vec);
			#endif
			
		}
		else
		{
			cout<<"Problem loading file"<<endl;
		}
				
		delete off;
		off = NULL;

		return true;
	}
	
	///< Callback function used by GLFW to capture some possible error.
	void errorCB(int error, const char* description)
	{
		cout << description << endl;
	}

	///
	/// The keyboard function call back
	/// @param window id of the window that received the event
	/// @param iKey the key pressed or released
	/// @param iScancode the system-specific scancode of the key.
	/// @param iAction can be GLFW_PRESS, GLFW_RELEASE or GLFW_REPEAT
	/// @param iMods Bit field describing which modifier keys were held down (Shift, Alt, & so on)
	///
	void keyboardCB(GLFWwindow* window, int iKey, int iScancode, int iAction, int iMods)
	{
		if (iAction == GLFW_PRESS)
		{
			switch (iKey)
			{
				case GLFW_KEY_ESCAPE:
				case GLFW_KEY_Q:
					glfwSetWindowShouldClose(window, GL_TRUE);
					break;
			}
		}
	}

	int TwEventMousePosGLFW3(GLFWwindow* window, double xpos, double ypos)
	{ 
		if(pres == 0){	
			//Rotation
			float dx = float(xpos - lastx);
			float dy = - float(ypos - lasty);

			//Calculate angle and rotation axis
			float angle = sqrtf(dx*dx + dy*dy)/50.0f;
					
			//Acumulate rotation with quaternion multiplication
			if(abs(dx) + abs(dy) > 0.01f){
				q2 = glm::angleAxis(angle, glm::normalize(glm::vec3(dy,dx,0.0f)));
				quater = glm::cross(q2, quater);
			}


			lastx = xpos;
			lasty = ypos;
		}else if(pres == 1){
			//Translate point
			tx += float(xpos - lastx) / 100.0f;
			ty += float(ypos - lasty) / 100.0f;
	
			lastx = xpos;
			lasty = ypos;
		}
		return true;
	}

	int TwEventMouseButtonGLFW3(GLFWwindow* window, int button, int action, int mods)
	{ 
		double x, y;   
		glfwGetCursorPos(window, &x, &y);  
			
		if(button == GLFW_MOUSE_BUTTON_LEFT){
			if(action == GLFW_PRESS){
				lastx = x;
				lasty = y;
				pres = 0;
			}else{				
				pres = -1;
			}
			return true;
		}else if(button == GLFW_MOUSE_BUTTON_RIGHT){
			if(action == GLFW_PRESS){
				lastx = x;
				lasty = y;
				pres = 1;
			}else{				
				pres = -1;
			}
			return true;
		}
			
		return false;
	}

	int TwEventMouseWheelGLFW3(GLFWwindow* window, double xoffset, double yoffset){
		s += float(yoffset) / 10.0f;
		return true;
	}
	
	///< The resizing function
	void resizeCB(GLFWwindow* window, int iWidth, int iHeight)
	{
		if(iHeight == 0) iHeight = 1;
		float ratio = iWidth / float(iHeight);
		glViewport(0, 0, iWidth, iHeight);
		//mProjMatrix = glm::perspective(fAngle, ratio, NCP, FCP);
		mProjMatrix = glm::ortho(-1.0f, 1.0f, -1.0f, 1.0f);

		m_Options.priX = -ratio * tan(fAngle/2.0f*float(MYPI)/180.0f) * NCP;
		m_Options.priY = -tan(fAngle/2.0f*float(MYPI)/180.0f) * NCP;
	}

	#ifdef GPU
		void displayKernel() 
		{
			cudaGraphicsMapResources(1, &cudaPboResource, 0);
			size_t num_bytes;
			uchar4 *d_textureBufferData;

			cudaGraphicsResourceGetMappedPointer((void**)&d_textureBufferData, &num_bytes, cudaPboResource);
 
			m_Options.incX = - 2.0f * m_Options.priX/float(WINDOW_WIDTH);
			m_Options.incY = - 2.0f * m_Options.priY/float(WINDOW_HEIGHT);

			glm::mat4 rot = glm::mat4_cast(glm::normalize(quater));
			glm::mat4 trans = glm::translate(glm::mat4(), glm::vec3(0,0,-10.f));
			glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(s));
			glm::mat4 trans2 = glm::translate(glm::mat4(), glm::vec3(tx, ty, tz));

			memcpy(m_Options.modelView, 
				glm::value_ptr(glm::inverse(trans * trans2 * scale * rot)), 
				16 * sizeof(float));

			cuda.cudaRC(d_textureBufferData, WINDOW_WIDTH, WINDOW_HEIGHT, m_Options);
 
			cudaGraphicsUnmapResources(1, &cudaPboResource, 0);
		}
	#else
		void displayKernel() 
		{
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl_pixelBufferObject);
			uchar4* ptr = (uchar4*)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
 
			m_Options.incX = - 2.0f * m_Options.priX/float(WINDOW_WIDTH);
			m_Options.incY = - 2.0f * m_Options.priY/float(WINDOW_HEIGHT);

			glm::mat4 rot = glm::mat4_cast(glm::normalize(quater));
			glm::mat4 trans = glm::translate(glm::mat4(), glm::vec3(0,0,-10.0f));
			glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(s));
			glm::mat4 trans2 = glm::translate(glm::mat4(), glm::vec3(tx, ty, tz));

			memcpy(m_Options.modelView, 
				glm::value_ptr(glm::inverse(trans * trans2 * scale * rot)), 
				16 * sizeof(float));

			if(ptr)
			{
			
				cpuclass.RC(ptr, WINDOW_WIDTH, WINDOW_HEIGHT, m_Options);
				glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
			}

		
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		}
	#endif
	///< The main rendering function.
	void draw()
	{
		displayKernel();
		
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glClearColor(0.15f, 0.15f, 0.15f, 1.f);
		
		//mModelViewMatrix = glm::translate(glm::mat4(), glm::vec3(0,0,-2.f)); 
		mModelViewMatrix = glm::mat4(); 
		
		//Bind buffers and textures to render
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl_pixelBufferObject);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, gl_texturePtr);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);

		//Display the triangle
		m_program.enable();
			glUniformMatrix4fv(m_program.getLocation("mModelView"), 1, GL_FALSE, glm::value_ptr(mModelViewMatrix));
			glUniformMatrix4fv(m_program.getLocation("mProjection"), 1, GL_FALSE, glm::value_ptr(mProjMatrix));
			glUniform1ui(m_program.getLocation("width"), WINDOW_WIDTH);
			glUniform1ui(m_program.getLocation("height"), WINDOW_HEIGHT);

			glBindVertexArray(m_idVAO);
				glDrawArrays(GL_QUADS, 0, 4);
			glBindVertexArray(0);
		m_program.disable();

		//Unbind buffers
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
		glBindTexture(GL_TEXTURE_2D, 0);

		//Swap buffer
		glfwSwapBuffers(glfwFunc::glfwWindow);
	}

	
	
	/// Here all data must be destroyed + glfwTerminate
	void destroy()
	{
		if(glIsVertexArray(m_idVAO)) glDeleteVertexArrays(1, &m_idVAO);
		glfwDestroyWindow(glfwFunc::glfwWindow);
		glfwTerminate();
	}
};

int main(int argc, char** argv)
{
	glfwSetErrorCallback(glfwFunc::errorCB);
	if (!glfwInit())	exit(EXIT_FAILURE);
	glfwFunc::glfwWindow = glfwCreateWindow(glfwFunc::WINDOW_WIDTH, glfwFunc::WINDOW_HEIGHT, glfwFunc::strNameWindow.c_str(), NULL, NULL);
	if (!glfwFunc::glfwWindow)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwMakeContextCurrent(glfwFunc::glfwWindow);
	if(!glfwFunc::initialize()) exit(EXIT_FAILURE);
	glfwFunc::resizeCB(glfwFunc::glfwWindow, glfwFunc::WINDOW_WIDTH, glfwFunc::WINDOW_HEIGHT);	//just the 1st time
	glfwSetKeyCallback(glfwFunc::glfwWindow, glfwFunc::keyboardCB);
	glfwSetWindowSizeCallback(glfwFunc::glfwWindow, glfwFunc::resizeCB);
	glfwSetMouseButtonCallback(glfwFunc::glfwWindow, (GLFWmousebuttonfun)glfwFunc::TwEventMouseButtonGLFW3);
	glfwSetScrollCallback(glfwFunc::glfwWindow, (GLFWscrollfun)glfwFunc::TwEventMouseWheelGLFW3);
	glfwSetCursorPosCallback(glfwFunc::glfwWindow, (GLFWcursorposfun)glfwFunc::TwEventMousePosGLFW3);

	// main loop!
	while (!glfwWindowShouldClose(glfwFunc::glfwWindow))
	{
		glfwFunc::draw();
		glfwPollEvents();	//or glfwWaitEvents()
	}

	glfwFunc::destroy();
	return EXIT_SUCCESS;
}