#define GLFW_DLL
#include "include/GL/glew.h"
#include "include/GLFW/glfw3.h"
#include "include/glm/glm.hpp"
#include "include/glm/gtc/matrix_transform.hpp"
#include "include/glm/gtc/type_ptr.hpp"
#include "include/glm/gtx/quaternion.hpp"
#include "GLSLProgram.h"
#include "ObjectOFF.h"
#include <stdlib.h>
#include <string>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"
#include "kernel.cuh"

#pragma comment(lib, "lib/glfw3dll.lib")
#pragma comment(lib, "lib/glew32.lib")
#pragma comment(lib, "opengl32.lib")

#define BUFFER_OFFSET(i) ((char *)NULL + (i))
#define MYPI 3.14159265

using namespace std;


///< Only wrapping the glfw functions
namespace glfwFunc
{
	CObjectOFF * off;
	CUDAClass cuda;
	Options m_Options;
	GLFWwindow* glfwWindow;
	const unsigned int WINDOW_WIDTH = 1280;
	const unsigned int WINDOW_HEIGHT = 768;
	const float NCP = 0.5f;
	const float FCP = 5.0f;
	const float fAngle = 45.f;
	double lastx, lasty;
	bool pres = false;
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
		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cudaPboResource, gl_pixelBufferObject, cudaGraphicsMapFlagsWriteDiscard));
		

		quater = glm::quat();

		//Load an object
		off = new CObjectOFF();
			
		//Set the image to the  class
		if(off->openFile("E:/Users/franjav/Desktop/Modelos/off/cube.off")){
			/*m_translate.x = 0.0;
			m_translate.y = 0.0;
			m_translate.z = 0.0;
			m_translate.w = 0.0;
			m_scale = 1.0f;
			m_rotatemat = Eyes();*/
			((CObjectOFF *)off)->center();
			((CObjectOFF *)off)->normalize();
			((CObjectOFF *)off)->norm();

			cuda.cudaSetObject(off->getVertex(), off->getFaces());
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
		if(pres){	
			//Rotation
			float dx = float(xpos - lastx);
			float dy = float(ypos - lasty);

			//Calculate angle and rotation axis
			float angle = sqrtf(dx*dx + dy*dy)/50.0f;
					
			//Acumulate rotation with quaternion multiplication
			q2 = glm::angleAxis(angle, glm::normalize(glm::vec3(dy,dx,0.0f)));
			quater = glm::cross(q2, quater);
			
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
				pres = true;
			}else{				
				pres = false;
			}
			return true;
		}
			
		return false;
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

		memcpy(m_Options.modelView, 
			glm::value_ptr(trans * rot), 
			16 * sizeof(float));

		cuda.cudaRC(d_textureBufferData, WINDOW_WIDTH, WINDOW_HEIGHT, m_Options);
 
		cudaGraphicsUnmapResources(1, &cudaPboResource, 0);
	}

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