#ifndef UMML_GLPLOT_INCLUDED
#define UMML_GLPLOT_INCLUDED

/*
 Machine Learning Artificial Intelligence Library
 2D, 3D plots and image display using OpenGL.

 FILE:   glplot.hpp
 AUTHOR: Anastasios Milikas (amilikas@csd.auth.gr)
 YEAR:   2021-2022
 
 Namespace
 ~~~~~~~~~
 umml
 
 Dependencies
 ~~~~~~~~~~~~
 OpenGL
 FreeGLUT
 GLEW
 STL vector
 STL string
*/

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif

#include <vector>
#include <string>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <string.h>

#include <iostream>


namespace umml {


/* GIMP RGBA C-Source image dump (cross.c) */

static const struct {
  unsigned int 	 width;
  unsigned int 	 height;
  unsigned int 	 bytes_per_pixel; /* 3:RGB, 4:RGBA */ 
  unsigned char	 pixel_data[15 * 15 * 4 + 1];
} img_texture = {
  15, 15, 4,
  "\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377\0\377"
  "\377\377\0\0\0\0\377\377\377\377\377\0\0\0\377\377\377\377\0\377\377\377"
  "\0\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377\0"
  "\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377\0\0"
  "\0\0\377\377\377\377\377\0\0\0\377\377\377\377\0\377\377\377\0\377\377\377"
  "\0\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377\0"
  "\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377\0\0\0\0\377\377\377"
  "\377\377\0\0\0\377\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377"
  "\0\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377\0"
  "\377\377\377\0\377\377\377\0\377\377\377\0\0\0\0\377\377\377\377\377\0\0"
  "\0\377\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377"
  "\0\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377\0"
  "\377\377\377\0\377\377\377\0\0\0\0\377\377\377\377\377\0\0\0\377\377\377"
  "\377\0\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377"
  "\0\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377\0"
  "\377\377\377\0\0\0\0\377\377\377\377\377\0\0\0\377\377\377\377\0\377\377"
  "\377\0\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377\0\0\0\0\377"
  "\0\0\0\377\0\0\0\377\0\0\0\377\0\0\0\377\0\0\0\377\0\0\0\377\377\377\377"
  "\377\0\0\0\377\0\0\0\377\0\0\0\377\0\0\0\377\0\0\0\377\0\0\0\377\0\0\0\377"
  "\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377"
  "\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377"
  "\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377"
  "\377\377\377\377\377\377\0\0\0\377\0\0\0\377\0\0\0\377\0\0\0\377\0\0\0\377"
  "\0\0\0\377\0\0\0\377\377\377\377\377\0\0\0\377\0\0\0\377\0\0\0\377\0\0\0"
  "\377\0\0\0\377\0\0\0\377\0\0\0\377\377\377\377\0\377\377\377\0\377\377\377"
  "\0\377\377\377\0\377\377\377\0\0\0\0""2\0\0\0\377\377\377\377\377\0\0\0\377"
  "\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377\0\377"
  "\377\377\0\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377\0\377\377"
  "\377\0\377\377\377\0\0\0\0\377\377\377\377\377\0\0\0\377\377\377\377\0\377"
  "\377\377\0\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377\0\377\377"
  "\377\0\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377"
  "\0\0\0\0\377\377\377\377\377\0\0\0\377\377\377\377\0\377\377\377\0\377\377"
  "\377\0\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377"
  "\0\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377\0\0\0\0\377\377"
  "\377\377\377\0\0\0\377\377\377\377\0\377\377\377\0\377\377\377\0\377\377"
  "\377\0\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377"
  "\0\377\377\377\0\377\377\377\0\377\377\377\0\0\0\0\377\377\377\377\377\0"
  "\0\0\377\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377\0\377\377"
  "\377\0\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377"
  "\0\377\377\377\0\377\377\377\0\0\0\0\377\377\377\377\377\0\0\0\377\377\377"
  "\377\0\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377\0\377\377\377"
  "\0",
};


class glplot {
 public:
	enum { Line, Scatter, Sprite };     // plot style
	enum { Plot2D, Plot3D, RGBImage };  // modes
	typedef unsigned char pixel;
	struct point2d { GLfloat x, y; };
	struct point3d { GLfloat x, y, z; };
	struct rgbpixel { pixel r, g, b; };
	struct rgbcolor { GLfloat c[3]; };
	struct graph_attr { rgbcolor color; int type; int ptsize; };
	
	// colors
	enum { 
		Black = 0x000000,
		White = 0xffffff,
		Gray = 0x808080,
		DarkGray = 0x404040,
		Yellow = 0xffff00,
		Orange = 0xffa500,
		Red = 0xff0000,
		Crimson	= 0xdc143c,
		Magenta	= 0xff00ff,
		Cyan = 0x00ffff,
		DarkCyan = 0x008b8b,
		Blue = 0x0000ff,
		Lime = 0x00ff00,
		Green = 0x008000,
		GreenYellow	= 0xadff2f,
	};

	glplot() {
		mode     = Plot2D;
		offset_x = 0.0;
		offset_y = 0.0;
		scale    = 1.0;
		program  = 0;
		win_x    = win_y = 5;
		win_w    = 640;
		win_h    = 480;
		mouse_x  = mouse_y = -1;
		mouse_left_down = false;
		grid_x = grid_y = 1;
		grid_padding = 0;
		xpos = ypos = 0.0f;
		zpos = -40.0f;
		xang = yang = 0.0f;
		bgcolor.c[0] = bgcolor.c[1] = bgcolor.c[2] = 0;
	}
	
	virtual ~glplot() {
		if (program != 0) glDeleteProgram(program);
	}
	
	// get a descriptive error text string in case an error occured.
	std::string error_description() const { return err_str; }
	
	// window geometry and position
	void    set_window_geometry(int w, int h) { win_w = w; win_h = h; }
	void    set_window_position(int x, int y) { win_x = x; win_y = y; }
	void    set_window_title(const std::string& title) { win_title = title; }

	// set initial zoom and view
	void    zoom(float _scale, float _zpos) { scale = scale; zpos = _zpos; }
	void    lookat(float ofsx, float angx) { offset_x = ofsx; xang = angx; }
	
	// add 2d or 3d plots
	void    add_2d(const std::vector<point2d>& points, int color, int type, int ptsize=1);
	void    add_3d(const std::vector<point3d>& points, int color, int type, int ptsize=1);

	// adds a grid to hold multiple images (padding is in pixels)
	void    add_image_grid(int nrows, int ncols, int padding=0);
	
	// add grayscale or grb image
	template <typename T>
	void    add_grayscale_image(const T* pixels, int width, int height);

	template <typename T>
	void    add_rgb_image(const T* pixels, int width, int height);

	template <typename T>
	void    add_rgb_image(const T* r, const T* g, const T* b, int width, int height);
	
	// display the plot or the image
	bool    show();

	// colors
	void    set_bkgnd_color(GLfloat r, GLfloat g, GLfloat b);
	void    set_bkgnd_color(int rgb);
	void    set_axis_color(GLfloat r, GLfloat g, GLfloat b);
	void    set_axis_color(int rgb);
	bool    is_rgb_dark(int rgb);
	static  int  make_rgb(GLfloat r, GLfloat g, GLfloat b);
	static  void from_rgb(int rgb, GLfloat& r, GLfloat& g, GLfloat& b);
	static  std::vector<int> make_palette(int n, int step=2, bool light=true);
	static  std::vector<int> make_palette16(bool light=true);

 protected:
	virtual void on_display();
	virtual void on_reshape(int width, int height);
	virtual void on_idle();
	virtual void on_mouse_button(int button, int state, int x, int y);
	virtual void on_mouse_move(int x, int y);
	virtual void on_kbd_down(unsigned char key, int x, int y);
	virtual void on_kbd_up(unsigned char key, int x, int y);
	virtual void on_special_kbd_down(int key, int x, int y);
	virtual void on_special_kbd_up(int key, int x, int y); 
	
	static void glut_display() { 
		instance->on_display(); 
	}
	static void glut_reshape(int width, int height) { 
		instance->on_reshape(width, height); 
	}
	static void glut_idle() { 
		instance->on_idle(); 
	}
	static void glut_mouse_button(int button, int state, int x, int y) { 
		instance->on_mouse_button(button, state, x, y); 
	}
	static void glut_mouse_move(int x, int y) {
		instance->on_mouse_move(x, y); 
	}
	static void glut_kbd_down(unsigned char key, int x, int y) { 
		instance->on_kbd_down(key, x, y);
	}
	static void glut_kbd_up(unsigned char key, int x, int y) { 
		instance->on_kbd_up(key, x, y); 
	}
	static void glut_special_kbd_down(int key, int x, int y) { 
		instance->on_special_kbd_down(key, x, y);
	}
	static void glut_special_kbd_up(int key, int x, int y) { 
		instance->on_special_kbd_up(key, x, y);
	}

 protected:
 	bool   init_2d();
 	bool   init_3d();
 	bool   init_image2d();
	GLuint create_shader(const char* shader_src, GLenum type);
	GLuint create_program(const char* vertex_src, const char* fragment_src);
	
	// flips images in the y-axis
	void   flip_y(std::vector<rgbpixel>& rgbpixels, int width, int height);
 
 protected:
	typedef std::vector<point2d> vp2d;
	typedef std::vector<point3d> vp3d;
	typedef struct { int w, h; } dim2; 
	
	// member data
	static glplot* instance;
	int win_x, win_y, win_w, win_h;
	std::string win_title;
	std::string err_str;
	rgbcolor bgcolor;
	rgbcolor axcolor;

	int   mode;
	int   grid_x, grid_y, grid_padding;
	float offset_x, offset_y;
	float scale;
	bool  mouse_left_down;
	int   mouse_x, mouse_y;
	int   mouse_down_x, mouse_down_y;
	
	// plot data
	std::vector<std::pair<vp2d,graph_attr>> points2d;
	std::vector<std::pair<vp3d,graph_attr>> points3d;

	// image data
	std::vector<std::pair<dim2,std::vector<rgbpixel>>> images;

	GLuint program;
	GLint  a_coord2d;
	GLint  u_color;
	GLint  u_offset;
	GLint  u_scale_x;
	GLint  u_sprite;
	GLuint texture_id;
	GLint  u_texture;
	GLuint vao;
	GLuint vbo[8];
	point3d axis[6];
	point3d axis_marks[600];
	float xpos, ypos, zpos;
	float xang, yang;
	GLuint u_angle;
	GLuint u_rotation;
	GLuint u_perspective;
	GLuint vbo_axis;
	GLuint vbo_marks;
};

glplot* glplot::instance = nullptr;



int glplot::make_rgb(GLfloat r, GLfloat g, GLfloat b) 
{   
	return (((int)(r*255) & 0xff) << 16) + (((int)(g*255) & 0xff) << 8) + ((int)(b*255) & 0xff);
}

void glplot::from_rgb(int rgb, GLfloat& r, GLfloat& g, GLfloat& b) 
{
	r = ((rgb >> 16) & 0xff) / 255.0f;
	g = ((rgb >> 8) & 0xff) / 255.0f;
	b = (rgb & 0xff) / 255.0f;
}

std::vector<int> glplot::make_palette(int n, int step, bool light) 
{
	// this function generates vibrant, "evenly spaced" colors
	// original author: Adam Cole, 2011-Sept-14
	std::vector<int> colors;
	colors.reserve(n);
	for (int s=0; s<n; ++s) {
		float r=0, g=0, b=0;
		float h = static_cast<float>(s*step) / n;
		int c = static_cast<int>(h * 6);
		float f = h * 6 - c;
		float q = 1 - f;
		double v = light ? 0.9 : 0.6; // adjust value for light or dark colors
		switch (c % 6) {
			case 0: r = v; g = f * v; b = 0; break;
			case 1: r = q * v; g = v; b = 0; break;
			case 2: r = 0; g = v; b = f * v; break;
			case 3: r = 0; g = q * v; b = v; break;
			case 4: r = f * v; g = 0; b = v; break;
			case 5: r = v; g = 0; b = q * v; break;
		}
		colors.push_back(glplot::make_rgb(r, g, b));
	}
	return colors;
}

std::vector<int> glplot::make_palette16(bool light) 
{
	return std::vector<int>({ 
		0x04713B, 0x594AAC, 0x68432C, 0xBE7868, 0xCE1271, 0x37B69A, 0x8BE1F9, 0x387996, 
		0xFBC69D, 0xE16488, 0x767590, 0xA1CF94, 0xE16E43, 0x9AF2E8, 0xA5F961, 0x07A746});
}

void glplot::set_bkgnd_color(int rgb) 
{
	GLfloat r, g, b;
	from_rgb(rgb, r, g, b);
	set_bkgnd_color(r, g, b); 
}

void glplot::set_bkgnd_color(GLfloat r, GLfloat g, GLfloat b) 
{ 
	bgcolor.c[0]=r; bgcolor.c[1]=g; bgcolor.c[2]=b; 
}

void glplot::set_axis_color(int rgb) 
{
	GLfloat r, g, b;
	from_rgb(rgb, r, g, b);
	set_axis_color(r, g, b); 
}

void glplot::set_axis_color(GLfloat r, GLfloat g, GLfloat b) 
{ 
	axcolor.c[0]=r; axcolor.c[1]=g; axcolor.c[2]=b; 
}

bool glplot::is_rgb_dark(int rgb)
{
	GLfloat r, g, b;
	from_rgb(rgb, r, g, b);
	GLfloat hsp = 0.299*(r*r) + 0.587*(g*g) + 0.114*(b*b);
	if (hsp > 16256.0f) return false;
	return true;
}

void glplot::add_2d(const std::vector<point2d>& points, int color, int type, int ptsize)
{
	if (points2d.size() >= 8) return;
	mode = Plot2D;
	graph_attr attr;
	from_rgb(color, attr.color.c[0], attr.color.c[1], attr.color.c[2]);
	attr.type = type;
	attr.ptsize = ptsize;
	points2d.push_back(std::make_pair(points, attr));
}

void glplot::add_3d(const std::vector<point3d>& points, int color, int type, int ptsize)
{
	if (points3d.size() >= 8) return;
	mode = Plot3D;
	graph_attr attr;
	from_rgb(color, attr.color.c[0], attr.color.c[1], attr.color.c[2]);
	attr.type = type;
	attr.ptsize = ptsize;
	points3d.push_back(std::make_pair(points, attr));
}

void glplot::add_image_grid(int nrows, int ncols, int padding)
{
	grid_x = ncols;
	grid_y = nrows;
	grid_padding = padding;
}

void glplot::flip_y(std::vector<rgbpixel>& rgbpixels, int width, int height)
{
	for (int i=0; i<height/2; ++i) 
	for (int j=0; j<width; ++j) 
		std::swap(rgbpixels[i*width+j], rgbpixels[(height-i-1)*width+j]);
}

template <>
void glplot::add_grayscale_image<glplot::pixel>(const pixel* pixels, int width, int height)
{
	mode = RGBImage;
	dim2 dims = { width, height };
	std::vector<rgbpixel> rgbpixels;
	rgbpixels.clear();
	rgbpixels.reserve(width*height);
	for (int i=0; i<width*height; ++i) {
		rgbpixel pix;
		pix.r = pix.g = pix.b = pixels[i];
		rgbpixels.push_back(pix);
	}
	flip_y(rgbpixels, width, height);
	images.push_back(std::make_pair(dims, rgbpixels));
}

template <typename T>
void glplot::add_grayscale_image(const T* pixels, int width, int height)
{
	mode = RGBImage;
	dim2 dims = { width, height };
	std::vector<rgbpixel> rgbpixels;
	rgbpixels.clear();
	rgbpixels.reserve(width*height);
	for (int i=0; i<width*height; ++i) {
		rgbpixel pix;
		pix.r = pix.g = pix.b = pixel(pixels[i]*T(255));
		rgbpixels.push_back(pix);
	}
	flip_y(rgbpixels, width, height);
	images.push_back(std::make_pair(dims, rgbpixels));
}

template <>
void glplot::add_rgb_image<glplot::rgbpixel>(const rgbpixel* pixels, int width, int height)
{
	mode = RGBImage;
	dim2 dims = { width, height };
	std::vector<rgbpixel> rgbpixels;
	rgbpixels.clear();
	rgbpixels.reserve(width*height);
	for (int i=0; i<width*height; ++i) rgbpixels.push_back(pixels[i]);
	flip_y(rgbpixels, width, height);
	images.push_back(std::make_pair(dims, rgbpixels));
}

template <typename T>
void glplot::add_rgb_image(const T* pixels, int width, int height)
{
	mode = RGBImage;
	dim2 dims = { width, height };
	std::vector<rgbpixel> rgbpixels;
	rgbpixels.clear();
	rgbpixels.reserve(width*height);
	for (int i=0; i<width*height; ++i) {
		rgbpixel pix;
		pix.r = pixel(*pixels++ * T(255));
		pix.g = pixel(*pixels++ * T(255));
		pix.b = pixel(*pixels++ * T(255));
		rgbpixels.push_back(pix);
	}
	flip_y(rgbpixels, width, height);
	images.push_back(std::make_pair(dims, rgbpixels));
}

template <>
void glplot::add_rgb_image<glplot::pixel>(const pixel* r, const pixel* g, const pixel* b, int width, int height)
{
	mode = RGBImage;
	dim2 dims = { width, height };
	std::vector<rgbpixel> rgbpixels;
	rgbpixels.clear();
	rgbpixels.reserve(width*height);
	for (int i=0; i<width*height; ++i) {
		rgbpixel pix;
		pix.r = r[i];
		pix.g = g[i];
		pix.b = b[i];
		rgbpixels.push_back(pix);
	}
	flip_y(rgbpixels, width, height);
	images.push_back(std::make_pair(dims, rgbpixels));
}

template <typename T>
void glplot::add_rgb_image(const T* r, const T* g, const T* b, int width, int height)
{
	mode = RGBImage;
	dim2 dims = { width, height };
	std::vector<rgbpixel> rgbpixels;
	rgbpixels.clear();
	rgbpixels.reserve(width*height);
	for (int i=0; i<width*height; ++i) {
		rgbpixel pix;
		pix.r = pixel(r[i] * T(255));
		pix.g = pixel(g[i] * T(255));
		pix.b = pixel(b[i] * T(255));
		rgbpixels.push_back(pix);
	}
	flip_y(rgbpixels, width, height);
	images.push_back(std::make_pair(dims, rgbpixels));
}


bool glplot::show()
{
	instance = this;

	// init glut and create main window
	int argc=0;
	glutInit(&argc, NULL);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	//glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowPosition(win_x, win_y);
	glutInitWindowSize(win_w, win_h);
	glutCreateWindow(win_title.c_str());
	GLenum glew_status = glewInit();
	if (glew_status != GLEW_OK) {
		err_str = std::string((const char*)glewGetErrorString(glew_status));
		return false;
	}
	if (!GLEW_VERSION_2_0) {
		err_str = "No support for OpenGL 2.0 found";
		return false;
	}

	// set glut's function callbacks
	glutReshapeFunc(glut_reshape);
	glutMouseFunc(glut_mouse_button);
	glutMotionFunc(glut_mouse_move);
	glutDisplayFunc(glut_display);
	glutKeyboardFunc(glut_kbd_down);
	glutKeyboardUpFunc(glut_kbd_up);
	glutSpecialFunc(glut_special_kbd_down);
	glutSpecialUpFunc(glut_special_kbd_up);
	
	// init
	if (mode==Plot2D && !init_2d()) return false;
	if (mode==Plot3D && !init_3d()) return false;
	if (mode==RGBImage && !init_image2d()) return false;

	// start main loop
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
	glutIdleFunc(glut_idle);
	glutMainLoop();
	return true;
}


bool glplot::init_2d()
{
	const char* VERTEX_SHADER = 
	"attribute vec2 coord2d;"
	"varying vec4 f_color;"
	"uniform vec3 color;"
	"uniform vec2 offset;"
	"uniform float scale_x;"
	"uniform float sprite;"
	"void main(void) {"
	"	gl_Position = vec4((coord2d.x+offset.x)*scale_x, (coord2d.y+offset.y)*scale_x, 0, 1);"
	//"	f_color = vec4(coord2d.xy / 2.0 + 0.5, 1, 1);"
	//"	f_color = vec4(coord2d.x/2.0+0.5, coord2d.y/2.0+0.5, 0.5, 1);"
	"	f_color = vec4(color, 1.0);"
	"	gl_PointSize = max(1.0, sprite);"
	"}";

	const char* FRAGMENT_SHADER =
	"uniform sampler2D texture;"
	"varying vec4 f_color;"
	"uniform float sprite;"
	"void main(void) {"
	"	if (sprite > 10.0) gl_FragColor = texture2D(texture, gl_PointCoord) * f_color;"
	"	else gl_FragColor = f_color;"
	"}";
	
	program = create_program(VERTEX_SHADER, FRAGMENT_SHADER);
	if (program == 0) return false;

	a_coord2d = glGetAttribLocation(program, "coord2d");
	u_color = glGetUniformLocation(program, "color");
	u_offset = glGetUniformLocation(program, "offset");
	u_scale_x = glGetUniformLocation(program, "scale_x");
	u_sprite = glGetUniformLocation(program, "sprite");
	u_texture = glGetUniformLocation(program, "texture");

	if (a_coord2d == -1 || u_color==-1 || u_offset == -1 || 
		u_scale_x == -1 || u_sprite == -1 || u_texture == -1)
		return false;

	// Enable blending
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glEnable(GL_POINT_SPRITE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

	// Upload the texture for point sprites
	glActiveTexture(GL_TEXTURE0);
	glGenTextures(1, &texture_id);
	glBindTexture(GL_TEXTURE_2D, texture_id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img_texture.width, img_texture.height, 0, 
				 GL_RGBA, GL_UNSIGNED_BYTE, img_texture.pixel_data);

	// Allocate and assign a Vertex Array Object to our handle 
	glGenVertexArrays(1, &vao);

	// Bind our Vertex Array Object as the current used object
	glBindVertexArray(vao);
    
    size_t n = points2d.size();
    
	// Create the vertex buffer object
	glGenBuffers(n, vbo);
	
	for (size_t i=0; i<n; i++) {
		// Bind our first VBO as being the active buffer and storing vertex attributes (coordinates)
		glBindBuffer(GL_ARRAY_BUFFER, vbo[i]);
		// Tell OpenGL to copy our array to the buffer object
		std::vector<point2d>& pts = points2d[i].first;
		glBufferData(GL_ARRAY_BUFFER, sizeof(point2d)*pts.size(), &pts[0], GL_STATIC_DRAW);
	}

	glClearColor(bgcolor.c[0], bgcolor.c[1], bgcolor.c[2], 1.0);

	return true;
}

bool glplot::init_3d()
{
	// vertex shader with two rotation matricies stored
	const char* VERTEX_SHADER = 
	"layout(location = 0) in vec3 position;"
	"uniform vec3 offset;"
	"uniform mat4 perspective;"
	"uniform vec2 angle;"
	"uniform vec3 color;"
	"smooth out vec4 theColor;"
	"void main() {"
	"	mat4 xRMatrix = mat4(cos(angle.x), 0.0, sin(angle.x), 0.0,"
    "                   0.0, 1.0, 0.0, 0.0,"
    "                   -sin(angle.x), 0.0, cos(angle.x), 0.0,"
    "                   0.0, 0.0, 0.0, 1.0);"
	"	mat4 yRMatrix = mat4(1.0, 0.0, 0.0, 0.0,"
    "              		0.0, cos(angle.y), -sin(angle.y), 0.0,"
    "                   0.0, sin(angle.y), cos(angle.y), 0.0,"
    "                   0.0, 0.0, 0.0, 1.0);"
	"	vec4 rotatedPosition = vec4(position.xyz, 1.0f) * xRMatrix * yRMatrix;"
	"	vec4 cameraPos = rotatedPosition + vec4(offset.x, offset.y, offset.z, 0.0);"
	"	gl_Position = perspective * cameraPos;"
	"	theColor = vec4(color, 1.0);"
//	"	theColor = mix(vec4(color, 1.0),"
//	"				   vec4(0.0f, color.y, color.z, 1.0), position.y/10.0);"
	"}";

	// fragment shader
	const char* FRAGMENT_SHADER =
	"smooth in vec4 theColor;"
	"out vec4 outputColor;"
	"void main() {"
	"	outputColor = theColor;"
	"}";
	
	program = create_program(VERTEX_SHADER, FRAGMENT_SHADER);
	if (program == 0) return false;

    //vertex array object state
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

    //get uniform locations in shaders
	u_offset = glGetUniformLocation(program, "offset");
	u_angle = glGetUniformLocation(program, "angle");
	u_color = glGetUniformLocation(program, "color");
	u_perspective = glGetUniformLocation(program, "perspective");

	//size of axis & marks
	GLfloat graph_size = 100;

	// setup perspective
	float frustumScale = 1.0f;
	float znear = 1.0f;
	float zfar = 100.0f;
	float M[16];
	memset(M, 0, sizeof(float)*16);
	// build perspective matrix
	M[0] = frustumScale;
	M[5] = frustumScale;
	M[10] = (zfar + znear) / (znear - zfar);
	M[14] = (2*zfar*znear) / (znear - zfar);
	M[11] = -1.0f;
    // bind data to shader
	glUseProgram(program);
	glUniformMatrix4fv(u_perspective, 1, GL_TRUE, M); // perspective matrix
	glUniform3f(u_offset, xpos, ypos, zpos); // position offset
	glUniform2f(u_angle, xang, yang); // rotation angles
	glUniform3f(u_color, 1, 1, 1); // default color
	glUseProgram(0);

	// axis-x lines
	axis[0] = { -graph_size, 0.0f, 0.0f };
	axis[1] = {  graph_size, 0.0f, 0.0f };
	// axis-y lines
	axis[2] = { 0.0f, -graph_size, 0.0f };
	axis[3] = { 0.0f,  graph_size, 0.0f };
	// axis-z lines
	axis[4] = { 0.0f, 0.0f,  graph_size };
	axis[5] = { 0.0f, 0.0f, -graph_size };
	//buffer for axis
	glGenBuffers(1, &vbo_axis);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_axis);
	glBufferData(GL_ARRAY_BUFFER, sizeof(axis), axis, GL_STATIC_DRAW);

	// axis marks
	int index = 0;
	for (int i=(int)-graph_size; i<(int)graph_size; i++) {
		if (i % 2 != 0) continue; //only every 2 points
		axis_marks[index++] = { GLfloat(i), 0.0f, 0.0f };
		axis_marks[index++] = { GLfloat(i), 1.0f, 0.0f };
	}
	for (int i=(int)-graph_size; i<(int)graph_size; i++) {
		if (i % 2 != 0) continue;
		axis_marks[index++] = { 0.0f, GLfloat(i), 0.0f };
		axis_marks[index++] = { 1.0f, GLfloat(i), 0.0f };
	}
	for (int i=(int)-graph_size; i<(int)graph_size; i++) {
		if (i % 2 != 0) continue;
		axis_marks[index++] = { 0.0f, 0.0f, GLfloat(i) };
		axis_marks[index++] = { 0.0f, 1.0f, GLfloat(i) };
	}

	// buffer for axis marks
	glGenBuffers(1, &vbo_marks);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_marks);
	glBufferData(GL_ARRAY_BUFFER, sizeof(axis_marks), axis_marks, GL_STATIC_DRAW);

    size_t n = points3d.size();
    
	// Create the vertex buffer object
	glGenBuffers(n, vbo);
	
	for (size_t i=0; i<n; i++) {
		// Bind our first VBO as being the active buffer and storing vertex attributes (coordinates)
		glBindBuffer(GL_ARRAY_BUFFER, vbo[i]);
		// Tell OpenGL to copy our array to the buffer object
		std::vector<point3d>& pts = points3d[i].first;
		glBufferData(GL_ARRAY_BUFFER, sizeof(point3d)*pts.size(), &pts[0], GL_STATIC_DRAW);
	}
	
	glClearColor(bgcolor.c[0], bgcolor.c[1], bgcolor.c[2], 1.0);
    	
	return true;
}

bool glplot::init_image2d()
{
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   	glShadeModel(GL_FLAT);
   	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glClearColor(bgcolor.c[0], bgcolor.c[1], bgcolor.c[2], 1.0);
	//gluOrtho2D(0, win_w, 0, win_h);
	return true;
}


void glplot::on_display()
{
	glClear(GL_COLOR_BUFFER_BIT);
	
	if (mode==Plot2D) {
		glUseProgram(program);
		glUniform1i(u_texture, 0);
		glUniform2f(u_offset, offset_x, offset_y);
		glUniform1f(u_scale_x, scale);
		for (size_t i=0; i<points2d.size(); i++) {
			size_t nvertices = points2d[i].first.size();
			glBindBuffer(GL_ARRAY_BUFFER, vbo[i]);
			glEnableVertexAttribArray(a_coord2d);
			glVertexAttribPointer(a_coord2d, 2, GL_FLOAT, GL_FALSE, 0, 0);
			GLfloat* color = points2d[i].second.color.c;
			glUniform3f(u_color, color[0], color[1], color[2]);
			switch (points2d[i].second.type) {
			case Line:
				glLineWidth(points2d[i].second.ptsize);
				glUniform1f(u_sprite, 0);
				glDrawArrays(GL_LINE_STRIP, 0, nvertices);
				glLineWidth(1);
				break;
			case Scatter:
				glUniform1f(u_sprite, points2d[i].second.ptsize);
				glDrawArrays(GL_POINTS, 0, nvertices);
				break;
			case Sprite:
				glUniform1f(u_sprite, img_texture.width);
				glDrawArrays(GL_POINTS, 0, nvertices);
				break;
			}
		}
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glUseProgram(0);

	} else if (mode==Plot3D) {
		glUseProgram(program);
		glUniform3f(u_offset, xpos, ypos, zpos);
		glUniform2f(u_angle, xang, yang);
		//draw axis
		glUniform3f(u_color, axcolor.c[0], axcolor.c[1], axcolor.c[2]);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_axis);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
		glDrawArrays(GL_LINES, 0, 9);
		glDisableVertexAttribArray(0);
		//axis marks
		glBindBuffer(GL_ARRAY_BUFFER, vbo_marks);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
		glDrawArrays(GL_LINES, 0, 600);
		glDisableVertexAttribArray(0);
		
		for (size_t i=0; i<points3d.size(); i++) {
			int nvertices = points3d[i].first.size();
			GLfloat* color = points3d[i].second.color.c;
			glUniform3f(u_color, color[0], color[1], color[2]);
			glBindBuffer(GL_ARRAY_BUFFER, vbo[i]);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
			glDrawArrays(GL_POINTS, 0, nvertices);
			glDisableVertexAttribArray(0);
		}
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glUseProgram(0);

	} else if (mode==RGBImage) {
		glPixelZoom(scale, scale);
		int n = (int)images.size();
		if (n > grid_x*grid_y) n = grid_x*grid_y;
		int wmax=-1, hmax=-1;
		for (int i=0; i<n; ++i) {
			dim2 dims = images[i].first;
			if (dims.w > wmax) wmax = dims.w;
			if (dims.h > hmax) hmax = dims.h;
		}
		float x_pos = offset_x - wmax*grid_x/2.0f/win_w*scale - grid_padding*grid_x/2.0f/win_w*scale;
		float y_pos = offset_y + hmax*grid_y/2.0f/win_h*scale + grid_padding*grid_y/2.0f/win_h*scale;;
   		int x=0, y=0;
   		for (int i=0; i<n; ++i) {
   			dim2 dims = images[i].first;
   			glRasterPos2f(x_pos+((x*wmax)*2.0f+x*grid_padding)/win_w*scale, y_pos-((y*hmax)*2.0f+y*grid_padding)/win_h*scale);
   			glDrawPixels(dims.w, dims.h, GL_RGB, GL_UNSIGNED_BYTE, &images[i].second[0]);
   			x++;
   			if (x >= grid_x) {
   				x = 0;
   				y++;
   			}
   		}
   	}
	
	glutSwapBuffers();
}

void glplot::on_reshape(int width, int height)
{
	win_w = width;
	win_h = height;
	glViewport(0, 0, win_w, win_h);
}

void glplot::on_idle()
{
}

void glplot::on_mouse_button(int button, int state, int x, int y)
{
	if (button==GLUT_LEFT_BUTTON) {
		mouse_left_down = (state==GLUT_DOWN);
		if (mouse_left_down) {
			mouse_down_x = x;
			mouse_down_y = y;
		}
		return;
	}
	if (button==3) {
		if (state==GLUT_UP) return;
		scale *= 1.5;
		zpos /= 1.2;
	}
	if (button==4) {
		if (state==GLUT_UP) return;
		scale /= 1.5;
		zpos *= 1.2;
	}
	//std::cout << "scale="<<scale<<", zpos="<<zpos<<"\n";		
	glutPostRedisplay();
}

void glplot::on_mouse_move(int x, int y)
{
	mouse_x = x;
	mouse_y = y;
	if (mouse_left_down) {
		if (mode==RGBImage) {
			offset_x += (mouse_x-mouse_down_x)*0.003;
			offset_y -= (mouse_y-mouse_down_y)*0.003;
		} else {
			offset_x += (mouse_x-mouse_down_x)*0.01;
			offset_y -= (mouse_y-mouse_down_y)*0.01;
		}
		xang += (mouse_x-mouse_down_x)*0.01;
		yang += (mouse_y-mouse_down_y)*0.01;
		mouse_down_x = mouse_x;
		mouse_down_y = mouse_y;
		glutPostRedisplay();
	}
}

void glplot::on_kbd_down(unsigned char key, int /*x*/, int /*y*/)
{
	if (key=='a') xpos -= 0.1;
	if (key=='d') xpos += 0.1;
	if (key=='w') ypos -= 0.1;
	if (key=='s') ypos += 0.1;
	if (key=='e') zpos -= 0.1;
	if (key=='q') zpos += 0.1;
	glutPostRedisplay();
}

void glplot::on_kbd_up(unsigned char /*key*/, int /*x*/, int /*y*/)
{
}

void glplot::on_special_kbd_down(int key, int /*x*/, int /*y*/)
{
	switch (key) {
	case GLUT_KEY_LEFT: 
		if (glutGetModifiers() == GLUT_ACTIVE_CTRL) {
			offset_x -= 2.0; xang = xang - 0.5; 
		} else {
			offset_x -= 0.1; xang = xang - 0.025; 
		}
		break;
	case GLUT_KEY_RIGHT: 
		if (glutGetModifiers() == GLUT_ACTIVE_CTRL) {
			offset_x += 2.0; xang = xang + 0.5; 
		} else {
			offset_x += 0.1; xang = xang + 0.025; 
		}
		break;
	case GLUT_KEY_UP: scale *= 1.5; yang = yang - 0.025; break;
	case GLUT_KEY_DOWN: scale /= 1.5; yang = yang + 0.025; break;
	}
	//std::cout << "ofs_x="<<offset_x<<", xang="<<xang<<", yang="<<yang<<"\n";		
	glutPostRedisplay();
}

void glplot::on_special_kbd_up(int key, int /*x*/, int /*y*/)
{
	bool exit = false;
	switch (key) {
	case GLUT_KEY_END: exit = true; break;
	case GLUT_KEY_HOME: 
		offset_x = offset_y = 0.0; scale  = 1.0; 
		xpos = ypos = 0.0f; zpos = -40.0f;
		xang = yang = 0.0f;
		break;
	}
	if (exit) glutLeaveMainLoop(); //glutExit();
	else glutPostRedisplay();
}


GLuint glplot::create_shader(const char* shader_src, GLenum type)
{
	std::string source = shader_src;
	GLuint res = glCreateShader(type);
	const GLchar* sources[] = { "#version 330\n", source.c_str() };
	glShaderSource(res, 2, sources, NULL);
	glCompileShader(res);
	GLint compiled = GL_FALSE;
	glGetShaderiv(res, GL_COMPILE_STATUS, &compiled);
	if (!compiled) {
		err_str = "glGetShaderiv failed.";
		glDeleteShader(res);
		return 0;
	}
	return res;
}

GLuint glplot::create_program(const char* vertex_src, const char* fragment_src)
{
	GLuint program = glCreateProgram();
	GLuint shader;
	if (vertex_src) {
		shader = create_shader(vertex_src, GL_VERTEX_SHADER);
		if (!shader) return 0;
		glAttachShader(program, shader);
	}
	if (fragment_src) {
		shader = create_shader(fragment_src, GL_FRAGMENT_SHADER);
		if (!shader) return 0;
		glAttachShader(program, shader);
	}
	glLinkProgram(program);
	GLint linked = GL_FALSE;
	glGetProgramiv(program, GL_LINK_STATUS, &linked);
	if (!linked) {
		err_str = "glGetProgramiv failed.";
		glDeleteProgram(program);
		return 0;
	}
	return program;
}


};     // namespace umml

#endif // UMML_GLPLOT_INCUDED
