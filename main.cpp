/*
  FLUIDS v.3 - SPH Fluid Simulator for CPU and GPU
  Copyright (C) 2012. Rama Hoetzlein, http://www.rchoetzlein.com

  ZLib license
  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

#include <time.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "common_defs.h"
#include "camera3d.h"

#ifdef BUILD_CUDA
	#include "fluid_system_host.cuh"	
#endif
#include "fluid_system.h"
#include "gl_helper.h"

#ifdef _MSC_VER						// Windows
	#include <gl/glut.h>
#else								// Linux
	#include <GL/glut.h>	
#endif

bool bTiming = true;
bool bRec = false;
int mFrame = 0;

// Globals
FluidSystem		psys;
Camera3D		cam;

float window_width  = 1024;
float window_height = 768;

Vector3DF	obj_from, obj_angs, obj_dang;
Vector4DF	light[2], light_to[2];				// Light stuff
float		light_fov;	

int		psys_rate = 0;							// Particle stuff
int		psys_freq = 1;
int		psys_playback;

bool	bHelp = false;					// Toggles
int		iShade = 0;						// Shading mode (default = no shadows)
int		iClrMode = 0;
bool    bPause = false;

// View matricies
float model_matrix[16];					// Model matrix (M)

// Mouse control
#define DRAG_OFF		0				// mouse states
#define DRAG_LEFT		1
#define DRAG_RIGHT		2
int		last_x = -1, last_y = -1;		// mouse vars
int		mode = 0;
int		dragging = 0;
int		psel;

GLuint	screen_id;
GLuint	depth_id;


// Different things we can move around
#define MODE_CAM		0
#define MODE_CAM_TO		1
#define MODE_OBJ		2
#define MODE_OBJPOS		3
#define MODE_OBJGRP		4
#define MODE_LIGHTPOS	5

#define MODE_DOF		6

GLuint screenBufferObject;
GLuint depthBufferObject;
GLuint envid;

void drawScene ( float* viewmat, bool bShade )
{
	if ( iShade <= 1 && bShade ) {		
	
		glEnable ( GL_LIGHTING );
		glEnable ( GL_LIGHT0 );
		glDisable ( GL_COLOR_MATERIAL );

		Vector4DF amb, diff, spec;
		float shininess = 5.0;
		
		glColor3f ( 1, 1, 1 );
		glLoadIdentity ();
		glLoadMatrixf ( viewmat );

		float pos[4];
		pos[0] = light[0].x;
		pos[1] = light[0].y;
		pos[2] = light[0].z;
		pos[3] = 1;
		amb.Set ( 0,0,0,1 ); diff.Set ( 1,1,1,1 ); spec.Set(1,1,1,1);
		glLightfv ( GL_LIGHT0, GL_POSITION, (float*) &pos[0]);
		glLightfv ( GL_LIGHT0, GL_AMBIENT, (float*) &amb.x );
		glLightfv ( GL_LIGHT0, GL_DIFFUSE, (float*) &diff.x );
		glLightfv ( GL_LIGHT0, GL_SPECULAR, (float*) &spec.x ); 
		
		amb.Set ( 0,0,0,1 ); diff.Set ( .3, .3, .3, 1); spec.Set (.1,.1,.1,1);
		glMaterialfv (GL_FRONT_AND_BACK, GL_AMBIENT, (float*) &amb.x );
		glMaterialfv (GL_FRONT_AND_BACK, GL_DIFFUSE, (float*) &diff.x);
		glMaterialfv (GL_FRONT_AND_BACK, GL_SPECULAR, (float*) &spec.x);
		glMaterialfv (GL_FRONT_AND_BACK, GL_SHININESS, (float*) &shininess);
		

		//glColorMaterial ( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE );
		
		glLoadMatrixf ( viewmat );
		
		glBegin ( GL_QUADS );
		glNormal3f ( 0, 1, 0.001  );
		for (float x=-1000; x <= 1000; x += 100.0 ) {
			for (float y=-1000; y <= 1000; y += 100.0 ) {
				glVertex3f ( x, 0.0, y );
				glVertex3f ( x+100, 0.0 , y );
				glVertex3f ( x+100, 0.0, y+100 );
				glVertex3f ( x, 0.0, y+100 );
			}
		}
		glEnd ();
		
		glColor3f ( 0.1, 0.1, 0.2 );
		glDisable ( GL_LIGHTING );
		glBegin ( GL_LINES );		
		for (float n=-100; n <= 100; n += 10.0 ) {
			glVertex3f ( -100, 0.1, n );
			glVertex3f ( 100, 0.1, n );
			glVertex3f ( n, 0.1, -100 );
			glVertex3f ( n, 0.1, 100 );
		}
		glVertex3f ( light[0].x, light[0].y, 0 );
		glVertex3f ( light[0].x, light[0].y, light[0].z );
		glEnd ();

		psys.Draw ( cam, 0.8 );				// Draw particles		

	} else {
		glDisable ( GL_LIGHTING );
		psys.Draw ( cam, 0.55 );			// Draw particles
	}
}

void draw2D ()
{
	
	mint::Time start, stop;

	#ifdef USE_SHADOWS
		disableShadows ();
	#endif
	glDisable ( GL_LIGHTING );  
	glDisable ( GL_DEPTH_TEST );

	glMatrixMode ( GL_PROJECTION );
	glLoadIdentity ();  
	glScalef ( 2.0/window_width, -2.0/window_height, 1 );		// Setup view (0,0) to (800,600)
	glTranslatef ( -window_width/2.0, -window_height/2, 0.0);


	float view_matrix[16];
	glMatrixMode ( GL_MODELVIEW );
	glLoadIdentity ();

	// Particle Information
	if ( psys.GetSelected() != -1 ) {	
		psys.DrawParticleInfo ();
		return;	
	}

	char disp[200];
		
	/*psys.getModeClr ();
	strcpy ( disp, psys.getModeStr().c_str() ); drawText ( 20, 40, disp );*/

	glColor4f ( 1.0, 1.0, 1.0, 1.0 );	
	strcpy ( disp, "Press H for help." );		drawText ( 10, 20, disp );  

	glColor4f ( 1.0, 1.0, 0.0, 1.0 );	
	strcpy ( disp, "" );
	if ( psys.GetToggle(PCAPTURE) ) strcpy ( disp, "CAPTURING VIDEO");
	drawText ( 200, 20, disp );

	if ( bHelp ) {	

		sprintf ( disp,	"Mode (f/g):                %s", psys.getModeStr().c_str() );					drawText ( 20, 40,  disp );
		
		sprintf ( disp,	"Scene:               %s (id: %d)", psys.getSceneName().c_str(), (int) psys.GetParam(PEXAMPLE) );				drawText ( 20, 60,  disp );

		sprintf ( disp,	"# Particles:         %d", psys.NumPoints() );					drawText ( 20, 80,  disp );
	
		sprintf ( disp,	"Grid Density:        %f", psys.GetParam (PGRID_DENSITY) );		drawText ( 20, 100,  disp );
		sprintf ( disp,	"Grid Count:          %f", (float) psys.GetParam( PSTAT_GRIDCNT ) / psys.GetParam(PSTAT_OCCUPY) );	drawText ( 20, 110,  disp );
		sprintf ( disp,	"Grid Occupancy:      %f%%", (float) psys.GetParam( PSTAT_OCCUPY ) / psys.getGridTotal() );		drawText ( 20, 130,  disp );
		sprintf ( disp,	"Grid Resolution:     %d x %d x %d (%d)", (int) psys.GetGridRes().x, (int) psys.GetGridRes().y, (int) psys.GetGridRes().z, psys.getGridTotal() );		drawText ( 20, 140,  disp );
		int nsrch = pow ( psys.getSearchCnt(), 1/3.0 );
		sprintf ( disp,	"Grid Search:         %d x %d x %d", nsrch, nsrch, nsrch );			drawText ( 20, 150,  disp );
		sprintf ( disp,	"Search Count:        %d, ave: %f, max: %f", (int) psys.GetParam(PSTAT_SRCH), psys.GetParam(PSTAT_SRCH)/psys.NumPoints(), psys.GetParam(PSTAT_SRCHMAX)/psys.NumPoints() );		drawText ( 20, 160,  disp );
		sprintf ( disp,	"Neighbor Count:      %d, ave: %f, max: %f", (int) psys.GetParam(PSTAT_NBR), psys.GetParam(PSTAT_NBR)/psys.NumPoints(), psys.GetParam(PSTAT_NBRMAX)/psys.NumPoints() );		drawText ( 20, 170,  disp );
		sprintf ( disp,	"Search Overhead:     %.2fx", psys.GetParam(PSTAT_SRCH)/psys.GetParam(PSTAT_NBR) );		drawText ( 20, 180,  disp );

		sprintf ( disp,	"Insert Time:         %.3f ms", psys.GetParam(PTIME_INSERT)  );			drawText ( 20, 200,  disp );
		sprintf ( disp,	"Sort Time:           %.3f ms", psys.GetParam(PTIME_SORT)  );			drawText ( 20, 210,  disp );
		sprintf ( disp,	"Count Time:          %.3f ms", psys.GetParam(PTIME_COUNT)  );			drawText ( 20, 220,  disp );
		sprintf ( disp,	"Pressure Time:       %.3f ms", psys.GetParam(PTIME_PRESS) );			drawText ( 20, 230,  disp );
		sprintf ( disp,	"Force Time:          %.3f ms", psys.GetParam(PTIME_FORCE) );			drawText ( 20, 240,  disp );
		sprintf ( disp,	"Advance Time:        %.3f ms", psys.GetParam(PTIME_ADVANCE) );			drawText ( 20, 250,  disp );
		float st = psys.GetParam(PTIME_INSERT) + psys.GetParam(PTIME_PRESS)+psys.GetParam(PTIME_FORCE) + psys.GetParam(PTIME_ADVANCE);
		sprintf ( disp,	"Total Sim Time:      %.3f ms, %.1f fps", st, 1000.0/st );									drawText ( 20, 260,  disp );
		sprintf ( disp,	"Performance:         %d particles/sec", (int) ((psys.NumPoints()*1000.0)/st) );			drawText ( 20, 270,  disp );

		sprintf ( disp,	"Particle Memory:     %.4f MB", (float) psys.GetParam(PSTAT_PMEM)/1000000.0f );		drawText ( 20, 290,  disp );
		sprintf ( disp,	"Grid Memory:         %.4f MB", (float) psys.GetParam(PSTAT_GMEM)/1000000.0f );		drawText ( 20, 300,  disp );
		
		
		/*sprintf ( disp,	"[ ]    Next/Prev Demo" );			drawText ( 20, 90,  disp );
		sprintf ( disp,	"N M    Adjust Max Particles" );	drawText ( 20, 100,  disp );
		sprintf ( disp,	"space  Pause" );					drawText ( 20, 110,  disp );
		sprintf ( disp,	"S      Shading mode" );			drawText ( 20, 120,  disp );	
		sprintf ( disp,	"G      Toggle CUDA vs CPU" );		drawText ( 20, 130,  disp );	
		sprintf ( disp,	"< >    Change emitter rate" );		drawText ( 20, 140,  disp );	
		sprintf ( disp,	"C      Move camera /w mouse" );	drawText ( 20, 150,  disp );	
		sprintf ( disp,	"I      Move emitter /w mouse" );	drawText ( 20, 160,  disp );	
		sprintf ( disp,	"O      Change emitter angle" );	drawText ( 20, 170,  disp );	
		sprintf ( disp,	"L      Move light /w mouse" );		drawText ( 20, 180,  disp );			
		sprintf ( disp,	"X      Draw velocity/pressure/color" );	drawText ( 20, 190,  disp );

		Vector3DF vol = psys.GetVec(PVOLMAX);
		vol -= psys.GetVec(PVOLMIN);
		sprintf ( disp,	"Volume Size:           %3.5f %3.2f %3.2f", vol.x, vol.y, vol.z );	drawText ( 20, 210,  disp );
		sprintf ( disp,	"Time Step (dt):        %3.5f", psys.GetDT () );					drawText ( 20, 220,  disp );
		
		sprintf ( disp,	"Simulation Scale:      %3.5f", psys.GetParam(PSIMSIZE) );		drawText ( 20, 240,  disp );
		sprintf ( disp,	"Simulation Size (m):   %3.5f", psys.GetParam(PSIMSCALE) );		drawText ( 20, 250,  disp );
		sprintf ( disp,	"Smooth Radius (m):     %3.3f", psys.GetParam(PSMOOTHRADIUS) );	drawText ( 20, 260,  disp );
		sprintf ( disp,	"Particle Radius (m):   %3.3f", psys.GetParam(PRADIUS) );		drawText ( 20, 270,  disp );
		sprintf ( disp,	"Particle Mass (kg):    %0.8f", psys.GetParam(PMASS) );			drawText ( 20, 280,  disp );
		sprintf ( disp,	"Rest Density (kg/m^3): %3.3f", psys.GetParam(PRESTDENSITY) );	drawText ( 20, 290,  disp );
		sprintf ( disp,	"Viscosity:             %3.3f", psys.GetParam(PVISC) );			drawText ( 20, 300,  disp );
		sprintf ( disp,	"Internal Stiffness:    %3.3f", psys.GetParam(PINTSTIFF) );		drawText ( 20, 310,  disp );
		sprintf ( disp,	"Boundary Stiffness:    %6.0f", psys.GetParam(PEXTSTIFF) );		drawText ( 20, 320,  disp );
		sprintf ( disp,	"Boundary Dampening:    %4.3f", psys.GetParam(PEXTDAMP) );		drawText ( 20, 330,  disp );
		sprintf ( disp,	"Speed Limiting:        %4.3f", psys.GetParam(PVEL_LIMIT) );		drawText ( 20, 340,  disp );
		vol = psys.GetVec ( PPLANE_GRAV_DIR );
		sprintf ( disp,	"Gravity:               %3.2f %3.2f %3.2f", vol.x, vol.y, vol.z );	drawText ( 20, 350,  disp );*/
	}
}

int frame;

void display () 
{
	mint::Time tstart, tstop;	
	mint::Time rstart, rstop;	

	tstart.SetSystemTime ( ACC_NSEC );

//	iso = sin(frame*0.01f );
	
	// Do simulation!
	if ( !bPause ) psys.Run (window_width, window_height);

	frame++;
	measureFPS ();

	glEnable ( GL_DEPTH_TEST );

	// Render depth map shadows
	rstart.SetSystemTime ( ACC_NSEC );
	disableShadows ();
	#ifdef USE_SHADOWS
		if ( iShade==1 ) {
			renderDepthMap_FrameBuffer ( 0, window_width, window_height );
		} else {
			renderDepthMap_Clear ( window_width, window_height );		
		}
	#endif	

	// Clear frame buffer
	if ( iShade<=1 ) 	glClearColor( 0.1, 0.1, 0.1, 1.0 );
	else				glClearColor ( 0, 0, 0, 0 );
	glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	glDisable ( GL_CULL_FACE );
	glShadeModel ( GL_SMOOTH );

	// Compute camera view
	cam.updateMatricies ();
	glMatrixMode ( GL_PROJECTION );
	glLoadMatrixf ( cam.getProjMatrix().GetDataF() );
	
	// Draw Shadows (if on)
	#ifdef USE_SHADOWS	
		if ( iShade==1 )	renderShadows ( cam.getViewMatrix().GetDataF()  );			
	#endif

	// Draw 3D	
	glEnable ( GL_LIGHTING );  
	glMatrixMode ( GL_MODELVIEW );
	glLoadMatrixf ( cam.getViewMatrix().GetDataF() );
	drawScene ( cam.getViewMatrix().GetDataF() , true );

	// Draw 2D overlay
	draw2D ();

	if ( psys.GetToggle(PPROFILE) ) { rstop.SetSystemTime ( ACC_NSEC ); rstop = rstop - rstart; printf ( "RENDER: %s\n", rstop.GetReadableTime().c_str() ); }
 
	// Swap buffers
	glutSwapBuffers();  
	glutPostRedisplay();

	if ( psys.GetToggle(PPROFILE) ) { 
		tstop.SetSystemTime ( ACC_NSEC ); tstop = tstop - tstart; 
		printf ( "TOTAL:  %s, %f fps\n", tstop.GetReadableTime().c_str(), 1000.0/tstop.GetMSec() ); 
		printf ( "PERFORMANCE:  %d particles/sec, %d\n", (int) (psys.NumPoints() * 1000.0)/tstop.GetMSec(), psys.NumPoints() ); 
	}
}

void reshape ( int width, int height ) 
{
  // set window height and width
  window_width  = (float) width;
  window_height = (float) height;
  glViewport( 0, 0, width, height );  
}

void UpdateEmit ()
{	
	obj_from = psys.GetVec ( PEMIT_POS );
	obj_angs = psys.GetVec ( PEMIT_ANG );
	obj_dang = psys.GetVec ( PEMIT_RATE );
}


void keyboard_func ( unsigned char key, int x, int y )
{
	Vector3DF fp = cam.getPos ();
	Vector3DF tp = cam.getToPos ();

	switch( key ) {
	case 'R': case 'r': {
		psys.StartRecord ();
		} break;
	//case 'P': case 'p': 	
		/*switch ( psys.getMode() ) {
		case MODE_RECORD: case MODE_SIM:	psys_playback = psys.getLastRecording();	break;
		case MODE_PLAYBACK:					psys_playback--; if ( psys_playback < 0 ) psys_playback = psys.getLastRecording();	break;
		};
		psys.StartPlayback ( psys_playback );
		} break;*/
	case 'M': case 'm': {
		psys.SetParam ( PNUM, psys.GetParam(PNUM)*2, 4, 40000000 );
		psys.Setup ( false );
		} break;
	case 'N': case 'n': {
		psys.SetParam ( PNUM, psys.GetParam(PNUM)/2, 4, 40000000 );
		psys.Setup ( false );
		} break;
	case '0':
		UpdateEmit ();
		psys_freq++;	
		psys.SetVec ( PEMIT_RATE, Vector3DF(psys_freq, psys_rate, 0) );
		break;  
	case '9':
		UpdateEmit ();
		psys_freq--;  if ( psys_freq < 0 ) psys_freq = 0;
		psys.SetVec ( PEMIT_RATE, Vector3DF(psys_freq, psys_rate, 0) );
		break;
	case '.': case '>':
		UpdateEmit ();
		if ( ++psys_rate > 100 ) psys_rate = 100;
		psys.SetVec ( PEMIT_RATE, Vector3DF(psys_freq, psys_rate, 0) );
		break;
	case ',': case '<':
		UpdateEmit ();
		if ( --psys_rate < 0 ) psys_rate = 0;
		psys.SetVec ( PEMIT_RATE, Vector3DF(psys_freq, psys_rate, 0) );
		break;
	
	case 'f': case 'F':		psys.IncParam ( PMODE, -1, 1, 8 );		psys.Setup (false); break;
	case 'g': case 'G':		psys.IncParam ( PMODE, 1, 1, 8 );		psys.Setup (false); break;
	case ' ':				psys.SetParam ( PMODE, RUN_PAUSE, 0, 8 );	break;		// pause

	case '1':				psys.IncParam ( PDRAWGRID, 1, 0, 1 );		break;
	case '2':				psys.IncParam ( PDRAWTEXT, 1, 0, 1 );		break;

	case 'C':	mode = MODE_CAM_TO;	break;
	case 'c': 	mode = MODE_CAM;	break; 
	case 'h': case 'H':	bHelp = !bHelp; break;
	case 'i': case 'I':	
		UpdateEmit ();
		mode = MODE_OBJPOS;	
		break;
	case 'o': case 'O':	
		UpdateEmit ();
		mode = MODE_OBJ;
		break;  
	case 'x': case 'X':
		if ( ++iClrMode > 2) iClrMode = 0;
		psys.SetParam ( PCLR_MODE, iClrMode );
		break;
	case 'l': case 'L':	mode = MODE_LIGHTPOS;	break;
	case 'j': case 'J': {
		int d = psys.GetParam ( PDRAWMODE ) + 1;
		if ( d > 2 ) d = 0;
		psys.SetParam ( PDRAWMODE, d );
		} break;	
	case 'k': case 'K':	if ( ++iShade > 3 ) iShade = 0;		break;

	case 'a': case 'A':		cam.setToPos( tp.x - 1, tp.y, tp.z ); break;
	case 'd': case 'D':		cam.setToPos( tp.x + 1, tp.y, tp.z ); break;
	case 'w': case 'W':		cam.setToPos( tp.x, tp.y - 1, tp.z ); break;
	case 's': case 'S':		cam.setToPos( tp.x, tp.y + 1, tp.z ); break;
	case 'q': case 'Q':		cam.setToPos( tp.x, tp.y, tp.z + 1 ); break;
	case 'z': case 'Z':		cam.setToPos( tp.x, tp.y, tp.z - 1 ); break;

		
	case 27:			    exit( 0 ); break;
	
	case '`':				psys.Toggle ( PCAPTURE ); break;
	
	case 't': case 'T':		psys.Setup (true); break;  

	case '-':  case '_':
		psys.IncParam ( PGRID_DENSITY, -0.2, 1, 10 );	
		psys.Setup (true);
		break;
	case '+': case '=':
		psys.IncParam ( PGRID_DENSITY, 0.2, 1, 10 );	
		psys.Setup (true);
		break;
	case '[':
		psys.IncParam ( PEXAMPLE, -1, 0, 10 );
		psys.Setup (true);
		UpdateEmit ();
		break;
	case ']':
		psys.IncParam ( PEXAMPLE, +1, 0, 10 );
		psys.Setup (true);
		UpdateEmit ();
		break;  
	default:
	break;
  }
}

Vector3DF cangs;
Vector3DF ctp;
float cdist;

void mouse_click_func ( int button, int state, int x, int y )
{
  cangs = cam.getAng();
  ctp = cam.getToPos();
  cdist = cam.getOrbitDist();

  if( state == GLUT_DOWN ) {
    if ( button == GLUT_LEFT_BUTTON )		dragging = DRAG_LEFT;
    else if ( button == GLUT_RIGHT_BUTTON ) dragging = DRAG_RIGHT;	
    last_x = x;
    last_y = y;	
  } else if ( state==GLUT_UP ) {
    dragging = DRAG_OFF;
  }
}

void mouse_move_func ( int x, int y )
{
	//psys.SelectParticle ( x, y, window_width, window_height, cam );
}

void mouse_drag_func ( int x, int y )
{
	int dx = x - last_x;
	int dy = y - last_y;

	switch ( mode ) {
	case MODE_CAM:
		if ( dragging == DRAG_LEFT ) {
			cam.moveOrbit ( dx, dy, 0, 0 );
		} else if ( dragging == DRAG_RIGHT ) {
			cam.moveOrbit ( 0, 0, 0, dy*0.15 );
		} 
		break;
	case MODE_CAM_TO:
		if ( dragging == DRAG_LEFT ) {
			cam.moveToPos ( dx*0.1, 0, dy*0.1 );
		} else if ( dragging == DRAG_RIGHT ) {
			cam.moveToPos ( 0, dy*0.1, 0 );
		}
		break;	
	case MODE_OBJ:
		if ( dragging == DRAG_LEFT ) {
			obj_angs.x -= dx*0.1;
			obj_angs.y += dy*0.1;
			printf ( "Obj Angs:  %f %f %f\n", obj_angs.x, obj_angs.y, obj_angs.z );
			//force_x += dx*.1;
			//force_y += dy*.1;
		} else if (dragging == DRAG_RIGHT) {
			obj_angs.z -= dy*.005;			
			printf ( "Obj Angs:  %f %f %f\n", obj_angs.x, obj_angs.y, obj_angs.z );
		}
		psys.SetVec ( PEMIT_ANG, Vector3DF ( obj_angs.x, obj_angs.y, obj_angs.z ) );
		break;
	case MODE_OBJPOS:
		if ( dragging == DRAG_LEFT ) {
			obj_from.x -= dx*.1;
			obj_from.y += dy*.1;
			printf ( "Obj:  %f %f %f\n", obj_from.x, obj_from.y, obj_from.z );
		} else if (dragging == DRAG_RIGHT) {
			obj_from.z -= dy*.1;
			printf ( "Obj:  %f %f %f\n", obj_from.x, obj_from.y, obj_from.z );
		}
		psys.SetVec ( PEMIT_POS, Vector3DF ( obj_from.x, obj_from.y, obj_from.z ) );
		//psys.setPos ( obj_x, obj_y, obj_z, obj_ang, obj_tilt, obj_dist );
		break;
	case MODE_LIGHTPOS:
		if ( dragging == DRAG_LEFT ) {
			light[0].x -= dx*.1;
			light[0].z -= dy*.1;		
			printf ( "Light: %f %f %f\n", light[0].x, light[0].y, light[0].z );
		} else if (dragging == DRAG_RIGHT) {
			light[0].y -= dy*.1;			
			printf ( "Light: %f %f %f\n", light[0].x, light[0].y, light[0].z );
		}	
		#ifdef USE_SHADOWS
			setShadowLight ( light[0].x, light[0].y, light[0].z, light_to[0].x, light_to[0].y, light_to[0].z, light_fov );
		#endif
		break;
	}

	if ( x < 10 || y < 10 || x > 1000 || y > 700 ) {
		glutWarpPointer ( 1024/2, 768/2 );
		last_x = 1024/2;
		last_y = 768/2;
	} else {
		last_x = x;
		last_y = y;
	}
}


void idle_func ()
{
}

void init ()
{
	
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);	
	glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);	

	srand ( time ( 0x0 ) );

	glClearColor( 0.49, 0.49, 0.49, 1.0 );
	glShadeModel( GL_SMOOTH );

	glEnable ( GL_COLOR_MATERIAL );
	glEnable (GL_DEPTH_TEST);  
	glEnable (GL_BLEND);
	glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 
	glDepthMask ( 1 );
	glEnable ( GL_TEXTURE_2D );

	// callbacks
	glutDisplayFunc( display );
	glutReshapeFunc( reshape );
	glutKeyboardFunc( keyboard_func );
	glutMouseFunc( mouse_click_func );  
	glutMotionFunc( mouse_drag_func );
	glutPassiveMotionFunc ( mouse_move_func );
	glutIdleFunc( idle_func );

	// glutSetCursor ( GLUT_CURSOR_NONE );
	
	// Initialize camera
	cam.setOrbit ( Vector3DF(200,30,0), Vector3DF(2,2,2), 400, 400 );
	cam.setFov ( 35 );
	cam.updateMatricies ();
	
	light[0].x = 0;		light[0].y = 200;	light[0].z = 0; light[0].w = 1;
	light_to[0].x = 0;	light_to[0].y = 0;	light_to[0].z = 0; light_to[0].w = 1;

	light[1].x = 55;		light[1].y = 140;	light[1].z = 50;	light[1].w = 1;
	light_to[1].x = 0;	light_to[1].y = 0;	light_to[1].z = 0;		light_to[1].w = 1;

	light_fov = 45;

	#ifdef USE_SHADOWS
		createShadowTextures();
		createFrameBuffer ();
		setShadowLight ( light[0].x, light[0].y, light[0].z, light_to[0].x, light_to[0].y, light_to[0].z, light_fov );
		setShadowLightColor ( .7, .7, .7, 0.2, 0.2, 0.2 );		
	#endif

	obj_from.x = 0;		obj_from.y = 0;		obj_from.z = 20;		// emitter
	obj_angs.x = 118.7;	obj_angs.y = 200;	obj_angs.z = 1.0;
	obj_dang.x = 1;	obj_dang.y = 1;		obj_dang.z = 0;

	psys.Setup (true);
	psys.SetVec ( PEMIT_ANG, Vector3DF ( obj_angs.x, obj_angs.y, obj_angs.z ) );
	psys.SetVec ( PEMIT_POS, Vector3DF ( obj_from.x, obj_from.y, obj_from.z ) );

	psys.SetParam ( PCLR_MODE, iClrMode );	

	// Get last recording
	psys_playback = psys.getLastRecording ();
}


int main ( int argc, char **argv )
{
	#ifdef BUILD_CUDA
		// Initialize CUDA
		cudaInit( argc, argv );
	#endif

	// set up the window
	glutInit( &argc, &argv[0] ); 
	glutInitDisplayMode( GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);
	glutInitWindowPosition( 100, 100 );
	glutInitWindowSize( (int) window_width, (int) window_height );
	glutCreateWindow ( "Fluids v.3 (c) 2012, R.C. Hoetzlein (ZLib)" );

//	glutFullScreen ();

	init();	
	
	psys.SetupRender ();
	glutMainLoop();

	psys.Exit ();

	return 0;
}

