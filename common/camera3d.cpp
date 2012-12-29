/*
  FLUIDS v.3 - SPH Fluid Simulator for CPU and GPU
  Copyright (C) 2012. Rama Hoetzlein, http://fluids3.com

  Fluids-ZLib license (* see part 1 below)
  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. Acknowledgement of the
	 original author is required if you publish this in a paper, or use it
	 in a product. (See fluids3.com for details)
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/


// CGL - Complex Graphics Language
// Camera3D Class
//
// 3D Camera3D IMPLEMENTATION 
// 
// The Camera3D transformation of an arbitrary point is:
//
//		Q' = Q * T * R * P
//
// where Q  = 3D point
//		 Q' = Screen point
//		 T = Camera3D Translation (moves Camera3D to origin)
//		 R = Camera3D Rotation (rotates Camera3D to point 'up' along Z axis)
//       P = Projection (projects points onto XY plane)
// 
// T is a unit-coordinate system translated to origin from Camera3D:
//		[1	0  0  0]
//		[0	1  0  0]
//		[0	0  1  0]
//		[-cx -cy -cz 0]		where c is Camera3D location
// R is a basis matrix:	
//
// P is a projection matrix:
//

#include "gl_helper.h"

#include "camera3d.h"

Camera3D::Camera3D ()
{	
	
	mProjType = Perspective;
	mWire = 0;

	up_dir.Set ( 0.0, 1.0, 0 );				// frustum params
	mAspect = 800/600.0;
	mDolly = 5.0;
	mFov = 40.0;	
	mNear = 0.1;
	mFar = 6000.0;
	mTile.Set ( 0, 0, 1, 1 );

	for (int n=0; n < 8; n++ ) mOps[n] = false;	
	mOps[0] = false;

//	mOps[0] = true;		
//	mOps[1] = true;

	setOrbit ( 0, 45, 0, Vector3DF(0,0,0), 120.0, 5.0 );
	updateMatricies ();

}

void Camera3D::draw_gl ()
{
	Vector3DF pnt; 
	int va, vb;
	
	if ( !mOps[0] ) return;

	// Box testing
	//
	// NOTES: This demonstrates AABB box testing against the frustum 
	// Boxes tested are 10x10x10 size, spaced apart from each other so we can see them.
	if ( mOps[5] ) {
		glPushMatrix ();
		glEnable ( GL_LIGHTING );
		glColor3f ( 1, 1, 1 );	
		Vector3DF bmin, bmax, vmin, vmax;
		int lod;
		for (float y=0; y < 100; y += 10.0 ) {
		for (float z=-100; z < 100; z += 10.0 ) {
			for (float x=-100; x < 100; x += 10.0 ) {
				bmin.Set ( x, y, z );
				bmax.Set ( x+8, y+8, z+8 );
				if ( boxInFrustum ( bmin, bmax ) ) {				
					lod = (int) calculateLOD ( bmin, 1, 5, 300.0 );
					//rendGL->drawCube ( bmin, bmax, Vector3DF(1,1,1) );
				}
			}
		}
		}
		glPopMatrix ();
	}

	glDisable ( GL_LIGHTING );	
	glLoadMatrixf ( getViewMatrix().GetDataF() );

	// Frustum planes (world space)
	//
	// NOTE: The frustum planes are drawn as discs because
	// they are boundless (infinite). The minimum information contained in the
	// plane equation is normal direction and distance from plane to origin.
	// This sufficiently defines infinite planes for inside/outside testing,
	// but cannot be used to draw the view frustum without more information.
	// Drawing is done as discs here to verify the frustum plane equations.
	if ( mOps[2] ) {
		glBegin ( GL_POINTS );
		glColor3f ( 1, 1, 0 );
		Vector3DF norm;
		Vector3DF side, up;
		for (int n=0; n < 6; n++ ) {
			norm.Set ( frustum[n][0], frustum[n][1], frustum[n][2] );
			glColor3f ( n/6.0, 1.0- (n/6.0), 0.5 );
			side = Vector3DF(0,1,0); side.Cross ( norm ); side.Normalize ();	
			up = side; up.Cross ( norm ); up.Normalize();
			norm *= frustum[n][3];
			for (float y=-50; y < 50; y += 1.0 ) {
				for (float x=-50; x < 50; x += 1.0 ) {
					if ( x*x+y*y < 1000 ) {
						//pnt = side * x + up * y - norm; 
                        pnt = side;
                        Vector3DF tv = up;

                        tv *= y;
                        pnt *= x;
                        pnt += tv;
                        pnt -= norm;

						glVertex3f ( pnt.x, pnt.y, pnt.z );
					}
				}
			}
		}
		glEnd (); 
	}

	// Inside/outside testing
	//
	// NOTES: This code demonstrates frustum clipping 
	// tests on individual points.
	if ( mOps[4] ) {
		glColor3f ( 1, 1, 1 );
		glBegin ( GL_POINTS );
		for (float z=-100; z < 100; z += 4.0 ) {
			for (float y=0; y < 100; y += 4.0 ) {
				for (float x=-100; x < 100; x += 4.0 ) {
					if ( pointInFrustum ( x, y, z) ) {
						glVertex3f ( x, y, z );
					}
				}
			}
		}
		glEnd ();
	}
	
	// Inverse rays (world space)
	//
	// NOTES: This code demonstrates drawing 
	// inverse camera rays, as might be needed for raytracing or hit testing.
	if ( mOps[3] ) {
		glBegin ( GL_LINES );
		glColor3f ( 0, 1, 0);
		for (float x = 0; x <= 1.0; x+= 0.5 ) {
			for (float y = 0; y <= 1.0; y+= 0.5 ) {
				pnt = inverseRay ( x, y, mFar );
				pnt += from_pos;
				glVertex3f ( from_pos.x, from_pos.y, from_pos.z );		// all inverse rays originate at the camera center
				glVertex3f ( pnt.x, pnt.y, pnt.z );
			}
		}
		glEnd ();
	}

	// Projection
	//
	// NOTES: This code demonstrates 
	// perspective projection _without_ using the OpenGL pipeline.
	// Projection is done by the camera class. A cube is drawn on the near plane.
	
	// Cube geometry
	Vector3DF pnts[8];
	Vector3DI edge[12];
	pnts[0].Set (  0,  0,  0 );	pnts[1].Set ( 10,  0,  0 ); pnts[2].Set ( 10,  0, 10 ); pnts[3].Set (  0,  0, 10 );		// lower points (y=0)
	pnts[4].Set (  0, 10,  0 );	pnts[5].Set ( 10, 10,  0 ); pnts[6].Set ( 10, 10, 10 ); pnts[7].Set (  0, 10, 10 );		// upper points (y=10)
	edge[0].Set ( 0, 1, 0 ); edge[1].Set ( 1, 2, 0 ); edge[2].Set ( 2, 3, 0 ); edge[3].Set ( 3, 0, 0 );					// 4 lower edges
	edge[4].Set ( 4, 5, 0 ); edge[5].Set ( 5, 6, 0 ); edge[6].Set ( 6, 7, 0 ); edge[7].Set ( 7, 4, 0 );					// 4 upper edges
	edge[8].Set ( 0, 4, 0 ); edge[9].Set ( 1, 5, 0 ); edge[10].Set ( 2, 6, 0 ); edge[11].Set ( 3, 7, 0 );				// 4 vertical edges
	
	// -- White cube is drawn using OpenGL projection
	if ( mOps[6] ) {
		glBegin ( GL_LINES );
		glColor3f ( 1, 1, 1);
		for (int e = 0; e < 12; e++ ) {
			va = edge[e].x;
			vb = edge[e].y;
			glVertex3f ( pnts[va].x, pnts[va].y, pnts[va].z );
			glVertex3f ( pnts[vb].x, pnts[vb].y, pnts[vb].z );
		}
		glEnd ();	
	}

	//---- Draw the following in camera space..
	// NOTES:
	// The remainder drawing steps are done in 
	// camera space. This is done by multiplying by the
	// inverse_rotation matrix, which transforms from camera to world space.
	// The camera axes, near, and far planes can now be drawn in camera space.
	glPushMatrix ();
	glLoadMatrixf ( getViewMatrix().GetDataF() );
	glTranslatef ( from_pos.x, from_pos.y, from_pos.z );
	glMultMatrixf ( invrot_matrix.GetDataF() );				// camera space --to--> world space

	// -- Red cube is drawn on the near plane using software projection pipeline. See Camera3D::project
	if ( mOps[6] ) {
		glBegin ( GL_LINES );
		glColor3f ( 1, 0, 0);
		Vector4DF proja, projb;
		for (int e = 0; e < 12; e++ ) {
			va = edge[e].x;
			vb = edge[e].y;
			proja = project ( pnts[va] );
			projb = project ( pnts[vb] );
			if ( proja.w > 0 && projb.w > 0 && proja.w < 1 && projb.w < 1) {	// Very simple Z clipping  (try commenting this out and see what happens)
				glVertex3f ( proja.x, proja.y, proja.z );
				glVertex3f ( projb.x, projb.y, projb.z );
			}
		}
		glEnd ();
	}
	// Camera axes
	glBegin ( GL_LINES );
	float to_d = (from_pos - to_pos).Length();
	glColor3f ( .8,.8,.8); glVertex3f ( 0, 0, 0 );	glVertex3f ( 0, 0, -to_d );
	glColor3f ( 1,0,0); glVertex3f ( 0, 0, 0 );		glVertex3f ( 10, 0, 0 );
	glColor3f ( 0,1,0); glVertex3f ( 0, 0, 0 );		glVertex3f ( 0, 10, 0 );
	glColor3f ( 0,0,1); glVertex3f ( 0, 0, 0 );		glVertex3f ( 0, 0, 10 );
	glEnd ();

	if ( mOps[1] ) {
		// Near plane
		float sy = tan ( mFov * DEGtoRAD / 2.0);
		float sx = sy * mAspect;
		glColor3f ( 0.8, 0.8, 0.8 );
		glBegin ( GL_LINE_LOOP );
		glVertex3f ( -mNear*sx,  mNear*sy, -mNear );
		glVertex3f (  mNear*sx,  mNear*sy, -mNear );
		glVertex3f (  mNear*sx, -mNear*sy, -mNear );
		glVertex3f ( -mNear*sx, -mNear*sy, -mNear );
		glEnd ();
		// Far plane
		glBegin ( GL_LINE_LOOP );
		glVertex3f ( -mFar*sx,  mFar*sy, -mFar );
		glVertex3f (  mFar*sx,  mFar*sy, -mFar );
		glVertex3f (  mFar*sx, -mFar*sy, -mFar );
		glVertex3f ( -mFar*sx, -mFar*sy, -mFar );
		glEnd ();

		// Subview Near plane
		float l, r, t, b;
		l = -sx + 2.0*sx*mTile.x;						// Tile is in range 0 <= x,y <= 1
		r = -sx + 2.0*sx*mTile.z;
		t =  sy - 2.0*sy*mTile.y;
		b =  sy - 2.0*sy*mTile.w;
		glColor3f ( 0.8, 0.8, 0.0 );
		glBegin ( GL_LINE_LOOP );
		glVertex3f ( l * mNear, t * mNear, -mNear );
		glVertex3f ( r * mNear, t * mNear, -mNear );
		glVertex3f ( r * mNear, b * mNear, -mNear );
		glVertex3f ( l * mNear, b * mNear, -mNear );		
		glEnd ();
		// Subview Far plane
		glBegin ( GL_LINE_LOOP );
		glVertex3f ( l * mFar, t * mFar, -mFar );
		glVertex3f ( r * mFar, t * mFar, -mFar );
		glVertex3f ( r * mFar, b * mFar, -mFar );
		glVertex3f ( l * mFar, b * mFar, -mFar );		
		glEnd ();
	}

	glPopMatrix ();
}

bool Camera3D::pointInFrustum ( float x, float y, float z )
{
	int p;
	for ( p = 0; p < 6; p++ )
		if( frustum[p][0] * x + frustum[p][1] * y + frustum[p][2] * z + frustum[p][3] <= 0 )
			return false;
	return true;
}

bool Camera3D::boxInFrustum ( Vector3DF bmin, Vector3DF bmax)
{
	Vector3DF vmin, vmax;
	int p;
	bool ret = true;	
	for ( p = 0; p < 6; p++ ) {
		vmin.x = ( frustum[p][0] > 0 ) ? bmin.x : bmax.x;		// Determine nearest and farthest point to plane
		vmax.x = ( frustum[p][0] > 0 ) ? bmax.x : bmin.x;		
		vmin.y = ( frustum[p][1] > 0 ) ? bmin.y : bmax.y;
		vmax.y = ( frustum[p][1] > 0 ) ? bmax.y : bmin.y;
		vmin.z = ( frustum[p][2] > 0 ) ? bmin.z : bmax.z;
		vmax.z = ( frustum[p][2] > 0 ) ? bmax.z : bmin.z;
		if ( frustum[p][0]*vmax.x + frustum[p][1]*vmax.y + frustum[p][2]*vmax.z + frustum[p][3] <= 0 ) return false;		// If nearest point is outside, Box is outside
		else if ( frustum[p][0]*vmin.x + frustum[p][1]*vmin.y + frustum[p][2]*vmin.z + frustum[p][3] <= 0 ) ret = true;		// If nearest inside and farthest point is outside, Box intersects
	}
	return ret;			// No points found outside. Box must be inside.
	
	/* --- Original method - Slow yet simpler.
	int p;
	for ( p = 0; p < 6; p++ ) {
		if( frustum[p][0] * bmin.x + frustum[p][1] * bmin.y + frustum[p][2] * bmin.z + frustum[p][3] > 0 ) continue;
		if( frustum[p][0] * bmax.x + frustum[p][1] * bmin.y + frustum[p][2] * bmin.z + frustum[p][3] > 0 ) continue;
		if( frustum[p][0] * bmax.x + frustum[p][1] * bmin.y + frustum[p][2] * bmax.z + frustum[p][3] > 0 ) continue;
		if( frustum[p][0] * bmin.x + frustum[p][1] * bmin.y + frustum[p][2] * bmax.z + frustum[p][3] > 0 ) continue;
		if( frustum[p][0] * bmin.x + frustum[p][1] * bmax.y + frustum[p][2] * bmin.z + frustum[p][3] > 0 ) continue;
		if( frustum[p][0] * bmax.x + frustum[p][1] * bmax.y + frustum[p][2] * bmin.z + frustum[p][3] > 0 ) continue;
		if( frustum[p][0] * bmax.x + frustum[p][1] * bmax.y + frustum[p][2] * bmax.z + frustum[p][3] > 0 ) continue;
		if( frustum[p][0] * bmin.x + frustum[p][1] * bmax.y + frustum[p][2] * bmax.z + frustum[p][3] > 0 ) continue;
		return false;
	}
	return true;*/
}

void Camera3D::setOrbit  ( Vector3DF angs, Vector3DF tp, float dist, float dolly )
{
	setOrbit ( angs.x, angs.y, angs.z, tp, dist, dolly );
}

void Camera3D::setOrbit ( float ax, float ay, float az, Vector3DF tp, float dist, float dolly )
{
	ang_euler.Set ( ax, ay, az );
	mOrbitDist = dist;
	mDolly = dolly;
	float dx, dy, dz;
	dx = cos ( ang_euler.y * DEGtoRAD ) * sin ( ang_euler.x * DEGtoRAD ) ;
	dy = sin ( ang_euler.y * DEGtoRAD );
	dz = cos ( ang_euler.y * DEGtoRAD ) * cos ( ang_euler.x * DEGtoRAD );
	from_pos.x = tp.x + dx * mOrbitDist;
	from_pos.y = tp.y + dy * mOrbitDist;
	from_pos.z = tp.z + dz * mOrbitDist;
	to_pos.x = from_pos.x - dx * mDolly;
	to_pos.y = from_pos.y - dy * mDolly;
	to_pos.z = from_pos.z - dz * mDolly;
	updateMatricies ();
}

void Camera3D::moveOrbit ( float ax, float ay, float az, float dd )
{
	ang_euler += Vector3DF(ax,ay,az);
	mOrbitDist += dd;
	
	float dx, dy, dz;
	dx = cos ( ang_euler.y * DEGtoRAD ) * sin ( ang_euler.x * DEGtoRAD ) ;
	dy = sin ( ang_euler.y * DEGtoRAD );
	dz = cos ( ang_euler.y * DEGtoRAD ) * cos ( ang_euler.x * DEGtoRAD );
	from_pos.x = to_pos.x + dx * mOrbitDist;
	from_pos.y = to_pos.y + dy * mOrbitDist;
	from_pos.z = to_pos.z + dz * mOrbitDist;
	updateMatricies ();
}

void Camera3D::moveToPos ( float tx, float ty, float tz )
{
	to_pos += Vector3DF(tx,ty,tz);

	float dx, dy, dz;
	dx = cos ( ang_euler.y * DEGtoRAD ) * sin ( ang_euler.x * DEGtoRAD ) ;
	dy = sin ( ang_euler.y * DEGtoRAD );
	dz = cos ( ang_euler.y * DEGtoRAD ) * cos ( ang_euler.x * DEGtoRAD );
	from_pos.x = to_pos.x + dx * mOrbitDist;
	from_pos.y = to_pos.y + dy * mOrbitDist;
	from_pos.z = to_pos.z + dz * mOrbitDist;
	updateMatricies ();
}

void Camera3D::setAngles ( float ax, float ay, float az )
{
	ang_euler = Vector3DF(ax,ay,az);
	to_pos.x = from_pos.x - cos ( ang_euler.y * DEGtoRAD ) * sin ( ang_euler.x * DEGtoRAD ) * mDolly;
	to_pos.y = from_pos.y - sin ( ang_euler.y * DEGtoRAD ) * mDolly;
	to_pos.z = from_pos.z - cos ( ang_euler.y * DEGtoRAD ) * cos ( ang_euler.x * DEGtoRAD ) * mDolly;
	updateMatricies ();
}


void Camera3D::moveRelative ( float dx, float dy, float dz )
{
	Vector3DF vec ( dx, dy, dz );
	vec *= invrot_matrix;
	to_pos += vec;
	from_pos += vec;
	updateMatricies ();
}

void Camera3D::setProjection (eProjection proj_type)
{
	mProjType = proj_type;
}

void Camera3D::updateMatricies ()
{
	Matrix4F basis;
	Vector3DF temp;	
	
	// compute camera direction vectors	--- MATCHES OpenGL's gluLookAt function (DO NOT MODIFY)
	dir_vec = to_pos;					// f vector in gluLookAt docs						
	dir_vec -= from_pos;				// eye = from_pos in gluLookAt docs
	dir_vec.Normalize ();
	side_vec = dir_vec;
	side_vec.Cross ( up_dir );
	side_vec.Normalize ();
	up_vec = side_vec;
	up_vec.Cross ( dir_vec );
	up_vec.Normalize();
	dir_vec *= -1;
	
	// construct view matrix
	rotate_matrix.Basis (side_vec, up_vec, dir_vec );
	view_matrix = rotate_matrix;
	//Matrix4F trans;
	//trans.Translate ( -from_pos.x, -from_pos.y, -from_pos.z );		// !efficiency
	view_matrix.PreTranslate ( Vector3DF(-from_pos.x, -from_pos.y, -from_pos.z ) );

	// construct projection matrix  --- MATCHES OpenGL's gluPerspective function (DO NOT MODIFY)
	float sx = tan ( mFov * 0.5 * DEGtoRAD ) * mNear;
	float sy = sx / mAspect;
	proj_matrix = 0.0;
	proj_matrix(0,0) = 2.0*mNear / sx;				// matches OpenGL definition
	proj_matrix(1,1) = 2.0*mNear / sy;
	proj_matrix(2,2) = -(mFar + mNear)/(mFar - mNear);			// C
	proj_matrix(2,3) = -(2.0*mFar * mNear)/(mFar - mNear);		// D
	proj_matrix(3,2) = -1.0;

	// construct tile projection matrix --- MATCHES OpenGL's glFrustum function (DO NOT MODIFY) 
	float l, r, t, b;
	l = -sx + 2.0*sx*mTile.x;						// Tile is in range 0 <= x,y <= 1
	r = -sx + 2.0*sx*mTile.z;
	t =  sy - 2.0*sy*mTile.y;
	b =  sy - 2.0*sy*mTile.w;
	tileproj_matrix = 0.0;
	tileproj_matrix(0,0) = 2.0*mNear / (r - l);
	tileproj_matrix(1,1) = 2.0*mNear / (t - b);
	tileproj_matrix(0,2) = (r + l) / (r - l);		// A
	tileproj_matrix(1,2) = (t + b) / (t - b);		// B
	tileproj_matrix(2,2) = proj_matrix(2,2);		// C
	tileproj_matrix(2,3) = proj_matrix(2,3);		// D
	tileproj_matrix(3,2) = -1.0;

	// construct inverse rotate and inverse projection matrix
    Vector3DF tvz(0, 0, 0);
	//invrot_matrix.InverseView ( rotate_matrix.GetDataF(), Vector3DF(0,0,0) );		// Computed using rule: "Inverse of a basis rotation matrix is its transpose." (So long as translation is taken out)
	invrot_matrix.InverseView ( rotate_matrix.GetDataF(), tvz );		// Computed using rule: "Inverse of a basis rotation matrix is its transpose." (So long as translation is taken out)
	invproj_matrix.InverseProj ( tileproj_matrix.GetDataF() );							// Computed using rule: 

	updateFrustum ();
}

void Camera3D::updateFrustum ()
{
	Matrix4F mv;
	mv = tileproj_matrix;					// Compute the model-view-projection matrix
	mv *= view_matrix;
	float* mvm = mv.GetDataF();
	float t;

	// Right plane
   frustum[0][0] = mvm[ 3] - mvm[ 0];
   frustum[0][1] = mvm[ 7] - mvm[ 4];
   frustum[0][2] = mvm[11] - mvm[ 8];
   frustum[0][3] = mvm[15] - mvm[12];
   t = sqrt( frustum[0][0] * frustum[0][0] + frustum[0][1] * frustum[0][1] + frustum[0][2] * frustum[0][2] );
   frustum[0][0] /= t; frustum[0][1] /= t; frustum[0][2] /= t; frustum[0][3] /= t;
	// Left plane
   frustum[1][0] = mvm[ 3] + mvm[ 0];
   frustum[1][1] = mvm[ 7] + mvm[ 4];
   frustum[1][2] = mvm[11] + mvm[ 8];
   frustum[1][3] = mvm[15] + mvm[12];
   t = sqrt( frustum[1][0] * frustum[1][0] + frustum[1][1] * frustum[1][1] + frustum[1][2]    * frustum[1][2] );
   frustum[1][0] /= t; frustum[1][1] /= t; frustum[1][2] /= t; frustum[1][3] /= t;
	// Bottom plane
   frustum[2][0] = mvm[ 3] + mvm[ 1];
   frustum[2][1] = mvm[ 7] + mvm[ 5];
   frustum[2][2] = mvm[11] + mvm[ 9];
   frustum[2][3] = mvm[15] + mvm[13];
   t = sqrt( frustum[2][0] * frustum[2][0] + frustum[2][1] * frustum[2][1] + frustum[2][2]    * frustum[2][2] );
   frustum[2][0] /= t; frustum[2][1] /= t; frustum[2][2] /= t; frustum[2][3] /= t;
	// Top plane
   frustum[3][0] = mvm[ 3] - mvm[ 1];
   frustum[3][1] = mvm[ 7] - mvm[ 5];
   frustum[3][2] = mvm[11] - mvm[ 9];
   frustum[3][3] = mvm[15] - mvm[13];
   t = sqrt( frustum[3][0] * frustum[3][0] + frustum[3][1] * frustum[3][1] + frustum[3][2]    * frustum[3][2] );
   frustum[3][0] /= t; frustum[3][1] /= t; frustum[3][2] /= t; frustum[3][3] /= t;
	// Far plane
   frustum[4][0] = mvm[ 3] - mvm[ 2];
   frustum[4][1] = mvm[ 7] - mvm[ 6];
   frustum[4][2] = mvm[11] - mvm[10];
   frustum[4][3] = mvm[15] - mvm[14];
   t = sqrt( frustum[4][0] * frustum[4][0] + frustum[4][1] * frustum[4][1] + frustum[4][2]    * frustum[4][2] );
   frustum[4][0] /= t; frustum[4][1] /= t; frustum[4][2] /= t; frustum[4][3] /= t;
	// Near plane
   frustum[5][0] = mvm[ 3] + mvm[ 2];
   frustum[5][1] = mvm[ 7] + mvm[ 6];
   frustum[5][2] = mvm[11] + mvm[10];
   frustum[5][3] = mvm[15] + mvm[14];
   t = sqrt( frustum[5][0] * frustum[5][0] + frustum[5][1] * frustum[5][1] + frustum[5][2]    * frustum[5][2] );
   frustum[5][0] /= t; frustum[5][1] /= t; frustum[5][2] /= t; frustum[5][3] /= t;
}

float Camera3D::calculateLOD ( Vector3DF pnt, float minlod, float maxlod, float maxdist )
{
	Vector3DF vec = pnt;
	vec -= from_pos;
	float lod = minlod + (vec.Length() * (maxlod-minlod) / maxdist );	
	lod = (lod < minlod) ? minlod : lod;
	lod = (lod > maxlod) ? maxlod : lod;
	return lod;
}

void Camera3D::setModelMatrix ()
{
	glGetFloatv ( GL_MODELVIEW_MATRIX, model_matrix.GetDataF() );
}

void Camera3D::setModelMatrix ( Matrix4F& model )
{
	model_matrix = model;
	mv_matrix = model;
	mv_matrix *= view_matrix;
	glLoadMatrixf ( mv_matrix.GetDataF() );
}

Vector3DF Camera3D::inverseRay (float x, float y, float z)
{	
	float sx = tan ( mFov * 0.5 * DEGtoRAD);
	float sy = sx / mAspect;
	float tu, tv;
	tu = mTile.x + x * (mTile.z-mTile.x);
	tv = mTile.y + y * (mTile.w-mTile.y);
	Vector4DF pnt ( (tu*2.0-1.0) * z*sx, (1.0-tv*2.0) * z*sy, -z, 1 );
	pnt *= invrot_matrix;
	return pnt;
}

Vector4DF Camera3D::project ( Vector3DF& p, Matrix4F& vm )
{
	Vector4DF q = p;								// World coordinates
	
	q *= vm;										// Eye coordinates
	
	q *= proj_matrix;								// Projection 

	q /= q.w;										// Normalized Device Coordinates (w-divide)
	
	q.x *= 0.5;
	q.y *= -0.5;
	q.z = q.z*0.5 + 0.5;							// Stored depth buffer value
		
	return q;
}

Vector4DF Camera3D::project ( Vector3DF& p )
{
	Vector4DF q = p;								// World coordinates
	q *= view_matrix;								// Eye coordinates

	q *= proj_matrix;								// Clip coordinates
	
	q /= q.w;										// Normalized Device Coordinates (w-divide)

	q.x *= 0.5;
	q.y *= -0.5;
	q.z = q.z*0.5 + 0.5;							// Stored depth buffer value
		
	return q;
}


