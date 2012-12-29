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


#ifndef DEF_CAMERA_3D
	#define	DEF_CAMERA_3D

	#include "matrix.h"
	#include "vector.h"
	#include "pivotx.h"	
	
	#define DEG_TO_RAD			(3.141592/180.0)

	class  Camera3D : public PivotX {
	public:
		enum eProjection {
			Perspective = 0,
			Parallel = 1
		};
		Camera3D ();

		void draw_gl();

		// Camera settings
		void setAspect ( float asp )					{ mAspect = asp;			updateMatricies(); }
		void setPos ( float x, float y, float z )		{ from_pos.Set(x,y,z);		updateMatricies(); }
		void setToPos ( float x, float y, float z )		{ to_pos.Set(x,y,z);		updateMatricies(); }
		void setFov (float fov)							{ mFov = fov;				updateMatricies(); }
		void setNearFar (float n, float f )				{ mNear = n; mFar = f;		updateMatricies(); }
		void setTile ( float x1, float y1, float x2, float y2 )		{ mTile.Set ( x1, y1, x2, y2 );		updateMatricies(); }
		void setProjection (eProjection proj_type);
		void setModelMatrix ();
		void setModelMatrix ( Matrix4F& model );
		
		// Camera motion
		void setOrbit  ( float ax, float ay, float az, Vector3DF tp, float dist, float dolly );
		void setOrbit  ( Vector3DF angs, Vector3DF tp, float dist, float dolly );
		void setAngles ( float ax, float ay, float az );
		void moveOrbit ( float ax, float ay, float az, float dist );		
		void moveToPos ( float tx, float ty, float tz );		
		void moveRelative ( float dx, float dy, float dz );

		// Frustum testing
		bool pointInFrustum ( float x, float y, float z );
		bool boxInFrustum ( Vector3DF bmin, Vector3DF bmax);
		float calculateLOD ( Vector3DF pnt, float minlod, float maxlod, float maxdist );

		// Utility functions
		void updateMatricies ();					// Updates camera axes and projection matricies
		void updateFrustum ();						// Updates frustum planes
		Vector3DF inverseRay ( float x, float y, float z );
		Vector4DF project ( Vector3DF& p );
		Vector4DF project ( Vector3DF& p, Matrix4F& vm );		// Project point - override view matrix

		void getVectors ( Vector3DF& dir, Vector3DF& up, Vector3DF& side )	{ dir = dir_vec; up = up_vec; side = side_vec; }
		void getBounds ( float dst, Vector3DF& min, Vector3DF& max );
		float getNear ()				{ return mNear; }
		float getFar ()					{ return mFar; }
		float getFov ()					{ return mFov; }
		float getDolly()				{ return mDolly; }	
		float getOrbitDist()			{ return mOrbitDist; }
		Vector3DF& getUpDir ()			{ return up_dir; }
		Vector4DF& getTile ()			{ return mTile; }
		Matrix4F& getViewMatrix ()		{ return view_matrix; }
		Matrix4F& getInvView ()			{ return invrot_matrix; }
		Matrix4F& getProjMatrix ()		{ return tileproj_matrix; }	
		Matrix4F& getFullProjMatrix ()	{ return proj_matrix; }
		Matrix4F& getModelMatrix()		{ return model_matrix; }
		Matrix4F& getMVMatrix()			{ return mv_matrix; }
		float getAspect ()				{ return mAspect; }

	public:
		eProjection		mProjType;								// Projection type

		// Camera Parameters									// NOTE: Pivot maintains camera from and orientation
		float			mDolly;									// Camera to distance
		float			mOrbitDist;
		float			mFov, mAspect;							// Camera field-of-view
		float			mNear, mFar;							// Camera frustum planes
		Vector3DF		dir_vec, side_vec, up_vec;				// Camera aux vectors (N, V, and U)
		Vector3DF		up_dir;
		Vector4DF		mTile;
		
		// Transform Matricies
		Matrix4F		rotate_matrix;							// Vr matrix (rotation only)
		Matrix4F		view_matrix;							// V matrix	(rotation + translation)
		Matrix4F		proj_matrix;							// P matrix
		Matrix4F		invrot_matrix;							// Vr^-1 matrix
		Matrix4F		invproj_matrix;
		Matrix4F		tileproj_matrix;						// tiled projection matrix
		Matrix4F		model_matrix;
		Matrix4F		mv_matrix;
		float			frustum[6][4];							// frustum plane equations

		bool			mOps[8];
		int				mWire;
				
	};

#endif


