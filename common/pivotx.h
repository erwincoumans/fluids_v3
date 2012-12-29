
#ifndef DEF_PIVOT_H
	#define DEF_PIVOT_H

	#include <string>

	#include "vector.h"
	#include "matrix.h"

	class PivotX {
	public:
		PivotX()	{ from_pos.Set(0,0,0); to_pos.Set(0,0,0); ang_euler.Set(0,0,0); scale.Set(1,1,1); trans.Identity(); }
		PivotX( Vector3DF& f, Vector3DF& t, Vector3DF& s, Vector3DF& a) { from_pos=f; to_pos=t; scale=s; ang_euler=a; }

		void setPivot ( float x, float y, float z, float rx, float ry, float rz );
		void setPivot ( Vector3DF& pos, Vector3DF& ang ) { from_pos = pos; ang_euler = ang; }
		void setPivot ( PivotX  piv )	{ from_pos = piv.from_pos; to_pos = piv.to_pos; ang_euler = piv.ang_euler; updateTform(); }		
		void setPivot ( PivotX& piv )	{ from_pos = piv.from_pos; to_pos = piv.to_pos; ang_euler = piv.ang_euler; updateTform(); }

		void setIdentity ()		{ from_pos.Set(0,0,0); to_pos.Set(0,0,0); ang_euler.Set(0,0,0); scale.Set(1,1,1); trans.Identity(); }

		void setAng ( float rx, float ry, float rz )	{ ang_euler.Set(rx,ry,rz);	updateTform(); }
		void setAng ( Vector3DF& a )					{ ang_euler = a;			updateTform(); }

		void setPos ( float x, float y, float z )		{ from_pos.Set(x,y,z);		updateTform(); }
		void setPos ( Vector3DF& p )					{ from_pos = p;				updateTform(); }

		void setToPos ( float x, float y, float z )		{ to_pos.Set(x,y,z);		updateTform(); }
		
		void updateTform ();
		void setTform ( Matrix4F& t )		{ trans = t; }
		inline Matrix4F& getTform ()		{ return trans; }
		inline float* getTformData ()		{ return trans.GetDataF(); }

		// Pivot		
		PivotX getPivot ()	{ return PivotX(from_pos, to_pos, scale, ang_euler); }
		Vector3DF& getPos ()			{ return from_pos; }
		Vector3DF& getToPos ()			{ return to_pos; }
		Vector3DF& getAng ()			{ return ang_euler; }
		Vector3DF getDir ()			{ 
			return to_pos - from_pos; 
		}

		Vector3DF	from_pos;
		Vector3DF	to_pos;
		Vector3DF	scale;
		Vector3DF	ang_euler;
		Matrix4F	trans;
		
		//Quatern	ang_quat;
		//Quatern	dang_quat;
	};

#endif



