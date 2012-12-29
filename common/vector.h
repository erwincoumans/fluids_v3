
// ** NOTES **
// Vector code CANNOT be inlined in header file because of dependencies
//    across vector classes (error generated: "Use of undeclared class..")
// 

#ifndef VECTOR_DEF
	#define VECTOR_DEF

	#include "common_defs.h"


	#include <stdlib.h>
	//#define VECTOR_INITIALIZE				// Initializes vectors	
																
	class Vector2DC;						// Forward Referencing
	class Vector2DI;
	class Vector2DF;
	class Vector3DC;
	class Vector3DI;
	class Vector3DF;
	class Vector4DC;
	class Vector4DF;	
	class MatrixF;
	class Matrix4F;

	// Vector2DC Declaration
	
	#define VNAME		2DC
	#define VTYPE		unsigned char
	
	#define LUNA_CORE

	class LUNA_CORE Vector2DC {
	public:
		VTYPE x, y;

		// Constructors/Destructors
		inline Vector2DC();
		inline ~Vector2DC();
		inline Vector2DC (VTYPE xa, VTYPE ya);
		inline Vector2DC (Vector2DC &op);	
		inline Vector2DC (Vector2DI &op);	
		inline Vector2DC (Vector2DF &op);	
		inline Vector2DC (Vector3DC &op);	
		inline Vector2DC (Vector3DI &op);	
		inline Vector2DC (Vector3DF &op);	
		inline Vector2DC (Vector4DF &op);

		inline Vector2DC &Set (VTYPE xa, VTYPE ya)	{x=xa; y=ya; return *this;}

		// Member Functions
		inline Vector2DC &operator= (Vector2DC &op);
		inline Vector2DC &operator= (Vector2DI &op);
		inline Vector2DC &operator= (Vector2DF &op);
		inline Vector2DC &operator= (Vector3DC &op);
		inline Vector2DC &operator= (Vector3DI &op);
		inline Vector2DC &operator= (Vector3DF &op);
		inline Vector2DC &operator= (Vector4DF &op);
		
		inline Vector2DC &operator+= (Vector2DC &op);
		inline Vector2DC &operator+= (Vector2DI &op);
		inline Vector2DC &operator+= (Vector2DF &op);
		inline Vector2DC &operator+= (Vector3DC &op);
		inline Vector2DC &operator+= (Vector3DI &op);
		inline Vector2DC &operator+= (Vector3DF &op);
		inline Vector2DC &operator+= (Vector4DF &op);

		inline Vector2DC &operator-= (Vector2DC &op);
		inline Vector2DC &operator-= (Vector2DI &op);
		inline Vector2DC &operator-= (Vector2DF &op);
		inline Vector2DC &operator-= (Vector3DC &op);
		inline Vector2DC &operator-= (Vector3DI &op);
		inline Vector2DC &operator-= (Vector3DF &op);
		inline Vector2DC &operator-= (Vector4DF &op);
	
		inline Vector2DC &operator*= (Vector2DC &op);
		inline Vector2DC &operator*= (Vector2DI &op);
		inline Vector2DC &operator*= (Vector2DF &op);
		inline Vector2DC &operator*= (Vector3DC &op);
		inline Vector2DC &operator*= (Vector3DI &op);
		inline Vector2DC &operator*= (Vector3DF &op);
		inline Vector2DC &operator*= (Vector4DF &op);

		inline Vector2DC &operator/= (Vector2DC &op);
		inline Vector2DC &operator/= (Vector2DI &op);
		inline Vector2DC &operator/= (Vector2DF &op);
		inline Vector2DC &operator/= (Vector3DC &op);
		inline Vector2DC &operator/= (Vector3DI &op);
		inline Vector2DC &operator/= (Vector3DF &op);
		inline Vector2DC &operator/= (Vector4DF &op);

		// Note: Cross product does not exist for 2D vectors (only 3D)
		
		inline double Dot(Vector2DC &v);
		inline double Dot(Vector2DI &v);
		inline double Dot(Vector2DF &v);

		inline double Dist (Vector2DC &v);
		inline double Dist (Vector2DI &v);
		inline double Dist (Vector2DF &v);
		inline double Dist (Vector3DC &v);
		inline double Dist (Vector3DI &v);
		inline double Dist (Vector3DF &v);
		inline double Dist (Vector4DF &v);

		inline double DistSq (Vector2DC &v);		
		inline double DistSq (Vector2DI &v);		
		inline double DistSq (Vector2DF &v);		
		inline double DistSq (Vector3DC &v);		
		inline double DistSq (Vector3DI &v);		
		inline double DistSq (Vector3DF &v);		
		inline double DistSq (Vector4DF &v);

		inline Vector2DC &Normalize (void);
		inline double Length (void);
		inline VTYPE *Data (void);
	};
	
	#undef VNAME
	#undef VTYPE

	// Vector2DI Declaration

	#define VNAME		2DI
	#define VTYPE		int

	class LUNA_CORE Vector2DI {
	public:
		VTYPE x, y;

		// Constructors/Destructors
		inline Vector2DI();							
		inline ~Vector2DI();			
		inline Vector2DI (const VTYPE xa, const VTYPE ya);
		inline Vector2DI (const Vector2DC &op);				
		inline Vector2DI (const Vector2DI &op);				// *** THESE SHOULD ALL BE const
		inline Vector2DI (const Vector2DF &op);				
		inline Vector2DI (const Vector3DC &op);				
		inline Vector2DI (const Vector3DI &op);				
		inline Vector2DI (const Vector3DF &op);				
		inline Vector2DI (const Vector4DF &op);

		// Member Functions
		inline Vector2DI &operator= (const Vector2DC &op);
		inline Vector2DI &operator= (const Vector2DI &op);
		inline Vector2DI &operator= (const Vector2DF &op);
		inline Vector2DI &operator= (const Vector3DC &op);
		inline Vector2DI &operator= (const Vector3DI &op);
		inline Vector2DI &operator= (const Vector3DF &op);
		inline Vector2DI &operator= (const Vector4DF &op);

		inline Vector2DI &operator+= (const Vector2DC &op);
		inline Vector2DI &operator+= (const Vector2DI &op);
		inline Vector2DI &operator+= (const Vector2DF &op);
		inline Vector2DI &operator+= (const Vector3DC &op);
		inline Vector2DI &operator+= (const Vector3DI &op);
		inline Vector2DI &operator+= (const Vector3DF &op);
		inline Vector2DI &operator+= (const Vector4DF &op);

		inline Vector2DI &operator-= (const Vector2DC &op);
		inline Vector2DI &operator-= (const Vector2DI &op);
		inline Vector2DI &operator-= (const Vector2DF &op);
		inline Vector2DI &operator-= (const Vector3DC &op);
		inline Vector2DI &operator-= (const Vector3DI &op);
		inline Vector2DI &operator-= (const Vector3DF &op);
		inline Vector2DI &operator-= (const Vector4DF &op);
	
		inline Vector2DI &operator*= (const Vector2DC &op);
		inline Vector2DI &operator*= (const Vector2DI &op);
		inline Vector2DI &operator*= (const Vector2DF &op);
		inline Vector2DI &operator*= (const Vector3DC &op);
		inline Vector2DI &operator*= (const Vector3DI &op);
		inline Vector2DI &operator*= (const Vector3DF &op);
		inline Vector2DI &operator*= (const Vector4DF &op);

		inline Vector2DI &operator/= (const Vector2DC &op);
		inline Vector2DI &operator/= (const Vector2DI &op);
		inline Vector2DI &operator/= (const Vector2DF &op);
		inline Vector2DI &operator/= (const Vector3DC &op);
		inline Vector2DI &operator/= (const Vector3DI &op);
		inline Vector2DI &operator/= (const Vector3DF &op);
		inline Vector2DI &operator/= (const Vector4DF &op);


		// Note: Cross product does not exist for 2D vectors (only 3D)
		
		inline double Dot (const Vector2DC &v);
		inline double Dot (const Vector2DI &v);
		inline double Dot (const Vector2DF &v);

		inline double Dist (const Vector2DC &v);
		inline double Dist (const Vector2DI &v);
		inline double Dist (const Vector2DF &v);
		inline double Dist (const Vector3DC &v);
		inline double Dist (const Vector3DI &v);
		inline double Dist (const Vector3DF &v);
		inline double Dist (const Vector4DF &v);

		inline double DistSq (const Vector2DC &v);
		inline double DistSq (const Vector2DI &v);
		inline double DistSq (const Vector2DF &v);
		inline double DistSq (const Vector3DC &v);
		inline double DistSq (const Vector3DI &v);
		inline double DistSq (const Vector3DF &v);
		inline double DistSq (const Vector4DF &v);
		
		inline Vector2DI &Normalize (void);
		inline double Length (void);
		inline VTYPE *Data (void);
	};
	
	#undef VNAME
	#undef VTYPE

	// Vector2DF Declarations

	#define VNAME		2DF
	#define VTYPE		float

	class LUNA_CORE Vector2DF {
	public:
		VTYPE x, y;

		// Constructors/Destructors
		 Vector2DF ();
		 ~Vector2DF ();
		 Vector2DF (const VTYPE xa, const VTYPE ya);
		 Vector2DF (const Vector2DC &op);
		 Vector2DF (const Vector2DI &op);
		 Vector2DF (const Vector2DF &op);
		 Vector2DF (const Vector3DC &op);
		 Vector2DF (const Vector3DI &op);
		 Vector2DF (const Vector3DF &op);
		 Vector2DF (const Vector4DF &op);

		 inline Vector2DF &Set (const float xa, const float ya);
		 
		 // Member Functions
		 Vector2DF &operator= (const Vector2DC &op);
		 Vector2DF &operator= (const Vector2DI &op);
		 Vector2DF &operator= (const Vector2DF &op);
		 Vector2DF &operator= (const Vector3DC &op);
		 Vector2DF &operator= (const Vector3DI &op);
		 Vector2DF &operator= (const Vector3DF &op);
		 Vector2DF &operator= (const Vector4DF &op);
		
		 Vector2DF &operator+= (const Vector2DC &op);
		 Vector2DF &operator+= (const Vector2DI &op);
		 Vector2DF &operator+= (const Vector2DF &op);
		 Vector2DF &operator+= (const Vector3DC &op);
		 Vector2DF &operator+= (const Vector3DI &op);
		 Vector2DF &operator+= (const Vector3DF &op);
		 Vector2DF &operator+= (const Vector4DF &op);

		 Vector2DF &operator-= (const Vector2DC &op);
		 Vector2DF &operator-= (const Vector2DI &op);
		 Vector2DF &operator-= (const Vector2DF &op);
		 Vector2DF &operator-= (const Vector3DC &op);
		 Vector2DF &operator-= (const Vector3DI &op);
		 Vector2DF &operator-= (const Vector3DF &op);
		 Vector2DF &operator-= (const Vector4DF &op);

		 Vector2DF &operator*= (const Vector2DC &op);
		 Vector2DF &operator*= (const Vector2DI &op);
		 Vector2DF &operator*= (const Vector2DF &op);
		 Vector2DF &operator*= (const Vector3DC &op);
		 Vector2DF &operator*= (const Vector3DI &op);
		 Vector2DF &operator*= (const Vector3DF &op);
		 Vector2DF &operator*= (const Vector4DF &op);

		 Vector2DF &operator/= (const Vector2DC &op);
		 Vector2DF &operator/= (const Vector2DI &op);
		 Vector2DF &operator/= (const Vector2DF &op);
		 Vector2DF &operator/= (const Vector3DC &op);
		 Vector2DF &operator/= (const Vector3DI &op);
		 Vector2DF &operator/= (const Vector3DF &op);
		 Vector2DF &operator/= (const Vector4DF &op);

		 Vector2DF &operator/= (const double v)		{x /= (float) v; y /= (float) v; return *this;}

		// Note: Cross product does not exist for 2D vectors (only 3D)
		
		 double Dot(const Vector2DC &v);
		 double Dot(const Vector2DI &v);
		 double Dot(const Vector2DF &v);

		 double Dist (const Vector2DC &v);
		 double Dist (const Vector2DI &v);
		 double Dist (const Vector2DF &v);
		 double Dist (const Vector3DC &v);
		 double Dist (const Vector3DI &v);
		 double Dist (const Vector3DF &v);
		 double Dist (const Vector4DF &v);

		 double DistSq (const Vector2DC &v);
		 double DistSq (const Vector2DI &v);
		 double DistSq (const Vector2DF &v);
		 double DistSq (const Vector3DC &v);
		 double DistSq (const Vector3DI &v);
		 double DistSq (const Vector3DF &v);
		 double DistSq (const Vector4DF &v);

		 Vector2DF &Normalize (void);
		 double Length (void);
		 VTYPE *Data (void);
	};
	
	#undef VNAME
	#undef VTYPE

	// Vector3DC Declaration
	
	#define VNAME		3DC
	#define VTYPE		unsigned char

	class LUNA_CORE Vector3DC {
	public:	
		VTYPE x, y, z;
	
		// Constructors/Destructors
		inline Vector3DC();
		inline ~Vector3DC();
		inline Vector3DC (const VTYPE xa, const VTYPE ya, const VTYPE za);
		inline Vector3DC  ( const Vector2DC &op);
		inline Vector3DC  ( const Vector2DI &op);
		inline Vector3DC  ( const Vector2DF &op);
		inline Vector3DC  ( const Vector3DC &op);
		inline Vector3DC  ( const Vector3DI &op);
		inline Vector3DC  ( const Vector3DF &op);
		inline Vector3DC  ( const Vector4DF &op);

		// Member Functions
		inline Vector3DC &Set (VTYPE xa, VTYPE ya, VTYPE za);
		
		inline Vector3DC &operator=  ( const Vector2DC &op);
		inline Vector3DC &operator=  ( const Vector2DI &op);
		inline Vector3DC &operator=  ( const Vector2DF &op);
		inline Vector3DC &operator=  ( const Vector3DC &op);
		inline Vector3DC &operator=  ( const Vector3DI &op);
		inline Vector3DC &operator=  ( const Vector3DF &op);
		inline Vector3DC &operator=  ( const Vector4DF &op);
		
		inline Vector3DC &operator+=  ( const Vector2DC &op);
		inline Vector3DC &operator+=  ( const Vector2DI &op);
		inline Vector3DC &operator+=  ( const Vector2DF &op);
		inline Vector3DC &operator+=  ( const Vector3DC &op);
		inline Vector3DC &operator+=  ( const Vector3DI &op);
		inline Vector3DC &operator+=  ( const Vector3DF &op);
		inline Vector3DC &operator+=  ( const Vector4DF &op);

		inline Vector3DC &operator-=  ( const Vector2DC &op);
		inline Vector3DC &operator-=  ( const Vector2DI &op);
		inline Vector3DC &operator-=  ( const Vector2DF &op);
		inline Vector3DC &operator-=  ( const Vector3DC &op);
		inline Vector3DC &operator-=  ( const Vector3DI &op);
		inline Vector3DC &operator-=  ( const Vector3DF &op);
		inline Vector3DC &operator-=  ( const Vector4DF &op);
	
		inline Vector3DC &operator*=  ( const Vector2DC &op);
		inline Vector3DC &operator*=  ( const Vector2DI &op);
		inline Vector3DC &operator*=  ( const Vector2DF &op);
		inline Vector3DC &operator*=  ( const Vector3DC &op);
		inline Vector3DC &operator*=  ( const Vector3DI &op);
		inline Vector3DC &operator*=  ( const Vector3DF &op);
		inline Vector3DC &operator*=  ( const Vector4DF &op);

		inline Vector3DC &operator/=  ( const Vector2DC &op);
		inline Vector3DC &operator/=  ( const Vector2DI &op);
		inline Vector3DC &operator/=  ( const Vector2DF &op);
		inline Vector3DC &operator/=  ( const Vector3DC &op);
		inline Vector3DC &operator/=  ( const Vector3DI &op);
		inline Vector3DC &operator/=  ( const Vector3DF &op);
		inline Vector3DC &operator/=  ( const Vector4DF &op);

		inline Vector3DC &Cross  ( const Vector3DC &v);
		inline Vector3DC &Cross  ( const Vector3DI &v);
		inline Vector3DC &Cross  ( const Vector3DF &v);	
		
		inline double Dot ( const Vector3DC &v);
		inline double Dot ( const Vector3DI &v);
		inline double Dot ( const Vector3DF &v);

		inline double Dist  ( const Vector2DC &v);
		inline double Dist  ( const Vector2DI &v);
		inline double Dist  ( const Vector2DF &v);
		inline double Dist  ( const Vector3DC &v);
		inline double Dist  ( const Vector3DI &v);
		inline double Dist  ( const Vector3DF &v);
		inline double Dist  ( const Vector4DF &v);

		inline double DistSq  ( const Vector2DC &v);
		inline double DistSq  ( const Vector2DI &v);
		inline double DistSq  ( const Vector2DF &v);
		inline double DistSq  ( const Vector3DC &v);
		inline double DistSq  ( const Vector3DI &v);
		inline double DistSq  ( const Vector3DF &v);
		inline double DistSq  ( const Vector4DF &v);

		inline Vector3DC &Normalize (void);
		inline double Length (void);
		inline VTYPE *Data (void);
	};
	
	#undef VNAME
	#undef VTYPE

	// Vector3DI Declaration

	#define VNAME		3DI
	#define VTYPE		int

	class LUNA_CORE Vector3DI {
	public:
		VTYPE x, y, z;
	
		// Constructors/Destructors
		inline Vector3DI();
		inline ~Vector3DI();
		inline Vector3DI (const VTYPE xa, const VTYPE ya, const VTYPE za);
		inline Vector3DI (const Vector2DC &op);
		inline Vector3DI (const Vector2DI &op);
		inline Vector3DI (const Vector2DF &op);
		inline Vector3DI (const Vector3DC &op);
		inline Vector3DI (const Vector3DI &op);
		inline Vector3DI (const Vector3DF &op);
		inline Vector3DI (const Vector4DF &op);

		// Set Functions
		inline Vector3DI &Set (const int xa, const int ya, const int za);

		// Member Functions
		inline Vector3DI &operator= (const Vector2DC &op);
		inline Vector3DI &operator= (const Vector2DI &op);
		inline Vector3DI &operator= (const Vector2DF &op);
		inline Vector3DI &operator= (const Vector3DC &op);
		inline Vector3DI &operator= (const Vector3DI &op);
		inline Vector3DI &operator= (const Vector3DF &op);
		inline Vector3DI &operator= (const Vector4DF &op);
		
		inline Vector3DI &operator+= (const Vector2DC &op);
		inline Vector3DI &operator+= (const Vector2DI &op);
		inline Vector3DI &operator+= (const Vector2DF &op);
		inline Vector3DI &operator+= (const Vector3DC &op);
		inline Vector3DI &operator+= (const Vector3DI &op);
		inline Vector3DI &operator+= (const Vector3DF &op);
		inline Vector3DI &operator+= (const Vector4DF &op);

		inline Vector3DI &operator-= (const Vector2DC &op);
		inline Vector3DI &operator-= (const Vector2DI &op);
		inline Vector3DI &operator-= (const Vector2DF &op);
		inline Vector3DI &operator-= (const Vector3DC &op);
		inline Vector3DI &operator-= (const Vector3DI &op);
		inline Vector3DI &operator-= (const Vector3DF &op);
		inline Vector3DI &operator-= (const Vector4DF &op);
	
		inline Vector3DI &operator*= (const Vector2DC &op);
		inline Vector3DI &operator*= (const Vector2DI &op);
		inline Vector3DI &operator*= (const Vector2DF &op);
		inline Vector3DI &operator*= (const Vector3DC &op);
		inline Vector3DI &operator*= (const Vector3DI &op);
		inline Vector3DI &operator*= (const Vector3DF &op);
		inline Vector3DI &operator*= (const Vector4DF &op);

		inline Vector3DI &operator/= (const Vector2DC &op);
		inline Vector3DI &operator/= (const Vector2DI &op);
		inline Vector3DI &operator/= (const Vector2DF &op);
		inline Vector3DI &operator/= (const Vector3DC &op);
		inline Vector3DI &operator/= (const Vector3DI &op);
		inline Vector3DI &operator/= (const Vector3DF &op);
		inline Vector3DI &operator/= (const Vector4DF &op);

		inline Vector3DI operator+ (const int op)			{ return Vector3DI(x+(VTYPE) op, y+(VTYPE) op, z+(VTYPE) op); }
		inline Vector3DI operator+ (const float op)		{ return Vector3DI(x+(VTYPE) op, y+(VTYPE) op, z+(VTYPE) op); }
		inline Vector3DI operator+ (const Vector3DI &op)	{ return Vector3DI(x+op.x, y+op.y, z+op.z); }
		inline Vector3DI operator- (const int op)			{ return Vector3DI(x-(VTYPE) op, y-(VTYPE) op, z-(VTYPE) op); }
		inline Vector3DI operator- (const float op)		{ return Vector3DI(x-(VTYPE) op, y-(VTYPE) op, z-(VTYPE) op); }
		inline Vector3DI operator- (const Vector3DI &op)	{ return Vector3DI(x-op.x, y-op.y, z-op.z); }
		inline Vector3DI operator* (const int op)			{ return Vector3DI(x*(VTYPE) op, y*(VTYPE) op, z*(VTYPE) op); }
		inline Vector3DI operator* (const float op)		{ return Vector3DI(x*(VTYPE) op, y*(VTYPE) op, z*(VTYPE) op); }
		inline Vector3DI operator* (const Vector3DI &op)	{ return Vector3DI(x*op.x, y*op.y, z*op.z); }		

		inline Vector3DI &Cross (const Vector3DC &v);
		inline Vector3DI &Cross (const Vector3DI &v);
		inline Vector3DI &Cross (const Vector3DF &v);	
		
		inline double Dot(const Vector3DC &v);
		inline double Dot(const Vector3DI &v);
		inline double Dot(const Vector3DF &v);

		inline double Dist (const Vector2DC &v);
		inline double Dist (const Vector2DI &v);
		inline double Dist (const Vector2DF &v);
		inline double Dist (const Vector3DC &v);
		inline double Dist (const Vector3DI &v);
		inline double Dist (const Vector3DF &v);
		inline double Dist (const Vector4DF &v);

		inline double DistSq (const Vector2DC &v);
		inline double DistSq (const Vector2DI &v);
		inline double DistSq (const Vector2DF &v);
		inline double DistSq (const Vector3DC &v);
		inline double DistSq (const Vector3DI &v);
		inline double DistSq (const Vector3DF &v);
		inline double DistSq (const Vector4DF &v);

		inline Vector3DI &Normalize (void);
		inline double Length (void);
		inline VTYPE *Data (void);
	};
	
	#undef VNAME
	#undef VTYPE

	// Vector3DF Declarations

	#define VNAME		3DF
	#define VTYPE		float

	class LUNA_CORE Vector3DF {
	public:
		VTYPE x, y, z;
	
		// Constructors/Destructors
		inline Vector3DF();
		inline ~Vector3DF();
		inline Vector3DF (const VTYPE xa, const VTYPE ya, const VTYPE za);
		inline Vector3DF (const Vector2DC &op);
		inline Vector3DF (const Vector2DI &op);
		inline Vector3DF (const Vector2DF &op);
		inline Vector3DF (const Vector3DC &op);
		inline Vector3DF (const Vector3DI &op);
		inline Vector3DF (const Vector3DF &op);
		inline Vector3DF (const Vector4DF &op);

		// Set Functions
		inline Vector3DF &Set (const double xa, const double ya, const double za);
		
		// Member Functions
		inline Vector3DF &operator= (const int op);
		inline Vector3DF &operator= (const double op);
		inline Vector3DF &operator= (const Vector2DC &op);
		inline Vector3DF &operator= (const Vector2DI &op);
		inline Vector3DF &operator= (const Vector2DF &op);
		inline Vector3DF &operator= (const Vector3DC &op);
		inline Vector3DF &operator= (const Vector3DI &op);
		inline Vector3DF &operator= (const Vector3DF &op);
		inline Vector3DF &operator= (const Vector4DF &op);

		inline Vector3DF &operator+= (const int op);
		inline Vector3DF &operator+= (const double op);
		inline Vector3DF &operator+= (const Vector2DC &op);
		inline Vector3DF &operator+= (const Vector2DI &op);
		inline Vector3DF &operator+= (const Vector2DF &op);
		inline Vector3DF &operator+= (const Vector3DC &op);
		inline Vector3DF &operator+= (const Vector3DI &op);
		inline Vector3DF &operator+= (const Vector3DF &op);
		inline Vector3DF &operator+= (const Vector4DF &op);

		inline Vector3DF &operator-= (const int op);
		inline Vector3DF &operator-= (const double op);
		inline Vector3DF &operator-= (const Vector2DC &op);
		inline Vector3DF &operator-= (const Vector2DI &op);
		inline Vector3DF &operator-= (const Vector2DF &op);
		inline Vector3DF &operator-= (const Vector3DC &op);
		inline Vector3DF &operator-= (const Vector3DI &op);
		inline Vector3DF &operator-= (const Vector3DF &op);
		inline Vector3DF &operator-= (const Vector4DF &op);
	
		inline Vector3DF &operator*= (const int op);
		inline Vector3DF &operator*= (const double op);
		inline Vector3DF &operator*= (const Vector2DC &op);
		inline Vector3DF &operator*= (const Vector2DI &op);
		inline Vector3DF &operator*= (const Vector2DF &op);
		inline Vector3DF &operator*= (const Vector3DC &op);
		inline Vector3DF &operator*= (const Vector3DI &op);
		inline Vector3DF &operator*= (const Vector3DF &op);
		inline Vector3DF &operator*= (const Vector4DF &op);
		Vector3DF &operator*= (const Matrix4F &op);
		Vector3DF &operator*= (const MatrixF &op);				// see vector.cpp

		inline Vector3DF &operator/= (const int op);
		inline Vector3DF &operator/= (const double op);
		inline Vector3DF &operator/= (const Vector2DC &op);
		inline Vector3DF &operator/= (const Vector2DI &op);
		inline Vector3DF &operator/= (const Vector2DF &op);
		inline Vector3DF &operator/= (const Vector3DC &op);
		inline Vector3DF &operator/= (const Vector3DI &op);
		inline Vector3DF &operator/= (const Vector3DF &op);
		inline Vector3DF &operator/= (const Vector4DF &op);

		// Slow operations - require temporary variables
		inline Vector3DF operator+ (int op)			{ return Vector3DF(x+float(op), y+float(op), z+float(op)); }
		inline Vector3DF operator+ (float op)		{ return Vector3DF(x+op, y+op, z+op); }
		inline Vector3DF operator+ (Vector3DF &op)	{ return Vector3DF(x+op.x, y+op.y, z+op.z); }
		inline Vector3DF operator- (int op)			{ return Vector3DF(x-float(op), y-float(op), z-float(op)); }
		inline Vector3DF operator- (float op)		{ return Vector3DF(x-op, y-op, z-op); }
		inline Vector3DF operator- (Vector3DF &op)	{ return Vector3DF(x-op.x, y-op.y, z-op.z); }
		inline Vector3DF operator* (int op)			{ return Vector3DF(x*float(op), y*float(op), z*float(op)); }
		inline Vector3DF operator* (float op)		{ return Vector3DF(x*op, y*op, z*op); }
		inline Vector3DF operator* (Vector3DF &op)	{ return Vector3DF(x*op.x, y*op.y, z*op.z); }		
		// --


		inline Vector3DF &Cross (const Vector3DC &v);
		inline Vector3DF &Cross (const Vector3DI &v);
		inline Vector3DF &Cross (const Vector3DF &v);	
		
		inline double Dot(const Vector3DC &v);
		inline double Dot(const Vector3DI &v);
		inline double Dot(const Vector3DF &v);

		inline double Dist (const Vector2DC &v);
		inline double Dist (const Vector2DI &v);
		inline double Dist (const Vector2DF &v);
		inline double Dist (const Vector3DC &v);
		inline double Dist (const Vector3DI &v);
		inline double Dist (const Vector3DF &v);
		inline double Dist (const Vector4DF &v);

		inline double DistSq (const Vector2DC &v);
		inline double DistSq (const Vector2DI &v);
		inline double DistSq (const Vector2DF &v);
		inline double DistSq (const Vector3DC &v);
		inline double DistSq (const Vector3DI &v);
		inline double DistSq (const Vector3DF &v);
		inline double DistSq (const Vector4DF &v);

		inline Vector3DF &Random ()		{ x=float(rand())/RAND_MAX; y=float(rand())/RAND_MAX; z=float(rand())/RAND_MAX;  return *this;}
		inline Vector3DF &Random (Vector3DF a, Vector3DF b)		{ x=a.x+float(rand()*(b.x-a.x))/RAND_MAX; y=a.y+float(rand()*(b.y-a.y))/RAND_MAX; z=a.z+float(rand()*(b.z-a.z))/RAND_MAX;  return *this;}
		inline Vector3DF &Random (float x1,float x2, float y1, float y2, float z1, float z2)	{ x=x1+float(rand()*(x2-x1))/RAND_MAX; y=y1+float(rand()*(y2-y1))/RAND_MAX; z=z1+float(rand()*(z2-z1))/RAND_MAX;  return *this;}

		Vector3DF RGBtoHSV ();
		Vector3DF HSVtoRGB ();

		inline Vector3DF &Normalize (void);
		inline double Length (void);
		inline VTYPE *Data ();
	};
	
	#undef VNAME
	#undef VTYPE

	// Vector4DC Declarations

	#define VNAME		4DC
	#define VTYPE		unsigned char

	class LUNA_CORE Vector4DC {
	public:
		VTYPE x, y, z, w;
	
		inline Vector4DC &Set (const float xa, const float ya, const float za)	{ x = (VTYPE) xa; y= (VTYPE) ya; z=(VTYPE) za; w=1; return *this;}
		inline Vector4DC &Set (const float xa, const float ya, const float za, const float wa )	{ x =(VTYPE) xa; y= (VTYPE) ya; z=(VTYPE) za; w=(VTYPE) wa; return *this;}
		inline Vector4DC &Set (const VTYPE xa, const VTYPE ya, const VTYPE za)	{ x = (VTYPE) xa; y= (VTYPE) ya; z=(VTYPE) za; w=1; return *this;}
		inline Vector4DC &Set (const VTYPE xa, const VTYPE ya, const VTYPE za, const VTYPE wa )	{ x =(VTYPE) xa; y= (VTYPE) ya; z=(VTYPE) za; w=(VTYPE) wa; return *this;}

		// Constructors/Destructors
		inline Vector4DC();
		inline ~Vector4DC();
		inline Vector4DC (const VTYPE xa, const VTYPE ya, const VTYPE za, const VTYPE wa);
		inline Vector4DC (const Vector2DC &op);
		inline Vector4DC (const Vector2DI &op);
		inline Vector4DC (const Vector2DF &op);
		inline Vector4DC (const Vector3DC &op);
		inline Vector4DC (const Vector3DI &op);
		inline Vector4DC (const Vector3DF &op);
		inline Vector4DC (const Vector4DC &op);
		inline Vector4DC (const Vector4DF &op);

		// Member Functions
		inline Vector4DC &operator= ( const int op);
		inline Vector4DC &operator= ( const double op);
		inline Vector4DC &operator= ( const Vector2DC &op);
		inline Vector4DC &operator= ( const Vector2DI &op);
		inline Vector4DC &operator= ( const Vector2DF &op);
		inline Vector4DC &operator= ( const Vector3DC &op);
		inline Vector4DC &operator= ( const Vector3DI &op);
		inline Vector4DC &operator= ( const Vector3DF &op);
		inline Vector4DC &operator= ( const Vector4DC &op);
		inline Vector4DC &operator= ( const Vector4DF &op);

		inline Vector4DC &operator+= ( const int op);
		inline Vector4DC &operator+= ( const double op);
		inline Vector4DC &operator+= ( const Vector2DC &op);
		inline Vector4DC &operator+= ( const Vector2DI &op);
		inline Vector4DC &operator+= ( const Vector2DF &op);
		inline Vector4DC &operator+= ( const Vector3DC &op);
		inline Vector4DC &operator+= ( const Vector3DI &op);
		inline Vector4DC &operator+= ( const Vector3DF &op);
		inline Vector4DC &operator+= ( const Vector4DC &op);
		inline Vector4DC &operator+= ( const Vector4DF &op);

		inline Vector4DC &operator-= ( const int op);
		inline Vector4DC &operator-= ( const double op);
		inline Vector4DC &operator-= ( const Vector2DC &op);
		inline Vector4DC &operator-= ( const Vector2DI &op);
		inline Vector4DC &operator-= ( const Vector2DF &op);
		inline Vector4DC &operator-= ( const Vector3DC &op);
		inline Vector4DC &operator-= ( const Vector3DI &op);
		inline Vector4DC &operator-= ( const Vector3DF &op);
		inline Vector4DC &operator-= ( const Vector4DC &op);
		inline Vector4DC &operator-= ( const Vector4DF &op);

		inline Vector4DC &operator*= ( const int op);
		inline Vector4DC &operator*= ( const double op);
		inline Vector4DC &operator*= ( const Vector2DC &op);
		inline Vector4DC &operator*= ( const Vector2DI &op);
		inline Vector4DC &operator*= ( const Vector2DF &op);
		inline Vector4DC &operator*= ( const Vector3DC &op);
		inline Vector4DC &operator*= ( const Vector3DI &op);
		inline Vector4DC &operator*= ( const Vector3DF &op);
		inline Vector4DC &operator*= ( const Vector4DC &op);
		inline Vector4DC &operator*= ( const Vector4DF &op);
		
		inline Vector4DC &operator/= ( const int op);
		inline Vector4DC &operator/= ( const double op);
		inline Vector4DC &operator/= ( const Vector2DC &op);
		inline Vector4DC &operator/= ( const Vector2DI &op);
		inline Vector4DC &operator/= ( const Vector2DF &op);
		inline Vector4DC &operator/= ( const Vector3DC &op);
		inline Vector4DC &operator/= ( const Vector3DI &op);
		inline Vector4DC &operator/= ( const Vector3DF &op);
		inline Vector4DC &operator/= ( const Vector4DC &op);
		inline Vector4DC &operator/= ( const Vector4DF &op);

		// Slow operations - require temporary variables
		inline Vector4DC operator+ ( const int op);
		inline Vector4DC operator+ ( const float op);
		inline Vector4DC operator+ ( const Vector4DC &op);
		inline Vector4DC operator- ( const int op);
		inline Vector4DC operator- ( const float op);
		inline Vector4DC operator- ( const Vector4DC &op);
		inline Vector4DC operator* ( const int op);
		inline Vector4DC operator* ( const float op);
		inline Vector4DC operator* ( const Vector4DC &op);
		// --

		inline double Dot( const Vector4DF &v);
		inline double Dist ( const Vector4DF &v);
		inline double DistSq ( const Vector4DF &v);
		inline Vector4DC &Normalize (void);
		inline double Length (void);

		inline Vector4DC &Random ()		{ x=(VTYPE) float(rand()*255)/RAND_MAX; y=(VTYPE) float(rand()*255)/RAND_MAX; z=(VTYPE) float(rand()*255)/RAND_MAX; w = 1;  return *this;}
		inline VTYPE *Data (void);
	};
	#undef VNAME
	#undef VTYPE


	// Vector4DF Declarations

	#define VNAME		4DF
	#define VTYPE		float

	class LUNA_CORE Vector4DF {
	public:
		VTYPE x, y, z, w;
	
		inline Vector4DF &Set (const float xa, const float ya, const float za)	{ x =xa; y= ya; z=za; w=1; return *this;}
		inline Vector4DF &Set (const float xa, const float ya, const float za, const float wa )	{ x =xa; y= ya; z=za; w=wa; return *this;}

		// Constructors/Destructors
		inline Vector4DF();
		inline ~Vector4DF();
		inline Vector4DF (const VTYPE xa, const VTYPE ya, const VTYPE za, const VTYPE wa);
		inline Vector4DF (const Vector2DC &op);
		inline Vector4DF (const Vector2DI &op);
		inline Vector4DF (const Vector2DF &op);
		inline Vector4DF (const Vector3DC &op);
		inline Vector4DF (const Vector3DI &op);
		inline Vector4DF (const Vector3DF &op);
		inline Vector4DF (const Vector4DF &op);

		// Member Functions
		inline Vector4DF &operator= (const int op);
		inline Vector4DF &operator= (const double op);
		inline Vector4DF &operator= (const Vector2DC &op);
		inline Vector4DF &operator= (const Vector2DI &op);
		inline Vector4DF &operator= (const Vector2DF &op);
		inline Vector4DF &operator= (const Vector3DC &op);
		inline Vector4DF &operator= (const Vector3DI &op);
		inline Vector4DF &operator= (const Vector3DF &op);
		inline Vector4DF &operator= (const Vector4DF &op);

		inline Vector4DF &operator+= (const int op);
		inline Vector4DF &operator+= (const float op);
		inline Vector4DF &operator+= (const double op);
		inline Vector4DF &operator+= (const Vector2DC &op);
		inline Vector4DF &operator+= (const Vector2DI &op);
		inline Vector4DF &operator+= (const Vector2DF &op);
		inline Vector4DF &operator+= (const Vector3DC &op);
		inline Vector4DF &operator+= (const Vector3DI &op);
		inline Vector4DF &operator+= (const Vector3DF &op);
		inline Vector4DF &operator+= (const Vector4DF &op);

		inline Vector4DF &operator-= (const int op);
		inline Vector4DF &operator-= (const double op);
		inline Vector4DF &operator-= (const Vector2DC &op);
		inline Vector4DF &operator-= (const Vector2DI &op);
		inline Vector4DF &operator-= (const Vector2DF &op);
		inline Vector4DF &operator-= (const Vector3DC &op);
		inline Vector4DF &operator-= (const Vector3DI &op);
		inline Vector4DF &operator-= (const Vector3DF &op);
		inline Vector4DF &operator-= (const Vector4DF &op);

		inline Vector4DF &operator*= (const int op);
		inline Vector4DF &operator*= (const double op);
		inline Vector4DF &operator*= (const Vector2DC &op);
		inline Vector4DF &operator*= (const Vector2DI &op);
		inline Vector4DF &operator*= (const Vector2DF &op);
		inline Vector4DF &operator*= (const Vector3DC &op);
		inline Vector4DF &operator*= (const Vector3DI &op);
		inline Vector4DF &operator*= (const Vector3DF &op);
		inline Vector4DF &operator*= (const Vector4DF &op);
		Vector4DF &operator*= (const float* op );
		Vector4DF &operator*= (const Matrix4F &op);
		Vector4DF &operator*= (const MatrixF &op);				// see vector.cpp

		inline Vector4DF &operator/= (const int op);
		inline Vector4DF &operator/= (const double op);
		inline Vector4DF &operator/= (const Vector2DC &op);
		inline Vector4DF &operator/= (const Vector2DI &op);
		inline Vector4DF &operator/= (const Vector2DF &op);
		inline Vector4DF &operator/= (const Vector3DC &op);
		inline Vector4DF &operator/= (const Vector3DI &op);
		inline Vector4DF &operator/= (const Vector3DF &op);
		inline Vector4DF &operator/= (const Vector4DF &op);

		// Slow operations - require temporary variables
		inline Vector4DF operator+ (const int op)			{ return Vector4DF(x+float(op), y+float(op), z+float(op), w+float(op)); }
		inline Vector4DF operator+ (const float op)		{ return Vector4DF(x+op, y+op, z+op, w*op); }
		inline Vector4DF operator+ (const Vector4DF &op)	{ return Vector4DF(x+op.x, y+op.y, z+op.z, w+op.w); }
		inline Vector4DF operator- (const int op)			{ return Vector4DF(x-float(op), y-float(op), z-float(op), w-float(op)); }
		inline Vector4DF operator- (const float op)		{ return Vector4DF(x-op, y-op, z-op, w*op); }
		inline Vector4DF operator- (const Vector4DF &op)	{ return Vector4DF(x-op.x, y-op.y, z-op.z, w-op.w); }
		inline Vector4DF operator* (const int op)			{ return Vector4DF(x*float(op), y*float(op), z*float(op), w*float(op)); }
		inline Vector4DF operator* (const float op)		{ return Vector4DF(x*op, y*op, z*op, w*op); }
		inline Vector4DF operator* (const Vector4DF &op)	{ return Vector4DF(x*op.x, y*op.y, z*op.z, w*op.w); }		
		// --

		inline Vector4DF &Set ( CLRVAL clr )	{
			x = RED(clr);		// (float( c      & 0xFF)/255.0)	
			y = GRN(clr);		// (float((c>>8)  & 0xFF)/255.0)
			z = BLUE(clr);		// (float((c>>16) & 0xFF)/255.0)
			w = ALPH(clr);		// (float((c>>24) & 0xFF)/255.0)
			return *this;
		}
		inline Vector4DF& fromClr ( CLRVAL clr ) { return Set (clr); }
		inline CLRVAL toClr () { return (CLRVAL) COLORA( x, y, z, w ); }

		inline Vector4DF& Clamp ( float xc, float yc, float zc, float wc )
		{
			x = (x > xc) ? xc : x;
			y = (y > yc) ? yc : y;
			z = (z > zc) ? zc : z;
			w = (w > wc) ? wc : w;
			return *this;
		}

		inline Vector4DF &Cross (const Vector4DF &v);	
		
		inline double Dot (const Vector4DF &v);

		inline double Dist (const Vector4DF &v);

		inline double DistSq (const Vector4DF &v);

		inline Vector4DF &Normalize (void);
		inline double Length (void);

		inline Vector4DF &Random ()		{ x=float(rand())/RAND_MAX; y=float(rand())/RAND_MAX; z=float(rand())/RAND_MAX; w = 1;  return *this;}
		inline VTYPE *Data (void);
	};
	
	#undef VNAME
	#undef VTYPE

    // Vector Code Definitions (Inlined)
	#include "vector_inline.h"

#endif



