// Vector Operations Implemented:
//		=, +, -, *, / (on vectors and scalars)
//		Cross			Cross product vector with op
//		Dot				Dot product vector with op
//		Dist (op)		Distance from vector to op
//		DistSq			Distance^2 from vector to op
//		Length ()		Length of vector
//		Normalize ()	Normalizes vector
//

#include <math.h>

// Vector2DC Code Definition

#define VTYPE		unsigned char
#define VNAME		2DC

// Constructors/Destructors
inline Vector2DC::Vector2DC() {x=0; y=0;}
inline Vector2DC::~Vector2DC() {}
inline Vector2DC::Vector2DC (VTYPE xa, VTYPE ya) {x=xa; y=ya;}
inline Vector2DC::Vector2DC (Vector2DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
inline Vector2DC::Vector2DC (Vector2DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
inline Vector2DC::Vector2DC (Vector2DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
inline Vector2DC::Vector2DC (Vector3DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
inline Vector2DC::Vector2DC (Vector3DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
inline Vector2DC::Vector2DC (Vector3DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
inline Vector2DC::Vector2DC (Vector4DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}

// Member Functions
inline Vector2DC &Vector2DC::operator= (Vector2DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
inline Vector2DC &Vector2DC::operator= (Vector2DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
inline Vector2DC &Vector2DC::operator= (Vector2DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
inline Vector2DC &Vector2DC::operator= (Vector3DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
inline Vector2DC &Vector2DC::operator= (Vector3DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
inline Vector2DC &Vector2DC::operator= (Vector3DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
inline Vector2DC &Vector2DC::operator= (Vector4DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}	
	
inline Vector2DC &Vector2DC::operator+= (Vector2DC &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
inline Vector2DC &Vector2DC::operator+= (Vector2DI &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
inline Vector2DC &Vector2DC::operator+= (Vector2DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
inline Vector2DC &Vector2DC::operator+= (Vector3DC &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
inline Vector2DC &Vector2DC::operator+= (Vector3DI &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
inline Vector2DC &Vector2DC::operator+= (Vector3DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
inline Vector2DC &Vector2DC::operator+= (Vector4DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}

inline Vector2DC &Vector2DC::operator-= (Vector2DC &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
inline Vector2DC &Vector2DC::operator-= (Vector2DI &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
inline Vector2DC &Vector2DC::operator-= (Vector2DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
inline Vector2DC &Vector2DC::operator-= (Vector3DC &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
inline Vector2DC &Vector2DC::operator-= (Vector3DI &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
inline Vector2DC &Vector2DC::operator-= (Vector3DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
inline Vector2DC &Vector2DC::operator-= (Vector4DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
	
inline Vector2DC &Vector2DC::operator*= (Vector2DC &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
inline Vector2DC &Vector2DC::operator*= (Vector2DI &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
inline Vector2DC &Vector2DC::operator*= (Vector2DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
inline Vector2DC &Vector2DC::operator*= (Vector3DC &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
inline Vector2DC &Vector2DC::operator*= (Vector3DI &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
inline Vector2DC &Vector2DC::operator*= (Vector3DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
inline Vector2DC &Vector2DC::operator*= (Vector4DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}

inline Vector2DC &Vector2DC::operator/= (Vector2DC &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
inline Vector2DC &Vector2DC::operator/= (Vector2DI &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
inline Vector2DC &Vector2DC::operator/= (Vector2DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
inline Vector2DC &Vector2DC::operator/= (Vector3DC &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
inline Vector2DC &Vector2DC::operator/= (Vector3DI &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
inline Vector2DC &Vector2DC::operator/= (Vector3DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
inline Vector2DC &Vector2DC::operator/= (Vector4DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}

// Note: Cross product does not exist for 2D vectors (only 3D)
		
inline double Vector2DC::Dot(Vector2DC &v)			{double dot; dot = (double) x*v.x + (double) y*v.y; return dot;}
inline double Vector2DC::Dot(Vector2DI &v)			{double dot; dot = (double) x*v.x + (double) y*v.y; return dot;}
inline double Vector2DC::Dot(Vector2DF &v)			{double dot; dot = (double) x*v.x + (double) y*v.y; return dot;}

inline double Vector2DC::Dist (Vector2DC &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector2DC::Dist (Vector2DI &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector2DC::Dist (Vector2DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector2DC::Dist (Vector3DC &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector2DC::Dist (Vector3DI &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector2DC::Dist (Vector3DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector2DC::Dist (Vector4DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector2DC::DistSq (Vector2DC &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
inline double Vector2DC::DistSq (Vector2DI &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
inline double Vector2DC::DistSq (Vector2DF &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
inline double Vector2DC::DistSq (Vector3DC &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
inline double Vector2DC::DistSq (Vector3DI &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
inline double Vector2DC::DistSq (Vector3DF &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
inline double Vector2DC::DistSq (Vector4DF &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}

inline Vector2DC &Vector2DC::Normalize (void) {
	double n = (double) x*x + (double) y*y;
	if (n!=0.0) {
		n = sqrt(n);
		x = (VTYPE) (((double) x*255)/n); 
		y = (VTYPE) (((double) y*255)/n);				
	}
	return *this;
}
inline double Vector2DC::Length (void) { double n; n = (double) x*x + (double) y*y; if (n != 0.0) return sqrt(n); return 0.0; }
inline VTYPE *Vector2DC::Data (void)			{return &x;}

#undef VTYPE
#undef VNAME

// Vector2DI Code Definition

#define VNAME		2DI
#define VTYPE		int

// Constructors/Destructors
inline Vector2DI::Vector2DI() {x=0; y=0;}
inline Vector2DI::~Vector2DI() {}
inline Vector2DI::Vector2DI (const VTYPE xa, const VTYPE ya) {x=xa; y=ya;}
inline Vector2DI::Vector2DI (const Vector2DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
inline Vector2DI::Vector2DI (const Vector2DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
inline Vector2DI::Vector2DI (const Vector2DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
inline Vector2DI::Vector2DI (const Vector3DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
inline Vector2DI::Vector2DI (const Vector3DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
inline Vector2DI::Vector2DI (const Vector3DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
inline Vector2DI::Vector2DI (const Vector4DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}

// Member Functions
inline Vector2DI &Vector2DI::operator= (const Vector2DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
inline Vector2DI &Vector2DI::operator= (const Vector2DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
inline Vector2DI &Vector2DI::operator= (const Vector2DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
inline Vector2DI &Vector2DI::operator= (const Vector3DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
inline Vector2DI &Vector2DI::operator= (const Vector3DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
inline Vector2DI &Vector2DI::operator= (const Vector3DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
inline Vector2DI &Vector2DI::operator= (const Vector4DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}	
	
inline Vector2DI &Vector2DI::operator+= (const Vector2DC &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
inline Vector2DI &Vector2DI::operator+= (const Vector2DI &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
inline Vector2DI &Vector2DI::operator+= (const Vector2DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
inline Vector2DI &Vector2DI::operator+= (const Vector3DC &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
inline Vector2DI &Vector2DI::operator+= (const Vector3DI &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
inline Vector2DI &Vector2DI::operator+= (const Vector3DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
inline Vector2DI &Vector2DI::operator+= (const Vector4DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}

inline Vector2DI &Vector2DI::operator-= (const Vector2DC &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
inline Vector2DI &Vector2DI::operator-= (const Vector2DI &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
inline Vector2DI &Vector2DI::operator-= (const Vector2DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
inline Vector2DI &Vector2DI::operator-= (const Vector3DC &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
inline Vector2DI &Vector2DI::operator-= (const Vector3DI &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
inline Vector2DI &Vector2DI::operator-= (const Vector3DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
inline Vector2DI &Vector2DI::operator-= (const Vector4DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
	
inline Vector2DI &Vector2DI::operator*= (const Vector2DC &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
inline Vector2DI &Vector2DI::operator*= (const Vector2DI &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
inline Vector2DI &Vector2DI::operator*= (const Vector2DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
inline Vector2DI &Vector2DI::operator*= (const Vector3DC &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
inline Vector2DI &Vector2DI::operator*= (const Vector3DI &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
inline Vector2DI &Vector2DI::operator*= (const Vector3DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
inline Vector2DI &Vector2DI::operator*= (const Vector4DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}

inline Vector2DI &Vector2DI::operator/= (const Vector2DC &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
inline Vector2DI &Vector2DI::operator/= (const Vector2DI &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
inline Vector2DI &Vector2DI::operator/= (const Vector2DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
inline Vector2DI &Vector2DI::operator/= (const Vector3DC &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
inline Vector2DI &Vector2DI::operator/= (const Vector3DI &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
inline Vector2DI &Vector2DI::operator/= (const Vector3DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
inline Vector2DI &Vector2DI::operator/= (const Vector4DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}

// Note: Cross product does not exist for 2D vectors (only 3D)
		
inline double Vector2DI::Dot(const Vector2DC &v)			{double dot; dot = (double) x*v.x + (double) y*v.y; return dot;}
inline double Vector2DI::Dot(const Vector2DI &v)			{double dot; dot = (double) x*v.x + (double) y*v.y; return dot;}
inline double Vector2DI::Dot(const Vector2DF &v)			{double dot; dot = (double) x*v.x + (double) y*v.y; return dot;}

inline double Vector2DI::Dist (const Vector2DC &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector2DI::Dist (const Vector2DI &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector2DI::Dist (const Vector2DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector2DI::Dist (const Vector3DC &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector2DI::Dist (const Vector3DI &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector2DI::Dist (const Vector3DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector2DI::Dist (const Vector4DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector2DI::DistSq (const Vector2DC &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
inline double Vector2DI::DistSq (const Vector2DI &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
inline double Vector2DI::DistSq (const Vector2DF &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
inline double Vector2DI::DistSq (const Vector3DC &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
inline double Vector2DI::DistSq (const Vector3DI &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
inline double Vector2DI::DistSq (const Vector3DF &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
inline double Vector2DI::DistSq (const Vector4DF &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}

inline Vector2DI &Vector2DI::Normalize (void) {
	double n = (double) x*x + (double) y*y;
	if (n!=0.0) {
		n = sqrt(n);
		x = (VTYPE) (((double) x*255)/n);
		y = (VTYPE) (((double) y*255)/n);				
	}
	return *this;
}
inline double Vector2DI::Length (void) { double n; n = (double) x*x + (double) y*y; if (n != 0.0) return sqrt(n); return 0.0; }
inline VTYPE *Vector2DI::Data (void)			{return &x;}

#undef VTYPE
#undef VNAME

// Vector2DF Code Definition

#define VNAME		2DF
#define VTYPE		float

// Constructors/Destructors
inline Vector2DF::Vector2DF() {x=0; y=0;}
inline Vector2DF::~Vector2DF() {}
inline Vector2DF::Vector2DF (const VTYPE xa, const VTYPE ya) {x=xa; y=ya;}
inline Vector2DF::Vector2DF (const Vector2DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
inline Vector2DF::Vector2DF (const Vector2DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
inline Vector2DF::Vector2DF (const Vector2DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
inline Vector2DF::Vector2DF (const Vector3DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
inline Vector2DF::Vector2DF (const Vector3DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
inline Vector2DF::Vector2DF (const Vector3DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
inline Vector2DF::Vector2DF (const Vector4DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}

inline Vector2DF &Vector2DF::Set (const float xa, const float ya) { x = (VTYPE) xa; y = (VTYPE) ya; return *this; }

// Member Functions
inline Vector2DF &Vector2DF::operator= (const Vector2DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
inline Vector2DF &Vector2DF::operator= (const Vector2DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
inline Vector2DF &Vector2DF::operator= (const Vector2DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
inline Vector2DF &Vector2DF::operator= (const Vector3DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
inline Vector2DF &Vector2DF::operator= (const Vector3DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
inline Vector2DF &Vector2DF::operator= (const Vector3DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
inline Vector2DF &Vector2DF::operator= (const Vector4DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}	
	
inline Vector2DF &Vector2DF::operator+= (const Vector2DC &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
inline Vector2DF &Vector2DF::operator+= (const Vector2DI &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
inline Vector2DF &Vector2DF::operator+= (const Vector2DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
inline Vector2DF &Vector2DF::operator+= (const Vector3DC &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
inline Vector2DF &Vector2DF::operator+= (const Vector3DI &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
inline Vector2DF &Vector2DF::operator+= (const Vector3DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
inline Vector2DF &Vector2DF::operator+= (const Vector4DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}

inline Vector2DF &Vector2DF::operator-= (const Vector2DC &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
inline Vector2DF &Vector2DF::operator-= (const Vector2DI &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
inline Vector2DF &Vector2DF::operator-= (const Vector2DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
inline Vector2DF &Vector2DF::operator-= (const Vector3DC &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
inline Vector2DF &Vector2DF::operator-= (const Vector3DI &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
inline Vector2DF &Vector2DF::operator-= (const Vector3DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
inline Vector2DF &Vector2DF::operator-= (const Vector4DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
	
inline Vector2DF &Vector2DF::operator*= (const Vector2DC &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
inline Vector2DF &Vector2DF::operator*= (const Vector2DI &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
inline Vector2DF &Vector2DF::operator*= (const Vector2DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
inline Vector2DF &Vector2DF::operator*= (const Vector3DC &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
inline Vector2DF &Vector2DF::operator*= (const Vector3DI &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
inline Vector2DF &Vector2DF::operator*= (const Vector3DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
inline Vector2DF &Vector2DF::operator*= (const Vector4DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}

inline Vector2DF &Vector2DF::operator/= (const Vector2DC &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
inline Vector2DF &Vector2DF::operator/= (const Vector2DI &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
inline Vector2DF &Vector2DF::operator/= (const Vector2DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
inline Vector2DF &Vector2DF::operator/= (const Vector3DC &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
inline Vector2DF &Vector2DF::operator/= (const Vector3DI &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
inline Vector2DF &Vector2DF::operator/= (const Vector3DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
inline Vector2DF &Vector2DF::operator/= (const Vector4DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}

// Note: Cross product does not exist for 2D vectors (only 3D)
		
inline double Vector2DF::Dot(const Vector2DC &v)			{double dot; dot = (double) x*v.x + (double) y*v.y; return dot;}
inline double Vector2DF::Dot(const Vector2DI &v)			{double dot; dot = (double) x*v.x + (double) y*v.y; return dot;}
inline double Vector2DF::Dot(const Vector2DF &v)			{double dot; dot = (double) x*v.x + (double) y*v.y; return dot;}

inline double Vector2DF::Dist (const Vector2DC &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector2DF::Dist (const Vector2DI &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector2DF::Dist (const Vector2DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector2DF::Dist (const Vector3DC &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector2DF::Dist (const Vector3DI &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector2DF::Dist (const Vector3DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector2DF::Dist (const Vector4DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector2DF::DistSq (const Vector2DC &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
inline double Vector2DF::DistSq (const Vector2DI &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
inline double Vector2DF::DistSq (const Vector2DF &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
inline double Vector2DF::DistSq (const Vector3DC &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
inline double Vector2DF::DistSq (const Vector3DI &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
inline double Vector2DF::DistSq (const Vector3DF &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
inline double Vector2DF::DistSq (const Vector4DF &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}

inline Vector2DF &Vector2DF::Normalize (void) {
	VTYPE n = (VTYPE) x*x + (VTYPE) y*y;
	if (n!=0.0) {
		n = sqrt(n);
		x /= n;
		y /= n;
	}
	return *this;
}
inline double Vector2DF::Length (void) { double n; n = (double) x*x + (double) y*y; if (n != 0.0) return sqrt(n); return 0.0; }
inline VTYPE *Vector2DF::Data (void)			{return &x;}

#undef VTYPE
#undef VNAME

// Vector3DC Code Definition

#define VNAME		3DC
#define VTYPE		unsigned char

// Constructors/Destructors
inline Vector3DC::Vector3DC() {x=0; y=0; z=0;}
inline Vector3DC::~Vector3DC() {}
inline Vector3DC::Vector3DC (const VTYPE xa, const VTYPE ya, const VTYPE za) {x=xa; y=ya; z=za;}
inline Vector3DC::Vector3DC (const Vector2DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0;}
inline Vector3DC::Vector3DC (const Vector2DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0;}
inline Vector3DC::Vector3DC (const Vector2DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0;}
inline Vector3DC::Vector3DC (const Vector3DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z;}
inline Vector3DC::Vector3DC (const Vector3DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z;}
inline Vector3DC::Vector3DC (const Vector3DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z;}
inline Vector3DC::Vector3DC (const Vector4DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z;}

// Member Functions
inline Vector3DC &Vector3DC::Set (const VTYPE xa, const VTYPE ya, const VTYPE za) {x=xa; y=ya; z=za; return *this;}

inline Vector3DC &Vector3DC::operator= (const Vector2DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0; return *this;}
inline Vector3DC &Vector3DC::operator= (const Vector2DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0; return *this;}
inline Vector3DC &Vector3DC::operator= (const Vector2DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0; return *this;}
inline Vector3DC &Vector3DC::operator=  ( const Vector3DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; return *this;}
inline Vector3DC &Vector3DC::operator=  ( const Vector3DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; return *this;}
inline Vector3DC &Vector3DC::operator=  ( const Vector3DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; return *this;}
inline Vector3DC &Vector3DC::operator=  ( const Vector4DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; return *this;}	
	
inline Vector3DC &Vector3DC::operator+=  ( const Vector2DC &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
inline Vector3DC &Vector3DC::operator+=  ( const Vector2DI &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
inline Vector3DC &Vector3DC::operator+=  ( const Vector2DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
inline Vector3DC &Vector3DC::operator+=  ( const Vector3DC &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}
inline Vector3DC &Vector3DC::operator+=  ( const Vector3DI &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}
inline Vector3DC &Vector3DC::operator+=  ( const Vector3DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}
inline Vector3DC &Vector3DC::operator+=  ( const Vector4DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}

inline Vector3DC &Vector3DC::operator-=  ( const Vector2DC &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
inline Vector3DC &Vector3DC::operator-=  ( const Vector2DI &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
inline Vector3DC &Vector3DC::operator-=  ( const Vector2DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
inline Vector3DC &Vector3DC::operator-=  ( const Vector3DC &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
inline Vector3DC &Vector3DC::operator-=  ( const Vector3DI &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
inline Vector3DC &Vector3DC::operator-=  ( const Vector3DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
inline Vector3DC &Vector3DC::operator-=  ( const Vector4DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
	
inline Vector3DC &Vector3DC::operator*=  ( const Vector2DC &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
inline Vector3DC &Vector3DC::operator*=  ( const Vector2DI &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
inline Vector3DC &Vector3DC::operator*=  ( const Vector2DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
inline Vector3DC &Vector3DC::operator*=  ( const Vector3DC &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}
inline Vector3DC &Vector3DC::operator*=  ( const Vector3DI &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}
inline Vector3DC &Vector3DC::operator*=  ( const Vector3DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}
inline Vector3DC &Vector3DC::operator*=  ( const Vector4DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}

inline Vector3DC &Vector3DC::operator/=  ( const Vector2DC &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
inline Vector3DC &Vector3DC::operator/=  ( const Vector2DI &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
inline Vector3DC &Vector3DC::operator/=  ( const Vector2DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
inline Vector3DC &Vector3DC::operator/=  ( const Vector3DC &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}
inline Vector3DC &Vector3DC::operator/=  ( const Vector3DI &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}
inline Vector3DC &Vector3DC::operator/=  ( const Vector3DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}
inline Vector3DC &Vector3DC::operator/=  ( const Vector4DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}

inline Vector3DC &Vector3DC::Cross  ( const Vector3DC &v) {double ax = x, ay = y, az = z; x = (VTYPE) (ay * (double) v.z - az * (double) v.y); y = (VTYPE) (-ax * (double) v.z + az * (double) v.x); z = (VTYPE) (ax * (double) v.y - ay * (double) v.x); return *this;}
inline Vector3DC &Vector3DC::Cross  ( const Vector3DI &v) {double ax = x, ay = y, az = z; x = (VTYPE) (ay * (double) v.z - az * (double) v.y); y = (VTYPE) (-ax * (double) v.z + az * (double) v.x); z = (VTYPE) (ax * (double) v.y - ay * (double) v.x); return *this;}
inline Vector3DC &Vector3DC::Cross  ( const Vector3DF &v) {double ax = x, ay = y, az = z; x = (VTYPE) (ay * (double) v.z - az * (double) v.y); y = (VTYPE) (-ax * (double) v.z + az * (double) v.x); z = (VTYPE) (ax * (double) v.y - ay * (double) v.x); return *this;}

inline double Vector3DC::Dot ( const Vector3DC &v)			{double dot; dot = (double) x*v.x + (double) y*v.y + (double) z*v.z; return dot;}
inline double Vector3DC::Dot ( const Vector3DI &v)			{double dot; dot = (double) x*v.x + (double) y*v.y + (double) z*v.z; return dot;}
inline double Vector3DC::Dot ( const Vector3DF &v)			{double dot; dot = (double) x*v.x + (double) y*v.y + (double) z*v.z; return dot;}

inline double Vector3DC::Dist  ( const Vector2DC &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector3DC::Dist  ( const Vector2DI &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector3DC::Dist  ( const Vector2DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector3DC::Dist  ( const Vector3DC &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector3DC::Dist  ( const Vector3DI &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector3DC::Dist  ( const Vector3DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector3DC::Dist  ( const Vector4DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector3DC::DistSq  ( const Vector2DC &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z; return (a*a + b*b + c*c);}
inline double Vector3DC::DistSq  ( const Vector2DI &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z; return (a*a + b*b + c*c);}
inline double Vector3DC::DistSq  ( const Vector2DF &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z; return (a*a + b*b + c*c);}
inline double Vector3DC::DistSq  ( const Vector3DC &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z - (double) v.z; return (a*a + b*b + c*c);}
inline double Vector3DC::DistSq  ( const Vector3DI &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z - (double) v.z; return (a*a + b*b + c*c);}
inline double Vector3DC::DistSq  ( const Vector3DF &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z - (double) v.z; return (a*a + b*b + c*c);}
inline double Vector3DC::DistSq  ( const Vector4DF &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z - (double) v.z; return (a*a + b*b + c*c);}

inline Vector3DC &Vector3DC::Normalize (void) {
	double n = (double) x*x + (double) y*y + (double) z*z;
	if (n!=0.0) {
		n = sqrt(n);
		x = (VTYPE) (((double) x*255)/n);
		y = (VTYPE) (((double) y*255)/n);
		z = (VTYPE) (((double) z*255)/n);
	}
	return *this;
}
inline double Vector3DC::Length (void) { double n; n = (double) x*x + (double) y*y + (double) z*z; if (n != 0.0) return sqrt(n); return 0.0; }
inline VTYPE *Vector3DC::Data (void)			{return &x;}

#undef VTYPE
#undef VNAME

// Vector3DI Code Definition

#define VNAME		3DI
#define VTYPE		int

// Constructors/Destructors
inline Vector3DI::Vector3DI() {x=0; y=0; z=0;}
inline Vector3DI::~Vector3DI() {}
inline Vector3DI::Vector3DI (const VTYPE xa, const  VTYPE ya, const VTYPE za) {x=xa; y=ya; z=za;}
inline Vector3DI::Vector3DI (const Vector2DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0;}
inline Vector3DI::Vector3DI (const Vector2DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0;}
inline Vector3DI::Vector3DI (const Vector2DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0;}
inline Vector3DI::Vector3DI (const Vector3DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z;}
inline Vector3DI::Vector3DI (const Vector3DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z;}
inline Vector3DI::Vector3DI (const Vector3DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z;}
inline Vector3DI::Vector3DI (const Vector4DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z;}

// Set Functions
inline Vector3DI &Vector3DI::Set (const int xa, const int ya, const int za)
{
	x = xa; y = ya; z = za;
	return *this;
}

// Member Functions
inline Vector3DI &Vector3DI::operator= (const Vector2DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
inline Vector3DI &Vector3DI::operator= (const Vector2DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
inline Vector3DI &Vector3DI::operator= (const Vector2DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
inline Vector3DI &Vector3DI::operator= (const Vector3DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; return *this;}
inline Vector3DI &Vector3DI::operator= (const Vector3DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; return *this;}
inline Vector3DI &Vector3DI::operator= (const Vector3DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; return *this;}
inline Vector3DI &Vector3DI::operator= (const Vector4DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; return *this;}	
	
inline Vector3DI &Vector3DI::operator+= (const Vector2DC &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
inline Vector3DI &Vector3DI::operator+= (const Vector2DI &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
inline Vector3DI &Vector3DI::operator+= (const Vector2DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
inline Vector3DI &Vector3DI::operator+= (const Vector3DC &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}
inline Vector3DI &Vector3DI::operator+= (const Vector3DI &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}
inline Vector3DI &Vector3DI::operator+= (const Vector3DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}
inline Vector3DI &Vector3DI::operator+= (const Vector4DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}

inline Vector3DI &Vector3DI::operator-= (const Vector2DC &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
inline Vector3DI &Vector3DI::operator-= (const Vector2DI &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
inline Vector3DI &Vector3DI::operator-= (const Vector2DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
inline Vector3DI &Vector3DI::operator-= (const Vector3DC &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
inline Vector3DI &Vector3DI::operator-= (const Vector3DI &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
inline Vector3DI &Vector3DI::operator-= (const Vector3DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
inline Vector3DI &Vector3DI::operator-= (const Vector4DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
	
inline Vector3DI &Vector3DI::operator*= (const Vector2DC &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
inline Vector3DI &Vector3DI::operator*= (const Vector2DI &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
inline Vector3DI &Vector3DI::operator*= (const Vector2DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
inline Vector3DI &Vector3DI::operator*= (const Vector3DC &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}
inline Vector3DI &Vector3DI::operator*= (const Vector3DI &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}
inline Vector3DI &Vector3DI::operator*= (const Vector3DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}
inline Vector3DI &Vector3DI::operator*= (const Vector4DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}

inline Vector3DI &Vector3DI::operator/= (const Vector2DC &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
inline Vector3DI &Vector3DI::operator/= (const Vector2DI &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
inline Vector3DI &Vector3DI::operator/= (const Vector2DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
inline Vector3DI &Vector3DI::operator/= (const Vector3DC &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}
inline Vector3DI &Vector3DI::operator/= (const Vector3DI &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}
inline Vector3DI &Vector3DI::operator/= (const Vector3DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}
inline Vector3DI &Vector3DI::operator/= (const Vector4DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}

inline Vector3DI &Vector3DI::Cross (const Vector3DC &v) {double ax = x, ay = y, az = z; x = (VTYPE) (ay * (double) v.z - az * (double) v.y); y = (VTYPE) (-ax * (double) v.z + az * (double) v.x); z = (VTYPE) (ax * (double) v.y - ay * (double) v.x); return *this;}
inline Vector3DI &Vector3DI::Cross (const Vector3DI &v) {double ax = x, ay = y, az = z; x = (VTYPE) (ay * (double) v.z - az * (double) v.y); y = (VTYPE) (-ax * (double) v.z + az * (double) v.x); z = (VTYPE) (ax * (double) v.y - ay * (double) v.x); return *this;}
inline Vector3DI &Vector3DI::Cross (const Vector3DF &v) {double ax = x, ay = y, az = z; x = (VTYPE) (ay * (double) v.z - az * (double) v.y); y = (VTYPE) (-ax * (double) v.z + az * (double) v.x); z = (VTYPE) (ax * (double) v.y - ay * (double) v.x); return *this;}
		
inline double Vector3DI::Dot(const Vector3DC &v)			{double dot; dot = (double) x*v.x + (double) y*v.y + (double) z*v.z; return dot;}
inline double Vector3DI::Dot(const Vector3DI &v)			{double dot; dot = (double) x*v.x + (double) y*v.y + (double) z*v.z; return dot;}
inline double Vector3DI::Dot(const Vector3DF &v)			{double dot; dot = (double) x*v.x + (double) y*v.y + (double) z*v.z; return dot;}

inline double Vector3DI::Dist (const Vector2DC &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector3DI::Dist (const Vector2DI &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector3DI::Dist (const Vector2DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector3DI::Dist (const Vector3DC &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector3DI::Dist (const Vector3DI &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector3DI::Dist (const Vector3DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector3DI::Dist (const Vector4DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector3DI::DistSq (const Vector2DC &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z; return (a*a + b*b + c*c);}
inline double Vector3DI::DistSq (const Vector2DI &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z; return (a*a + b*b + c*c);}
inline double Vector3DI::DistSq (const Vector2DF &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z; return (a*a + b*b + c*c);}
inline double Vector3DI::DistSq (const Vector3DC &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z - (double) v.z; return (a*a + b*b + c*c);}
inline double Vector3DI::DistSq (const Vector3DI &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z - (double) v.z; return (a*a + b*b + c*c);}
inline double Vector3DI::DistSq (const Vector3DF &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z - (double) v.z; return (a*a + b*b + c*c);}
inline double Vector3DI::DistSq (const Vector4DF &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z - (double) v.z; return (a*a + b*b + c*c);}

inline Vector3DI &Vector3DI::Normalize (void) {
	double n = (double) x*x + (double) y*y + (double) z*z;
	if (n!=0.0) {
		n = sqrt(n);
		x = (VTYPE) (((double) x*255)/n);
		y = (VTYPE) (((double) y*255)/n);
		z = (VTYPE) (((double) z*255)/n);
	}
	return *this;
}
inline double Vector3DI::Length (void) { double n; n = (double) x*x + (double) y*y + (double) z*z; if (n != 0.0) return sqrt(n); return 0.0; }
inline VTYPE *Vector3DI::Data (void)			{return &x;}

#undef VTYPE
#undef VNAME

// Vector3DF Code Definition

#define VNAME		3DF
#define VTYPE		float

// Constructors/Destructors
inline Vector3DF::Vector3DF() {x=0; y=0; z=0;}
inline Vector3DF::~Vector3DF() {}
inline Vector3DF::Vector3DF (const VTYPE xa, const VTYPE ya, const VTYPE za) {x=xa; y=ya; z=za;}
inline Vector3DF::Vector3DF (const Vector2DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0;}
inline Vector3DF::Vector3DF (const Vector2DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0;}
inline Vector3DF::Vector3DF (const Vector2DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0;}
inline Vector3DF::Vector3DF (const Vector3DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z;}
inline Vector3DF::Vector3DF (const Vector3DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z;}
inline Vector3DF::Vector3DF (const Vector3DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z;}
inline Vector3DF::Vector3DF (const Vector4DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z;}

// Set Functions
inline Vector3DF &Vector3DF::Set (const double xa, const double ya, const double za)
{
	x = (VTYPE) xa; y = (VTYPE) ya; z = (VTYPE) za;
	return *this;
}

// Member Functions
inline Vector3DF &Vector3DF::operator= (const int op) {x= (VTYPE) op; y= (VTYPE) op; z= (VTYPE) op; return *this;}
inline Vector3DF &Vector3DF::operator= (const double op) {x= (VTYPE) op; y= (VTYPE) op; z= (VTYPE) op; return *this;}
inline Vector3DF &Vector3DF::operator= (const Vector2DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
inline Vector3DF &Vector3DF::operator= (const Vector2DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
inline Vector3DF &Vector3DF::operator= (const Vector2DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
inline Vector3DF &Vector3DF::operator= (const Vector3DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; return *this;}
inline Vector3DF &Vector3DF::operator= (const Vector3DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; return *this;}
inline Vector3DF &Vector3DF::operator= (const Vector3DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; return *this;}
inline Vector3DF &Vector3DF::operator= (const Vector4DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; return *this;}	
	
inline Vector3DF &Vector3DF::operator+= (const int op) {x+= (VTYPE) op; y+= (VTYPE) op; z+= (VTYPE) op; return *this;}
inline Vector3DF &Vector3DF::operator+= (const double op) {x+= (VTYPE) op; y+= (VTYPE) op; z+= (VTYPE) op; return *this;}
inline Vector3DF &Vector3DF::operator+= (const Vector2DC &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
inline Vector3DF &Vector3DF::operator+= (const Vector2DI &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
inline Vector3DF &Vector3DF::operator+= (const Vector2DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
inline Vector3DF &Vector3DF::operator+= (const Vector3DC &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}
inline Vector3DF &Vector3DF::operator+= (const Vector3DI &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}
inline Vector3DF &Vector3DF::operator+= (const Vector3DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}
inline Vector3DF &Vector3DF::operator+= (const Vector4DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}

inline Vector3DF &Vector3DF::operator-= (const int op) {x-= (VTYPE) op; y-= (VTYPE) op; z-= (VTYPE) op; return *this;}
inline Vector3DF &Vector3DF::operator-= (const double op) {x-= (VTYPE) op; y-= (VTYPE) op; z-= (VTYPE) op; return *this;}
inline Vector3DF &Vector3DF::operator-= (const Vector2DC &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
inline Vector3DF &Vector3DF::operator-= (const Vector2DI &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
inline Vector3DF &Vector3DF::operator-= (const Vector2DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
inline Vector3DF &Vector3DF::operator-= (const Vector3DC &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
inline Vector3DF &Vector3DF::operator-= (const Vector3DI &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
inline Vector3DF &Vector3DF::operator-= (const Vector3DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
inline Vector3DF &Vector3DF::operator-= (const Vector4DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
	
inline Vector3DF &Vector3DF::operator*= (const int op) {x*= (VTYPE) op; y*= (VTYPE) op; z*= (VTYPE) op; return *this;}
inline Vector3DF &Vector3DF::operator*= (const double op) {x*= (VTYPE) op; y*= (VTYPE) op; z*= (VTYPE) op; return *this;}
inline Vector3DF &Vector3DF::operator*= (const Vector2DC &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
inline Vector3DF &Vector3DF::operator*= (const Vector2DI &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
inline Vector3DF &Vector3DF::operator*= (const Vector2DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
inline Vector3DF &Vector3DF::operator*= (const Vector3DC &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}
inline Vector3DF &Vector3DF::operator*= (const Vector3DI &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}
inline Vector3DF &Vector3DF::operator*= (const Vector3DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}
inline Vector3DF &Vector3DF::operator*= (const Vector4DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}

inline Vector3DF &Vector3DF::operator/= (const int op) {x/= (VTYPE) op; y/= (VTYPE) op; z/= (VTYPE) op; return *this;}
inline Vector3DF &Vector3DF::operator/= (const double op) {x/= (VTYPE) op; y/= (VTYPE) op; z/= (VTYPE) op; return *this;}
inline Vector3DF &Vector3DF::operator/= (const Vector2DC &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
inline Vector3DF &Vector3DF::operator/= (const Vector2DI &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
inline Vector3DF &Vector3DF::operator/= (const Vector2DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
inline Vector3DF &Vector3DF::operator/= (const Vector3DC &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}
inline Vector3DF &Vector3DF::operator/= (const Vector3DI &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}
inline Vector3DF &Vector3DF::operator/= (const Vector3DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}
inline Vector3DF &Vector3DF::operator/= (const Vector4DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}

inline Vector3DF &Vector3DF::Cross (const Vector3DC &v) {double ax = x, ay = y, az = z; x = (VTYPE) (ay * (double) v.z - az * (double) v.y); y = (VTYPE) (-ax * (double) v.z + az * (double) v.x); z = (VTYPE) (ax * (double) v.y - ay * (double) v.x); return *this;}
inline Vector3DF &Vector3DF::Cross (const Vector3DI &v) {double ax = x, ay = y, az = z; x = (VTYPE) (ay * (double) v.z - az * (double) v.y); y = (VTYPE) (-ax * (double) v.z + az * (double) v.x); z = (VTYPE) (ax * (double) v.y - ay * (double) v.x); return *this;}
inline Vector3DF &Vector3DF::Cross (const Vector3DF &v) {double ax = x, ay = y, az = z; x = (VTYPE) (ay * (double) v.z - az * (double) v.y); y = (VTYPE) (-ax * (double) v.z + az * (double) v.x); z = (VTYPE) (ax * (double) v.y - ay * (double) v.x); return *this;}
		
inline double Vector3DF::Dot(const Vector3DC &v)			{double dot; dot = (double) x*v.x + (double) y*v.y + (double) z*v.z; return dot;}
inline double Vector3DF::Dot(const Vector3DI &v)			{double dot; dot = (double) x*v.x + (double) y*v.y + (double) z*v.z; return dot;}
inline double Vector3DF::Dot(const Vector3DF &v)			{double dot; dot = (double) x*v.x + (double) y*v.y + (double) z*v.z; return dot;}

inline double Vector3DF::Dist (const Vector2DC &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector3DF::Dist (const Vector2DI &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector3DF::Dist (const Vector2DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector3DF::Dist (const Vector3DC &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector3DF::Dist (const Vector3DI &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector3DF::Dist (const Vector3DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector3DF::Dist (const Vector4DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector3DF::DistSq (const Vector2DC &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z; return (a*a + b*b + c*c);}
inline double Vector3DF::DistSq (const Vector2DI &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z; return (a*a + b*b + c*c);}
inline double Vector3DF::DistSq (const Vector2DF &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z; return (a*a + b*b + c*c);}
inline double Vector3DF::DistSq (const Vector3DC &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z - (double) v.z; return (a*a + b*b + c*c);}
inline double Vector3DF::DistSq (const Vector3DI &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z - (double) v.z; return (a*a + b*b + c*c);}
inline double Vector3DF::DistSq (const Vector3DF &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z - (double) v.z; return (a*a + b*b + c*c);}
inline double Vector3DF::DistSq (const Vector4DF &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z - (double) v.z; return (a*a + b*b + c*c);}

inline Vector3DF &Vector3DF::Normalize (void) {
	double n = (double) x*x + (double) y*y + (double)  z*z;
	if (n!=0.0) {
		n = sqrt(n);
		x = double(x)/n; y = double(y)/n; z = double(z)/n;
	}
	return *this;
}
inline double Vector3DF::Length (void) { double n; n = (double) x*x + (double) y*y + (double) z*z; if (n != 0.0) return sqrt(n); return 0.0; }
inline VTYPE *Vector3DF::Data ()			{return &x;}

#undef VTYPE
#undef VNAME

// Vector4DC Code Definition

#define VNAME		4DC
#define VTYPE		unsigned char

// Constructors/Destructors
inline Vector4DC::Vector4DC() {x=0; y=0; z=0; w=0;}
inline Vector4DC::~Vector4DC() {}
inline Vector4DC::Vector4DC (const VTYPE xa, const VTYPE ya, const VTYPE za, const VTYPE wa) {x=xa; y=ya; z=za; w=wa;}
inline Vector4DC::Vector4DC (const Vector2DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0; w=(VTYPE) 0;}
inline Vector4DC::Vector4DC (const Vector2DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0; w=(VTYPE) 0;}
inline Vector4DC::Vector4DC (const Vector2DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0; w=(VTYPE) 0;}
inline Vector4DC::Vector4DC (const Vector3DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; w=(VTYPE) 0;}
inline Vector4DC::Vector4DC (const Vector3DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; w=(VTYPE) 0;}
inline Vector4DC::Vector4DC (const Vector3DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; w=(VTYPE) 0;}
inline Vector4DC::Vector4DC (const Vector4DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; w=(VTYPE) op.w;}
inline Vector4DC::Vector4DC (const Vector4DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; w=(VTYPE) op.w;}

// Member Functions
inline Vector4DC &Vector4DC::operator= (const int op) {x= (VTYPE) op; y= (VTYPE) op; z= (VTYPE) op; w = (VTYPE) op; return *this;}
inline Vector4DC &Vector4DC::operator= (const double op) {x= (VTYPE) op; y= (VTYPE) op; z= (VTYPE) op; w = (VTYPE) op; return *this;}
inline Vector4DC &Vector4DC::operator=  ( const Vector2DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0; w=(VTYPE) 0; return *this;}
inline Vector4DC &Vector4DC::operator=  ( const Vector2DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0; w=(VTYPE) 0;  return *this;}
inline Vector4DC &Vector4DC::operator=  ( const Vector2DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0; w=(VTYPE) 0;  return *this;}
inline Vector4DC &Vector4DC::operator=  ( const Vector3DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; w=(VTYPE) 0;  return *this;}
inline Vector4DC &Vector4DC::operator=  ( const Vector3DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; w=(VTYPE) 0;  return *this;}
inline Vector4DC &Vector4DC::operator=  ( const Vector3DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; w=(VTYPE) 0; return *this;}
inline Vector4DC &Vector4DC::operator=  ( const Vector4DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; w=(VTYPE) op.w; return *this;}	
inline Vector4DC &Vector4DC::operator=  ( const Vector4DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; w=(VTYPE) op.w; return *this;}	
	
inline Vector4DC &Vector4DC::operator+= (const int op) {x+= (VTYPE) op; y+= (VTYPE) op; z+= (VTYPE) op; w += (VTYPE) op; return *this;}
inline Vector4DC &Vector4DC::operator+= (const double op) {x+= (VTYPE) op; y+= (VTYPE) op; z+= (VTYPE) op; w += (VTYPE) op; return *this;}
inline Vector4DC &Vector4DC::operator+=  ( const Vector2DC &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
inline Vector4DC &Vector4DC::operator+=  ( const Vector2DI &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
inline Vector4DC &Vector4DC::operator+=  ( const Vector2DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
inline Vector4DC &Vector4DC::operator+=  ( const Vector3DC &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}
inline Vector4DC &Vector4DC::operator+=  ( const Vector3DI &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}
inline Vector4DC &Vector4DC::operator+=  ( const Vector3DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}
inline Vector4DC &Vector4DC::operator+=  ( const Vector4DC &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; w+=(VTYPE) op.w; return *this;}	
inline Vector4DC &Vector4DC::operator+=  ( const Vector4DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; w+=(VTYPE) op.w; return *this;}	

inline Vector4DC &Vector4DC::operator-= (const int op) {x-= (VTYPE) op; y-= (VTYPE) op; z-= (VTYPE) op; w -= (VTYPE) op; return *this;}
inline Vector4DC &Vector4DC::operator-= (const double op) {x-= (VTYPE) op; y-= (VTYPE) op; z-= (VTYPE) op; w -= (VTYPE) op; return *this;}
inline Vector4DC &Vector4DC::operator-=  ( const Vector2DC &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
inline Vector4DC &Vector4DC::operator-=  ( const Vector2DI &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
inline Vector4DC &Vector4DC::operator-=  ( const Vector2DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
inline Vector4DC &Vector4DC::operator-=  ( const Vector3DC &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
inline Vector4DC &Vector4DC::operator-=  ( const Vector3DI &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
inline Vector4DC &Vector4DC::operator-=  ( const Vector3DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
inline Vector4DC &Vector4DC::operator-=  ( const Vector4DC &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; w-=(VTYPE) op.w; return *this;}	
inline Vector4DC &Vector4DC::operator-=  ( const Vector4DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; w-=(VTYPE) op.w; return *this;}	

inline Vector4DC &Vector4DC::operator*= (const int op) {x*= (VTYPE) op; y*= (VTYPE) op; z*= (VTYPE) op; w *= (VTYPE) op; return *this;}
inline Vector4DC &Vector4DC::operator*= (const double op) {x*= (VTYPE) op; y*= (VTYPE) op; z*= (VTYPE) op; w *= (VTYPE) op; return *this;}
inline Vector4DC &Vector4DC::operator*=  ( const Vector2DC &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
inline Vector4DC &Vector4DC::operator*=  ( const Vector2DI &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
inline Vector4DC &Vector4DC::operator*=  ( const Vector2DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
inline Vector4DC &Vector4DC::operator*=  ( const Vector3DC &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}
inline Vector4DC &Vector4DC::operator*=  ( const Vector3DI &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}
inline Vector4DC &Vector4DC::operator*=  ( const Vector3DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}
inline Vector4DC &Vector4DC::operator*=  ( const Vector4DC &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; w*=(VTYPE) op.w; return *this;}	
inline Vector4DC &Vector4DC::operator*=  ( const Vector4DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; w*=(VTYPE) op.w; return *this;}	

inline Vector4DC &Vector4DC::operator/= (const int op) {x/= (VTYPE) op; y/= (VTYPE) op; z/= (VTYPE) op; w /= (VTYPE) op; return *this;}
inline Vector4DC &Vector4DC::operator/= (const double op) {x/= (VTYPE) op; y/= (VTYPE) op; z/= (VTYPE) op; w /= (VTYPE) op; return *this;}
inline Vector4DC &Vector4DC::operator/=  ( const Vector2DC &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
inline Vector4DC &Vector4DC::operator/=  ( const Vector2DI &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
inline Vector4DC &Vector4DC::operator/=  ( const Vector2DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
inline Vector4DC &Vector4DC::operator/=  ( const Vector3DC &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}
inline Vector4DC &Vector4DC::operator/=  ( const Vector3DI &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}
inline Vector4DC &Vector4DC::operator/=  ( const Vector3DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}
inline Vector4DC &Vector4DC::operator/=  ( const Vector4DC &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; w/=(VTYPE) op.w; return *this;}	
inline Vector4DC &Vector4DC::operator/=  ( const Vector4DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; w/=(VTYPE) op.w; return *this;}

	inline Vector4DC Vector4DC::operator+ (const int op)			{ Vector4DC v(x,y,z,w); return v+=(VTYPE) op; }
	inline Vector4DC Vector4DC::operator+ (const float op)		{ Vector4DC v(x,y,z,w); return v+=(VTYPE) op; }
	inline Vector4DC Vector4DC::operator+  ( const Vector4DC &op)		{ Vector4DC v(x,y,z,w); return v+= op; }
	inline Vector4DC Vector4DC::operator- (const int op)			{ Vector4DC v(x,y,z,w); return v-=(VTYPE) op; }
	inline Vector4DC Vector4DC::operator- (const float op)		{ Vector4DC v(x,y,z,w); return v-=(VTYPE) op; }
	inline Vector4DC Vector4DC::operator-  ( const Vector4DC &op)		{ Vector4DC v(x,y,z,w); return v-= op; }
	inline Vector4DC Vector4DC::operator* (const int op)			{ Vector4DC v(x,y,z,w); return v*=(VTYPE) op; }
	inline Vector4DC Vector4DC::operator* (const float op)		{ Vector4DC v(x,y,z,w); return v*=(VTYPE) op; }
	inline Vector4DC Vector4DC::operator*  ( const Vector4DC &op)		{ Vector4DC v(x,y,z,w); return v*= op; }

inline double Vector4DC::Dot ( const Vector4DF &v)			{double dot; dot = (double) x*v.x + (double) y*v.y + (double) z*v.z + (double) w*v.w; return dot;}
inline double Vector4DC::Dist  ( const Vector4DF &v)		{double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
inline double Vector4DC::DistSq  ( const Vector4DF &v)		{double a,b,c,d; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z - (double) v.z; d = (double) w - (double) v.w; return (a*a + b*b + c*c + d*d);}

inline Vector4DC &Vector4DC::Normalize (void) {
	float n = (float) x*x + (float) y*y + (float) z*z + (float) w*w;
	if (n!=0.0) {
		n = sqrt( n);
		x = (VTYPE) (255.0 / n); y = (VTYPE) (255 / n); z = (VTYPE) (255 / n); w = (VTYPE) (255 / n);
	}
	return *this;
}
inline double Vector4DC::Length (void) { double n; n = (double) x*x + (double) y*y + (double) z*z + (double) w*w; if (n != 0.0) return sqrt(n); return 0.0; }
inline VTYPE *Vector4DC::Data (void)			{return &x;}

#undef VTYPE
#undef VNAME


// Vector4DF Code Definition

#define VNAME		4DF
#define VTYPE		float

// Constructors/Destructors
inline Vector4DF::Vector4DF() {x=0; y=0; z=0; w=0;}
inline Vector4DF::~Vector4DF() {}
inline Vector4DF::Vector4DF (const VTYPE xa, const VTYPE ya, const VTYPE za, const VTYPE wa) {x=xa; y=ya; z=za; w=wa;}
inline Vector4DF::Vector4DF (const Vector2DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0; w=(VTYPE) 1;}
inline Vector4DF::Vector4DF (const Vector2DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0; w=(VTYPE) 1;}
inline Vector4DF::Vector4DF (const Vector2DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0; w=(VTYPE) 1;}
inline Vector4DF::Vector4DF (const Vector3DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; w=(VTYPE) 1;}
inline Vector4DF::Vector4DF (const Vector3DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; w=(VTYPE) 1;}
inline Vector4DF::Vector4DF (const Vector3DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; w=(VTYPE) 1;}
inline Vector4DF::Vector4DF (const Vector4DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; w=(VTYPE) op.w;}

// Member Functions
inline Vector4DF &Vector4DF::operator= (const int op) {x= (VTYPE) op; y= (VTYPE) op; z= (VTYPE) op; w = (VTYPE) op; return *this;}
inline Vector4DF &Vector4DF::operator= (const double op) {x= (VTYPE) op; y= (VTYPE) op; z= (VTYPE) op; w = (VTYPE) op; return *this;}
inline Vector4DF &Vector4DF::operator= (const Vector2DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0; w=(VTYPE) 0; return *this;}
inline Vector4DF &Vector4DF::operator= (const Vector2DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0; w=(VTYPE) 0;  return *this;}
inline Vector4DF &Vector4DF::operator= (const Vector2DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0; w=(VTYPE) 0;  return *this;}
inline Vector4DF &Vector4DF::operator= (const Vector3DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; w=(VTYPE) 0;  return *this;}
inline Vector4DF &Vector4DF::operator= (const Vector3DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; w=(VTYPE) 0;  return *this;}
inline Vector4DF &Vector4DF::operator= (const Vector3DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; w=(VTYPE) 0; return *this;}
inline Vector4DF &Vector4DF::operator= (const Vector4DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; w=(VTYPE) op.w; return *this;}	
	
inline Vector4DF &Vector4DF::operator+= (const int op) {x+= (VTYPE) op; y+= (VTYPE) op; z+= (VTYPE) op; w += (VTYPE) op; return *this;}
inline Vector4DF &Vector4DF::operator+= (const float op) {x+= (VTYPE) op; y+= (VTYPE) op; z+= (VTYPE) op; w += (VTYPE) op; return *this;}
inline Vector4DF &Vector4DF::operator+= (const double op) {x+= (VTYPE) op; y+= (VTYPE) op; z+= (VTYPE) op; w += (VTYPE) op; return *this;}
inline Vector4DF &Vector4DF::operator+= (const Vector2DC &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
inline Vector4DF &Vector4DF::operator+= (const Vector2DI &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
inline Vector4DF &Vector4DF::operator+= (const Vector2DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
inline Vector4DF &Vector4DF::operator+= (const Vector3DC &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}
inline Vector4DF &Vector4DF::operator+= (const Vector3DI &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}
inline Vector4DF &Vector4DF::operator+= (const Vector3DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}
inline Vector4DF &Vector4DF::operator+= (const Vector4DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; w+=(VTYPE) op.w; return *this;}	

inline Vector4DF &Vector4DF::operator-= (const int op) {x-= (VTYPE) op; y-= (VTYPE) op; z-= (VTYPE) op; w -= (VTYPE) op; return *this;}
inline Vector4DF &Vector4DF::operator-= (const double op) {x-= (VTYPE) op; y-= (VTYPE) op; z-= (VTYPE) op; w -= (VTYPE) op; return *this;}
inline Vector4DF &Vector4DF::operator-= (const Vector2DC &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
inline Vector4DF &Vector4DF::operator-= (const Vector2DI &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
inline Vector4DF &Vector4DF::operator-= (const Vector2DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
inline Vector4DF &Vector4DF::operator-= (const Vector3DC &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
inline Vector4DF &Vector4DF::operator-= (const Vector3DI &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
inline Vector4DF &Vector4DF::operator-= (const Vector3DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
inline Vector4DF &Vector4DF::operator-= (const Vector4DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; w-=(VTYPE) op.w; return *this;}	

inline Vector4DF &Vector4DF::operator*= (const int op) {x*= (VTYPE) op; y*= (VTYPE) op; z*= (VTYPE) op; w *= (VTYPE) op; return *this;}
inline Vector4DF &Vector4DF::operator*= (const double op) {x*= (VTYPE) op; y*= (VTYPE) op; z*= (VTYPE) op; w *= (VTYPE) op; return *this;}
inline Vector4DF &Vector4DF::operator*= (const Vector2DC &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
inline Vector4DF &Vector4DF::operator*= (const Vector2DI &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
inline Vector4DF &Vector4DF::operator*= (const Vector2DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
inline Vector4DF &Vector4DF::operator*= (const Vector3DC &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}
inline Vector4DF &Vector4DF::operator*= (const Vector3DI &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}
inline Vector4DF &Vector4DF::operator*= (const Vector3DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}
inline Vector4DF &Vector4DF::operator*= (const Vector4DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; w*=(VTYPE) op.w; return *this;}	

inline Vector4DF &Vector4DF::operator/= (const int op) {x/= (VTYPE) op; y/= (VTYPE) op; z/= (VTYPE) op; w /= (VTYPE) op; return *this;}
inline Vector4DF &Vector4DF::operator/= (const double op) {x/= (VTYPE) op; y/= (VTYPE) op; z/= (VTYPE) op; w /= (VTYPE) op; return *this;}
inline Vector4DF &Vector4DF::operator/= (const Vector2DC &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
inline Vector4DF &Vector4DF::operator/= (const Vector2DI &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
inline Vector4DF &Vector4DF::operator/= (const Vector2DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
inline Vector4DF &Vector4DF::operator/= (const Vector3DC &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}
inline Vector4DF &Vector4DF::operator/= (const Vector3DI &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}
inline Vector4DF &Vector4DF::operator/= (const Vector3DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}
inline Vector4DF &Vector4DF::operator/= (const Vector4DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; w/=(VTYPE) op.w; return *this;}	

inline Vector4DF &Vector4DF::Cross (const Vector4DF &v) {double ax = x, ay = y, az = z, aw = w; x = (VTYPE) (ay * (double) v.z - az * (double) v.y); y = (VTYPE) (-ax * (double) v.z + az * (double) v.x); z = (VTYPE) (ax * (double) v.y - ay * (double) v.x); w = (VTYPE) 0; return *this;}
		
inline double Vector4DF::Dot (const Vector4DF &v)			{double dot; dot = (double) x*v.x + (double) y*v.y + (double) z*v.z + (double) w*v.w; return dot;}

inline double Vector4DF::Dist (const Vector4DF &v)		{double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}

inline double Vector4DF::DistSq (const Vector4DF &v)		{double a,b,c,d; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z - (double) v.z; d = (double) w - (double) v.w; return (a*a + b*b + c*c + d*d);}

inline Vector4DF &Vector4DF::Normalize (void) {
	VTYPE n = (VTYPE) x*x + (VTYPE) y*y + (VTYPE) z*z + (VTYPE) w*w;
	if (n!=0.0) {
		n = sqrt(n);
		x /= n; y /= n; z /= n; w /= n;
	}
	return *this;
}
inline double Vector4DF::Length (void) { double n; n = (double) x*x + (double) y*y + (double) z*z + (double) w*w; if (n != 0.0) return sqrt(n); return 0.0; }
inline VTYPE *Vector4DF::Data (void)			{return &x;}

#undef VTYPE
#undef VNAME
