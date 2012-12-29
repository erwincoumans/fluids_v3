

//*********** NOTE
//
// LOOK AT MovieTrackPoint. IN ORDER FOR VECTORS AND MATRICIES TO BE USED IN OBJECTS 
// THAT WILL BE USED IN stl::vectors, THEIR CONSTRUCTORS AND OPERATORS MUST TAKE ONLY
// const PARAMETERS. LOOK AT MatrixF and Vector2DF.. THIS WAS NOT YET DONE WITH
// THE OTHER MATRIX AND VECTOR CLASSES (Vector2DC, Vector2DI, MatrixC, MatrixI, ...)
//


#ifndef MATRIX_DEF
	#define MATRIX_DEF
		
	#include <stdio.h>
	#include <iostream>
	#include <memory.h>
	#include <math.h>
	#include <string>

	#include "vector.h"
	#include "mdebug.h"	
	
	//#define MATRIX_INITIALIZE				// Initializes vectors	

	class MatrixC;							// Forward Referencing
	class MatrixI;
	class MatrixF;

	#define LUNA_CORE

	class LUNA_CORE Matrix {
	public:
		// Member Virtual Functions		
		virtual Matrix &operator= (unsigned char c)=0;
		virtual Matrix &operator= (int c)=0;
		virtual Matrix &operator= (double c)=0;		
		virtual Matrix &operator= (MatrixC &op)=0;
		virtual Matrix &operator= (MatrixI &op)=0;
		virtual Matrix &operator= (MatrixF &op)=0;
		
		virtual Matrix &operator+= (unsigned char c)=0;
		virtual Matrix &operator+= (int c)=0;
		virtual Matrix &operator+= (double c)=0;		
		virtual Matrix &operator+= (MatrixC &op)=0;
		virtual Matrix &operator+= (MatrixI &op)=0;
		virtual Matrix &operator+= (MatrixF &op)=0;

		virtual Matrix &operator-= (unsigned char c)=0;
		virtual Matrix &operator-= (int c)=0;
		virtual Matrix &operator-= (double c)=0;		
		virtual Matrix &operator-= (MatrixC &op)=0;
		virtual Matrix &operator-= (MatrixI &op)=0;
		virtual Matrix &operator-= (MatrixF &op)=0;

		virtual Matrix &operator*= (unsigned char c)=0;
		virtual Matrix &operator*= (int c)=0;
		virtual Matrix &operator*= (double c)=0;		
		virtual Matrix &operator*= (MatrixC &op)=0;
		virtual Matrix &operator*= (MatrixI &op)=0;
		virtual Matrix &operator*= (MatrixF &op)=0;

		virtual Matrix &operator/= (unsigned char c)=0;
		virtual Matrix &operator/= (int c)=0;
		virtual Matrix &operator/= (double c)=0;		
		virtual Matrix &operator/= (MatrixC &op)=0;
		virtual Matrix &operator/= (MatrixI &op)=0;
		virtual Matrix &operator/= (MatrixF &op)=0;

		virtual Matrix &Multiply (MatrixF &op)=0;
		virtual Matrix &Resize (int x, int y)=0;
		virtual Matrix &ResizeSafe (int x, int y)=0;
		virtual Matrix &InsertRow (int r)=0;
		virtual Matrix &InsertCol (int c)=0;
		virtual Matrix &Transpose (void)=0;
		virtual Matrix &Identity (int order)=0;
		/*inline Matrix &RotateX (double ang);
		inline Matrix &RotateY (double ang);
		inline Matrix &RotateZ (double ang); */
		virtual Matrix &Basis (Vector3DF &c1, Vector3DF &c2, Vector3DF &c3)=0;
		virtual Matrix &GaussJordan (MatrixF &b)		{ return *this; }
		virtual Matrix &ConjugateGradient (MatrixF &b)	{ return *this; }

		virtual int GetRows(void)=0;
		virtual int GetCols(void)=0;
		virtual int GetLength(void)=0;		

		virtual unsigned char *GetDataC (void)=0;
		virtual int	*GetDataI (void)=0;
		virtual double *GetDataF (void)=0;
	};
	
	// MatrixC Declaration	
	#define VNAME		C
	#define VTYPE		unsigned char

	class LUNA_CORE MatrixC {
	public:
		VTYPE *data;
		int rows, cols, len;		

		// Constructors/Destructors
		inline MatrixC ();
		inline ~MatrixC ();
		inline MatrixC (int r, int c);

		// Member Functions
		inline VTYPE &operator () (int c, int r);
		inline MatrixC &operator= (unsigned char c);
		inline MatrixC &operator= (int c);
		inline MatrixC &operator= (double c);		
		inline MatrixC &operator= (MatrixC &op);
		inline MatrixC &operator= (MatrixI &op);
		inline MatrixC &operator= (MatrixF &op);
		
		inline MatrixC &operator+= (unsigned char c);
		inline MatrixC &operator+= (int c);
		inline MatrixC &operator+= (double c);		
		inline MatrixC &operator+= (MatrixC &op);
		inline MatrixC &operator+= (MatrixI &op);
		inline MatrixC &operator+= (MatrixF &op);

		inline MatrixC &operator-= (unsigned char c);
		inline MatrixC &operator-= (int c);
		inline MatrixC &operator-= (double c);		
		inline MatrixC &operator-= (MatrixC &op);
		inline MatrixC &operator-= (MatrixI &op);
		inline MatrixC &operator-= (MatrixF &op);

		inline MatrixC &operator*= (unsigned char c);
		inline MatrixC &operator*= (int c);
		inline MatrixC &operator*= (double c);		
		inline MatrixC &operator*= (MatrixC &op);
		inline MatrixC &operator*= (MatrixI &op);
		inline MatrixC &operator*= (MatrixF &op);

		inline MatrixC &operator/= (unsigned char c);
		inline MatrixC &operator/= (int c);
		inline MatrixC &operator/= (double c);		
		inline MatrixC &operator/= (MatrixC &op);
		inline MatrixC &operator/= (MatrixI &op);
		inline MatrixC &operator/= (MatrixF &op);

		inline MatrixC &Multiply (MatrixF &op);
		inline MatrixC &Resize (int x, int y);
		inline MatrixC &ResizeSafe (int x, int y);
		inline MatrixC &InsertRow (int r);
		inline MatrixC &InsertCol (int c);
		inline MatrixC &Transpose (void);
		inline MatrixC &Identity (int order);		
		inline MatrixC &Basis (Vector3DF &c1, Vector3DF &c2, Vector3DF &c3);
		inline MatrixC &GaussJordan (MatrixF &b);

		inline int GetX();
		inline int GetY();	
		inline int GetRows(void);
		inline int GetCols(void);
		inline int GetLength(void);
		inline VTYPE *GetData(void);

		inline unsigned char *GetDataC (void)	{return data;}
		inline int *GetDataI (void)				{return NULL;}
		inline double *GetDataF (void)			{return NULL;}		

		inline double GetF (int r, int c);
	};
	#undef VNAME
	#undef VTYPE

	// MatrixI Declaration	
	#define VNAME		I
	#define VTYPE		int

	class LUNA_CORE MatrixI {
	public:
		VTYPE *data;
		int rows, cols, len;		
	
		// Constructors/Destructors
		inline MatrixI ();
		inline ~MatrixI ();
		inline MatrixI (int r, int c);

		// Member Functions
		inline VTYPE &operator () (int c, int r);
		inline MatrixI &operator= (unsigned char c);
		inline MatrixI &operator= (int c);
		inline MatrixI &operator= (double c);		
		inline MatrixI &operator= (MatrixC &op);
		inline MatrixI &operator= (MatrixI &op);
		inline MatrixI &operator= (MatrixF &op);
		
		inline MatrixI &operator+= (unsigned char c);
		inline MatrixI &operator+= (int c);
		inline MatrixI &operator+= (double c);		
		inline MatrixI &operator+= (MatrixC &op);
		inline MatrixI &operator+= (MatrixI &op);
		inline MatrixI &operator+= (MatrixF &op);

		inline MatrixI &operator-= (unsigned char c);
		inline MatrixI &operator-= (int c);
		inline MatrixI &operator-= (double c);		
		inline MatrixI &operator-= (MatrixC &op);
		inline MatrixI &operator-= (MatrixI &op);
		inline MatrixI &operator-= (MatrixF &op);

		inline MatrixI &operator*= (unsigned char c);
		inline MatrixI &operator*= (int c);
		inline MatrixI &operator*= (double c);		
		inline MatrixI &operator*= (MatrixC &op);
		inline MatrixI &operator*= (MatrixI &op);
		inline MatrixI &operator*= (MatrixF &op);

		inline MatrixI &operator/= (unsigned char c);
		inline MatrixI &operator/= (int c);
		inline MatrixI &operator/= (double c);		
		inline MatrixI &operator/= (MatrixC &op);
		inline MatrixI &operator/= (MatrixI &op);
		inline MatrixI &operator/= (MatrixF &op);

		inline MatrixI &Multiply (MatrixF &op);
		inline MatrixI &Resize (int x, int y);
		inline MatrixI &ResizeSafe (int x, int y);
		inline MatrixI &InsertRow (int r);
		inline MatrixI &InsertCol (int c);
		inline MatrixI &Transpose (void);
		inline MatrixI &Identity (int order);		
		inline MatrixI &Basis (Vector3DF &c1, Vector3DF &c2, Vector3DF &c3);
		inline MatrixI &GaussJordan (MatrixF &b);

		inline int GetX();
		inline int GetY();	
		inline int GetRows(void);
		inline int GetCols(void);
		inline int GetLength(void);
		inline VTYPE *GetData(void);

		inline unsigned char *GetDataC (void)	{return NULL;}
		inline int *GetDataI (void)				{return data;}
		inline double *GetDataF (void)			{return NULL;}
		
		inline double GetF (int r, int c);
	};
	#undef VNAME
	#undef VTYPE

	// MatrixF Declaration	
	#define VNAME		F
	#define VTYPE		double

	class LUNA_CORE MatrixF {
	public:	
		VTYPE *data;
		int rows, cols, len;		

		// Constructors/Destructors		
		inline MatrixF ();
		inline ~MatrixF ();
		inline MatrixF (const int r, const int c);

		// Member Functions
		inline VTYPE GetVal ( int c, int r );
		inline VTYPE &operator () (const int c, const int r);
		inline MatrixF &operator= (const unsigned char c);
		inline MatrixF &operator= (const int c);
		inline MatrixF &operator= (const double c);		
		inline MatrixF &operator= (const MatrixC &op);
		inline MatrixF &operator= (const MatrixI &op);
		inline MatrixF &operator= (const MatrixF &op);
		
		inline MatrixF &operator+= (const unsigned char c);
		inline MatrixF &operator+= (const int c);
		inline MatrixF &operator+= (const double c);		
		inline MatrixF &operator+= (const MatrixC &op);
		inline MatrixF &operator+= (const MatrixI &op);
		inline MatrixF &operator+= (const MatrixF &op);

		inline MatrixF &operator-= (const unsigned char c);
		inline MatrixF &operator-= (const int c);
		inline MatrixF &operator-= (const double c);		
		inline MatrixF &operator-= (const MatrixC &op);
		inline MatrixF &operator-= (const MatrixI &op);
		inline MatrixF &operator-= (const MatrixF &op);

		inline MatrixF &operator*= (const unsigned char c);
		inline MatrixF &operator*= (const int c);
		inline MatrixF &operator*= (const double c);		
		inline MatrixF &operator*= (const MatrixC &op);
		inline MatrixF &operator*= (const MatrixI &op);
		inline MatrixF &operator*= (const MatrixF &op);		

		inline MatrixF &operator/= (const unsigned char c);
		inline MatrixF &operator/= (const int c);
		inline MatrixF &operator/= (const double c);		
		inline MatrixF &operator/= (const MatrixC &op);
		inline MatrixF &operator/= (const MatrixI &op);
		inline MatrixF &operator/= (const MatrixF &op);

		inline MatrixF &Multiply4x4 (const MatrixF &op);
		inline MatrixF &Multiply (const MatrixF &op);
		inline MatrixF &Resize (const int x, const int y);
		inline MatrixF &ResizeSafe (const int x, const int y);
		inline MatrixF &InsertRow (const int r);
		inline MatrixF &InsertCol (const int c);
		inline MatrixF &Transpose (void);
		inline MatrixF &Identity (const int order);
		inline MatrixF &RotateX (const double ang);
		inline MatrixF &RotateY (const double ang);
		inline MatrixF &RotateZ (const double ang);
		inline MatrixF &Ortho (double sx, double sy, double n, double f);		
		inline MatrixF &Translate (double tx, double ty, double tz);
		inline MatrixF &Basis (const Vector3DF &c1, const Vector3DF &c2, const Vector3DF &c3);
		inline MatrixF &GaussJordan (MatrixF &b);
		inline MatrixF &ConjugateGradient (MatrixF &b);
		inline MatrixF &Submatrix ( MatrixF& b, int mx, int my);
		inline MatrixF &MatrixVector5 (MatrixF& x, int mrows, MatrixF& b );
		inline MatrixF &ConjugateGradient5 (MatrixF &b, int mrows );
		inline double Dot ( MatrixF& b );

		inline void Print ( char* fname );

		inline int GetX();
		inline int GetY();	
		inline int GetRows(void);
		inline int GetCols(void);
		inline int GetLength(void);
		inline VTYPE *GetData(void);
		inline void GetRowVec (int r, Vector3DF &v);

		inline unsigned char *GetDataC (void) const	{return NULL;}
		inline int *GetDataI (void)	const			{return NULL;}
		inline double *GetDataF (void) const		{return data;}

		inline double GetF (const int r, const int c);
	};
	#undef VNAME
	#undef VTYPE

	// MatrixF Declaration	
	#define VNAME		F
	#define VTYPE		float

	class LUNA_CORE Matrix4F {
	public:	
		VTYPE	data[16];		

		// Constructors/Destructors
		inline Matrix4F ( float* dat );
		inline Matrix4F ();		
		inline Matrix4F ( float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7, float f8, float f9, float f10, float f11,	float f12, float f13, float f14, float f15 );

		// Member Functions
		inline VTYPE &operator () (const int n)					{ return data[n]; }
		inline VTYPE &operator () (const int c, const int r)	{ return data[ (r<<2)+c ]; }		
		inline Matrix4F &operator= (const unsigned char c);
		inline Matrix4F &operator= (const int c);
		inline Matrix4F &operator= (const double c);				
		inline Matrix4F &operator+= (const unsigned char c);
		inline Matrix4F &operator+= (const int c);
		inline Matrix4F &operator+= (const double c);				
		inline Matrix4F &operator-= (const unsigned char c);
		inline Matrix4F &operator-= (const int c);
		inline Matrix4F &operator-= (const double c);
		inline Matrix4F &operator*= (const unsigned char c);
		inline Matrix4F &operator*= (const int c);
		inline Matrix4F &operator*= (const double c);
		inline Matrix4F &operator/= (const unsigned char c);
		inline Matrix4F &operator/= (const int c);
		inline Matrix4F &operator/= (const double c);		

		inline Matrix4F &operator=  (const float* op);
		inline Matrix4F &operator*= (const Matrix4F& op);
		inline Matrix4F &operator*= (const float* op);	

		inline Matrix4F &PreTranslate (const Vector3DF& t);
		inline Matrix4F &operator+= (const Vector3DF& t);		// quick translate
		inline Matrix4F &operator*= (const Vector3DF& t);		// quick scale
		
		inline Matrix4F &Transpose (void);
		inline Matrix4F &Identity ();
		inline Matrix4F &RotateZYX ( const Vector3DF& angs );
		inline Matrix4F &RotateZYXT (const Vector3DF& angs, const Vector3DF& t);
		inline Matrix4F &RotateTZYX (const Vector3DF& angs, const Vector3DF& t);
		inline Matrix4F &RotateX (const double ang);
		inline Matrix4F &RotateY (const double ang);
		inline Matrix4F &RotateZ (const double ang);
		inline Matrix4F &Ortho (double sx, double sy, double n, double f);		
		inline Matrix4F &Translate (double tx, double ty, double tz);
		inline Matrix4F &Scale (double sx, double sy, double sz);
		inline Matrix4F &Basis (const Vector3DF &yaxis);
		inline Matrix4F &Basis (const Vector3DF &c1, const Vector3DF &c2, const Vector3DF &c3);		
		inline Matrix4F &InvertTRS ();

		inline void Print ();
		inline std::string WriteToStr ();

		inline Matrix4F operator* (const float &op);	
		inline Matrix4F operator* (const Vector3DF &op);	

		// Scale-Rotate-Translate (compound matrix)
		inline Matrix4F &SRT (const Vector3DF &c1, const Vector3DF &c2, const Vector3DF &c3, const Vector3DF& t, const Vector3DF& s);
		inline Matrix4F &SRT (const Vector3DF &c1, const Vector3DF &c2, const Vector3DF &c3, const Vector3DF& t, const float s);

		// invTranslate-invRotate-invScale (compound matrix)
		inline Matrix4F &InvTRS (const Vector3DF &c1, const Vector3DF &c2, const Vector3DF &c3, const Vector3DF& t, const Vector3DF& s);
		inline Matrix4F &InvTRS (const Vector3DF &c1, const Vector3DF &c2, const Vector3DF &c3, const Vector3DF& t, const float s);

		inline Matrix4F &operator= ( float* mat);
		inline Matrix4F &InverseProj ( float* mat );
		inline Matrix4F &InverseView ( float* mat, Vector3DF& pos );
		inline Vector4DF GetT ( float* mat );

		inline int GetX()			{ return 4; }
		inline int GetY()			{ return 4; }
		inline int GetRows(void)	{ return 4; }
		inline int GetCols(void)	{ return 4; }	
		inline int GetLength(void)	{ return 16; }
		inline VTYPE *GetData(void)	{ return data; }
		inline void GetRowVec (int r, Vector3DF &v);

		inline unsigned char *GetDataC (void) const	{return NULL;}
		inline int *GetDataI (void)	const			{return NULL;}
		inline float *GetDataF (void) const		{return (float*) data;}

		inline float GetF (const int r, const int c);
	};
	#undef VNAME
	#undef VTYPE


    // Matrix Code Definitions (Inlined)

	#include "matrix_inline.h"

#endif



