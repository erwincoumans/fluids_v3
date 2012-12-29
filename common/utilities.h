

#ifndef DEF_UTILS
	#define DEF_UTILS


	#include <string>

	class Vector3DI;
	class Vector3DF;
	class ObjectX;
	
	#define LUNA_CORE
	#define	fmin(a,b)		( min (a,b) )
	#define fmax(a,b)		( max (a,b) )	
	typedef	unsigned short int		objType;
	typedef unsigned char			uchar;

	// Math utilities
	unsigned long xorshf96 (void);
	LUNA_CORE void random_seed ( unsigned long a );
	LUNA_CORE float random (float xmin, float xmax);
	inline float cotf(float a) { return cosf(a)/sinf(a); };
	inline float fractf(float a) { return a - floorf(a); };
	inline float sqrf(float a) { return a*a; };
	inline int sqri(int a) { return a*a; };
	inline int powi(int base, int exp) { return int(powf((float)base, (float)exp)); };
	inline float logf(float base, float x) { return logf(x)/logf(base); };
	inline int logi(int base, int x) { return int(logf((float)x)/logf((float)base)); };
	inline float signf(float x) { if (x > 0) return 1; else if (x < 0) return -1; else return 0; };
	inline float minf(float a, float b) { if (a < b) return a; else return b; };
	inline float maxf(float a, float b) { if (a < b) return b; else return a; };	
	inline int maxi(int a, int b) { if (a < b) return b; else return a; };
	inline int mini(int a, int b) { if (a < b) return a; else return b; };
	inline int mini(int a, int b, int c) { if (a < b) { if(a < c) return a; else return c; } else { if (b < c) return b; else return c; } };
	inline int absi(int a) { if (a < 0) return -a; else return a; };
	inline short abss(short a) { if (a < 0) return -a; else return a; };
	inline void swapi(int & a, int & b) { int t = a; a = b; b = t; };
	inline int floori(float a) { return int(floor(a)); };
	inline int ceili(float a) { return int(ceil(a)); };
	
	// File utilities
	LUNA_CORE int freadi(FILE * fIn);
	LUNA_CORE float freadf(FILE * fIn);
	LUNA_CORE bool readword ( char *line, char *word, char delim  );
	LUNA_CORE std::string readword ( char *line, char delim );

	// String utilities
	LUNA_CORE objType strToType ( std::string str );
	LUNA_CORE std::string typeToStr ( objType t );
	LUNA_CORE std::string cToStr ( char c );
	LUNA_CORE std::string iToStr ( int i );
	LUNA_CORE std::string fToStr ( float f );
	LUNA_CORE int strToI (std::string s);
	LUNA_CORE float strToF (std::string s);
	LUNA_CORE unsigned char strToC ( std::string s );
	LUNA_CORE std::string strParse ( std::string& str, std::string lsep, std::string rsep );
	LUNA_CORE std::string strSplit ( std::string& str, std::string sep );
	LUNA_CORE std::string strReplace ( std::string str, std::string delim, std::string ins );
	LUNA_CORE std::string strTrim ( std::string str, std::string ch );	
	LUNA_CORE std::string wsToStr ( const std::wstring& str );
	LUNA_CORE std::wstring strToWs (const std::string& s);

	// Geometry utilities
	LUNA_CORE Vector3DI vecmin ( Vector3DI a, Vector3DI b );
	LUNA_CORE Vector3DI vecmax ( Vector3DI a, Vector3DI b );
	LUNA_CORE Vector3DF intersectLineLine ( Vector3DF p1, Vector3DF p2, Vector3DF p3, Vector3DF p4 );
	LUNA_CORE Vector3DF intersectLineBox ( Vector3DF p1, Vector3DF p2, Vector3DF bmin, Vector3DF bmax );
	LUNA_CORE Vector3DF intersectLinePlane ( Vector3DF p1, Vector3DF p2, Vector3DF p0, Vector3DF pnorm );
	LUNA_CORE bool intersectLineLine ( Vector3DF p1, Vector3DF p2, Vector3DF p3, Vector3DF p4, Vector3DF& pa,Vector3DF& pb, double& mua, double& mub);

	
#endif
