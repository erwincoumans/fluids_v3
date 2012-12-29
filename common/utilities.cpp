

#include "vector.h"
#include "utilities.h"

#include <sstream>

#include <stdlib.h>

// Math utilities --------------------------------------------------------------------------------------

static unsigned long x=123456789, y=362436069, z=521288629;

unsigned long xorshf96 (void) {          //period 2^96-1
	unsigned long t;
    x ^= x << 16;
    x ^= x >> 5;
    x ^= x << 1;
   t = x;
   x = y;
   y = z;
   z = t ^ x ^ y;
  return z;
}
void random_seed ( unsigned long a )
{
	x = a;
	y = 362436069;
	z = 521288629;
}
float random (float xmin, float xmax)
{
	return xmin + (xorshf96()*(xmax-xmin)) / 4294967295.0f ;
}
Vector3DI vecmin ( Vector3DI a, Vector3DI b )
{
	Vector3DI c;
	c.x = fmin ( a.x, b.x );
	c.y = fmin ( a.y, b.y );
	c.z = fmin ( a.z, b.z );
	return c;
}
Vector3DI vecmax ( Vector3DI a, Vector3DI b )
{
	Vector3DI c;
	c.x = fmax ( a.x, b.x );
	c.y = fmax ( a.y, b.y );
	c.z = fmax ( a.z, b.z );
	return c;
}

/*float cotf(float a)  { return cosf(a)/sinf(a); };
float fractf(float a) { return a - floorf(a); };
float sqrf(float a) { return a*a; };
int sqri(int a) { return a*a; };
int powi(int base, int exp) { return int(powf((float)base, (float)exp)); };
float logf(float base, float x) { return logf(x)/logf(base); };
int logi(int base, int x) { return int(logf((float)x)/logf((float)base)); };
float signf(float x) { if (x > 0) return 1; else if (x < 0) return -1; else return 0; };
float maxf(float a, float b) { if (a < b) return b; else return a; };
float minf(float a, float b) { if (a < b) return a; else return b; };
int maxi(int a, int b) { if (a < b) return b; else return a; };
int mini(int a, int b) { if (a < b) return a; else return b; };
int mini(int a, int b, int c) { if (a < b) { if(a < c) return a; else return c; } else { if (b < c) return b; else return c; } };
int absi(int a) { if (a < 0) return -a; else return a; };
short abss(short a) { if (a < 0) return -a; else return a; };
void swapi(int & a, int & b) { int t = a; a = b; b = t; };
int floori(float a) { return int(floor(a)); };
int ceili(float a) { return int(ceil(a)); };*/

// File utilities --------------------------------------------------------------------------------------

int freadi(FILE * fIn)
{
	int iTemp;
	fread(&iTemp, sizeof(int), 1, fIn);
	return iTemp;
}

float freadf(FILE * fIn)
{
	float fTemp;
	fread(&fTemp, sizeof(float), 1, fIn);
	return fTemp;
}

std::string readword ( char* line, char delim )
{
	char word_buf[8192];

	if ( readword ( line, word_buf, delim ) ) return word_buf;

	return "";
}


bool readword ( char *line, char *word, char delim )
{
	int max_size = 200;
	char *buf_pos;
	char *start_pos;	

	// read past spaces/tabs, or until end of line/string
	for (buf_pos=line; (*buf_pos==' ' || *buf_pos=='\t') && *buf_pos!='\n' && *buf_pos!='\0';)
		buf_pos++;
	
	// if end of line/string found, then no words found, return null
	if (*buf_pos=='\n' || *buf_pos=='\0') {*word = '\0'; return false;}

	// mark beginning of word, read until end of word
	for (start_pos = buf_pos; *buf_pos != delim && *buf_pos!='\t' && *buf_pos!='\n' && *buf_pos!='\0';)
		buf_pos++;
	
	if (*buf_pos=='\n' || *buf_pos=='\0') {	// buf_pos now points to the end of buffer
		//strcpy_s (word, max_size, start_pos);			// copy word to output string
        strncpy (word, start_pos, max_size);
		if ( *buf_pos=='\n') *(word + strlen(word)-1) = '\0';
		*line = '\0';						// clear input buffer
	} else {
											// buf_pos now points to the delimiter after word
		*buf_pos++ = '\0';					// replace delimiter with end-of-word marker
		//strcpy_s (word, max_size, start_pos);
		strncpy (word, start_pos, buf_pos-line );	// copy word(s) string to output string			
											// move start_pos to beginning of entire buffer
		strcpy ( start_pos, buf_pos );		// copy remainder of buffer to beginning of buffer
	}
	return true;						// return word(s) copied	
}


// String utilities --------------------------------------------------------------------------------------

objType strToType (	std::string str )
{
	char buf[5];
	strcpy ( buf, str.c_str() );
	objType name;
	((char*) &name)[3] = buf[0];
	((char*) &name)[2] = buf[1];
	((char*) &name)[1] = buf[2];
	((char*) &name)[0] = buf[3];
	return name;
}

std::string typeToStr ( objType name )			// static function
{
	char buf[5];	
	buf[0] = ((char*) &name)[3];
	buf[1] = ((char*) &name)[2];
	buf[2] = ((char*) &name)[1];
	buf[3] = ((char*) &name)[0];
	buf[4] = '\0';
	return std::string ( buf );
}

std::string cToStr ( char c )
{
	char buf[2];
	buf[0] = c;
	buf[1] = '\0';
	return std::string ( buf );
}

std::string iToStr ( int i )
{
	std::ostringstream ss;
	ss << i;
	return ss.str();
}
std::string fToStr ( float f )
{
	std::ostringstream ss;
	ss << f;
	return ss.str();
}

int strToI (std::string s) {
	//return ::atoi ( s.c_str() );
	std::istringstream str_stream ( s ); 
	int x; 
	if (str_stream >> x) return x;		// this is the correct way to convert std::string to int, do not use atoi
	return 0;
};
float strToF (std::string s) {
	//return ::atof ( s.c_str() );
	std::istringstream str_stream ( s ); 
	float x; 
	if (str_stream >> x) return x;		// this is the correct way to convert std::string to float, do not use atof
	return 0;
};
unsigned char strToC ( std::string s ) {
	char c;
	memcpy ( &c, s.c_str(), 1 );		// cannot use atoi here. atoi only returns numbers for strings containing ascii numbers.
	return c;
};
std::string strParse ( std::string& str, std::string lsep, std::string rsep )
{
	std::string result;
	size_t lfound, rfound;

	lfound = str.find_first_of ( lsep );
	if ( lfound != std::string::npos) {
		rfound = str.find_first_of ( rsep, lfound+1 );
		if ( rfound != std::string::npos ) {
			result = str.substr ( lfound+1, rfound-lfound-1 );					// return string strickly between lsep and rsep
			str = str.substr ( 0, lfound ) + str.substr ( rfound+1 );
			return result;
		} 
	}
	return "";
}

std::string strSplit ( std::string& str, std::string sep )
{
	std::string result;
	size_t found;

	found = str.find_first_of ( sep );
	if ( found != std::string::npos) {
		result = str.substr ( 0, found );
		str = str.substr ( found+1 );
	} else {
		result = str;
		str = "";
	}
	return result;
}
std::string strReplace ( std::string str, std::string delim, std::string ins )
{
	size_t found = str.find_first_of ( delim );
	while ( found != std::string::npos ) {
		str = str.substr ( 0, found ) + ins + str.substr ( found+1 );
		found = str.find_first_of ( delim );
	}
	return str;
}
std::string strTrim ( std::string str, std::string ch )
{
	size_t found = str.find_first_not_of ( ch );
	if ( found != std::string::npos ) str = str.substr ( found );
	found = str.find_last_not_of ( ch );
	if ( found != std::string::npos ) str = str.substr ( 0, found+1 );

	return str;
}

std::string wsToStr ( const std::wstring& str )
{
#ifdef _MSC_VER
	int len = WideCharToMultiByte ( CP_ACP, 0, str.c_str(), str.length(), 0, 0, 0, 0);
	char* buf = new char[ len+1 ];
	memset ( buf, '\0', len+1 );
	WideCharToMultiByte ( CP_ACP, 0, str.c_str(), str.length(), buf, len+1, 0, 0);	
#else
    int len = wcstombs( NULL, str.c_str(), 0 );
	char* buf = new char[ len ];
	wcstombs( buf, str.c_str(), len );
#endif
	std::string r(buf);
	delete[] buf;
	return r;
}

std::wstring strToWs (const std::string& s)
{
	wchar_t* buf = new wchar_t[ s.length()+1 ];

#ifdef _MSC_VER
	MultiByteToWideChar(CP_ACP, 0, s.c_str(), -1, buf, (int) s.length()+1 );
#else
    mbstowcs( buf, s.c_str(), s.length() + 1 );
#endif
	std::wstring r(buf);
	delete[] buf;
	return r;
}


// Geometry utilities --------------------------------------------------------------------------------------

#define EPS		0.00001

Vector3DF intersectLineLine (  Vector3DF p1, Vector3DF p2, Vector3DF p3, Vector3DF p4 )
{
	Vector3DF pa, pb;
	double ma, mb;
	if ( intersectLineLine ( p1, p2, p3, p4, pa, pb, ma, mb ) ) {
		return pa;
	}
	return p2;
}


// Line A: p1 to p2
// Line B: p3 to p4
bool intersectLineLine ( Vector3DF p1, Vector3DF p2, Vector3DF p3, Vector3DF p4, Vector3DF& pa, Vector3DF& pb, double& mua, double& mub)
{
   Vector3DF p13,p43,p21;
   double d1343,d4321,d1321,d4343,d2121;
   double numer,denom;

   p13 = p1;	p13 -= p3;   
   p43 = p4;	p43 -= p3;
   if (fabs(p43.x) < EPS && fabs(p43.y) < EPS && fabs(p43.z) < EPS) return false;
   p21 = p2;	p21 -= p1;
   if (fabs(p21.x) < EPS && fabs(p21.y) < EPS && fabs(p21.z) < EPS) return false;

   d1343 = p13.Dot ( p43 );
   d4321 = p43.Dot ( p21 );
   d1321 = p13.Dot ( p21 );
   d4343 = p43.Dot ( p43 );
   d2121 = p21.Dot ( p21 );

   denom = d2121 * d4343 - d4321 * d4321;
   if (fabs(denom) < EPS) return false;
   numer = d1343 * d4321 - d1321 * d4343;

   mua = numer / denom;
   mub = (d1343 + d4321 * (mua)) / d4343;

   pa = p21;	pa *= mua;		pa += p1;
   pb = p43;	pb *= mub;		pb += p3;
   
   return true;
}

Vector3DF intersectLineBox ( Vector3DF p1, Vector3DF p2, Vector3DF bmin, Vector3DF bmax )
{
	Vector3DF p;
	Vector3DF nearp, farp;
	float t[6];
	Vector3DF dir;
	dir = p2; dir -= p1;
	
	int bst1 = -1, bst2 = -1;		// bst1 = front face hit, bst2 = back face hit

	t[0] = ( bmax.y - p1.y ) / dir.y;			// 0 = max y
	t[1] = ( bmin.x - p1.x ) / dir.x;			// 1 = min x
	t[2] = ( bmin.z - p1.z ) / dir.z;			// 2 = min z
	t[3] = ( bmax.x - p1.x ) / dir.x;			// 3 = max x
	t[4] = ( bmax.z - p1.z ) / dir.z;			// 4 = max z
	t[5] = ( bmin.y - p1.y ) / dir.y;			// 5 = min y
	
    p = dir * t[0]; p += p1;    if ( p.x < bmin.x || p.x > bmax.x || p.z < bmin.z || p.z > bmax.z ) t[0] = -1;
    p = dir * t[1]; p += p1;    if ( p.y < bmin.y || p.y > bmax.y || p.z < bmin.z || p.z > bmax.z ) t[1] = -1;
    p = dir * t[2]; p += p1;    if ( p.x < bmin.x || p.x > bmax.x || p.y < bmin.y || p.y > bmax.y ) t[2] = -1;
    p = dir * t[3]; p += p1;    if ( p.y < bmin.y || p.y > bmax.y || p.z < bmin.z || p.z > bmax.z ) t[3] = -1;
    p = dir * t[4]; p += p1;    if ( p.x < bmin.x || p.x > bmax.x || p.y < bmin.y || p.y > bmax.y ) t[4] = -1;
    p = dir * t[5]; p += p1;    if ( p.x < bmin.x || p.x > bmax.x || p.z < bmin.z || p.z > bmax.z ) t[5] = -1;

	for (int j=0; j < 6; j++) 
		if ( t[j] > 0.0 && ( t[j] < t[bst1] || bst1==-1 ) ) bst1=j;
	for (int j=0; j < 6; j++)
		if ( t[j] > 0.0 && ( t[j] < t[bst2] || bst2==-1 ) && j!=bst1 ) bst2=j;

	if ( bst1 == -1 ) return p1;

    nearp = dir * t[bst1];
    nearp += p1;
    
    farp = dir * t[bst2];
    farp += p1;
	
	if ( p1.x >= bmin.x && p1.y >= bmin.y && p1.z >= bmin.z && p1.x <= bmax.x && p1.y <= bmax.y && p1.z <= bmax.z ) {
		return farp;
	}
	return nearp;
}

Vector3DF intersectLinePlane ( Vector3DF p1, Vector3DF p2, Vector3DF p0, Vector3DF pnorm )
{
	Vector3DF u, w;
	u = p2;	u -= p1;					// ray direction
	w = p1;	w -= p0;

    float dval = pnorm.Dot( u );
    float nval = -pnorm.Dot( w );

    if (fabs(dval) < EPS ) {			// segment is parallel to plane
        if (nval == 0) return p1;       // segment lies in plane
		else			return p1;      // no intersection
    }
    // they are not parallel, compute intersection
    float t = nval / dval;
    u *= t;
	u += p1;
    return u;
}

