
#include "vector.h"
#include "matrix.h"

#define VTYPE	float

Vector3DF &Vector3DF::operator*= (const MatrixF &op)
{
	double *m = op.GetDataF ();
	float xa, ya, za;
	xa = x * float(*m++);	ya = x * float(*m++);	za = x * float(*m++);	m++;
	xa += y * float(*m++);	ya += y * float(*m++);	za += y * float(*m++);	m++;
	xa += z * float(*m++);	ya += z * float(*m++);	za += z * float(*m++);	m++;
	xa += float(*m++);		ya += float(*m++);		za += float(*m++);
	x = xa; y = ya; z = za;
	return *this;
}

// p' = Mp
Vector3DF &Vector3DF::operator*= (const Matrix4F &op)
{
	float xa, ya, za;
	xa = x * op.data[0] + y * op.data[4] + z * op.data[8] + op.data[12];
	ya = x * op.data[1] + y * op.data[5] + z * op.data[9] + op.data[13];
	za = x * op.data[2] + y * op.data[6] + z * op.data[10] + op.data[14];
	x = xa; y = ya; z = za;
	return *this;
}

	
#define min3(a,b,c)		( (a<b) ? ((a<c) ? a : c) : ((b<c) ? b : c) )
#define max3(a,b,c)		( (a>b) ? ((a>c) ? a : c) : ((b>c) ? b : c) )

Vector3DF Vector3DF::RGBtoHSV ()
{
	float h,s,v;
	float minv, maxv;
	int i;
	float f;

	minv = min3(x, y, z);
	maxv = max3(x, y, z);
	if (minv==maxv) {
		v = (float) maxv;
		h = 0.0; 
		s = 0.0;			
	} else {
		v = (float) maxv;
		s = (maxv - minv) / maxv;
		f = (x == minv) ? y - z : ((y == minv) ? z - x : x - y); 	
		i = (x == minv) ? 3 : ((y == minv) ? 5 : 1);
		h = (i - f / (maxv - minv) ) / 6.0f;	
	}
	return Vector3DF(h,s,v);
}

Vector3DF Vector3DF::HSVtoRGB ()
{
	double m, n, f;
	int i = floor ( x*6.0 );
	f = x*6.0 - i;
	if ( i % 2 == 0 ) f = 1.0 - f;	
	m = z * (1.0 - y );
	n = z * (1.0 - y * f );	
	switch ( i ) {
	case 6: 
	case 0: return Vector3DF( z, n, m );	break;
	case 1: return Vector3DF( n, z, m );	break;
	case 2: return Vector3DF( m, z, n );	break;
	case 3: return Vector3DF( m, n, z );	break;
	case 4: return Vector3DF( n, m, z );	break;
	case 5: return Vector3DF( z, m, n );	break;
	};
	return Vector3DF(1,1,1);
}

Vector4DF &Vector4DF::operator*= (const MatrixF &op)
{
	double *m = op.GetDataF ();
	VTYPE xa, ya, za, wa;
	xa = x * float(*m++);	ya = x * float(*m++);	za = x * float(*m++);	wa = x * float(*m++);
	xa += y * float(*m++);	ya += y * float(*m++);	za += y * float(*m++);	wa += y * float(*m++);
	xa += z * float(*m++);	ya += z * float(*m++);	za += z * float(*m++);	wa += z * float(*m++);
	xa += w * float(*m++);	ya += w * float(*m++);	za += w * float(*m++);	wa += w * float(*m++);
	x = xa; y = ya; z = za; w = wa;
	return *this;
}

Vector4DF &Vector4DF::operator*= (const Matrix4F &op)
{
	float xa, ya, za, wa;
	xa = x * op.data[0] + y * op.data[4] + z * op.data[8] + w * op.data[12];
	ya = x * op.data[1] + y * op.data[5] + z * op.data[9] + w * op.data[13];
	za = x * op.data[2] + y * op.data[6] + z * op.data[10] + w * op.data[14];
	wa = x * op.data[3] + y * op.data[7] + z * op.data[11] + w * op.data[15];
	x = xa; y = ya; z = za; w = wa;
	return *this;
}


Vector4DF &Vector4DF::operator*= (const float* op)
{
	float xa, ya, za, wa;
	xa = x * op[0] + y * op[4] + z * op[8] + w * op[12];
	ya = x * op[1] + y * op[5] + z * op[9] + w * op[13];
	za = x * op[2] + y * op[6] + z * op[10] + w * op[14];
	wa = x * op[3] + y * op[7] + z * op[11] + w * op[15];
	x = xa; y = ya; z = za; w = wa;
	return *this;
}
