

#ifndef DEF_EVENT_BUFFER_H
	#define DEF_EVENT_BUFFER_H

	#include "luna_config.h"

	#include <string>
	#include "event.h"
	#include "matrix.h"

	class LUNA_CORE EventBuffer {
	public:
		EventBuffer();		

		void resize ( int s );
		void expand ( int s );
		void initread ();

		void attachBool (bool b);
		void attachInt (int i);	
		void attachUInt ( unsigned int i );
		void attachULong ( unsigned long i );
		void attachVec3 ( Vector3DF& v );
		void attachVec4 ( Vector4DF& v );
		void attachM4 ( Matrix4F& m );
		void attachInt64 (xlong i);		
		void attachFloat (float f);		
		void attachDouble (double d);
		void attachStr ( std::string str );	
		void attachEvent ( luna::Event* e, int es );		
		void attachMem ( char* buf, int len );
		
		bool getBool ();
		int getInt (void);	
		unsigned int getUInt ();
		unsigned long getULong ();
		Vector3DF getVec3 ();
		Vector4DF getVec4 ();
		Matrix4F getM4 ();
		xlong getInt64 ();
		float getFloat ();		
		double getDouble ();
		std::string getStr ();
		void getStr ( char* str );
		void getMem ( char* buf, int len );
		void getEvent ( luna::Event* e, std::string& str );

		int getMax ()		{ return int(dat_end-dat); }
		int getUsed ()		{ return int(pos-dat); }
		int getRemain ()	{ return int(pos-rpos); }

	private:
		char*	dat;		
		char*	dat_end;
		char*	pos;
		char*	rpos;		
		
		static char mbuf [ 16384 ];
	};

#endif
