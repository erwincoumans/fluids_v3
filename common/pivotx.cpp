
#include "pivotx.h"

void PivotX::setPivot ( float x, float y, float z, float rx, float ry, float rz )
{
	from_pos.Set ( x,y,z);
	ang_euler.Set ( rx,ry,rz );
}

void PivotX::updateTform ()
{
	trans.RotateZYXT ( ang_euler, from_pos );
}