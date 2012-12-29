
struct fragment
{
	float4 pos	: POSITION;
	float4 color    : COLOR0;
	float2 texCoord : TEXCOORD0;
	float3 Vdir     : TEXCOORD3;
};
struct pixel
{
	float4 color : COLOR;
};


pixel main ( fragment IN )
{	
	pixel OUT;
	OUT.color = IN.color;
	return OUT;
}

