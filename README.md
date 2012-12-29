
Fluids v.3
----------
R. Hoetzlein (c) 2012, http://fluids3.com

Fluids v.3 is a large-scale, open source fluid simulator for the CPU and GPU 
using the smooth particle hydrodynamics method. 
Fluids is capable of efficiently simulating 
up to 8 million particles on the GPU (on 1500 MB of ram).

Requirements:
Fluids v.3 requires a CUDA-capable NVIDIA graphics card /w Compute capability 2.1 or higher. 
Builds tested on Windows 7 with Visual Studio 2010

Start Fluids by running fluids_v3.exe
See below for License terms.
See website for software details.

Fluids makes use of the following libraries:
 CUDA - http://www.nvidia.com/object/cuda_home_new.html
 OpenGL - http://www.opengl.org/
 FreeGlut - http://freeglut.sourceforge.net/
 TinyXML - http://www.grinninglizard.com/tinyxml/index.html
 Glee - http://elf-stone.com/glee.php

Keyboard commands:
------------------
H		Turn scene info on/off
N, M		Change number of particles
[, ]		Change example scene (loaded from scene.xml)
F, G		Change algorithm (CPU, GPU, etc.)
J		Change rendering mode (points, sprites, spheres)
C		Adjust camera (using mouse)
L		Adjust light (using mouse)
A,S,W,D	Move camera target
1		Draw acceleration grid
2		Draw particle IDs (be sure # < 4096 first)
~		Start video capture to disk (tilde key)
-, +		Change grid density

Scene parameters:
-----------------
* Example scenes are loaded from the scene.xml file
All parameters are permitted in either Fluid or Scene sections.
DT			Simulation time step
SimScale		Simulation scale (see website)
Viscosity		Fluid viscosity coefficient
RestDensity	Fluid rest density
Mass			Fluid particle mass
Radius		Fluid particle radius, only for boundary tests
IntStiff		Fluid internal stiffness (non-boundary)
BoundStiff		Fluid stiffness at boundary
BoundDamp		Fluid damping at boundary
AccelLimit		Acceleration limit (for stability)
VelLimit		Velocity limit (for stability)
PointGravAmt	Strength of point gravity
PointGravPos	Position of point gravity
PlaneGravDir	Direction of plane gravity
GroundSlope	Slope of the ground (Y- plane)
WaveForceFre	Frequency of wave forcing
WaveForceMin	Amplitude of wave forcing from X- plane
WaveForceMax	Amplitude of wave forcing from X+ plane
Name			Name of scene example
Num			Number of particles to simulate
VolMin		Start corner of Domain Volume
VolMax		End corner of Domain Volume
InitMin		Start corner of Initial Particle placement
InitMax		End corner of Initial Particle placement


Fluids-Zlib License
-------------------
FLUIDS v.3 - SPH Fluid Simulator for CPU and GPU
Copyright (C) 2012. Rama Hoetzlein, http://fluids3.com

* NOTE: Acknowledgement of the original author is required in professional papers or software products as:
2012 Hoetzlein, Rama. Fluids v.3 - A Large-Scale, Open Source Fluid Simulator. Published online at: fluids3.com. December 2012.
(See part 1 below)

  Fluids-ZLib license 
  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. Acknowledgement of the
	 original author is required if you publish this in a paper, or use it
	 in a product. (See fluids3.com for details)
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

