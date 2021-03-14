
cmake_minimum_required(VERSION 2.8)

#add_definitions(-g)
#add_definitions(-w)

include_directories(.)
include_directories(./common)
include_directories(./fluids)

#list all source files here
add_executable(	fluids_v3
		main.cpp
		common/camera3d.cpp
		common/geomx.cpp
		common/gl_helper.cpp
		common/image.cpp
		common/matrix.cpp
		common/mdebug.cpp
		common/mesh.cpp
		common/mtime.cpp
		common/particle.cpp
		common/pivotx.cpp
		common/point_set.cpp
		common/tinystr.cpp
		common/tinyxml.cpp
		common/tinyxmlerror.cpp
		common/tinyxmlparser.cpp
		common/utilities.cpp
		common/vector.cpp
		common/xml_settings.cpp
		fluids/bank_checker.cpp
		fluids/cmd_arg_reader.cpp
		fluids/cutil.cpp
		fluids/fluid.cpp
		fluids/fluid_system.cpp
)

target_link_libraries(fluids_v3 GL)
target_link_libraries(fluids_v3 GLU)
target_link_libraries(fluids_v3 glut)
target_link_libraries(fluids_v3 Cg)
target_link_libraries(fluids_v3 CgGL)
target_link_libraries(fluids_v3 GLee)
target_link_libraries(fluids_v3 OpenCL)
