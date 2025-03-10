cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SAMurai)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Coen de Vente")
set(EXTENSION_DESCRIPTION "An intuitive 3D slicer plugin for efficient SAM2-based segmentation.")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/SAMurai.png")
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/SAMurai")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/SAMurai/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SAMurai)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
