cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(nnInteractiveSlicer)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Coen de Vente")
set(EXTENSION_DESCRIPTION "A 3D Slicer plugin for nnInteractive")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/nnInteractiveSlicer.png")
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/nnInteractiveSlicer")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/nnInteractiveSlicer/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(nnInteractiveSlicer)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
