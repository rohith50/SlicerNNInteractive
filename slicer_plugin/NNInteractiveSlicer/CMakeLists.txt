set(MODULE_NAME "NNInteractiveSlicer")

#-----------------------------------------------------------------------------
# Configure the module
#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

# If you have additional resources (icons, .ui files, etc.), list them here:
set(MODULE_PYTHON_RESOURCES
	Resources/Icons/${MODULE_NAME}.png
	Resources/UI/${MODULE_NAME}.ui
)

#-----------------------------------------------------------------------------
# Add the Python scripted module
#-----------------------------------------------------------------------------
slicer_add_python_script(
  TARGET ${MODULE_NAME}
  MODULE_NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
)