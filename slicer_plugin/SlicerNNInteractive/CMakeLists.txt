set(MODULE_NAME "SlicerNNInteractive")

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/Icons/prompts/dark/bbox_icon.svg
  Resources/Icons/prompts/dark/lasso_icon.svg
  Resources/Icons/prompts/dark/point_icon.svg
  Resources/Icons/prompts/dark/scribble_icon.svg
  Resources/Icons/prompts/light/bbox_icon.svg
  Resources/Icons/prompts/light/lasso_icon.svg
  Resources/Icons/prompts/light/point_icon.svg
  Resources/Icons/prompts/light/scribble_icon.svg
  Resources/UI/${MODULE_NAME}.ui
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )
