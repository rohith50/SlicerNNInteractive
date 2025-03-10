import slicer
import qt
import vtk
import logging

from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *

# Assume this function is defined somewhere in your module.
def convert_device_to_image_pixel(sliceWidget):
    sliceLogic = sliceWidget.sliceLogic()

    # Get the RAS coordinates from the Crosshair node
    crosshairNode = slicer.util.getNode("Crosshair")
    point_Ras = [0, 0, 0]
    crosshairNode.GetCursorPositionRAS(point_Ras)
    
    # Get the volume node from the slice logic
    volumeNode = sliceLogic.GetBackgroundLayer().GetVolumeNode()

    # If the volume node is transformed, apply that transform to get volume's RAS coordinates
    transformRasToVolumeRas = vtk.vtkGeneralTransform()
    slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, volumeNode.GetParentTransformNode(), transformRasToVolumeRas)
    point_VolumeRas = transformRasToVolumeRas.TransformPoint(point_Ras)

    # Get voxel coordinates from physical coordinates
    volumeRasToIjk = vtk.vtkMatrix4x4()
    volumeNode.GetRASToIJKMatrix(volumeRasToIjk)
    point_Ijk = [0, 0, 0, 1]
    volumeRasToIjk.MultiplyPoint(list(point_VolumeRas) + [1.0], point_Ijk)
    point_Ijk = [int(round(c)) for c in point_Ijk[0:3]]
    
    print("convert_device_to_image_pixel:", point_Ijk)
    return point_Ijk


#
# SAMurai
#

class SAMurai(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("SAMurai")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Segmentation")]
        self.parent.dependencies = []  # List other modules if needed
        self.parent.contributors = ["Coen de Vente"]
        self.parent.helpText = """
            This is an intuitive 3D Slicer plugin for efficient SAM2-based segmentation.
            """
        self.parent.acknowledgementText = ""


class SAMuraiWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        ui_widget = slicer.util.loadUI(self.resourcePath("UI/SAMurai.ui"))
        self.layout.addWidget(ui_widget)
        self.ui = slicer.util.childWidgetVariables(ui_widget)
        
        self.add_segmentation_widget()
        self.add_module_icon_to_toolbar()
        self.setup_shortcuts()
    
    def setup_shortcuts(self):
        """
        Install a Qt event filter on the Red, Green, and Yellow slice views.
        The filter tracks key press/release events to update a flag.
        When a mouse button press occurs:
          - If the left button is pressed with Meta (or Control) held, print the x,y,z location and trigger positive_point_prompt().
          - If the right button is pressed with Meta (or Control) held, do the same and trigger negative_point_prompt().
        """
        self._qt_event_filters = []
        layout_manager = slicer.app.layoutManager()
        for slice_name in ['Red', 'Green', 'Yellow']:
            slice_widget = layout_manager.sliceWidget(slice_name)
            if not slice_widget:
                continue
            # Get the slice view widget (a Qt widget)
            slice_view = slice_widget.sliceView()
            # Create the event filter and pass the slice widget for coordinate conversion.
            event_filter = SAMuraiQtEventFilter(self, slice_widget)
            slice_view.installEventFilter(event_filter)
            self._qt_event_filters.append((slice_view, event_filter))
    
    def positive_point_prompt(self, xyz):
        print("Positive point prompt triggered!", xyz)
    
    def negative_point_prompt(self, xyz):
        print("Negative point prompt triggered!", xyz)
    
    def add_segmentation_widget(self):
        import qSlicerSegmentationsModuleWidgetsPythonQt
        self.editor = qSlicerSegmentationsModuleWidgetsPythonQt.qMRMLSegmentEditorWidget()
        self.editor.setMaximumNumberOfUndoStates(10)
        self.editor.setMRMLScene(slicer.mrmlScene)
        self.ui.clbtnOperation.layout().addWidget(self.editor, 1, 0, 1, 2)
    
    def add_module_icon_to_toolbar(self):
        toolbar = slicer.util.mainWindow().findChild(qt.QToolBar, "ModuleSelectorToolBar")
        if not toolbar:
            logging.warning("Could not find 'ModuleSelectorToolBar'.")
            return

        for existing_action in toolbar.actions():
            if existing_action.objectName == "samurai_action":
                return

        action = qt.QAction(qt.QIcon(self.resourcePath("Icons/SAMurai.png")), "SAMurai", toolbar)
        action.setObjectName("samurai_action")
        action.setToolTip("Switch to SAMurai module")
        action.triggered.connect(lambda: slicer.util.selectModule("SAMurai"))
        toolbar.addAction(action)
    
    def cleanup(self):
        if hasattr(self, "_qt_event_filters"):
            for slice_view, event_filter in self._qt_event_filters:
                slice_view.removeEventFilter(event_filter)
            self._qt_event_filters = []
        return


class SAMuraiQtEventFilter(qt.QObject):
    """
    A Qt event filter for slice view widgets.
    Tracks key press and release events to update a flag indicating whether the Meta (Command)
    or Control key is pressed. When a mouse button press occurs, it converts the current click
    position to image (x,y,z) coordinates and prints them. Then:
      - If left-click is detected, it calls positive_point_prompt().
      - If right-click is detected, it calls negative_point_prompt().
    """
    def __init__(self, samurai_widget, slice_widget):
        super(SAMuraiQtEventFilter, self).__init__()
        self.samurai_widget = samurai_widget
        self.slice_widget = slice_widget
        self._meta_pressed = False

    def eventFilter(self, obj, event):
        if event.type() == qt.QEvent.KeyPress:
            # Check if Meta (Command) or Control key is pressed.
            if event.key() in [qt.Qt.Key_Meta, qt.Qt.Key_Control]:
                self._meta_pressed = True
        elif event.type() == qt.QEvent.KeyRelease:
            if event.key() in [qt.Qt.Key_Meta, qt.Qt.Key_Control]:
                self._meta_pressed = False
        elif event.type() == qt.QEvent.MouseButtonPress:
            # When a mouse button is pressed, check if the meta flag is set.
            if self._meta_pressed:
                # Convert the click to image pixel coordinates using the provided conversion function.
                xyz = convert_device_to_image_pixel(self.slice_widget)
                if event.button() == qt.Qt.LeftButton:
                    self.samurai_widget.positive_point_prompt(xyz)
                    return True  # Consume the event.
                elif event.button() == qt.Qt.RightButton:
                    self.samurai_widget.negative_point_prompt(xyz)
                    return True  # Consume the event.
        return False
