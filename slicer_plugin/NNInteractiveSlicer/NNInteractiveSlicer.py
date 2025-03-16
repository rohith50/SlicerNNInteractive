import slicer
import qt
import vtk
import logging

from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *

import numpy as np

import threading
import time

import io
import gzip
import requests
import copy

import importlib.util
import time


def ensure_synched(func):
    def inner(self, *args, **kwargs):
        # def compute_checksum():
        try:                
            if getattr(self, "_sync_in_progress", False):
                print("Sync already in progress; skipping checksum computation.")
                return
            self._sync_in_progress = True

            if self.image_changed():
                print("Image changed (or not previously set). Calling sync_image_with_server()")
                self.upload_image_to_server()
            
            selected_segment_changed = self.selected_segment_changed()
            if 'override_selected_segment_changed' in kwargs and kwargs['override_selected_segment_changed'] is not None:
                selected_segment_changed = kwargs['override_selected_segment_changed']
            
            if selected_segment_changed:
                print("Segment changed (or not previously set). Calling sync_segment_with_server()")
                self.upload_segment_to_server()
                
        except Exception as e:
            print("Error in ensure_synched:", e)
        finally:
            self._sync_in_progress = False
            
            return func(self, *args, **kwargs)
        # threading.Thread(target=compute_checksum, daemon=True).start()
    return inner

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
# NNInteractiveSlicer
#

class NNInteractiveSlicer(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("NNInteractiveSlicer")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Segmentation")]
        self.parent.dependencies = []  # List other modules if needed
        self.parent.contributors = ["Coen de Vente"]
        self.parent.helpText = """
            This is an intuitive 3D Slicer plugin for efficient SAM2-based segmentation.
            """
        self.parent.acknowledgementText = ""


class NNInteractiveSlicerWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        
        self.install_dependencies()
        
        ui_widget = slicer.util.loadUI(self.resourcePath("UI/NNInteractiveSlicer.ui"))
        self.layout.addWidget(ui_widget)
        self.ui = slicer.util.childWidgetVariables(ui_widget)
        
        self.add_segmentation_widget()
        self.add_module_icon_to_toolbar()
        self.setup_shortcuts()
        self.setup_markups_points()
        self.update_server()
        self.init_ui_functionality()
        
        _ = self.get_current_segment_id()
        self.previous_states = {}
        self._sync_in_progress = False
    
    def update_server(self):
        self.server = self.ui.Server.text
    
    def init_ui_functionality(self):
        self.ui.Server.editingFinished.connect(self.update_server)
        self.ui.pbPositivePoint.clicked.connect(self.on_positive_point_clicked)
        self.ui.pbNegativePoint.clicked.connect(self.on_negative_point_clicked)
    
    def install_dependencies(self):
        dependencies = {
            'xxhash': 'xxhash==3.5.0',
            'requests_toolbelt': 'requests_toolbelt==1.0.0'
        }

        for dependency in dependencies:
            if self.check_dependency_installed(dependencies[dependency]):
                continue
            self.run_with_progress_bar(self.pip_install_wrapper, (dependencies[dependency],), 'Installing dependencies: %s'%dependency)
    
    def check_dependency_installed(self, module_name_and_version):
        module_name, module_version = module_name_and_version.split('==')
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return False
        else:
            try:
                # For Python 3.8+; if using an older Python version, you might need to install importlib-metadata.
                import importlib.metadata as metadata
            except ImportError:
                print('Use Python 3.8+')

            try:
                version = metadata.version(module_name)
                if version != module_version:
                    return False
            except metadata.PackageNotFoundError:
                print(f"Could not determine version for {module_name}.")
            return True
    
    def pip_install_wrapper(self, command, event):
        slicer.util.pip_install(command)
        event.set()

    def run_with_progress_bar(self, target, args, title):
        self.progressbar = slicer.util.createProgressDialog(autoClose=False)
        self.progressbar.minimum = 0
        self.progressbar.maximum = 100
        self.progressbar.setLabelText(title)

        parallel_event = threading.Event()
        dep_thread = threading.Thread(
            target=target,
            args=(
                *args,
                parallel_event,
            ),
        )
        dep_thread.start()
        while not parallel_event.is_set():
            slicer.app.processEvents()
        dep_thread.join()

        self.progressbar.close()
        
    def setup_shortcuts(self):
        """
        Install a Qt event filter on the Red, Green, and Yellow slice views.
        The filter tracks key press/release events to update a flag.
        When a mouse button press occurs:
          - If the left button is pressed with Meta (or Control) held, print the x,y,z location and trigger point_prompt().
          - If the right button is pressed with Meta (or Control) held, do the same and trigger point_prompt().
        """        
        self._qt_event_filters = []
        self._meta_pressed = False
        layout_manager = slicer.app.layoutManager()
        for slice_name in ['Red', 'Green', 'Yellow']:
            slice_widget = layout_manager.sliceWidget(slice_name)
            if not slice_widget:
                continue
            # Get the slice view widget (a Qt widget)
            slice_view = slice_widget.sliceView()
            # Create the event filter and pass the slice widget for coordinate conversion.
            event_filter = NNInteractiveSlicerQtEventFilter(self, slice_widget)
            slice_view.installEventFilter(event_filter)
            self._qt_event_filters.append((slice_view, event_filter))
        
        main_window = slicer.util.mainWindow()
        event_filter = NNInteractiveSlicerQtEventFilterMainWindow(self, main_window)
        main_window.installEventFilter(event_filter)
        self._qt_event_filters.append((main_window, event_filter))
    
    def add_segmentation_widget(self):
        import qSlicerSegmentationsModuleWidgetsPythonQt

        self.editor = qSlicerSegmentationsModuleWidgetsPythonQt.qMRMLSegmentEditorWidget()
        self.editor.setMaximumNumberOfUndoStates(10)
        
        segment_editor_singleton_tag = "SegmentEditor"
        self.segment_editor_node = slicer.mrmlScene.GetSingletonNode(
            segment_editor_singleton_tag, "vtkMRMLSegmentEditorNode"
        )
        
        if self.segment_editor_node is None:
            self.segment_editor_node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLSegmentEditorNode")
            self.segment_editor_node.UnRegister(None)
            self.segment_editor_node.SetSingletonTag(segment_editor_singleton_tag)
            self.segment_editor_node = slicer.mrmlScene.AddNode(self.segment_editor_node)
        
        self.editor.setMRMLSegmentEditorNode(self.segment_editor_node)
        self.editor.setMRMLScene(slicer.mrmlScene)
        
        # Add the editor widget to the segmentation group
        if hasattr(self.ui, 'segmentationGroup'):
            # Create a new layout if needed
            if self.ui.segmentationGroup.layout() is None:
                layout = qt.QVBoxLayout(self.ui.segmentationGroup)
                self.ui.segmentationGroup.setLayout(layout)
            else:
                layout = self.ui.segmentationGroup.layout()
            
            # Clear any existing widgets in the layout
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().setParent(None)
                    
            # Add the editor widget
            layout.addWidget(self.editor)
        else:
            print("Could not find segmentationGroup in UI")
    
    def add_module_icon_to_toolbar(self):
        toolbar = slicer.util.mainWindow().findChild(qt.QToolBar, "ModuleSelectorToolBar")
        if not toolbar:
            logging.warning("Could not find 'ModuleSelectorToolBar'.")
            return

        for existing_action in toolbar.actions():
            if existing_action.objectName == "nninteractive_slicer_action":
                return

        action = qt.QAction(qt.QIcon(self.resourcePath("Icons/NNInteractiveSlicer.png")), "NNInteractiveSlicer", toolbar)
        action.setObjectName("nninteractive_slicer_action")
        action.setToolTip("Switch to NNInteractiveSlicer module")
        action.triggered.connect(lambda: slicer.util.selectModule("NNInteractiveSlicer"))
        toolbar.addAction(action)
    
    def get_volume_node(self):
        volume_nodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        
        if not volume_nodes:
            return None
        
        volume_node = volume_nodes[-1]
        
        return volume_node
    
    def get_image_data(self):
        # Get the current volume node (adjust as needed if you have multiple volumes)
        volume_node = self.get_volume_node()
        
        image_data = slicer.util.arrayFromVolume(volume_node)
        
        return image_data
    
    def get_segment_data(self):
        segmentation_node, selected_segment_id = self.get_selected_segmentation_node_and_segment_id()
        
        mask = slicer.util.arrayFromSegmentBinaryLabelmap(segmentation_node, selected_segment_id, self.get_volume_node())
        seg_data_bool = mask.astype(bool)
        
        return seg_data_bool

    def upload_image_to_server(self):
        """
        Upload the current image data to a FastAPI endpoint in a separate thread.
        This function retrieves the image data, window, and level; converts the image data
        to a Base64-encoded string; and then makes a POST request to a fictive endpoint.
        """
        # def _upload():
        print("Syncing image with server...")
        try:
            # Retrieve image data, window, and level.
            t0 = time.time()
            image_data = self.get_image_data()  # Expected to return (image_data, window, level)
            print('self.get_image_data took', time.time() - t0)
            
            if image_data is None:
                print("No image data available to upload.")
                return
            
            t0 = time.time()
            
            buffer = io.BytesIO()
            np.save(buffer, image_data)
            compressed_data = gzip.compress(buffer.getvalue())
            print('len(compressed_data):', len(compressed_data))
            
            files = {
                'file': ('volume.npy.gz', compressed_data, 'application/octet-stream')
            }

            print("Uploading payload to server...")
            url = f"{self.server}/upload_image"  # Update this with your actual endpoint.
            print('url:', url)
            
            print('271 took', time.time() - t0)
            t0 = time.time()
            
            response = requests.post(
                url,
                files=files,
                headers={"Content-Encoding": "gzip"}
            )
            print('Response took', time.time() - t0)
            
            if response.status_code == 200:
                print("Image successfully uploaded to server.")
            else:
                print("Image upload failed with status code:", response.status_code)
        except Exception as e:
            print("Error in upload_image_to_server:", e)

    def upload_segment_to_server(self):
        print("Syncing segment with server...")
        try:
            t0 = time.time()
            segment_data = self.get_segment_data()  # Expected to return (image_data, window, level)
            print('self.segment_data() took', time.time() - t0)
            
            t0 = time.time()
            
            buffer = io.BytesIO()
            np.save(buffer, segment_data)
            compressed_data = gzip.compress(buffer.getvalue())
            print('len(compressed_data):', len(compressed_data))
            
            files = {
                'file': ('volume.npy.gz', compressed_data, 'application/octet-stream')
            }

            print("Uploading payload to server...")
            url = f"{self.server}/upload_segment"  # Update this with your actual endpoint.
            
            t0 = time.time()
            
            response = requests.post(
                url,
                files=files,
                headers={"Content-Encoding": "gzip"}
            )
            print('Response took', time.time() - t0)
            
            if response.status_code == 200:
                print("Image successfully uploaded to server.")
            else:
                print("Image upload failed with status code:", response.status_code)
        except Exception as e:
            print("Error in upload_image_to_server:", e)
    
    @ensure_synched
    def point_prompt(self, xyz=None, positive_click=False, override_selected_segment_changed=None):
        url = f"{self.server}/add_point_interaction"
        
        seg_response = requests.post(
            url, 
            json={'voxel_coord': xyz[::-1],
                  'positive_click': positive_click})
        
        unpacked_segmentation = self.unpack_binary_segmentation(seg_response.content, decompress=False)
        print(seg_response)
        print(f"{positive_click} point prompt triggered!", xyz)
        
        self.show_segmentation(unpacked_segmentation)
    
    def unpack_binary_segmentation(self, binary_data, decompress=False):
        """
        Unpacks binary data (1 bit per voxel) into a full 3D numpy array (bool type).
        
        Parameters:
            binary_data (bytes): The packed binary segmentation data.
        
        Returns:
            np.ndarray: The unpacked 3D boolean numpy array.
        """
        if decompress:
            binary_data = binary_data = gzip.decompress(binary_data)

        if self.get_image_data() is None:
            self.capture_image()

        # Get the shape of the original volume (same as image_data shape)
        vol_shape = self.get_image_data().shape
        
        # Calculate the total number of bits (voxels)
        total_voxels = np.prod(vol_shape)
        
        # Unpack the binary data (convert from bytes to bits)
        unpacked_bits = np.unpackbits(np.frombuffer(binary_data, dtype=np.uint8))
        
        # Trim any extra bits (in case the bit length is not perfectly divisible)
        unpacked_bits = unpacked_bits[:total_voxels]
        
        # Reshape into the original volume shape
        segmentation_mask = unpacked_bits.reshape(vol_shape).astype(np.bool_).astype(np.uint8)
        
        return segmentation_mask
    
    def show_segmentation(self, segmentation_mask):
        t0 = time.time()
        
        self.previous_states['segment_data'] = segmentation_mask
        
        segmentationNode, selectedSegmentID = self.get_selected_segmentation_node_and_segment_id()
        
        slicer.util.updateSegmentBinaryLabelmapFromArray(
            segmentation_mask, segmentationNode, selectedSegmentID, self.get_volume_node()
        )

        # Mark the segmentation as modified so the UI updates
        segmentationNode.Modified()

        segmentationNode.GetSegmentation().CollapseBinaryLabelmaps()
        del segmentation_mask
        
        print('show_segmentation took', time.time() - t0)
    
    def get_selected_segmentation_node_and_segment_id(self):
        """Retrieve the currently selected segmentation node and segment ID.
        If no segmentation exists, it creates a new one.
        """        
        
        # Get the current segmentation node (or create one if it does not exist)
        segmentationNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
        if segmentationNode is None:
            segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(self.get_volume_node())

        # Retrieve the currently selected segment ID from the Segment Editor
        segmentEditorNode = self.get_widget_segment_editor().mrmlSegmentEditorNode()
        selectedSegmentID = self.get_current_segment_id()
        # If no segment is selected, create a new segment
        if not selectedSegmentID:
            # Generate a new segment name
            segmentIDs = segmentationNode.GetSegmentation().GetSegmentIDs()
            if len(segmentIDs) == 0:
                newSegmentName = "Segment_1"
            else:
                # Find the next available number
                segmentNumbers = [int(seg.split('_')[-1]) for seg in segmentIDs if seg.startswith("Segment_") and seg.split('_')[-1].isdigit()]
                nextSegmentNumber = max(segmentNumbers) + 1 if segmentNumbers else 1
                newSegmentName = f"Segment_{nextSegmentNumber}"

            # Create and add the new segment
            newSegmentID = segmentationNode.GetSegmentation().AddEmptySegment(newSegmentName)
            segmentEditorNode.SetSelectedSegmentID(newSegmentID)
 
            return segmentationNode, newSegmentID

        return segmentationNode, selectedSegmentID
    
    def image_changed(self):
        image_data = self.get_image_data()
        if image_data is None:
            print("No volume node found")
            return
        
        old_image_data = self.previous_states.get("image_data", None)
        image_changed =  old_image_data is None or not np.all(old_image_data == image_data)
        self.previous_states["image_data"] = copy.deepcopy(image_data)

        return image_changed

    def selected_segment_changed(self):
        segment_data = self.get_segment_data()
        old_segment_data = self.previous_states.get("segment_data", None)
        selected_segment_changed = old_segment_data is None or not np.all(old_segment_data.astype(bool) == segment_data)
        self.previous_states["segment_data"] = copy.deepcopy(segment_data)

        return selected_segment_changed

    def get_widget_segment_editor(self):
        return slicer.modules.segmenteditor.widgetRepresentation().self().editor
    
    def get_current_segment_id(self):
        segment_editor_widget = self.get_widget_segment_editor()
        return segment_editor_widget.mrmlSegmentEditorNode().GetSelectedSegmentID()
        
    def cleanup(self):
        """Clean up resources when the module is closed"""
        self.removeObservers()

        if hasattr(self, "_qt_event_filters"):
            for slice_view, event_filter in self._qt_event_filters:
                slice_view.removeEventFilter(event_filter)
            self._qt_event_filters = []
        return

    def setup_markups_points(self):
        """Initialize the markups fiducial list for storing point prompts"""
        # Remove any existing points nodes first to avoid duplicates
        for node_name in ["PromptPointsPositive", "PromptPointsNegative"]:
            existing_points = slicer.mrmlScene.GetNodesByName(node_name)
            if existing_points and existing_points.GetNumberOfItems() > 0:
                for i in range(existing_points.GetNumberOfItems()):
                    slicer.mrmlScene.RemoveNode(existing_points.GetItemAsObject(i))
        
        # Create separate nodes for positive and negative points        
        self.positive_points_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "PromptPointsPositive")
        self.positive_points_node.CreateDefaultDisplayNodes()
        
        self.negative_points_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "PromptPointsNegative")
        self.negative_points_node.CreateDefaultDisplayNodes()
        
        # Configure display properties for positive points (green)
        pos_display_node = self.positive_points_node.GetDisplayNode()
        pos_display_node.SetTextScale(0)  # Hide text labels
        pos_display_node.SetGlyphScale(0.75)  # Make the points larger
        pos_display_node.SetColor(0.0, 1.0, 0.0)  # Green color
        pos_display_node.SetSelectedColor(0.0, 1.0, 0.0)
        pos_display_node.SetOpacity(1.0)  # Fully opaque
        pos_display_node.SetSliceProjection(False)  # Make points visible in all slice views
        
        # Configure display properties for negative points (red)
        neg_display_node = self.negative_points_node.GetDisplayNode()
        neg_display_node.SetTextScale(0)  # Hide text labels
        neg_display_node.SetGlyphScale(0.75)  # Make the points larger
        neg_display_node.SetColor(1.0, 0.0, 0.0)  # Red color
        neg_display_node.SetSelectedColor(1.0, 0.0, 0.0)
        neg_display_node.SetOpacity(1.0)  # Fully opaque
        neg_display_node.SetSliceProjection(False)  # Make points visible in all slice views

        self.clear_points()
        
        # Setup for interactive placement
        self.is_placing_positive = False
        self.is_placing_negative = False
        self.point_placement_observers = []
        
        # Flag to track if we've shown the placement mode warning
        self.shown_placement_warning = False
    
    def clear_points(self):
        # Track points
        self.positive_points = []
        self.negative_points = []
        
        # Clear the list widget
        self.ui.pointListWidget.clear()

    def on_positive_point_clicked(self):
        """Start interactive placement of a positive point"""
        if self.is_placing_positive or self.is_placing_negative:
            # Already in placement mode, cancel current placement
            self.stop_point_placement()
            return
            
        # Enter positive point placement mode
        print("Starting positive point placement - click in the view to place")
        self.is_placing_positive = True
        self.start_point_placement()
    
    def on_negative_point_clicked(self):
        """Start interactive placement of a negative point"""
        if self.is_placing_positive or self.is_placing_negative:
            # Already in placement mode, cancel current placement
            self.stop_point_placement()
            return
            
        # Enter negative point placement mode
        print("Starting negative point placement - click in the view to place")
        self.is_placing_negative = True
        self.start_point_placement()
    
    def start_point_placement(self):
        """Enter point placement mode"""
        selected_segment_changed = False
        if self.selected_segment_changed():
            selected_segment_changed = True
            self.clear_points()

        markups_logic = slicer.modules.markups.logic()
        
        # Use the appropriate node based on what we're placing
        active_node = self.positive_points_node if self.is_placing_positive else self.negative_points_node
        markups_logic.SetActiveListID(active_node)
        
        # Try to enter placement mode
        print("Starting placement mode...")
        markups_logic.StartPlaceMode(False)
        
        # Make sure the input is active
        active_node.SetLocked(False)
        
        # Clear any existing observers
        for observer in self.point_placement_observers:
            if observer[0]:
                observer[0].RemoveObserver(observer[1])
        self.point_placement_observers = []
        
        # Add observer to the active node
        observer_id = active_node.AddObserver(slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent, 
                                              lambda a, b: self.on_point_placed(a, b, override_selected_segment_changed=selected_segment_changed))
        self.point_placement_observers.append((active_node, observer_id))
        
        print("Placement mode started successfully")
    
    def stop_point_placement(self):
        """Exit point placement mode"""
        print("Stopping placement mode...")
        # Exit placement mode - try multiple methods for compatibility
        try:
            # Try various methods to stop place mode, as different versions use different methods
            markups_logic = slicer.modules.markups.logic()
            
            # First try StopPlaceMode() (newer versions)
            if hasattr(markups_logic, 'StopPlaceMode'):
                markups_logic.StopPlaceMode()
                print("Used StopPlaceMode()")
            # Then try EndPlaceMode() (some versions)
            elif hasattr(markups_logic, 'EndPlaceMode'):
                markups_logic.EndPlaceMode()
                print("Used EndPlaceMode()")
            # Then try DeactivatePointModePlace() (older versions)
            elif hasattr(markups_logic, 'DeactivatePointModePlace'):
                markups_logic.DeactivatePointModePlace()
                print("Used DeactivatePointModePlace()")
            else:
                # As a fallback, just print a warning once
                if not hasattr(self, 'shown_placement_warning') or not self.shown_placement_warning:
                    print("Warning: Could not find a method to stop place mode, but placement will still work")
                    self.shown_placement_warning = True
            
            # Make sure the placement is actually stopped by deselecting the active node
            # This is a common workaround when the explicit stop methods aren't available
            markups_logic.SetActiveListID(None)
            print("Set active list to None")
        except Exception as e:
            print(f"Error stopping place mode: {e}")
            print("Placement functionality may still work")
        
        # Clean up observers
        for observer in self.point_placement_observers:
            try:
                if observer[0] and observer[1]:
                    observer[0].RemoveObserver(observer[1])
            except Exception as e:
                print(f"Error removing observer: {e}")
        self.point_placement_observers = []
            
        print("Removed observers")
        
        # Reset placement flags
        self.is_placing_positive = False
        self.is_placing_negative = False
        print("Placement mode stopped")
    
    def on_point_placed(self, caller, event, override_selected_segment_changed=None):
        """Called when a point is placed in the scene"""
        # Add debug information to help diagnose issues
        print(f"on_point_placed called with event: {event}")
        
        # Determine which node called this (positive or negative)
        active_node = caller
        is_positive = (active_node == self.positive_points_node)
            
        n = active_node.GetNumberOfControlPoints() - 1
        if n < 0:
            print("No control points found")
            return
            
        # Get the position
        pos = [0, 0, 0]
        active_node.GetNthControlPointPosition(n, pos)
        
        # Set the point label
        if is_positive:
            point_id = len(self.positive_points) + 1
            label_prefix = "P"
            label_type = "Positive"
        else:
            point_id = len(self.negative_points) + 1
            label_prefix = "N"
            label_type = "Negative"
            
        active_node.SetNthControlPointLabel(n, f"{label_prefix}-{point_id}")
        
        # Lock the point to prevent accidental movement
        active_node.SetNthControlPointLocked(n, True)
        
        # Store the point info
        point_info = {'id': n, 'position': pos}
        if is_positive:
            self.positive_points.append(point_info)
        else:
            self.negative_points.append(point_info)
        
        # Add to list widget
        self.ui.pointListWidget.addItem(f"{label_prefix}-{point_id}: {label_type} at ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
        
        print(f"Added {label_type} point at: {pos}")
        
        # Make sure the point is visible in the scene
        active_node.GetDisplayNode().SetVisibility(True)
        active_node.SetDisplayVisibility(True)
        
        # Force scene update
        slicer.mrmlScene.Modified()
        
        # Convert RAS coordinates to IJK (voxel) coordinates
        volumeNode = self.get_volume_node()
        if volumeNode:
            # Apply any transforms to get volume's RAS coordinates
            transformRasToVolumeRas = vtk.vtkGeneralTransform()
            slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, volumeNode.GetParentTransformNode(), transformRasToVolumeRas)
            point_VolumeRas = transformRasToVolumeRas.TransformPoint(pos)
            
            # Convert to IJK coordinates
            volumeRasToIjk = vtk.vtkMatrix4x4()
            volumeNode.GetRASToIJKMatrix(volumeRasToIjk)
            point_Ijk = [0, 0, 0, 1]
            volumeRasToIjk.MultiplyPoint(list(point_VolumeRas) + [1.0], point_Ijk)
            xyz = [int(round(c)) for c in point_Ijk[0:3]]
            
            print(f"Converted point to voxel coordinates: {xyz}")
            
            # Call point_prompt with the voxel coordinates
            self.point_prompt(xyz=xyz, positive_click=is_positive, override_selected_segment_changed=override_selected_segment_changed)
        
        # Exit placement mode after placing a point
        self.stop_point_placement()


class NNInteractiveSlicerQtEventFilter(qt.QObject):
    def __init__(self, nninteractive_slicer_widget, slice_widget):
        super(NNInteractiveSlicerQtEventFilter, self).__init__()
        self.nninteractive_slicer_widget = nninteractive_slicer_widget
        self.slice_widget = slice_widget

    def eventFilter(self, obj, event):
        if event.type() == qt.QEvent.MouseButtonPress:
            if self.nninteractive_slicer_widget._meta_pressed:
                xyz = convert_device_to_image_pixel(self.slice_widget)
                if event.button() == qt.Qt.LeftButton:
                    # Get the RAS position for adding to the markup node
                    ras_position = self.get_ras_from_ijk(xyz)
                    self.add_point_to_markup(ras_position, is_positive=True)
                    
                    # Call the prompt method to handle server interaction
                    self.nninteractive_slicer_widget.point_prompt(xyz=xyz, positive_click=True)
                    return True
                elif event.button() == qt.Qt.RightButton:
                    # Get the RAS position for adding to the markup node
                    ras_position = self.get_ras_from_ijk(xyz)
                    self.add_point_to_markup(ras_position, is_positive=False)
                    
                    # Call the prompt method to handle server interaction
                    self.nninteractive_slicer_widget.point_prompt(xyz=xyz, positive_click=False)
                    return True
        return False

    def get_ras_from_ijk(self, ijk_coords):
        """Convert IJK (voxel) coordinates to RAS coordinates"""
        volume_node = self.nninteractive_slicer_widget.get_volume_node()
        if not volume_node:
            return [0, 0, 0]
            
        # Convert IJK to RAS
        ijkToRas = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(ijkToRas)
        
        # Apply the transformation
        ijk_point = ijk_coords + [1.0]  # Add homogeneous coordinate
        ras_point = [0, 0, 0, 1]
        ijkToRas.MultiplyPoint(ijk_point, ras_point)
        
        return ras_point[0:3]
        
    def add_point_to_markup(self, ras_position, is_positive=True):
        """Add a point to the appropriate markup fiducial node"""
        widget = self.nninteractive_slicer_widget
        
        # Select the appropriate node
        if is_positive:
            active_node = widget.positive_points_node
            point_id = len(widget.positive_points) + 1
            label_prefix = "P"
            label_type = "Positive"
        else:
            active_node = widget.negative_points_node
            point_id = len(widget.negative_points) + 1
            label_prefix = "N"
            label_type = "Negative"
        
        # Add the point to the node
        n = active_node.AddControlPoint(ras_position)
        active_node.SetNthControlPointLabel(n, f"{label_prefix}-{point_id}")
        active_node.SetNthControlPointLocked(n, True)
        
        # Store the point info
        point_info = {'id': n, 'position': ras_position}
        if is_positive:
            widget.positive_points.append(point_info)
        else:
            widget.negative_points.append(point_info)
        
        # Add to list widget
        widget.ui.pointListWidget.addItem(f"{label_prefix}-{point_id}: {label_type} at ({ras_position[0]:.1f}, {ras_position[1]:.1f}, {ras_position[2]:.1f})")
        
        # Make sure the point is visible
        active_node.GetDisplayNode().SetVisibility(True)
        active_node.SetDisplayVisibility(True)
        
        # Force scene update
        slicer.mrmlScene.Modified()
        
        print(f"Added {label_type} point at: {ras_position}")


class NNInteractiveSlicerQtEventFilterMainWindow(qt.QObject):
    def __init__(self, nninteractive_slicer_widget, slice_widget):
        super(NNInteractiveSlicerQtEventFilterMainWindow, self).__init__()
        self.nninteractive_slicer_widget = nninteractive_slicer_widget
        self.slice_widget = slice_widget

    def eventFilter(self, obj, event):
        if event.type() == qt.QEvent.KeyPress:
            if event.key() in [qt.Qt.Key_Meta, qt.Qt.Key_Control]:
                self.nninteractive_slicer_widget._meta_pressed = True
        elif event.type() == qt.QEvent.KeyRelease:
            if event.key() in [qt.Qt.Key_Meta, qt.Qt.Key_Control]:
                self.nninteractive_slicer_widget._meta_pressed = False
        return False