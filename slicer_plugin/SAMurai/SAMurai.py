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
import zlib

import io
import gzip
import requests

import importlib.util


def ensure_synched(func):
    def inner(self, *args, **kwargs):
        def compute_checksum():
            try:
                if 'xyz' in kwargs:
                    x, y, z = kwargs['xyz']
                else:
                    x, y, z = None
                
                # Check if a sync is already in progress.
                if getattr(self, "_sync_in_progress", False):
                    print("Sync already in progress; skipping checksum computation.")
                    return
                self._sync_in_progress = True

                overall_start = time.time()
                # 1. Retrieve image data, window, and level.
                t0 = time.time()
                result = self.get_image_data()  # Expected to return (image_data, window, level)
                t1 = time.time()
                if result is None:
                    print("No volume node found")
                    return
                image_data, window, level = result
                print(f"Time to get image data: {t1 - t0:.4f} seconds")
                print(f"Window: {window}, Level: {level}")
                
                t2 = time.time()
                image_bytes = image_data[::10, ::10, ::10].tobytes()
                t3 = time.time()
                print(f"Time to create byte representation: {t3 - t2:.4f} seconds")
                
                # 2. Convert window and level to bytes.
                window_bytes = str(window).encode("utf-8")
                level_bytes = str(level).encode("utf-8")
                
                import xxhash

                # 3. Compute xxhash over the image bytes and update with window and level.
                t4 = time.time()
                hasher = xxhash.xxh64()
                hasher.update(image_bytes)
                hasher.update(window_bytes)
                hasher.update(level_bytes)
                hash_value = hasher.intdigest()
                t5 = time.time()
                print("Current image xxhash:", format(hash_value, '016x'))
                print(f"xxhash computation took: {t5 - t4:.4f} seconds")
                print(f"Total computation time: {t5 - overall_start:.4f} seconds")
                
                # 4. Check the previous state and sync if needed.
                old_hash = self.previous_states.get("image_crc", None)
                if old_hash is None or old_hash != hash_value:
                    print("Image changed (or not previously set). Calling sync_image_with_server()")
                    self.upload_image_to_server(z=z)
                else:
                    print("Image unchanged.")
                # Update the previous state.
                self.previous_states["image_crc"] = hash_value
            except Exception as e:
                print("Error in ensure_synched:", e)
            finally:
                self._sync_in_progress = False
        threading.Thread(target=compute_checksum, daemon=True).start()
        return func(self, *args, **kwargs)
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
        
        self.install_dependencies()
        
        ui_widget = slicer.util.loadUI(self.resourcePath("UI/SAMurai.ui"))
        self.layout.addWidget(ui_widget)
        self.ui = slicer.util.childWidgetVariables(ui_widget)
        
        self.add_segmentation_widget()
        self.add_module_icon_to_toolbar()
        self.setup_shortcuts()
        
        _ = self.get_current_segment_id()
        self.previous_states = {}
        self._sync_in_progress = False
        
    def install_dependencies(self):
        dependencies = {
            'xxhash': 'xxhash==3.5.0'
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
          - If the left button is pressed with Meta (or Control) held, print the x,y,z location and trigger positive_point_prompt().
          - If the right button is pressed with Meta (or Control) held, do the same and trigger negative_point_prompt().
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
            event_filter = SAMuraiQtEventFilter(self, slice_widget)
            slice_view.installEventFilter(event_filter)
            self._qt_event_filters.append((slice_view, event_filter))
        
        main_window = slicer.util.mainWindow()
        event_filter = SAMuraiQtEventFilterMainWindow(self, main_window)
        main_window.installEventFilter(event_filter)
        self._qt_event_filters.append((main_window, event_filter))
    
    def add_segmentation_widget(self):
        import qSlicerSegmentationsModuleWidgetsPythonQt

        self.editor = qSlicerSegmentationsModuleWidgetsPythonQt.qMRMLSegmentEditorWidget()
        self.editor.setMaximumNumberOfUndoStates(10)
        self.editor.setMRMLScene(slicer.mrmlScene)
        self.ui.clbtnOperation.layout().addWidget(self.editor, 1, 0, 1, 2)
        self.segment_editor_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        self.editor.setMRMLSegmentEditorNode(self.segment_editor_node)
        seg_nodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
        if seg_nodes:
            self.segmentation_node = seg_nodes[-1]
        else:
            self.segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.editor.setSegmentationNode(self.segmentation_node)
    
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
    
    def get_image_data(self):
        # Get the current volume node (adjust as needed if you have multiple volumes)
        volume_nodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        
        if not volume_nodes:
            return None
        
        volume_node = volume_nodes[0]
        
        image_data = slicer.util.arrayFromVolume(volume_node)
        display_node = volume_node.GetDisplayNode()
        
        # Capture the current window/level settings
        window = display_node.GetWindow()
        level = display_node.GetLevel()
        
        return image_data, window, level

    def upload_image_to_server(self, z=None):
        """
        Upload the current image data to a FastAPI endpoint in a separate thread.
        This function retrieves the image data, window, and level; converts the image data
        to a Base64-encoded string; and then makes a POST request to a fictive endpoint.
        """
        def _upload():
            print("Syncing image with server...")
            try:
                # Retrieve image data, window, and level.
                t0 = time.time()
                result = self.get_image_data()  # Expected to return (image_data, window, level)
                print('self.get_image_data took', time.time() - t0)
                
                if result is None:
                    print("No image data available to upload.")
                    return
                
                t0 = time.time()
                image_data, window, level = result
                if z is not None:
                    image_data = image_data[z]
                
                print('image_data.shape:', image_data.shape)
                print('241 took', time.time() - t0)
                t0 = time.time()
                
                # Adjust the image based on the new window/level settings
                lower_bound, upper_bound = level - window / 2, level + window / 2
                image_data_pre = np.clip(image_data, lower_bound, upper_bound)
                image_data_pre = (
                    (image_data_pre - np.min(image_data_pre))
                    / (np.max(image_data_pre) - np.min(image_data_pre))
                    * 255.0
                )
                image_data_pre = np.uint8(image_data_pre)
                
                print('norm took', time.time() - t0)
                t0 = time.time()
                
                buffer = io.BytesIO()
                np.save(buffer, image_data_pre)
                compressed_data = gzip.compress(buffer.getvalue())
                print('len(compressed_data):', len(compressed_data))
                
                files = {
                    'file': ('volume.npy.gz', compressed_data, 'application/octet-stream')
                }
                
                data = {
                    'z': z
                }

                print("Uploading payload to server...")
                url = "http://0.0.0.0:1526/api/upload_image"  # Update this with your actual endpoint.
                
                print('271 took', time.time() - t0)
                t0 = time.time()
                
                response = requests.post(
                    url,
                    files=files,
                    data=data,
                    headers={"Content-Encoding": "gzip"}
                )
                print('Response took', time.time() - t0)
                
                if response.status_code == 200:
                    print("Image successfully uploaded to server.")
                else:
                    print("Image upload failed with status code:", response.status_code)
            except Exception as e:
                print("Error in upload_image_to_server:", e)
        threading.Thread(target=_upload, daemon=True).start()
    
    @ensure_synched
    def positive_point_prompt(self, xyz=None):
        print("Positive point prompt triggered!", xyz)
    
    @ensure_synched
    def negative_point_prompt(self, xyz=None):
        print("Negative point prompt triggered!", xyz)
    
    def get_widget_segment_editor(self):
        return slicer.modules.segmenteditor.widgetRepresentation().self().editor
    
    def get_current_segment_id(self):
        segment_editor_widget = self.get_widget_segment_editor()
        return segment_editor_widget.mrmlSegmentEditorNode().GetSelectedSegmentID()
        
    def cleanup(self):
        if hasattr(self, "_qt_event_filters"):
            for slice_view, event_filter in self._qt_event_filters:
                slice_view.removeEventFilter(event_filter)
            self._qt_event_filters = []
        return


class SAMuraiQtEventFilter(qt.QObject):
    def __init__(self, samurai_widget, slice_widget):
        super(SAMuraiQtEventFilter, self).__init__()
        self.samurai_widget = samurai_widget
        self.slice_widget = slice_widget

    def eventFilter(self, obj, event):
        if event.type() == qt.QEvent.MouseButtonPress:
            if self.samurai_widget._meta_pressed:
                xyz = convert_device_to_image_pixel(self.slice_widget)
                if event.button() == qt.Qt.LeftButton:
                    self.samurai_widget.positive_point_prompt(xyz=xyz)
                    return True
                elif event.button() == qt.Qt.RightButton:
                    self.samurai_widget.negative_point_prompt(xyz=xyz)
                    return True
        return False


class SAMuraiQtEventFilterMainWindow(qt.QObject):
    def __init__(self, samurai_widget, slice_widget):
        super(SAMuraiQtEventFilterMainWindow, self).__init__()
        self.samurai_widget = samurai_widget
        self.slice_widget = slice_widget

    def eventFilter(self, obj, event):
        if event.type() == qt.QEvent.KeyPress:
            if event.key() in [qt.Qt.Key_Meta, qt.Qt.Key_Control]:
                self.samurai_widget._meta_pressed = True
        elif event.type() == qt.QEvent.KeyRelease:
            if event.key() in [qt.Qt.Key_Meta, qt.Qt.Key_Control]:
                self.samurai_widget._meta_pressed = False
        return False