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

from skimage.draw import polygon


def ensure_synched(func):
    def inner(self, *args, **kwargs):
        try:
            if self.image_changed():
                print("Image changed (or not previously set). Calling upload_segment_to_server()")
                self.upload_image_to_server()
            
            if self.selected_segment_changed():
                print("Segment changed (or not previously set). Calling upload_segment_to_server()")
                # self.clear_all_but_last_point()
                print('Calling self.remove_prompt_nodes!')
                self.remove_all_but_last_prompt()
                self.upload_segment_to_server()
            else:
                print("Segment did not change!")
                
        except Exception as e:
            print("Error in ensure_synched:", e)
        finally:            
            return func(self, *args, **kwargs)
    return inner


#
# nnInteractiveSlicer
#

class nnInteractiveSlicer(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        
        self.parent.title = _("nnInteractiveSlicer")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Segmentation")]
        self.parent.dependencies = []  # List other modules if needed
        self.parent.contributors = ["Coen de Vente"]
        self.parent.helpText = """
            This is an 3D Slicer plugin for using nnInteractive.
            """
        self.parent.acknowledgementText = ""


class nnInteractiveSlicerWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        
        self.install_dependencies()
        
        ui_widget = slicer.util.loadUI(self.resourcePath("UI/nnInteractiveSlicer.ui"))
        self.layout.addWidget(ui_widget)
        self.ui = slicer.util.childWidgetVariables(ui_widget)
        
        self.add_segmentation_widget()
        self.add_module_icon_to_toolbar()
        self.setup_shortcuts()
        
        self.prompt_types = {
            "point": {
                "node_class": "vtkMRMLMarkupsFiducialNode",
                "node": None,
                "name": "PointPrompt",
                "button_text": "Point",
                "display_node_markup_function": self.display_node_markup_point,
                "on_placed_function": self.on_point_placed,
                "place_widget": self.ui.pointPlaceWidget,
            },
            "bbox": {
                "node_class": "vtkMRMLMarkupsROINode",
                "node": None,
                "name": "BBoxPrompt",
                "button_text": "BBox",
                "display_node_markup_function": self.display_node_markup_bbox,
                "on_placed_function": self.on_bbox_placed,
                "place_widget": self.ui.bboxPlaceWidget,
            },
            "lasso": {
                "node_class": "vtkMRMLMarkupsClosedCurveNode",
                "node": None,
                "name": "LassoPrompt",
                "button_text": "Lasso",
                "display_node_markup_function": self.display_node_markup_lasso,
                "on_placed_function": self.on_lasso_placed,
                "place_widget": self.ui.lassoPlaceWidget,
            }
        }
        
        self.setup_markups_points()
        self.init_ui_functionality()
        
        _ = self.get_current_segment_id()
        self.previous_states = {}

    def update_server(self):
        # Get the updated server URL from the UI
        self.server = self.ui.Server.text
        
        # Save the server URL to QSettings
        settings = qt.QSettings()
        settings.setValue("nnInteractiveSlicer/server", self.server)
        
        print("Server URL updated and saved:", self.server)
    
    def init_ui_functionality(self):
        self.ui.uploadProgressGroup.setVisible(False)

        # Load the saved server URL (default to an empty string if not set)
        savedServer = slicer.util.settingsValue("nnInteractiveSlicer/server", "")
        self.ui.Server.text = savedServer
        self.server = savedServer

        self.ui.Server.editingFinished.connect(self.update_server)
        
        # Set up style sheets for selected/unselected buttons
        self.selected_style = "background-color: #3498db; color: white; min-height: 28px; font-size: 13pt;"
        self.unselected_style = "min-height: 28px; font-size: 13pt;"
        
        # Set initial prompt type
        self.current_prompt_type_positive = True
        self.ui.pbPromptTypePositive.setStyleSheet(self.selected_style)
        self.ui.pbPromptTypeNegative.setStyleSheet(self.unselected_style)
        
        # Connect Prompt Type buttons
        self.ui.pbPromptTypePositive.clicked.connect(self.on_prompt_type_positive_clicked)
        self.ui.pbPromptTypeNegative.clicked.connect(self.on_prompt_type_negative_clicked)
        
        self.prompt_types["lasso"]["place_widget"].placeButton().clicked.connect(self.on_lasso_clicked)
        
        self.interaction_tool_mode = None
        
    def display_node_markup_lasso(self, display_node):
        display_node.SetFillOpacity(0.)
        display_node.SetOutlineOpacity(.5)
        display_node.SetSelectedColor(0, 0, 1)
        display_node.SetColor(0, 0, 1)
        display_node.SetActiveColor(0, 0, 1)
        display_node.SetSliceProjectionColor(0, 0, 1)
        display_node.SetGlyphScale(2)
        display_node.SetLineThickness(.3)
        display_node.SetHandlesInteractive(False)
        display_node.SetTextScale(0)
    
    def display_node_markup_bbox(self, display_node):
        display_node.SetFillOpacity(0.)
        display_node.SetOutlineOpacity(.5)
        display_node.SetSelectedColor(0, 0, 1)
        display_node.SetColor(0, 0, 1)
        display_node.SetActiveColor(0, 0, 1)
        display_node.SetSliceProjectionColor(0, 0, 1)
        display_node.SetInteractionHandleScale(1)
        display_node.SetGlyphScale(0)
        display_node.SetHandlesInteractive(False)
        display_node.SetTextScale(0)
    
    def display_node_markup_point(self, display_node):
        display_node.SetTextScale(0)  # Hide text labels
        display_node.SetGlyphScale(0.75)  # Make the points larger
        display_node.SetColor(0.0, 1.0, 0.0)  # Green color
        display_node.SetSelectedColor(0.0, 1.0, 0.0)
        display_node.SetOpacity(1.0)  # Fully opaque
        display_node.SetSliceProjection(False)  # Make points visible in all slice views
    
    def setup_prompts(self):        
        for prompt_name, prompt_type in self.prompt_types.items():
            node = slicer.mrmlScene.AddNewNodeByClass(prompt_type["node_class"])
            node.SetName(prompt_type["name"])
            node.CreateDefaultDisplayNodes()
            
            display_node = node.GetDisplayNode()
            prompt_type["display_node_markup_function"](display_node)
            
            place_widget = prompt_type["place_widget"]
            place_widget.setMRMLScene(slicer.mrmlScene)
            place_widget.buttonsVisible = False
            place_widget.placeButton().show()
            # place_widget.deleteButton().hide()
            place_widget.setCurrentNode(node)

            place_button = place_widget.placeButton()
            place_button.setText(prompt_type["button_text"])
            place_button.setToolButtonStyle(qt.Qt.ToolButtonTextOnly)
            
            # place_button.setIcon(qt.QIcon())
            place_button.setStyleSheet("min-height: 22px; font-size: 13pt;")

            place_button = place_widget.placeButton()

            # 1) Make the button checkable (so it can appear "pressed"/selected)
            place_button.setCheckable(True)

            # 2) Define a style sheet for the checked state
            place_button.setStyleSheet("""
                QToolButton {
                    /* normal state: no special styling */
                    min-height: 22px;
                    font-size: 13pt;
                }
                QToolButton:checked {
                    background-color: #3498db;  /* blue */
                    color: white;
                }
            """)

            self.prev_caller = None        
            
            if prompt_type["on_placed_function"] is not None:
                node.AddObserver(slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent, 
                                 prompt_type["on_placed_function"])      
                
            prompt_type["node"] = node

    def remove_prompt_nodes(self):
        for prompt_type in self.prompt_types.values():
            existing_nodes = slicer.mrmlScene.GetNodesByName(prompt_type["name"])
            if existing_nodes and existing_nodes.GetNumberOfItems() > 0:
                for i in range(existing_nodes.GetNumberOfItems()):
                    node = existing_nodes.GetItemAsObject(i)
                    slicer.mrmlScene.RemoveNode(node)

    def remove_all_but_last_prompt(self):
        last_modified_node = None
        all_nodes = []
        
        for prompt_type in self.prompt_types.values():
            existing_nodes = slicer.mrmlScene.GetNodesByName(prompt_type["name"])
            if existing_nodes and existing_nodes.GetNumberOfItems() > 0:
                for i in range(existing_nodes.GetNumberOfItems()):
                    node = existing_nodes.GetItemAsObject(i)
                    
                    all_nodes.append(node)
                    if last_modified_node is None or node.GetMTime() > last_modified_node.GetMTime():
                        last_modified_node = node
        
        for node in all_nodes:
            n = node.GetNumberOfControlPoints()
            
            if node == last_modified_node:
                if node.GetName() == "LassoPrompt":
                    continue
                n -= 1
            
            for i in range(n):
                node.RemoveNthControlPoint(0)
        

    def setup_markups_points(self):
        """Initialize the markups fiducial list for storing point prompts"""
        self.remove_prompt_nodes()
        
        self.setup_prompts()
        
        # Setup for interactive placement
        self.is_placing_positive = False
        self.is_placing_negative = False
    
    def install_dependencies(self):
        dependencies = {
            'xxhash': 'xxhash==3.5.0',
            'requests_toolbelt': 'requests_toolbelt==1.0.0',
            'SimpleITK': 'SimpleITK==2.3.1'
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
        shortcuts = {
            "return": self.submit_lasso_if_present,
            "t": self.toggle_prompt_type,  # Add 'T' shortcut to toggle between positive/negative
        }
        self.shortcut_items = {}
        
        for shortcut_key, shortcut_event in shortcuts.items():
            print(f'Added shortcut for {shortcut_key}: {shortcut_event}')
            shortcut = qt.QShortcut(qt.QKeySequence(shortcut_key), slicer.util.mainWindow())
            shortcut.activated.connect(shortcut_event)
            self.shortcut_items[shortcut_key] = shortcut
    
    def remove_shortcut_items(self):
        if hasattr(self, 'shortcut_items'):
            for _, shortcut in self.shortcut_items.items():
                shortcut.setParent(None)
                shortcut.deleteLater()
                shortcut = None
    
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

        action = qt.QAction(qt.QIcon(self.resourcePath("Icons/nnInteractiveSlicer.png")), "nnInteractiveSlicer", toolbar)
        action.setObjectName("nninteractive_slicer_action")
        action.setToolTip("Switch to nnInteractiveSlicer module")
        action.triggered.connect(lambda: slicer.util.selectModule("nnInteractiveSlicer"))
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

            from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor

            # self.ui.uploadProgressGroup.setVisible(True)

            slicer.progress_window = slicer.util.createProgressDialog(autoClose=False)
            slicer.progress_window.minimum = 0
            slicer.progress_window.maximum = 100
            slicer.progress_window.setLabelText("Uploading image...")

            def my_callback(monitor):
                if not hasattr(monitor, 'last_update'):
                    monitor.last_update = time.time()

                if time.time() - monitor.last_update <= .2:
                    return
                monitor.last_update = time.time()

                slicer.progress_window.setValue(monitor.bytes_read / len(compressed_data) * 100)
                slicer.progress_window.show()
                slicer.progress_window.activateWindow()
                slicer.progress_window.setLabelText("Uploading image...")
                # Process events to allow screen to refresh
                slicer.app.processEvents()

            e = MultipartEncoder(
                fields=files
            )
            m = MultipartEncoderMonitor(e, my_callback)
            
            response = requests.post(
                url,
                data=m,
                headers={"Content-Encoding": "gzip", 'Content-Type': m.content_type}
            )
            print('Response took', time.time() - t0)
            
            if response.status_code == 200:
                print("Image successfully uploaded to server.")
            else:
                print("Image upload failed with status code:", response.status_code)

            # self.ui.uploadProgressGroup.setVisible(False)
            slicer.progress_window.close()
        except Exception as e:
            print("Error in upload_image_to_server:", e)

    def mask_to_np_upload_file(self, mask):
        buffer = io.BytesIO()
        np.save(buffer, mask)
        compressed_data = gzip.compress(buffer.getvalue())
        
        files = {
            'file': ('volume.npy.gz', compressed_data, 'application/octet-stream')
        }
        
        return files

    def upload_segment_to_server(self):
        print("Syncing segment with server...")
        # return
        try:
            segment_data = self.get_segment_data()
            files = self.mask_to_np_upload_file(segment_data)
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
    def point_prompt(self, xyz=None, positive_click=False):
        url = f"{self.server}/add_point_interaction"
        
        seg_response = requests.post(
            url, 
            json={'voxel_coord': xyz[::-1],
                  'positive_click': positive_click})
        
        unpacked_segmentation = self.unpack_binary_segmentation(seg_response.content, decompress=False)
        print(seg_response)
        print(f"{positive_click} point prompt triggered!", xyz)
        
        self.show_segmentation(unpacked_segmentation)
    
    @ensure_synched
    def bbox_prompt(self, outer_point_one, outer_point_two, positive_click=False):
        url = f"{self.server}/add_bbox_interaction"
        
        seg_response = requests.post(
            url, 
            json={'outer_point_one': outer_point_one[::-1],
                  'outer_point_two': outer_point_two[::-1],
                  'positive_click': positive_click})
        
        unpacked_segmentation = self.unpack_binary_segmentation(seg_response.content, decompress=False)
        print('np.sum(unpacked_segmentation):', np.sum(unpacked_segmentation))
        self.show_segmentation(unpacked_segmentation)
    
    @ensure_synched
    def lasso_prompt(self, mask, positive_click=False):
        url = f"{self.server}/add_lasso_interaction"
        print(url)
        try:
            import SimpleITK as sitk
            sim = sitk.GetImageFromArray(mask)
            sitk.WriteImage(sim, '/Users/coendevente/Desktop/a.nii.gz')
            
            buffer = io.BytesIO()
            np.save(buffer, mask)
            compressed_data = gzip.compress(buffer.getvalue())
            
            from requests_toolbelt import MultipartEncoder

            fields = {
                'file': ('volume.npy.gz', compressed_data, 'application/octet-stream'),
                'positive_click': str(positive_click)  # Make sure to send it as a string.
            }
            encoder = MultipartEncoder(fields=fields)
            seg_response = requests.post(
                url,
                data=encoder,
                headers={"Content-Type": encoder.content_type, "Content-Encoding": "gzip"}
            )
            
            if seg_response.status_code == 200:
                unpacked_segmentation = self.unpack_binary_segmentation(seg_response.content, decompress=False)
                print('np.sum(unpacked_segmentation):', np.sum(unpacked_segmentation))
                self.show_segmentation(unpacked_segmentation)
            else:
                print("Lasso prompt upload failed with status code:", seg_response.status_code)
        except Exception as e:
            print("Error in lasso_prompt:", e)
    
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
        
        self.previous_states["segment_data"] = segmentation_mask
        
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
    
    def image_changed(self, do_prev_image_update=True):
        image_data = self.get_image_data()
        if image_data is None:
            print("No volume node found")
            return
        
        old_image_data = self.previous_states.get("image_data", None)
        image_changed =  old_image_data is None or not np.all(old_image_data == image_data)

        if do_prev_image_update:
            self.previous_states["image_data"] = copy.deepcopy(image_data)

        return image_changed

    def selected_segment_changed(self):
        segment_data = self.get_segment_data()
        old_segment_data = self.previous_states.get("segment_data", None)
        selected_segment_changed = old_segment_data is None or not np.all(old_segment_data.astype(bool) == segment_data.astype(bool))
        
        print('segment_data.sum():', segment_data.sum())
        
        if old_segment_data is not None:
            print('old_segment_data.sum():', old_segment_data.sum())
        else:
            print('old_segment_data is None')

        print('selected_segment_changed:', selected_segment_changed)

        return selected_segment_changed

    def get_widget_segment_editor(self):
        return slicer.modules.segmenteditor.widgetRepresentation().self().editor
    
    def get_current_segment_id(self):
        segment_editor_widget = self.get_widget_segment_editor()
        return segment_editor_widget.mrmlSegmentEditorNode().GetSelectedSegmentID()
        
    def cleanup(self):
        """Clean up resources when the module is closed"""
        if hasattr(self, "_qt_event_filters"):
            for slice_view, event_filter in self._qt_event_filters:
                slice_view.removeEventFilter(event_filter)
            self._qt_event_filters = []

        self.remove_shortcut_items()
        return

    def __del__(self):
        # pass
        self.remove_shortcut_items()
    
    def ras_to_xyz(self, pos):
        volumeNode = self.get_volume_node()
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
        
        return xyz
    
    def xyz_from_caller(self, caller, lock_point=True, return_all=False):
        n = caller.GetNumberOfControlPoints()
        if n < 0:
            print("No control points found")
            return
        
        xyzs = []
        ids = range(n) if return_all else [n - 1]
        
        for i in ids:
            pos = [0, 0, 0]
            caller.GetNthControlPointPosition(i, pos)
            if lock_point:
                caller.SetNthControlPointLocked(i, True)
            xyz = self.ras_to_xyz(pos)
            xyzs.append(xyz)
        
        if not return_all:
            return xyzs[0]
        
        return xyzs
    
    @property
    def is_positive(self):
        return self.ui.pbPromptTypePositive.isChecked()
    
    def on_point_placed(self, caller, event):
        """Called when a point is placed in the scene"""        
        # Determine which node called this (positive or negative)
        xyz = self.xyz_from_caller(caller)
        
        volume_node = self.get_volume_node()
        if volume_node:
            
            # Call point_prompt with the voxel coordinates
            self.point_prompt(xyz=xyz, positive_click=self.is_positive)

            qt.QTimer.singleShot(0, self.ui.pointPlaceWidget.placeButton().click)
            print("Scheduled point placement restart with timer")
    
    def on_bbox_placed(self, caller, event):
        # This method will be called every time a point is defined or moved
        placeButton = self.ui.bboxPlaceWidget.placeButton()
        xyz = self.xyz_from_caller(caller)

        if self.prev_caller is not None and caller.GetID() == self.prev_caller.GetID():
            print("placed!")
            
            print(xyz, self.prev_bbox_xyz)
            
            roi_node = slicer.mrmlScene.GetNodeByID(caller.GetID())
            current_size = list(roi_node.GetSize())
            drawn_in_axis = np.argwhere(np.array(xyz) == self.prev_bbox_xyz).squeeze()
            current_size[drawn_in_axis] = 0 
            roi_node.SetSize(current_size)
            
            volume_node = self.get_volume_node()
            if volume_node:                
                outer_point_two=self.prev_bbox_xyz
                
                outer_point_one = [xyz[0] * 2 - outer_point_two[0],
                                   xyz[1] * 2 - outer_point_two[1],
                                   xyz[2] * 2 - outer_point_two[2]]
                
                
                self.bbox_prompt(outer_point_one=outer_point_one, 
                                 outer_point_two=outer_point_two, 
                                 positive_click=self.is_positive)

                def _next():
                    self.setup_prompts()
                    qt.QTimer.singleShot(0, placeButton.click)
                qt.QTimer.singleShot(0, _next)
            
            self.prev_caller = None
        else:
            self.prev_bbox_xyz = xyz

        self.prev_caller = caller
        
    def on_lasso_placed(self, caller, event):
        self.ui.lassoPlaceWidget.placeButton().setText("Lasso [Hit Enter to finish]")
        
    def on_lasso_clicked(self, checked=False):
        if checked:
            self.ui.lassoPlaceWidget.placeButton().setText("Lasso [Hit Enter to finish]")
        else:
            self.ui.lassoPlaceWidget.placeButton().setText(self.prompt_types["lasso"]["button_text"])

    def lasso_points_to_mask(self, points):
        shape = self.get_image_data().shape
        pts = np.array(points)  # shape (n, 3)
        
        # Determine which coordinate is constant
        const_axes = [i for i in range(3) if np.unique(pts[:, i]).size == 1]
        if len(const_axes) != 1:
            raise ValueError("Expected exactly one constant coordinate among the points")
        const_axis = const_axes[0]
        const_val = int(pts[0, const_axis])
        
        # Create a blank 3D mask
        mask = np.zeros(shape, dtype=np.uint8)
        
        # Depending on which axis is constant, extract the 2D polygon and fill the corresponding slice.
        # Note: our volume is ordered as (z, y, x)
        if const_axis == 2:
            x_coords = pts[:, 0]
            y_coords = pts[:, 1]
            rr, cc = polygon(y_coords, x_coords, shape=(shape[1], shape[2]))
            mask[const_val, rr, cc] = 1
        elif const_axis == 1:
            x_coords = pts[:, 0]
            z_coords = pts[:, 2]
            rr, cc = polygon(z_coords, x_coords, shape=(shape[0], shape[2]))
            mask[rr, const_val, cc] = 1
        elif const_axis == 0:
            y_coords = pts[:, 1]
            z_coords = pts[:, 2]
            rr, cc = polygon(z_coords, y_coords, shape=(shape[0], shape[1]))
            mask[rr, cc, const_val] = 1
            
        return mask

    def submit_lasso_if_present(self):
        caller = self.prompt_types["lasso"]["node"]            
        
        print('Lasso finished!')
        xyzs = self.xyz_from_caller(caller, return_all=True)
        mask = self.lasso_points_to_mask(xyzs)
        
        print('xyzs:', xyzs)
        print('mask.shape:', mask.shape)
        print('np.sum(mask):', np.sum(mask))
        
        volume_node = self.get_volume_node()
        if volume_node:
            self.lasso_prompt(mask=mask, positive_click=self.is_positive)

            def _next():
                self.setup_prompts()
                qt.QTimer.singleShot(0, self.ui.lassoPlaceWidget.placeButton().click)
                self.ui.lassoPlaceWidget.placeButton().setText(self.prompt_types["lasso"]["button_text"])
            
            print("Scheduled point placement restart with timer")
            qt.QTimer.singleShot(0, _next)

    def on_prompt_type_positive_clicked(self, checked=False):
        """Set the current prompt type to positive"""        
        # Update UI
        self.current_prompt_type_positive = True
        self.ui.pbPromptTypePositive.setStyleSheet(self.selected_style)
        self.ui.pbPromptTypeNegative.setStyleSheet(self.unselected_style)
        self.ui.pbPromptTypePositive.setChecked(True)
        self.ui.pbPromptTypeNegative.setChecked(False)
        print("Prompt type set to POSITIVE")
    
    def toggle_prompt_type(self, checked=False):
        """Toggle between positive and negative prompt types (triggered by 'T' key)"""
        print("Toggling prompt type (positive <> negative)")
        if self.current_prompt_type_positive:
            self.on_prompt_type_negative_clicked()
        else:
            self.on_prompt_type_positive_clicked()
    
    def on_prompt_type_negative_clicked(self, checked=False):
        """Set the current prompt type to negative"""
        
        # Update UI
        self.current_prompt_type_positive = False
        self.ui.pbPromptTypePositive.setStyleSheet(self.unselected_style)
        self.ui.pbPromptTypeNegative.setStyleSheet(self.selected_style)
        self.ui.pbPromptTypePositive.setChecked(False)
        self.ui.pbPromptTypeNegative.setChecked(True)
        print("Prompt type set to NEGATIVE")
