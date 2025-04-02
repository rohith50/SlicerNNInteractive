import io
import gzip
import requests
import copy
import threading
import time

import importlib.util

import numpy as np

import slicer
import qt
import vtk

from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from PythonQt.QtGui import QMessageBox



def ensure_synched(func):
    def inner(self, *args, **kwargs):
        failed_to_sync = False
        
        if self.image_changed():
            print("Image changed (or not previously set). Calling upload_segment_to_server()")
            result = self.upload_image_to_server()
            
            failed_to_sync = result is None
        
        if not failed_to_sync and self.selected_segment_changed():
            print("Segment changed (or not previously set). Calling upload_segment_to_server()")
            self.remove_all_but_last_prompt()
            result = self.upload_segment_to_server()
            
            failed_to_sync = result is None
        else:
            print("Segment did not change!")
        
        if not failed_to_sync:
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
        
        self.prompt_types = {
            "point": {
                "node_class": "vtkMRMLMarkupsFiducialNode",
                "node": None,
                "name": "PointPrompt",
                "display_node_markup_function": self.display_node_markup_point,
                "on_placed_function": self.on_point_placed,
                "place_widget": self.ui.pointPlaceWidget,
                "button": self.ui.pbInteractionPoint,
                "button_text": self.ui.pbInteractionPoint.text,
            },
            "bbox": {
                "node_class": "vtkMRMLMarkupsROINode",
                "node": None,
                "name": "BBoxPrompt",
                "display_node_markup_function": self.display_node_markup_bbox,
                "on_placed_function": self.on_bbox_placed,
                "place_widget": self.ui.bboxPlaceWidget,
                "button": self.ui.pbInteractionBBox,
                "button_text": self.ui.pbInteractionBBox.text,
            },
            "lasso": {
                "node_class": "vtkMRMLMarkupsClosedCurveNode",
                "node": None,
                "name": "LassoPrompt",
                "display_node_markup_function": self.display_node_markup_lasso,
                "on_placed_function": self.on_lasso_placed,
                "place_widget": self.ui.lassoPlaceWidget,
                "button": self.ui.pbInteractionLasso,
                "button_text": self.ui.pbInteractionLasso.text,
            }
        }
        
        self.setup_shortcuts()
        
        self.all_prompt_buttons = {}
        self.setup_prompts()
        
        self.init_ui_functionality()
        
        _ = self.get_current_segment_id()
        self.previous_states = {}

    def update_server(self):
        # Get the updated server URL from the UI
        self.server = self.ui.Server.text.rstrip("/")
        
        # Save the server URL to QSettings
        settings = qt.QSettings()
        settings.setValue("nnInteractiveSlicer/server", self.server)
        
        print(f"Server URL updated and saved: {self.server}")
        
    def request_to_server(self, *args, **kwargs):
        error_message = None
        try:
            response = requests.post(*args, **kwargs)
        except requests.exceptions.MissingSchema as e:
            if self.server == '':
                error_message = "It seems you have not set the server URL yet!"
            else:
                error_message = "It seems the Server URL is unreachable!"

            error_message += f"""
            
You can configure it in the 'Configuration' menu of the nnInteractiveSlicer plugin.

This is the error: {e}."""
        except Exception as e:
            error_message = f"""Your request was unsuccessful.


This is the error: {e}."""
        
        if error_message is None and response.status_code != 200:
            error_message = f"""Something seems to have gone wrong with your request (Status code {response.status_code})."""
        
        if error_message is not None:
            QMessageBox.warning(
                slicer.util.mainWindow(),
                "Error",
                error_message
            )
            return None
    
        return response

    def init_ui_functionality(self):
        self.ui.uploadProgressGroup.setVisible(False)

        # Load the saved server URL (default to an empty string if not set)
        savedServer = slicer.util.settingsValue("nnInteractiveSlicer/server", "")
        self.ui.Server.text = savedServer
        self.server = savedServer.rstrip("/")

        self.ui.Server.editingFinished.connect(self.update_server)
        
        # Set up style sheets for selected/unselected buttons
        self.selected_style = "background-color: #3498db; color: white; min-height: 28px; font-size: 13pt;"
        self.unselected_style = "min-height: 28px; font-size: 13pt;"
        
        # Set initial prompt type
        self.current_prompt_type_positive = True
        self.ui.pbPromptTypePositive.setStyleSheet(self.selected_style)
        self.ui.pbPromptTypeNegative.setStyleSheet(self.unselected_style)
        
        # Top buttons
        self.ui.pbResetSegment.clicked.connect(self.clear_current_segment)
        self.ui.pbNextSegment.clicked.connect(self.make_new_segment)
        
        # Connect Prompt Type buttons
        self.ui.pbPromptTypePositive.clicked.connect(self.on_prompt_type_positive_clicked)
        self.ui.pbPromptTypeNegative.clicked.connect(self.on_prompt_type_negative_clicked)
        
        self.ui.pbInteractionScribble.clicked.connect(self.on_scribble_clicked)
        
        self.ui.pbInteractionScribble.setCheckable(True)
        self.ui.pbInteractionLasso.clicked.connect(self.on_lasso_clicked)
        
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
        display_node.SetColor(0.0, 0.0, 1.0)  # Green color
        display_node.SetSelectedColor(0.0, 0.0, 1.0)
        display_node.SetActiveColor(0.0, 0.0, 1.0)
        display_node.SetOpacity(1.0)  # Fully opaque
        display_node.SetSliceProjection(False)  # Make points visible in all slice views
    
    def setup_prompts(self, skip_if_exists=False):
        if not skip_if_exists:
            self.remove_prompt_nodes()
        
        unselected_style = """
                    min-height: 30px;
                    font-size: 13pt;
        """
        
        selected_style = """
            min-height: 30px;
            background-color: #3498db;  /* blue */
            color: white;
        """
        
        for prompt_name, prompt_type in self.prompt_types.items():
            if skip_if_exists and slicer.mrmlScene.GetFirstNodeByName(prompt_type["name"]):
                print('Skipping', prompt_name)
                continue
            node = slicer.mrmlScene.AddNewNodeByClass(prompt_type["node_class"])
            node.SetName(prompt_type["name"])
            node.CreateDefaultDisplayNodes()
            
            display_node = node.GetDisplayNode()
            prompt_type["display_node_markup_function"](display_node)
            
            place_widget = prompt_type["place_widget"]
            place_widget.setMRMLScene(slicer.mrmlScene)
            place_widget.setCurrentNode(node)

            place_button = place_widget.placeButton()
            place_button.setText(prompt_type["button_text"])
            place_button.setToolButtonStyle(qt.Qt.ToolButtonTextOnly)

            place_button = place_widget.placeButton()

            place_button.setCheckable(True)

            prompt_type["button"].setStyleSheet(f"""
                QPushButton {{
                    {unselected_style}
                }}
                QPushButton:checked {{
                    {selected_style}
                }}
            """)
            
            place_widget.hide()

            self.prev_caller = None        
            
            if prompt_type["on_placed_function"] is not None:
                node.AddObserver(slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent, 
                                 prompt_type["on_placed_function"])      
                
            prompt_type["node"] = node
            prompt_type["button"].clicked.connect(
                self.get_on_button_clicked_function(place_widget, prompt_type["button"])
            )
            self.all_prompt_buttons[prompt_name] = prompt_type["button"]
        
        if not skip_if_exists or slicer.mrmlScene.GetFirstNodeByName("ScribbleSegmentNode") is None:
            self.setup_scribble_prompt()
        
            self.ui.pbInteractionScribble.setStyleSheet(f"""
                QPushButton {{
                    {unselected_style}
                }}
                QPushButton:checked {{
                    {selected_style}
                }}
            """)
            self.all_prompt_buttons["scribble"] = self.ui.pbInteractionScribble
        
        # To make sure that when segment is reset, no interaction is selected (without this code
        # the last interaction tool gets selected)
        interaction_node = slicer.app.applicationLogic().GetInteractionNode()
        interaction_node.SetCurrentInteractionMode(interaction_node.ViewTransform)
        
    def hide_all_but_this_button(self, this_button):
        for button in self.all_prompt_buttons.values():
            if button != this_button:
                button.setChecked(False)
            
        if this_button != self.ui.pbInteractionScribble:
            self.set_lasso_unselected_text()
    
    def get_on_button_clicked_function(self, place_widget, this_button):
        def on_button_clicked(checked=False):
            self.hide_all_but_this_button(this_button)
            self.setup_prompts(skip_if_exists=True)
            place_widget.setPlaceModeEnabled(checked)
        return on_button_clicked
    
    def on_scribble_clicked(self, checked=False):
        self.hide_all_but_this_button(self.ui.pbInteractionScribble)
        
        if not checked:
            # Deactivate paint effect
            if self.scribble_editor_widget:
                self.scribble_editor_widget.setActiveEffectByName("")  # Clears the active effect
            
            # Optionally clear or reset the segmentation node
            if hasattr(self, "_scribble_labelmap_callback_tag"):
                tag = self._scribble_labelmap_callback_tag.get("tag", None)
                if tag:
                    self.scribble_segment_node.RemoveObserver(tag)
                del self._scribble_labelmap_callback_tag

            return

        segment_id = "fg" if self.is_positive else "bg"

        # Set segmentation and segment
        self.scribble_editor_widget.setSegmentationNode(self.scribble_segment_node)
        self.scribble_editor_node.SetSelectedSegmentID(segment_id)

        # Set reference volume
        volume_node = self.get_volume_node()
        self.scribble_editor_widget.setSourceVolumeNode(volume_node)

        # Activate paint effect
        self.scribble_editor_widget.setActiveEffectByName("Paint")
        self.scribble_editor_widget.updateWidgetFromMRML()
        
        paint_effect = self.scribble_editor_widget.activeEffect()
        if paint_effect:
            paint_effect.setParameter("BrushUseAbsoluteSize", "0")  # Use relative mode
            paint_effect.setParameter("BrushSphere", "0")  # 2D brush
            paint_effect.setParameter("BrushRelativeDiameter", ".75")
            self._scribble_labelmap_callback_tag = {
                "tag": self.scribble_segment_node.AddObserver(vtk.vtkCommand.AnyEvent, self.on_scribble_finished),
                "label_name": segment_id
            }
        
        print(f"Scribble mode (hidden editor) activated on '{segment_id}'")

    
    def setup_scribble_prompt(self):
        import qSlicerSegmentationsModuleWidgetsPythonQt

        # Create a background (headless) segment editor
        self.scribble_editor_widget = qSlicerSegmentationsModuleWidgetsPythonQt.qMRMLSegmentEditorWidget()
        self.scribble_editor_widget.setMRMLScene(slicer.mrmlScene)
        self.scribble_editor_widget.setMaximumNumberOfUndoStates(10)

        # Create a separate SegmentEditorNode
        self.scribble_editor_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        self.scribble_editor_widget.setMRMLSegmentEditorNode(self.scribble_editor_node)
        
        self.scribble_segment_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.scribble_segment_node.SetReferenceImageGeometryParameterFromVolumeNode(self.get_volume_node())
        self.scribble_segment_node.SetName("ScribbleSegmentNode")

        # Make sure the node exists and is set
        self.scribble_editor_widget.setSegmentationNode(self.scribble_segment_node)
        # return

        self.scribble_segment_node.CreateDefaultDisplayNodes()
        # self.scribble_segment_node.GetSegmentation().AddEmptySegment("bg", "bg", [1.0, 0.0, 0.0])
        # self.scribble_segment_node.GetSegmentation().AddEmptySegment("fg", "fg", [0.0, 1.0, 0.0])
        self.scribble_segment_node.GetSegmentation().AddEmptySegment("bg", "bg", [0.0, 0.0, 1.0])
        self.scribble_segment_node.GetSegmentation().AddEmptySegment("fg", "fg", [0.0, 0.0, 1.0])
        dn = self.scribble_segment_node.GetDisplayNode()
        
        opacity = 0.2
        dn.SetSegmentOpacity2DFill("bg", opacity)
        dn.SetSegmentOpacity2DOutline("bg", opacity)
        dn.SetSegmentOpacity2DFill("fg", opacity)
        dn.SetSegmentOpacity2DOutline("fg", opacity)
        
    def on_scribble_finished(self, caller, event):
        print("Scribble stroke finished - labelmap modified!")
        
        # Clean up observer if you only want it once
        if hasattr(self, "_scribble_labelmap_callback_tag"):
            caller.RemoveObserver(self._scribble_labelmap_callback_tag["tag"])
            label_name = self._scribble_labelmap_callback_tag["label_name"]
            del self._scribble_labelmap_callback_tag
        else:
            return
            
        mask = slicer.util.arrayFromSegmentBinaryLabelmap(self.scribble_segment_node, 
                                                          label_name, 
                                                          self.get_volume_node())

        if hasattr(self, "_prev_scribble_mask"):
            prev_scribble_mask = self._prev_scribble_mask
        else:
            prev_scribble_mask = mask * 0
            
        diff_mask = mask - prev_scribble_mask
        self._prev_scribble_mask = mask
        
        self.lasso_or_scribble_prompt(mask=diff_mask, positive_click=self.is_positive, tp="scribble")
        
        self.ui.pbInteractionScribble.click()  # turn it off
        self.ui.pbInteractionScribble.click()  # turn it on
        
    def remove_prompt_nodes(self):
        def _remove(node_name):
            existing_nodes = slicer.mrmlScene.GetNodesByName(node_name)
            if existing_nodes and existing_nodes.GetNumberOfItems() > 0:
                for i in range(existing_nodes.GetNumberOfItems()):
                    node = existing_nodes.GetItemAsObject(i)
                    slicer.mrmlScene.RemoveNode(node)
        
        for prompt_type in list(self.prompt_types.values()):
            _remove(prompt_type["name"])
        
        _remove("ScribbleSegmentNode")
        
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
    
    def install_dependencies(self):
        dependencies = {
            'xxhash': 'xxhash==3.5.0',
            'requests_toolbelt': 'requests_toolbelt==1.0.0',
            'SimpleITK': 'SimpleITK==2.3.1',
            'skimage': 'skimage==0.22.0'
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
            "o": self.ui.pbInteractionPoint.click,
            "b": self.ui.pbInteractionBBox.click,
            "l": self.ui.pbInteractionLasso.click,
            "s": self.ui.pbInteractionScribble.click,
            "e": self.make_new_segment,
            "r": self.clear_current_segment,
            "Shift+L": self.submit_lasso_if_present,
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

        self.editor_widget = qSlicerSegmentationsModuleWidgetsPythonQt.qMRMLSegmentEditorWidget()
        self.editor_widget.setMaximumNumberOfUndoStates(10)
        
        segment_editor_singleton_tag = "SegmentEditor"
        self.segment_editor_node = slicer.mrmlScene.GetSingletonNode(
            segment_editor_singleton_tag, "vtkMRMLSegmentEditorNode"
        )
        
        if self.segment_editor_node is None:
            self.segment_editor_node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLSegmentEditorNode")
            self.segment_editor_node.UnRegister(None)
            self.segment_editor_node.SetSingletonTag(segment_editor_singleton_tag)
            self.segment_editor_node = slicer.mrmlScene.AddNode(self.segment_editor_node)
        
        self.editor_widget.setMRMLSegmentEditorNode(self.segment_editor_node)
        self.editor_widget.setMRMLScene(slicer.mrmlScene)
        
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
            layout.addWidget(self.editor_widget)
        else:
            print("Could not find segmentationGroup in UI")
    
    def get_volume_node(self):
        # Attempt to get the active volume from the selection node.
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        activeVolumeID = selectionNode.GetActiveVolumeID()
        if activeVolumeID:
            volumeNode = slicer.mrmlScene.GetNodeByID(activeVolumeID)
            if volumeNode:
                return volumeNode

        # Ff no active volume is selected, return the first available volume.
        volumeNodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        if volumeNodes:
            # Since getNodesByClass returns a dict, get one of the values.
            return list(volumeNodes.values())[0]

        return None
    
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
        print("Syncing image with server...")
        try:
            # Retrieve image data, window, and level.
            t0 = time.time()
            image_data = self.get_image_data()  # Expected to return (image_data, window, level)
            print(f'self.get_image_data took {time.time() - t0}')
            
            if image_data is None:
                print("No image data available to upload.")
                return
            
            t0 = time.time()
            url = f"{self.server}/upload_image"  # Update this with your actual endpoint.
            
            buffer = io.BytesIO()
            np.save(buffer, image_data)
            raw_data = buffer.getvalue()
            print(f'len(raw_data): {len(raw_data)}')

            files = {
                'file': ('volume.npy', raw_data, 'application/octet-stream')
            }

            # Create your MultipartEncoder without gzip headers
            from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor

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
                slicer.progress_window.setValue(monitor.bytes_read / len(raw_data) * 100)
                slicer.progress_window.show()
                slicer.progress_window.activateWindow()
                slicer.progress_window.setLabelText("Uploading image...")
                slicer.app.processEvents()

            encoder = MultipartEncoder(fields=files)
            monitor = MultipartEncoderMonitor(encoder, my_callback)

            result = self.request_to_server(
                url,
                data=monitor,
                headers={'Content-Type': monitor.content_type}
            )

            slicer.progress_window.close()
            
            return result
        except Exception as e:
            print(f"Error in upload_image_to_server: {e}")

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
        try:
            segment_data = self.get_segment_data()
            files = self.mask_to_np_upload_file(segment_data)
            url = f"{self.server}/upload_segment"  # Update this with your actual endpoint.
            
            result = self.request_to_server(
                url,
                files=files,
                headers={"Content-Encoding": "gzip"}
            )
            
            return result
        except Exception as e:
            print(f"Error in upload_image_to_server: {e}")
    
    @ensure_synched
    def point_prompt(self, xyz=None, positive_click=False):
        url = f"{self.server}/add_point_interaction"
        
        seg_response = self.request_to_server(
            url, 
            json={'voxel_coord': xyz[::-1],
                  'positive_click': positive_click})
        
        unpacked_segmentation = self.unpack_binary_segmentation(seg_response.content, decompress=False)
        print('unpacked_segmentation.sum():', unpacked_segmentation.sum())
        print(seg_response)
        print(f"{positive_click} point prompt triggered! {xyz}")
        
        self.show_segmentation(unpacked_segmentation)
    
    @ensure_synched
    def bbox_prompt(self, outer_point_one, outer_point_two, positive_click=False):
        url = f"{self.server}/add_bbox_interaction"
        
        seg_response = self.request_to_server(
            url, 
            json={'outer_point_one': outer_point_one[::-1],
                  'outer_point_two': outer_point_two[::-1],
                  'positive_click': positive_click})
        
        unpacked_segmentation = self.unpack_binary_segmentation(seg_response.content, decompress=False)
        self.show_segmentation(unpacked_segmentation)
    
    @ensure_synched
    def lasso_or_scribble_prompt(self, mask, positive_click=False, tp="lasso"):
        url = f"{self.server}/add_{tp}_interaction"
        try:            
            buffer = io.BytesIO()
            np.save(buffer, mask)
            compressed_data = gzip.compress(buffer.getvalue())
            
            from requests_toolbelt import MultipartEncoder

            fields = {
                'file': ('volume.npy.gz', compressed_data, 'application/octet-stream'),
                'positive_click': str(positive_click)  # Make sure to send it as a string.
            }
            encoder = MultipartEncoder(fields=fields)
            seg_response = self.request_to_server(
                url,
                data=encoder,
                headers={"Content-Type": encoder.content_type, "Content-Encoding": "gzip"}
            )
            
            if seg_response.status_code == 200:
                unpacked_segmentation = self.unpack_binary_segmentation(seg_response.content, decompress=False)
                self.show_segmentation(unpacked_segmentation)
            else:
                print(f"lasso_or_scribble_prompt upload failed with status code: {seg_response.status_code}")
        except Exception as e:
            print(f"Error in lasso_or_scribble_prompt: {e}")
    
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

        vol_shape = self.get_image_data().shape
        total_voxels = np.prod(vol_shape)
        unpacked_bits = np.unpackbits(np.frombuffer(binary_data, dtype=np.uint8))
        unpacked_bits = unpacked_bits[:total_voxels]
        
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
        
        print(f'show_segmentation took {time.time() - t0}')
    
    def make_new_segment(self):
        print('doing make_new_segment')
        segmentation_node = self.get_segmentation_node()
        # segment_editor_node = self.get_widget_segment_editor().mrmlSegmentEditorNode()
        
        # Generate a new segment name
        segment_ids = segmentation_node.GetSegmentation().GetSegmentIDs()
        if len(segment_ids) == 0:
            new_segment_name = "Segment_1"
        else:
            # Find the next available number
            segment_numbers = [int(seg.split('_')[-1]) for seg in segment_ids if seg.startswith("Segment_") and seg.split('_')[-1].isdigit()]
            next_segment_number = max(segment_numbers) + 1 if segment_numbers else 1
            new_segment_name = f"Segment_{next_segment_number}"

        # Create and add the new segment
        new_segment_id = segmentation_node.GetSegmentation().AddEmptySegment(new_segment_name)
        self.segment_editor_node.SetSelectedSegmentID(new_segment_id)
        
        # Make sure the right node is selected
        self.editor_widget.setSegmentationNode(segmentation_node)
        self.segment_editor_node.SetSelectedSegmentID(new_segment_id)
        self.editor_widget.updateWidgetFromMRML()

        return segmentation_node, new_segment_id
    
    def clear_current_segment(self):
        """
        Clears the contents (labelmap) of the currently selected segment.
        """
        _, selected_segment_id = self.get_selected_segmentation_node_and_segment_id()

        if selected_segment_id:
            print(f"Clearing segment: {selected_segment_id}")
            self.show_segmentation(np.zeros(self.get_image_data().shape, dtype=np.uint8))
            self.setup_prompts()
            self.upload_segment_to_server()
        else:
            print("No segment selected to clear.")
    
    def get_segmentation_node(self):
        # If the segmentation widget has a currently selected segmentation node, return it.
        if hasattr(self, 'editor_widget') and self.editor_widget.segmentationNode():
            seg_node = self.editor_widget.segmentationNode()
            if seg_node.GetName() != "ScribbleSegmentNode":
                return seg_node

        # Otherwise, fall back to getting the first segmentation node (or create one if none exists).
        segment_editor_node = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
        if segment_editor_node is None or segment_editor_node.GetName() == "ScribbleSegmentNode":
            segment_editor_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            segment_editor_node.SetReferenceImageGeometryParameterFromVolumeNode(self.get_volume_node())
            
        return segment_editor_node
    
    def get_selected_segmentation_node_and_segment_id(self):
        """Retrieve the currently selected segmentation node and segment ID.
        If no segmentation exists, it creates a new one.
        """
        print('doing get_selected_segmentation_node_and_segment_id')
        segmentationNode = self.get_segmentation_node()
        selected_segment_id = self.get_current_segment_id()
        if not selected_segment_id:
            return self.make_new_segment()

        return segmentationNode, selected_segment_id
    
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
        
        print(f'segment_data.sum(): {segment_data.sum()}')
        
        if old_segment_data is not None:
            print(f'old_segment_data.sum(): {old_segment_data.sum()}')
        else:
            print('old_segment_data is None')

        print(f'selected_segment_changed: {selected_segment_changed}')

        return selected_segment_changed

    def get_widget_segment_editor(self):
        return slicer.modules.segmenteditor.widgetRepresentation().self().editor
    
    def get_current_segment_id(self):        
        return self.editor_widget.mrmlSegmentEditorNode().GetSelectedSegmentID()
        
    def cleanup(self):
        """Clean up resources when the module is closed"""
        if hasattr(self, "_qt_event_filters"):
            for slice_view, event_filter in self._qt_event_filters:
                slice_view.removeEventFilter(event_filter)
            self._qt_event_filters = []

        self.remove_shortcut_items()
        return

    def __del__(self):
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
    
    def set_lasso_selected_text(self):
        self.ui.pbInteractionLasso.setText(f"{self.prompt_types['lasso']['button_text']} [Hit Shift+L to finish]")
    
    def set_lasso_unselected_text(self):
        self.ui.pbInteractionLasso.setText(f"{self.prompt_types['lasso']['button_text']}")
    
    def on_lasso_placed(self, caller, event):
        self.set_lasso_selected_text()
        
    def on_lasso_clicked(self, checked=False):
        if checked:
            self.set_lasso_selected_text()
        else:
            self.set_lasso_unselected_text()

    def lasso_points_to_mask(self, points):
        from skimage.draw import polygon

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
        xyzs = self.xyz_from_caller(caller, return_all=True)
        
        if len(xyzs) < 3:
            return
        
        mask = self.lasso_points_to_mask(xyzs)
        
        volume_node = self.get_volume_node()
        if volume_node:
            self.lasso_or_scribble_prompt(mask=mask, positive_click=self.is_positive, tp="lasso")

            def _next():
                self.setup_prompts()
                qt.QTimer.singleShot(0, self.ui.lassoPlaceWidget.placeButton().click)
                self.ui.lassoPlaceWidget.placeButton().setText(self.prompt_types["lasso"]["button_text"])
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
