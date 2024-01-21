# -*- coding:utf-8 -*-
"""
作者：荇子
日期：2023年03月31日
"""
import ctypes
ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)
import math
import os
import sys

import shutil
import time

import nibabel as nib
import numpy as np
import torch.cuda
from PIL import Image
from PIL import ImageTk
import imageio
from PyQt5 import uic, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from qt_material import apply_stylesheet
import qtawesome as qta
from threading import Thread
from functools import partial

from scripts.ImageWidget import NiiImageViewer
from scripts.ModelWidget import NiiModelCore, NiiModelWidget

from scripts.WaterProgress import WaterProgress


# # 获取 路径
# file_path = os.path.dirname(os.path.abspath(__file__))
# # 修改运行路径
# sys.path.append(file_path)
# print("sys path add:" + file_path)
# # 0 表示优先级，数字越大级别越低，修改模块的导入
# sys.path.insert(2, os.path.dirname(file_path))

from nnunet.inference.predict import predict_from_folder
from nnunet.paths import default_plans_identifier, network_training_output_dir, default_cascade_trainer, default_trainer
from batchgenerators.utilities.file_and_folder_operations import join, isdir
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name


class MouseTracker(QtCore.QObject):
    positionChanged = QtCore.pyqtSignal(QtCore.QPoint)

    def __init__(self, widget):
        super().__init__(widget)
        self._widget = widget
        self.widget.setMouseTracking(True)
        self.widget.installEventFilter(self)

    @property
    def widget(self):
        return self._widget

    def eventFilter(self, o, e):
        if o is self.widget and e.type() == QtCore.QEvent.MouseMove:
            self.positionChanged.emit(e.pos())
        return super().eventFilter(o, e)


class MyLabel(QLabel):
    clicked = pyqtSignal()
    right_clicked = pyqtSignal()
    double_clicked = pyqtSignal()  # 自定义信号
    wheel = pyqtSignal(QtGui.QWheelEvent)

    def __init__(self, width, height, parent=None) -> None:
        super().__init__(parent)
        self.resize(width, height)

    def mouseDoubleClickEvent(self, event):  # 双击事件的处理
        self.double_clicked.emit()

    def wheelEvent(self, a0: QtGui.QWheelEvent) -> None:
        self.wheel.emit(a0)

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        self.clicked.emit()
        if ev.button() == Qt.RightButton:
            self.right_clicked.emit()


class RenameWindow(QMainWindow):
    filepath = r'nnUNetFrame\DATASET\nnUNet_raw\nnUNet_raw_data\Task500_BraTS2021\tmp'
    window_data_dict = {
        "T1": "BraTS2021_99999_0000.nii.gz",
        "T1CE": "BraTS2021_99999_0001.nii.gz",
        "T2": "BraTS2021_99999_0002.nii.gz",
        "FLAIR": "BraTS2021_99999_0003.nii.gz"
    }
    options = ["T1", "T1CE", "T2", "FLAIR"]
    closed = pyqtSignal()


    def __init__(self, parent=None):
        super().__init__()
        self.setWindowTitle('Rename Uploads')
        self.resize(300, 100)
        self.setCentralWidget(QWidget())
        self.filenames = os.listdir(self.filepath)

        self.file_group = []
        self.radio_buttons = []
        self.finished = False

        self.main_layout = QGridLayout()
        self.main_layout.setColumnMinimumWidth(0, 150)
        self.main_layout.setColumnMinimumWidth(2, 150)
        self.main_layout.setColumnMinimumWidth(4, 150)
        self.main_layout.setColumnMinimumWidth(6, 150)
        self.main_layout.setColumnMinimumWidth(8, 150)
        self.main_layout.setColumnStretch(0, 1)
        self.main_layout.addWidget(QLabel("FileName"), 0, 0, 1, 1)
        for i in range(4):
            self.main_layout.addWidget(QLabel(self.options[i]), 0, 2*(i+1), 1, 1)
        self.set_radios()
        self.reset()
        self.button_confirm = QPushButton(self)
        self.button_confirm.clicked.connect(self.confirm)
        self.button_confirm.setText('Confirm')
        self.button_cancel = QPushButton(self)
        self.button_cancel.clicked.connect(self.cancel)
        self.button_cancel.setText("Cancel")

        self.main_layout.addWidget(self.button_confirm, len(self.filenames) + 1, 2, 1, 2)

        self.main_layout.addWidget(self.button_cancel, len(self.filenames) + 1, 6, 1, 2)
        self.centralWidget().setLayout(self.main_layout)

    def set_radios(self):
        for i in range(len(self.filenames)):
            self.main_layout.addWidget(QLabel(self.filenames[i]), i+1, 0, 1, 1)
            self.radio_buttons.append([])
            self.file_group.append(QButtonGroup(self))
            self.file_group[i].setExclusive(False)
            for j in range(len(self.options)):
                button = QRadioButton(self)
                button.clicked.connect(partial(self.click_button, i, j))
                self.radio_buttons[i].append(button)
                self.file_group[i].addButton(button, j)
                self.main_layout.addWidget(button, i+1, (j+1)*2, 1, 1)

    def click_button(self, row, column):
        for i in range(len(self.radio_buttons)):
            if i != row:
                self.radio_buttons[i][column].setChecked(False)
        for j in range(4):
            if j != column:
                self.radio_buttons[row][j].setChecked(False)


    def reset(self):
        for group in self.radio_buttons:
            for b in group:
                b.setChecked(False)

    def confirm(self):
        print("confirm")
        self.finished = True
        for file in self.file_group:
            if file.checkedId() == -1:
                self.finished = False
                break

        if not self.finished:
            reply = QMessageBox.question(self, 'confirm',
                                         'Are you sure? All the unselected files will not be uploaded.',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return
            else:
                self.finished = True
        self.rename()
        self.close()

    def cancel(self):
        print("cancel")
        self.finished = True
        shutil.rmtree(self.filepath)
        self.close()

    def rename(self):
        for i in range(len(self.file_group)):
            if self.file_group[i].checkedId() == -1:
                os.remove(os.path.join(self.filepath, self.filenames[i]))
            else:
                id = self.file_group[i].checkedId()
                label = self.options[id]
                os.rename(os.path.join(self.filepath, self.filenames[i]),
                          os.path.join(self.filepath, self.window_data_dict[label]))

    def closeEvent(self, event):
        if self.finished:
            self.closed.emit()
            event.accept()
        else:
            reply = QMessageBox.question(self, 'confirm', 'Are you sure to exit? Selected files will not be uploaded.',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                shutil.rmtree(self.filepath)
                self.closed.emit()
                event.accept()
            else:
                event.ignore()


class WindowQt(QWidget):
    img_group = []
    img_model_data = []
    predict_fdata = []
    data_name = []
    data_window_dict = {
        "BraTS2021_99999_0000.nii.gz": "T1",
        "BraTS2021_99999_0001.nii.gz": "T1CE",
        "BraTS2021_99999_0002.nii.gz": "T2",
        "BraTS2021_99999_0003.nii.gz": "FLAIR"
    }
    shape_x, shape_y, shape_z = 0, 0, 0
    shapes = []
    scale_x, scale_y, scale_z = 0, 0, 0
    scales = []
    status_x, status_y, status_z = 0, 0, 0
    sliders = []
    cross_x, cross_y, cross_z = 0, 0, 0
    points_for_dist = []
    cross_sign = False
    radioButtons = []
    currentIndex = 0
    predicted = False

    core = QWidget

    modal_name = ["t1", "t1ce", "t2", "flair"]
    tumor_color = {1: [249, 190, 0], 2: [24, 135, 73], 3: [1, 102, 184], 4: [164, 45, 232], 5: [233, 77, 44]}
    tumor_name = {1: "ED", 2: "NCR", 3: "ET", 4: "TC", 5: "WT"}
    tumor_buttons = [QCheckBox]
    tumor_show = [True, False, False, False, False, False]
    tumor_volume = [0]

    predict_buttons = []

    # progress_bar = WaterProgress(QThread(), 0)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = uic.loadUi("UI\\qtwindow2.0.ui")
        self.set_background()
        self.connect_functions()
        self.load_nii()
        self.predicted = False
        print("load data finish!")
        self.set_page()

    def connect_functions(self):
        self.ui.uploadButton.clicked.connect(self.upload_nii)
        self.ui.outlineButton.clicked.connect(self.outline)

        self.ui.actionUpload_Data.triggered.connect(self.upload_nii)
        self.ui.actionSave_Segment_Data.triggered.connect(self.export_prediction)
        self.ui.actionAutomatic_Segmentation.triggered.connect(self.outline)

        self.timer = QtCore.QTimer()  # 定义定时器，用于控制显示视频的帧率
        self.timer.start(10)

        self.sliders.append(self.ui.slider_x)
        self.sliders.append(self.ui.slider_y)
        self.sliders.append(self.ui.slider_z)

        self.radioButtons.append(self.ui.radioButton_0)
        self.radioButtons.append(self.ui.radioButton_1)
        self.radioButtons.append(self.ui.radioButton_2)
        self.radioButtons.append(self.ui.radioButton_3)
        self.radioButtons.append(self.ui.radioButton_4)

        # for index in range(5):
        #     self.radioButtons[index].toggled.connect(lambda: self.switch_page(index))

        self.canvas_x = MyLabel(self.ui.canvas_x.width(), self.ui.canvas_x.height(), parent=self.ui.canvas_x)
        self.canvas_y = MyLabel(self.ui.canvas_y.width(), self.ui.canvas_y.height(), parent=self.ui.canvas_y)
        self.canvas_z = MyLabel(self.ui.canvas_z.width(), self.ui.canvas_z.height(), parent=self.ui.canvas_z)

        self.tumor_buttons.append(self.ui.EDButton)
        self.tumor_buttons.append(self.ui.NCRButton)
        self.tumor_buttons.append(self.ui.ETButton)
        self.tumor_buttons.append(self.ui.TCButton)
        self.tumor_buttons.append(self.ui.WTButton)

        self.ui.textBrowser.setText("Eedema:ED —— Yellow\n\
Necrotic Tumor Core:NCR —— Green\n\
Enhancing Tumor:ET —— Blue\n\
Tumor Core:TC = ET + NCR —— Purple\n\
Whole Tumor:WT = ED + ET + NCR —— Red")

        self.statusBar = QLabel()
        self.statusBar.setStyleSheet('font: 25 11pt "Bookman Old Style";')
        self.ui.statusbar.addPermanentWidget(self.statusBar)

        self.predict_buttons.append(self.ui.BS_L_Button)
        self.predict_buttons.append(self.ui.BS_Button)
        self.predict_buttons.append(self.ui.BS_L_TS_Button)
        self.predict_buttons.append(self.ui.BS_TS_Button)

    def reset_page(self):
        self.cross_sign = False
        self.ui.name.setText(self.data_name[0])
        self.radioButtons[0].setChecked(True)
        self.set_status()
        self.ui.slider_x.setMaximum(self.shape_x - 1)
        self.ui.slider_y.setMaximum(self.shape_y - 1)
        self.ui.slider_z.setMaximum(self.shape_z - 1)
        # self.ui.slider_x.setMinimum(0)
        # self.ui.slider_y.setMinimum(1)
        # self.ui.slider_z.setMinimum(1)
        self.ui.slider_x.setValue(self.scale_x)
        self.ui.slider_y.setValue(self.scale_y)
        self.ui.slider_z.setValue(self.scale_z)

        self.scales = []
        self.scales.append(self.scale_x)
        self.scales.append(self.scale_y)
        self.scales.append(self.scale_z)
        print("sliders set!")

        self.ui.ButtonSectionImage.setVisible(True)

        self.timer.timeout.connect(lambda: self.fit_window(self.core, self.ui.canvas_threeD))
        self.ui.radioButton_4.toggled.connect(lambda: self.core.AddOrRemoveVolume(0))
        self.core.rightClickSignal.connect(self.select_point_3d)
        self.core.show()

        for i in range(1, 6):
            self.tumor_buttons[i].setChecked(False)
            self.tumor_buttons[i].setVisible(False)
        self.tumor_show = [True, False, False, False, False, False]
        self.set_icon()

        self.currentIndex = 0
        self.drawImages()

    def set_page(self):
        self.ui.actionxOy_Interface_2.triggered.connect(lambda: self.zoom(2))
        self.ui.actionxOz_Interface_2.triggered.connect(lambda: self.zoom(1))
        self.ui.actionyOz_Interface_4.triggered.connect(lambda: self.zoom(0))

        self.ui.radioButton_0.toggled.connect(lambda: self.switch_page(0))
        self.ui.radioButton_1.toggled.connect(lambda: self.switch_page(1))
        self.ui.radioButton_2.toggled.connect(lambda: self.switch_page(2))
        self.ui.radioButton_3.toggled.connect(lambda: self.switch_page(3))
        self.ui.radioButton_4.toggled.connect(lambda: self.switch_page(4))

        self.ui.slider_x.valueChanged.connect(self.scroll_x)
        self.ui.slider_y.valueChanged.connect(self.scroll_y)
        self.ui.slider_z.valueChanged.connect(self.scroll_z)

        self.canvas_x.right_clicked.connect(self.cross)
        self.canvas_y.right_clicked.connect(self.cross)
        self.canvas_z.right_clicked.connect(self.cross)

        self.canvas_x.double_clicked.connect(lambda: self.zoom(0))
        self.canvas_y.double_clicked.connect(lambda: self.zoom(1))
        self.canvas_z.double_clicked.connect(lambda: self.zoom(2))

        self.canvas_x.wheel.connect(lambda e: self.roll(e, 0))
        self.canvas_y.wheel.connect(lambda e: self.roll(e, 1))
        self.canvas_z.wheel.connect(lambda e: self.roll(e, 2))

        self.ui.EDButton.clicked.connect(lambda: self.view_tumor(1))
        self.ui.NCRButton.clicked.connect(lambda: self.view_tumor(2))
        self.ui.ETButton.clicked.connect(lambda: self.view_tumor(3))
        self.ui.TCButton.clicked.connect(lambda: self.view_tumor(4))
        self.ui.WTButton.clicked.connect(lambda: self.view_tumor(5))

        self.ui.EDButton.clicked.connect(lambda: self.core.AddOrRemoveVolume(1))
        self.ui.NCRButton.clicked.connect(lambda: self.core.AddOrRemoveVolume(2))
        self.ui.ETButton.clicked.connect(lambda: self.core.AddOrRemoveVolume(3))
        self.ui.TCButton.clicked.connect(lambda: self.core.AddOrRemoveVolume(4))
        self.ui.WTButton.clicked.connect(lambda: self.core.AddOrRemoveVolume(5))

        self.ui.zoom_threeD.clicked.connect(self.zoom_3d)
        # self.core.doubleClickSignal.connect(self.zoom_3d)

        self.ui.actionUpload_Segment_Data.triggered.connect(self.upload_prediction)
        self.ui.action3D_Interface.triggered.connect(self.zoom_3d)

    def set_background(self):
        self.ui.centralwidget.setStyleSheet("#centralwidget{background-image:color:html292929")

    def show_nii_warning_massage(self):
        message = QMessageBox.warning(self.ui, "Warning", "只能上传最多4个有效模态",
                                      QMessageBox.Retry | QMessageBox.Cancel, QMessageBox.Retry)
        if message == QMessageBox.Retry:
            self.upload_nii()
        elif message == QMessageBox.Cancel:
            print("取消上传")
        else:
            return

    def show_prediction_warning_massage(self):
        message = QMessageBox.warning(self.ui, "Warning", "上传的预测文件与模型不匹配",
                                      QMessageBox.Retry | QMessageBox.Cancel, QMessageBox.Retry)
        if message == QMessageBox.Retry:
            self.upload_prediction()
        elif message == QMessageBox.Cancel:
            print("取消上传")
        else:
            return

    def upload_nii(self):
        filepath_tmp = r'nnUNetFrame\DATASET\nnUNet_raw\nnUNet_raw_data\Task500_BraTS2021\tmp'

        dialog = QFileDialog(self, "", "../../", "")
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setFilter(QDir.Files)

        if dialog.exec():
            filelist = dialog.selectedFiles()
            print(filelist)
            if len(filelist) == 0:
                return
            if os.path.exists(filepath_tmp):
                shutil.rmtree(filepath_tmp)
            os.mkdir(filepath_tmp)
            for i in filelist:
                shutil.copy(i, filepath_tmp)
            rename_window = RenameWindow()
            rename_window.closed.connect(self.finish_rename_nii)
            rename_window.show()

    def finish_rename_nii(self):
        filepath = r'nnUNetFrame\DATASET\nnUNet_raw\nnUNet_raw_data\Task500_BraTS2021\imagesTs'
        filepath_tmp = r'nnUNetFrame\DATASET\nnUNet_raw\nnUNet_raw_data\Task500_BraTS2021\tmp'
        if not os.path.exists(filepath_tmp):
            return
        list = os.listdir(filepath_tmp)
        if len(list) == 0:
            shutil.rmtree(filepath_tmp)
            return
        if os.path.exists(filepath):
            shutil.rmtree(filepath)
        os.mkdir(filepath)

        for i in list:
            shutil.copy(os.path.join(filepath_tmp, i), filepath)
        shutil.rmtree(filepath_tmp)

        self.predicted = False
        self.load_nii()

    def upload_prediction(self):
        dialog = QFileDialog(self, "", "../../", "")
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setFilter(QDir.Files)

        if dialog.exec():
            filelist = dialog.selectedFiles()
            print(filelist)
            seg = filelist[0]
            s = seg[-6:]
            if s != 'nii.gz':
                return

            predict = nib.load(seg)
            predict_fdata = predict.get_fdata()
            if predict_fdata.shape != self.img_model_data.shape:
                self.show_prediction_warning_massage()
                return

            folder_path = r'nnUNetFrame\DATASET\nnUNet_raw\nnUNet_raw_data\Task500_BraTS2021\ensemble_result'
            filenames = os.listdir(folder_path)
            for f in filenames:
                s = f[-6:]
                if s != 'nii.gz':
                    continue
                img_path = os.path.join(folder_path, f)
                os.remove(img_path)
            shutil.copy(seg, folder_path)

            self.predict_fdata = np.uint8(predict_fdata)
            self.affine = predict.affine.copy()
            self.hdr = predict.header.copy()
            # self.core.change_seg_data(self.predict_fdata)
            self.show_tumor_buttons()

    # 读取.nii.gz文件
    def load_nii(self):
        img_group = []
        self.data_name = []
        filepath = r'nnUNetFrame\DATASET\nnUNet_raw\nnUNet_raw_data\Task500_BraTS2021\imagesTs'
        filenames = os.listdir(filepath)
        filenames.sort()
        # print(filenames)

        # 开始读取nii文件
        for f in filenames:
            s = f[-6:]
            # print(s)
            if s != 'nii.gz':
                continue
            s1 = f[:-7]
            # print(s1)
            self.data_name.append(self.data_window_dict[f])

            img_path = os.path.join(filepath, f)
            img = nib.load(img_path)
            self.affine = img.affine.copy()
            self.hdr = img.header.copy()
            img = img.get_fdata()
            img = np.uint8(img / np.max(img) * 255)
            if s1[-4:] == "0003":
                self.img_model_data = img
            img_group.append(img)

        self.ui.radioButton_0.setChecked(True)
        self.ui.radioButton_4.setVisible(False)
        self.ui.radioButton_3.setVisible(len(img_group) > 3)
        self.ui.radioButton_2.setVisible(len(img_group) > 2)
        self.ui.radioButton_1.setVisible(len(img_group) > 1)

        if len(img_group) < 5:
            for i in range(len(img_group), 5):
                img_group.append(np.uint8(np.zeros(img_group[0].shape)))
                self.data_name.append("")

        if len(self.img_model_data) == 0:
            self.img_model_data = img_group[0]

        if len(img_group) == 5:
            self.img_group = img_group
            self.shape_x, self.shape_y, self.shape_z = self.img_group[0].shape
            self.scale_x, self.scale_y, self.scale_z = int(self.shape_x / 2), int(self.shape_y / 2), int(
                self.shape_z / 2)
            self.status_x, self.status_y, self.status_z = self.scale_x, self.scale_y, self.scale_z

        else:
            print("没有找到4个有效文件")
            # self.ui.outlineButton.setEnable(False)
            return
        self.predict_fdata = np.zeros(self.img_group[0].shape)
        try:
            self.core.setVisible(False)
            del self.core
        except TypeError:
            print('core empty')
        except AttributeError:
            print('core empty')
        finally:
            self.core = NiiModelCore(self.img_model_data, self.predict_fdata, [1, 0, 0, 0, 0, 0],
                                     self.ui.canvas_threeD.width(),
                                     self.ui.canvas_threeD.height(), self.ui.canvas_threeD)
        self.shapes = []
        self.shapes.append(self.shape_x)
        self.shapes.append(self.shape_y)
        self.shapes.append(self.shape_z)
        self.reset_page()

    def set_icon(self):
        for index in range(4):
            self.currentIndex = index
            self.draw_z()
            pic = Image.open('./tmp/z.png')
            pic = pic.convert('RGBA')  # 转为RGBA模式
            width, height = pic.size
            array = pic.load()  # 获取图片像素操作入口
            for i in range(width):
                for j in range(height):
                    pos = array[i, j]  # 获得某个像素点，格式为(R,G,B,A)元组
                    # 黑色背景，白色背景改255或者>=240
                    isEdit = (sum([1 for x in pos[0:3] if x == 0]) == 3)
                    if isEdit:
                        # 更改为透明
                        array[i, j] = (255, 255, 255, 0)
            # 保存图片
            imageio.imwrite('./tmp/icon{}.png'.format(index), pic)
            img = QIcon('./tmp/icon{}.png'.format(index))
            self.radioButtons[index].setIcon(img)
        img = QIcon('./UI/pictures/左侧按钮5.png')
        self.radioButtons[4].setIcon(img)

    def drawImages(self):
        self.draw_x()
        self.draw_y()
        self.draw_z()

    def draw_x(self):
        index = self.currentIndex
        assert index <= 4, "modelWidget index out of range"
        slice_x = self.img_group[index][self.shape_x - self.scale_x, :, :]
        slice_x = np.rot90(slice_x, 1)
        imageio.imwrite('./tmp/x.png', slice_x)
        if self.predicted:
            predict_x = self.predict_fdata[self.shape_x - self.scale_x, :, :]
            predict_x = np.rot90(predict_x, 1)

            slice_x = Image.open('./tmp/x.png')
            slice_x = slice_x.convert("RGB")
            slice_x = self.add_mask(slice_x, predict_x)
            imageio.imwrite('./tmp/x.png', slice_x)

        if self.cross_sign:
            if 0 < self.cross_y < self.shape_y and 0 < self.cross_z < self.shape_z:
                slice_x = Image.open('./tmp/x.png')
                slice_x = slice_x.convert("RGB")
                slice_x = self.add_cross(slice_x, self.shape_z - self.cross_z, self.shape_y - self.cross_y)
                imageio.imwrite('./tmp/x.png', slice_x)

        img = QPixmap('./tmp/x.png')
        self.timer.timeout.connect(lambda: self.fit_window(self.canvas_x, self.ui.canvas_x, img=img))

        container = self.canvas_x
        tracker = MouseTracker(container)
        tracker.positionChanged.connect(
            lambda e: self.on_positionChanged(pos=e, img_h=img.height(), img_w=img.width(), con_h=container.height(),
                                              con_w=container.width(), sign="x"))
        return

    def draw_y(self):
        index = self.currentIndex
        assert index <= 4, "modelWidget index out of range"

        slice_y = self.img_group[index][:, self.shape_y - self.scale_y, :]
        slice_y = np.rot90(slice_y, 1)
        # slice_y = np.transpose(slice_y)
        # slice_y = np.rot90(slice_y, 2)
        imageio.imwrite('./tmp/y.png', slice_y)

        if self.predicted:
            predict_y = self.predict_fdata[:, self.shape_y - self.scale_y, :]
            predict_y = np.rot90(predict_y, 1)
            # predict_y = np.transpose(predict_y)
            # predict_y = np.rot90(predict_y, 2)

            slice_y = Image.open('./tmp/y.png')
            slice_y = slice_y.convert("RGB")
            slice_y = self.add_mask(slice_y, predict_y)
            imageio.imwrite('./tmp/y.png', slice_y)
        if self.cross_sign:
            if 0 <= self.cross_x < self.shape_x and 0 < self.cross_z <= self.shape_z:
                slice_y = Image.open('./tmp/y.png')
                slice_y = slice_y.convert("RGB")
                slice_y = self.add_cross(slice_y, self.shape_z - self.cross_z, self.shape_x - self.cross_x)
                imageio.imwrite('./tmp/y.png', slice_y)

        img = QPixmap('./tmp/y.png')
        self.timer.timeout.connect(lambda: self.fit_window(self.canvas_y, self.ui.canvas_y, img=img))

        container = self.canvas_y
        tracker = MouseTracker(container)
        tracker.positionChanged.connect(
            lambda e: self.on_positionChanged(pos=e, img_h=img.height(), img_w=img.width(), con_h=container.height(),
                                              con_w=container.width(), sign="y"))
        return

    def draw_z(self):
        index = self.currentIndex
        assert index <= 4, "modelWidget index out of range"

        slice_z = self.img_group[index][:, :, self.scale_z]
        slice_z = np.transpose(slice_z)
        # slice_z = np.rot90(slice_z, -1)

        imageio.imwrite('./tmp/z.png', slice_z)
        if self.predicted:
            predict_z = self.predict_fdata[:, :, self.scale_z]
            predict_z = np.transpose(predict_z)
            # predict_z = np.rot90(predict_z, -1)

            slice_z = Image.open('./tmp/z.png')
            slice_z = slice_z.convert("RGB")
            slice_z = self.add_mask(slice_z, predict_z)
            imageio.imwrite('./tmp/z.png', slice_z)

        if self.cross_sign:
            if 0 < self.cross_x < self.shape_x and 0 < self.cross_y <= self.shape_y:
                slice_z = Image.open('./tmp/z.png')
                slice_z = slice_z.convert("RGB")
                slice_z = self.add_cross(slice_z, self.shape_y - self.cross_y, self.shape_x - self.cross_x)
                imageio.imwrite('./tmp/z.png', slice_z)

        img = QPixmap('./tmp/z.png')
        self.timer.timeout.connect(lambda: self.fit_window(self.canvas_z, self.ui.canvas_z, img=img))

        container = self.canvas_z
        tracker = MouseTracker(container)
        tracker.positionChanged.connect(
            lambda e: self.on_positionChanged(pos=e, img_h=img.height(), img_w=img.width(), con_h=container.height(),
                                              con_w=container.width(), sign="z"))
        return

    def on_positionChanged(self, pos, img_w, img_h, con_w, con_h, sign):
        delta = QtCore.QPoint(con_w // 2, con_h // 2)
        pos -= delta
        scale = min([con_w / img_w, con_h / img_h])
        pos /= scale
        delta = QtCore.QPoint(img_w // 2, img_h // 2)
        pos += delta
        if sign == "x":
            self.status_x = self.scale_x
            self.status_y = int(self.shape_y - pos.x())
            self.status_z = int(self.shape_z - pos.y())
        elif sign == "y":
            self.status_x = int(self.shape_x - pos.x())
            self.status_y = self.scale_y
            self.status_z = int(self.shape_z - pos.y())
        elif sign == "z":
            self.status_x = int(self.shape_x - pos.x())
            self.status_y = int(self.shape_y - pos.y())
            self.status_z = self.scale_z
        self.set_status()

    def add_mask(self, img, data):
        img_array = np.array(img)
        w, h = img.size[0], img.size[1]
        img2_array = []
        assert data.shape == (h, w)
        for i in range(0, h):
            for j in range(0, w):
                if self.tumor_show[5] and data[i, j] != 0:
                    img_array[i, j] = self.tumor_color[5]
                if self.tumor_show[4] and data[i, j] >= 2:
                    img_array[i, j] = self.tumor_color[4]
                for k in range(1, 4):
                    if self.tumor_show[k] and data[i, j] == k:
                        img_array[i, j] = self.tumor_color[k]
                img2_array.append(img_array[i, j])
        img2_array = np.array(img2_array)
        img2_array = img2_array.reshape(h, w, 3)
        img3 = Image.fromarray(img2_array)
        return img3

    def add_cross(self, img, pos_x: int, pos_y: int):
        img_array = np.array(img)
        w, h = img.size[0], img.size[1]
        for i in range(h):
            img_array[i, pos_y] = [255, 0, 0]
        for j in range(w):
            img_array[pos_x, j] = [255, 0, 0]
        img2_array = np.array(img_array)
        img2_array = img2_array.reshape(h, w, 3)
        img2 = Image.fromarray(img2_array)
        return img2

    def scroll_x(self):
        self.scale_x = self.ui.slider_x.value()
        self.scales[0] = self.scale_x
        self.status_x = self.scale_x
        self.set_status()
        self.draw_x()
        return

    def scroll_y(self):
        self.scale_y = self.ui.slider_y.value()
        self.scales[1] = self.scale_y
        self.status_y = self.scale_y
        self.set_status()
        self.draw_y()
        return

    def scroll_z(self):
        self.scale_z = self.ui.slider_z.value()
        self.scales[2] = self.scale_z
        self.status_z = self.scale_z
        self.set_status()
        self.draw_z()
        return

    def roll(self, event, index):
        angle = event.angleDelta() / 8
        angleY = angle.y()

        self.scales[index] += int(angleY / 5)
        if self.scales[index] <= 0:
            self.scales[index] = 1
        if self.scales[index] >= self.shapes[index]:
            self.scales[index] = self.shapes[index] - 1
        self.sliders[index].setValue(self.scales[index])
        self.drawImages()

    def cross(self):
        self.cross_x, self.cross_y, self.cross_z = self.status_x, self.status_y, self.status_z
        self.cross_sign = True
        self.drawImages()

    def zoom(self, index):
        if self.currentIndex == 4:
            return
        try:
            self.canvas_zoom.setVisible(False)
            del self.canvas_zoom
        except TypeError:
            print('canvas_zoom empty')
        except AttributeError:
            print('canvas_zoom empty')
        finally:
            self.canvas_zoom = NiiImageViewer(self.img_group[3], self.img_group[0], self.img_group[1],
                                              self.img_group[2],
                                              self.modal_name[self.currentIndex], self.predict_fdata, index,
                                              parent=self.ui.centralwidget, first_img=self.sliders[index].value())
            print("canvas zoom loaded!")
        self.view_clean()
        print("view cleaned!")
        self.canvas_zoom.exit_action.triggered.connect(self.view_show)

        self.timer.timeout.connect(lambda: self.fit_window(self.canvas_zoom, self.ui.centralwidget))
        self.canvas_zoom.canvas.pixoffset = QPoint(
            self.ui.size().width() // 2 - self.canvas_zoom.canvas.pixmap.size().width() // 2,
            self.ui.size().height() // 2 - self.canvas_zoom.canvas.pixmap.size().height() // 2)
        self.canvas_zoom.show()
        self.ui.actionMain_Window.triggered.connect(self.view_show)

    def zoom_3d(self):
        self.view_clean()
        seg_mask = [1, 0, 0, 0, 0, 0]
        for i in range(1, 6):
            if self.tumor_buttons[i].isChecked():
                seg_mask[i] = 1
        if self.currentIndex == 4:
            seg_mask[0] = 0

        try:
            self.canvas_zoom.setVisible(False)
            del self.canvas_zoom
        except TypeError:
            print('canvas_zoom empty')
        except AttributeError:
            print('canvas_zoom empty')
        finally:
            self.canvas_zoom = NiiModelWidget(self.img_model_data, self.predict_fdata, seg_mask,
                                              self.ui.canvas_threeD.width(),
                                              self.ui.canvas_threeD.height(),
                                              parent=self.ui.centralwidget)
        self.timer.timeout.connect(lambda: self.fit_window(self.canvas_zoom, self.ui.centralwidget))
        self.canvas_zoom.show()
        self.ui.actionMain_Window.triggered.connect(self.view_show)

    def view_clean(self):
        # self.ui.canvas_threeD.setVisible(False)
        # self.ui.canvas_x.setVisible(False)
        # self.ui.canvas_y.setVisible(False)
        # self.ui.canvas_z.setVisible(False)
        # self.ui.bar_x.setVisible(False)
        # self.ui.bar_y.setVisible(False)
        # self.ui.bar_z.setVisible(False)
        self.ui.MainWidget.setVisible(False)
        self.statusBar.setVisible(False)
        self.ui.centralwidget.setStyleSheet("background-color:rgb(0, 0, 0);")

    def view_show(self):
        self.canvas_zoom.setVisible(False)
        # self.ui.canvas_threeD.setVisible(True)
        # self.ui.canvas_x.setVisible(True)
        # self.ui.canvas_y.setVisible(True)
        # self.ui.canvas_z.setVisible(True)
        # self.ui.bar_x.setVisible(True)
        # self.ui.bar_y.setVisible(True)
        # self.ui.bar_z.setVisible(True)
        self.ui.MainWidget.setVisible(True)
        self.statusBar.setVisible(True)
        self.core.change_seg_data(self.predict_fdata)
        self.show_tumor_buttons()
        self.ui.centralwidget.setStyleSheet("background-color:rgb(4, 16, 34);")

    def fit_window(self, container, ui_container, img=None):
        container.resize(ui_container.width(), ui_container.height())
        if img != None:
            pil_image = self.m_resize(container.width(), container.height(), img)
            container.setPixmap(pil_image)

    def m_resize(self, w_box, h_box, pil_image):  # 参数是：要适应的窗口宽、高、Image.open后的图片
        w, h = pil_image.width(), pil_image.height()  # 获取图像的原始大小
        f1 = 1.0 * w_box / w
        f2 = 1.0 * h_box / h
        factor = min([f1, f2])
        width = int(w * factor)
        height = int(h * factor)
        return pil_image.scaled(width, height)

    def switch_page(self, index):
        self.currentIndex = index

        if (index < 4):
            self.ui.name.setText(self.data_name[index])
        else:
            self.ui.name.setText("")
        self.drawImages()

    def view_tumor(self, index):
        # print(index)
        self.tumor_show[index] = self.tumor_buttons[index].isChecked()
        # print(self.tumor_show)
        self.drawImages()

    def select_point_3d(self, x, y, z):
        if x == -1 or y == -1 or z == -1:
            # 直接全部清理掉.
            self.ui.textBrowser.setText("Eedema:ED —— Yellow\n\
Non-Enhancing Tumor:NCR —— Green\n\
Enhancing Tumour:ET —— Blue\n\
Tumor Core:TC = ET + NCR —— Purple\n\
Whole Tumor:WT = ED + ET + NCR —— Red")
            self.points_for_dist.clear()
        else:
            has_dist = False
            if len(self.points_for_dist) == 2:
                self.points_for_dist.pop(0)
                has_dist = True
            elif len(self.points_for_dist) == 1:
                has_dist = True
            self.points_for_dist.append((x, y, z))
            self.cross_x, self.cross_y, self.cross_z = x, y, z
            self.scale_x, self.scale_y, self.scale_z = x, y, z
            self.ui.slider_x.setValue(self.scale_x)
            self.ui.slider_y.setValue(self.scale_y)
            self.ui.slider_z.setValue(self.scale_z)
            self.cross_sign = True
            self.drawImages()
            if has_dist:
                x0, y0, z0 = self.points_for_dist[0]
                self.points_for_dist[1] = (x, y, z)
                dist = np.sqrt((x0 - x) ** 2 + (y0 - y) ** 2 + (z0 - z) ** 2)
                self.ui.textBrowser.setText(
                    "x=%d, y=%d, z=%d\nx=%d, y=%d, z=%d\ndistance=%.2f (1mm isotropic)" % (x0, y0, z0, x, y, z, dist))
            else:
                self.ui.textBrowser.setText("x=%d, y=%d, z=%d (1mm isotropic)\n\n" % (x, y, z))

    def outline(self):
        if not torch.cuda.is_available():
            QMessageBox.warning(self.ui, "Warning", "没有可以使用的显卡，无法推理")
            return
        self.predict_thread = MyPredictThread(son=self)
        self.predict_thread.finish_signal.connect(self.show_tumor_buttons)

        num = 0
        for i in range(4):
            num += 1 if self.predict_buttons[i].isChecked() else 0
        if num == 0:
            return

        try:
            self.progress_bar.setVisible(False)
            del self.progress_bar
        except AttributeError:
            print('progress_bar empty')
        finally:
            self.progress_bar = WaterProgress(thread=self.predict_thread, parent=self.ui.progressBar, num=num)

        self.predict_thread.start()
        self.progress_bar.show()

        # self.predict_thread.join()

    def show_tumor_buttons(self):
        if np.sum(self.predict_fdata > 0) == 0:
            for i in range(1, 6):
                self.tumor_buttons[i].setVisible(False)
            self.ui.radioButton_4.setVisible(False)
            self.predicted = False
            return
        self.tumor_volume = [0]
        x, y, z = self.predict_fdata.shape
        print(x, y, z)
        self.tumor_volume.append(np.sum(self.predict_fdata == 1))
        self.tumor_volume.append(np.sum(self.predict_fdata == 2))
        self.tumor_volume.append(np.sum(self.predict_fdata == 3))
        self.tumor_volume.append(self.tumor_volume[2] + self.tumor_volume[3])
        self.tumor_volume.append(self.tumor_volume[1] + self.tumor_volume[4])
        total_volumn = np.sum(self.img_group[0] > 0)
        print("total brain volume:", total_volumn)
        print("total tumor volume:", np.sum(self.tumor_volume))

        self.tumor_volume = [x / total_volumn for x in self.tumor_volume]
        print("tumor volume percentage:", self.tumor_volume)

        self.core.change_seg_data(self.predict_fdata)

        self.ui.ButtonSectionImage.setVisible(False)
        for i in range(1, 6):
            self.tumor_buttons[i].setVisible(True)
            str = self.tumor_name.get(i) + ':{:.2%}'.format(self.tumor_volume[i])
            self.tumor_buttons[i].setText(str)
        self.ui.radioButton_4.setVisible(True)
        self.predicted = True
        self.drawImages()
        # self.reset_page()

    def set_status(self):
        Str = "x={}, y={}, z={}".format(self.status_x, self.status_y, self.status_z)
        self.statusBar.setText(Str)

    def export_prediction(self):
        savename, _ = QFileDialog.getSaveFileName(self, "save as...", "../../", "Nibabel files (*.nii.gz)")
        if savename != "":
            new_nii = nib.Nifti1Image(self.predict_fdata, self.affine, self.hdr)
            nib.save(new_nii, savename)


class MyPredictThread(QThread):
    finish_signal = pyqtSignal()
    isFinished = False

    def __init__(self, son=None):
        super().__init__()
        self.son = son

    def run(self):
        self.isFinished = False
        parent_folder = r'nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task500_BraTS2021'
        folders = ['inferTs_0', 'inferTs_1', 'inferTs_2', 'inferTs_3']
        ensemble_folder = 'ensemble_result'

        folder_path = os.path.join(parent_folder, ensemble_folder)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.mkdir(folder_path)

        # output_order = [ '-f']
        output_order = "..\\venv\\python.exe nnunet\\inference\\ensemble_predictions.py -f "
        source_file = "nnunet\\network_architecture\\generic_UNet_{}.py"
        destination_file = "nnunet\\network_architecture\\generic_UNet.py"
        for i in range(4):
            if self.son.predict_buttons[i].isChecked():
                print("start prediction--model_{}".format(i))

                # TODO:predict开关_begin
                folder_path = os.path.join(parent_folder, folders[i])
                if os.path.exists(folder_path):
                    shutil.rmtree(folder_path)
                os.mkdir(folder_path)
                shutil.copyfile(source_file.format(i), destination_file)
                os.system(
                    r'..\venv\python.exe nnunet\inference\predict_simple.py -i nnUNetFrame\DATASET\nnUNet_raw\nnUNet_raw_data\Task500_BraTS2021\imagesTs -o nnUNetFrame\DATASET\nnUNet_raw\nnUNet_raw_data\Task500_BraTS2021\inferTs_{} -t 500 -m 3d_fullres -f {} --save_npz'.format(
                        i, i))

                # predict_order = [
                #          '-i', r'nnUNetFrame\\DATASET\\nnUNet_raw\\nnUNet_raw_data\\Task500_BraTS2021\\imagesTs',
                #          '-o', r'nnUNetFrame\\DATASET\\nnUNet_raw\\nnUNet_raw_data\\Task500_BraTS2021\\inferTs_{}'.format(i),
                #           '-t', '500', '-m', '3d_fullres', '-f', '{}'.format(i), '--save_npz']
                # sys.argv = predict_order
                # predict_simple.main()

                # TODO:predict开关_end

                #         output_order.append("nnUNetFrame\\DATASET\\nnUNet_raw\\nnUNet_raw_data\\Task500_BraTS2021"
                output_order += "nnUNetFrame\\DATASET\\nnUNet_raw\\nnUNet_raw_data\\Task500_BraTS2021\\inferTs_{} ".format(
                    i)

        output_order += "-o nnUNetFrame\\DATASET\\nnUNet_raw\\nnUNet_raw_data\\Task500_BraTS2021\\ensemble_result"
        # output_order.extend(['-o', 'nnUNetFrame\\DATASET\\nnUNet_raw\\nnUNet_raw_data\\Task500_BraTS2021\\ensemble_result'])

        os.system(output_order)
        # sys.argv = output_order
        # ensemble_predictions.main()

        # time.sleep(2)
        predict_path = os.path.join(parent_folder, ensemble_folder)
        filenames = os.listdir(predict_path)
        for f in filenames:
            s = f[-6:]
            print(s)

            if s == 'nii.gz':
                break
        s1 = f[:-7]
        print(s1)
        predict_path = os.path.join(predict_path, f)
        predict = nib.load(predict_path)
        predict_fdata = predict.get_fdata()

        self.son.affine = predict.affine.copy()
        self.son.hdr = predict.header.copy()
        self.son.predict_fdata = predict_fdata
        self.son.predicted = True

        self.finish_signal.emit()
        self.isFinished = True

def main():
    print("sys path:", sys.path)
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_blue.xml')

    mywindow = WindowQt()
    mywindow.ui.setWindowTitle("GNav(GliomaNavigator)")
    icon = QIcon('UI/pictures/window_icon.png')
    mywindow.ui.setWindowIcon(icon)
    mywindow.ui.show()

    app.exec_()

if __name__ == '__main__':
    main()
    # app = QApplication(sys.argv)
    # rename = RenameWindow()
    # rename.show()
    # app.exec_()