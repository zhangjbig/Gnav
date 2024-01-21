#############
# author : JhLi
#############

from PyQt5.QtWidgets import (QApplication, QWidget, QTabBar, QComboBox, QHBoxLayout, QPushButton)
from PyQt5.QtCore import pyqtSignal,Qt
from PyQt5.QtGui import QIntValidator, QKeySequence
from PyQt5.Qt import *
from qt_material import apply_stylesheet
import sys
from scripts.Constant import MyIcons
from scripts.PaintingComponents import PaintingManager

class PaintingBar(QWidget):
    paintModeOn = pyqtSignal()
    eraseModeOn = pyqtSignal()
    dragModeOn = pyqtSignal()

    def __init__(self, parent=None, height=40, width=600) -> None:
        super().__init__(parent)
        self.resize(width, height)
        self.setStyleSheet('QWidget{background-color:#000000;}')
        hlayout = QHBoxLayout()
        # icons
        myicons = MyIcons()
        # tabBar for paintMode, eraseMode, dragMode
        self.mode_bar = QTabBar(self)
        self.mode_bar.setMovable(False)
        self.mode_bar.setTabsClosable(False)
        self.mode_bar.setUsesScrollButtons(False)
        self.mode_bar.setShape(QTabBar.Shape.RoundedNorth)
        self.mode_bar.addTab(myicons.brush, "Brush")
        self.mode_bar.addTab(myicons.eraser, "Eraser")
        self.mode_bar.addTab(myicons.drag, "Drag")
        self.mode_bar.currentChanged.connect(self.switchMode)

        # ComboBox for tumor regions and pen width.
        self.region_combo = QComboBox(self)
        self.region_combo.setStyleSheet('QWidget{color:#ffffff;}')
        self.region_combo.addItems(['ED','NCR','ET'])
        self.region_combo.setCurrentIndex(0)
        
        # pen width combobox
        self.width_combo = QComboBox(self)
        self.width_combo.setStyleSheet('QWidget{color:#ffffff;}')
        i = 10
        while i<=30:
            self.width_combo.addItem(str(i))
            i = i+2
        self.width_combo.setCurrentIndex(5)
        self.width_combo.setEditable(True)
        self.width_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.width_combo.setValidator(QIntValidator(10,99,self.width_combo))

        # save pushbutton
        self.save = QPushButton(myicons.save, "save", self)
        self.save.setShortcut(QKeySequence("Ctrl+S"))
        # ok pushbutton
        self.ok = QPushButton(myicons.ok, "finish", self)
        # fallback pushbutton
        self.fallback = QPushButton(myicons.fallback, "fallback", self)
        self.fallback.setShortcut(QKeySequence("Ctrl+Z"))

        hlayout.addWidget(self.mode_bar)
        hlayout.addWidget(self.region_combo)
        hlayout.addWidget(self.width_combo)
        hlayout.addWidget(self.fallback)
        hlayout.addWidget(self.save)
        hlayout.addWidget(self.ok)
        self.setLayout(hlayout)

        #保证在自己内部使用正常鼠标.
        self.setCursor(Qt.ArrowCursor)

    def restart(self):
        '''
        第二次打开paintingBar，必须和对应的paint_manager同步，调用这个
        '''
        self.width_combo.setCurrentIndex(5)  #20

    def switchMode(self, index):
        if index==0:
            self.paintModeOn.emit()
        elif index==1:
            self.eraseModeOn.emit()
        elif index==2:
            self.dragModeOn.emit()

    def connect(self, manager:PaintingManager):
        '''
        由于架构问题，这个废弃.  
        画笔、橡皮模式，调整区域模式，调整粗细.  
        注意还有一个拖拽模式是和外面的ImageViewer绑定的!
        '''
        self.paintModeOn.connect(manager.setPaintingMode)
        self.eraseModeOn.connect(manager.setEraseMode)
        self.region_combo.currentIndexChanged.connect(manager.bind)
        self.width_combo.currentTextChanged.connect(manager.setWidth_str)



if __name__=="__main__":
    app = QApplication(sys.argv)
    apply_stylesheet(app, 'dark_blue.xml')
    bar = PaintingBar()
    bar.show()
    sys.exit(app.exec_())