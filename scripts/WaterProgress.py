#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 2021/1/1
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: WaterProgress
@description: 
"""
import sys

from PyQt5.QtCore import QTimer, QThread
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
from scripts.DWaterProgress import DWaterProgress

class WaterProgress(QWidget):

    def __init__(self, thread:QThread, num:int, *args, **kwargs):
        super(WaterProgress, self).__init__(*args, **kwargs)
        layout = QVBoxLayout(self)

        self.thread = thread
        self.progress = DWaterProgress(self)
        self.progress.setFixedSize(100, 100)
        self.progress.setValue(0)
        self.progress.start()


        layout.addWidget(self.progress)

        self.timer = QTimer(self, timeout=self.updateProgress)
        self.timer.start(1000*num)

    def updateProgress(self):
        value = self.progress.value()
        if not self.thread.isFinished:
            if value == 100:
                self.progress.setValue(0)
            if value < 98:
                self.progress.setValue(value + 1)
        else:
            self.progress.setValue(100)


if __name__ == '__main__':
    import cgitb

    cgitb.enable(format='text')
    app = QApplication(sys.argv)
    w = WaterProgress()
    w.show()
    sys.exit(app.exec_())
