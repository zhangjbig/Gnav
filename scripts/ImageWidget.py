#############
# author : JhLi
#############

import sys
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.Qt import *
import qt_material as qm
import qtawesome as qta
import nibabel
import cv2
import copy
import os

from scripts.Constant import Constant, MyIcons
from scripts.util import add_mask, preprocessMRI4Image, postprocessMRI4Image, rotate
from scripts.PaintingComponents import PaintingManager
from scripts.PaintingBar import PaintingBar

class FixHeightWidthRateLabel(QLabel):
    '''
    保护长宽比的QLabel像普通的QLable那样申请它、操作它.
    但是保护效果并不好...可能还得用修改回调函数的方法.  
    废弃  
    '''
    def __init__(self, parent = None):
        super(FixHeightWidthRateLabel, self).__init__(parent)
        self.setScaledContents(True)

    def hasHeightForWidth(self) -> bool:
        return self.pixmap() is not None
    
    def heightForWidth(self, a0: int) -> int:
        if self.pixmap():
            return int(a0 * (self.pixmap().height() / self.pixmap().width()))
        
class Point3d():
    def __init__(self, x:int=int(-1), y:int=int(-1), z:int=int(-1)) -> None:
        self.x = x
        self.y = y
        self.z = z
    def __getitem__(self, item):
        if item==0:
            return self.x
        elif item==1:
            return self.y
        else:
            return self.z


class NiiImageCanvas(QWidget):
    '''
    nii大图查看器的画布,用来绘制图片.只是也只能是画布。  
    .nii本体也保存在这里,这个是Core  
    因为之前的架构，绘画部分主体(PaintingManager)也在这里,而PaintingBar在外面.  
    不会存放所有模态的脑，切换脑模态的时候在外部传入新的脑模态数据.
    在canvas里,将完全按照索引值来获取切片.不做取补处理等.
    '''
    def __init__(self, img_fdata, seg_fdata, index, init_seg_mask=[0,0,0,0,0,0], first_img=0, parent = None, height = 800, width = 800) -> None:
        super(NiiImageCanvas, self).__init__(parent)
        self.img_fdata = preprocessMRI4Image(img_fdata.copy())
        self.seg_fdata = preprocessMRI4Image(seg_fdata.copy())
        self.index = index
        self.img_num = img_fdata.shape[index]  # 图片数量，用于滑动条
        self.showing_img = None                # 正在展示的rgb图片
        self.cur_img = first_img               # 正在展示第几张图片.  

        self.focusPoint = Point3d()            # 它不应该被修改.

        self.seg_mask = copy.copy(init_seg_mask)

        #self.init_resize = True    #第一次resize，不能动self.pixoffset.
        self.resize(width, height)

        # 显示图片, Qpixmap，用绘制的方法进行，在一个额外的画布上面画（QWidget）
        self.ori_pixmap = QPixmap.fromImage(self.getQImage(first_img))
        self.scale = 2
        self.pixmap = self.scale_ori_pixmap()
        self.pixoffset = QPoint(self.size().width()//2-self.pixmap.size().width()//2, self.size().height()//2-self.pixmap.size().height()//2)  # 用来移动画面的偏移量.

        # 用于事件处理的一些状态变量.
        self.enableDragging = bool(True)   # 允许拖拽的标志绘画过程中不允许拖拽，除非用手工具.
        self.LeftPressed = bool(False)
        self.prevMousePos = QPoint(0,0)
        self.currentMousePos = QPoint(0,0) # 这是绘画时候要用的.

        # 绘画模式部件.
        self.paintingMode = bool(False)    # 开启绘画模式标记
        self.eraseMode = bool(False)      # 擦除模式.
        self.paint_manager = None          # 先为None.

    def scale_ori_pixmap(self):
        '''
        按照self.scale缩放self.showing_img(self.ori_pixmap)
        '''
        return self.ori_pixmap.scaled(int(self.showing_img.shape[1]*self.scale),int(self.showing_img.shape[0]*self.scale))

    def getQImage(self, index):
        # 获取rgb图片形式的 img_fdata[:, index, :]
        # 和外面的同步，x是np.rot90, y是np.rot90，z是转置.
        if self.index == 0:
            brain = rotate(self.img_fdata[index,:,:],0)
            seg = rotate(self.seg_fdata[index,:,:],0)
        elif self.index == 1:
            brain = rotate(self.img_fdata[:,index,:],1)
            seg = rotate(self.seg_fdata[:,index,:],1)
        else:
            brain = rotate(self.img_fdata[:,:,index], 2)
            seg = rotate(self.seg_fdata[:,:,index], 2)
        brain = cv2.cvtColor(np.uint8(brain), cv2.COLOR_GRAY2RGB)
        seg = cv2.cvtColor(np.uint8(seg), cv2.COLOR_GRAY2RGB)
        self.showing_img = add_mask(brain, seg, self.seg_mask)
        #self.showing_img = cv2.flip(brain, 0)  不用翻转，和主界面同步.
        #padh, padw = (self.maxH - brain.shape[0])//2, (self.maxW - brain.shape[1])//2
        #self.showing_img = np.pad(brain, ((padh, padh), (padw, padw),(0,0)), 'constant')  # 填充.
        qimg = QImage(
            self.showing_img.tobytes(),
            self.showing_img.shape[1], self.showing_img.shape[0],
            self.showing_img.shape[1]*3, QImage.Format.Format_RGB888
        )
        return qimg

    # 事件响应函数
    def paintEvent(self, a0: QPaintEvent) -> None:
        '''
        绘制pixmap, 即把self.pixmap绘制到画面上.  
        由于每个视角的旋转方法有多次来回变，下面关于十字架点的转换方法可能已经错误。
        由于大窗口中不使用十字架点，所以后面应该将focuspoint及相关移除.
        '''
        painter = QPainter()
        painter.begin(self)
        painter.drawPixmap(self.pixoffset, self.pixmap)
        # 下面是那个画十字架的，我这个窗口应该用不太上了.
        if self.focusPoint.x!=-1 and self.focusPoint.y!=-1 and self.focusPoint.z!=-1 and self.focusPoint[self.index]==self.cur_img:
            # 画一个十字焦点.小心图片在展示之前是逆时针转了90度过的.
            pen = QPen(Qt.red,1, Qt.SolidLine)
            painter.setPen(pen)
            # x:np.rot90, y:np.rot90, z:转置.
            # 3D里x,y传进来，好像直接是y,x了!
            # 像素坐标系和普通坐标系的y轴天然是翻转过的，但还是很怪.
            if self.index==0:
                # rot90: (y, -x+self.pix.height)
                x = self.focusPoint[1]
                y = -self.focusPoint[2]+self.showing_img.shape[0]-1  # showing_img是处理以后的!
            elif self.index==1:
                x = self.focusPoint[0]
                y = -self.focusPoint[2]+self.showing_img.shape[0]-1
            else:
                x = self.focusPoint[0]
                y = self.focusPoint[1]
            y = self.pixoffset.y() + int(y * self.scale)
            x = self.pixoffset.x() + int(x * self.scale)
            painter.drawLine(0,y,self.size().width(),y)
            painter.drawLine(x,0,x,self.size().height())
        # 如果正在绘画过程中（鼠标按着拖拽且paintMode or EraseMode.
        if self.isPainting():    #总之要把画出来的东西draw上去.
            if self.LeftPressed and (self.paintingMode or self.eraseMode):   #这是要画线的.
                self.paint_manager.paint(self.prevMousePos-self.pixoffset,self.currentMousePos-self.pixoffset)
            drawed_pixmap = self.paint_manager.getScaledPixmap()
            painter.drawPixmap(self.pixoffset, drawed_pixmap)
        painter.end()
        return super().paintEvent(a0)

    def PaintingModeOn(self, savePath = None, affine = None):
        # 不在这里bind.
        if self.paint_manager is None:
            self.paint_manager = PaintingManager(
                self.seg_fdata,self.index,self.cur_img,self.scale,
                savePath=savePath, affine=affine
            )
        else:
            self.paint_manager.initialize(
                self.seg_fdata, self.index, self.cur_img, self.scale
            )   # 会默认激活.
        self.paintingMode = True
        self.enableDragging = False
        self.eraseMode = False
        self.repaint()
        return self.paint_manager   # 返回自己的paint_manager给上层，进行绑定..

    def PaintingModeOff(self):
        '''
        试图关闭绘画模式，关闭成功返回0，否则返回非零，在外面应该打断关闭.
        如果在指定位置有新的.nii，就用这个新的代替原本那个.
        '''
        if self.paint_manager.onClose() != 0:
            return 1   # 打断关闭.
        self.paint_manager.disActivate()
        self.paintingMode = False
        self.eraseMode = False
        self.enableDragging = True
        self.repaint()
        return 0   # 继续关闭.

    def useBrush(self):
        self.paintingMode = True
        self.eraseMode = False
        self.enableDragging = False
        self.paint_manager.setPaintingMode()
    def useEraser(self):
        self.paintingMode = False
        self.eraseMode = True
        self.enableDragging = False
        self.paint_manager.setEraseMode()
    def useDragger(self):
        # drag模式仅仅由canvas控制，不下到painting_manager.
        self.paintingMode = False
        self.eraseMode = False
        self.enableDragging = True

    def isPainting(self):
        return self.paint_manager is not None and self.paint_manager.isActivated()

    def mousePressEvent(self, a0: QMouseEvent) -> None:
        '''
        按下鼠标时触发,这里只用考虑按下左键，移动图片.
        左键按下的时候，既可能是拖拽也可能是绘画，必为两者之一,不能同时有拖拽和绘画.
        都需要prevMousePos
        '''
        if a0.button() == Qt.MouseButton.LeftButton:
            self.LeftPressed = True
            self.prevMousePos = a0.pos()
        return super().mousePressEvent(a0)

    def mouseMoveEvent(self, a0: QMouseEvent) -> None:
        '''
        拖拽鼠标的时候,实时修改offset
        '''
        if self.LeftPressed:         # 按下左键，要么拖拽要么绘画.
            if self.enableDragging:
                mousePosDelta = a0.pos() - self.prevMousePos
                self.pixoffset = self.pixoffset + mousePosDelta
                self.prevMousePos = a0.pos()
                self.repaint()     # 重绘
            elif self.paintingMode or self.eraseMode:
                self.currentMousePos = a0.pos()
                self.repaint()
                self.prevMousePos = self.currentMousePos
        return super().mouseMoveEvent(a0)

    def mouseReleaseEvent(self, a0: QMouseEvent) -> None:
        '''
        松开左键的时候, 取消左键按下状态状态
        松开右键, 将图片放回到中间.
        '''
        if a0.button() == Qt.MouseButton.LeftButton:
            self.LeftPressed = False
        elif a0.button() == Qt.MouseButton.RightButton:
            self.pixoffset = QPoint(self.size().width()//2-self.pixmap.size().width()//2, self.size().height()//2-self.pixmap.size().height()//2)
            self.repaint()
        return super().mouseReleaseEvent(a0)

    def wheelEvent(self, a0: QWheelEvent) -> None:
        '''
        滚轮事件：缩放.
        '''
        if self.LeftPressed and (self.paintingMode or self.eraseMode):
            # 正在绘制过程中，禁止滚轮缩放!
            return super().wheelEvent(a0)
        delta = a0.angleDelta().y()
        prev_scale = self.scale
        self.scale = self.scale + float(delta) / 500
        if self.scale <= 0.5:
            self.scale = 0.5
        elif self.scale >= 8.0:
            self.scale = 8.0
        self.pixmap = self.scale_ori_pixmap()
        # 保证鼠标选中的地方不动，边offset.
        # 在缩放之前，距离图片左上角的相对坐标.
        x,y = a0.pos().x() - self.pixoffset.x(), a0.pos().y() - self.pixoffset.y()
        x, y = int(x*self.scale/prev_scale), int(y*self.scale/prev_scale)
        newx = a0.pos().x() - x
        newy = a0.pos().y() - y
        self.pixoffset = QPoint(newx, newy)
        # 如果是在激活状态（注意，激活不一定有paintMode and eraseMode!
        if self.paint_manager is not None and self.paint_manager.isActivated():
            self.paint_manager.setScale(self.scale)  # 那边也刷新scale!
        self.repaint()
        return super().wheelEvent(a0)

    # def resizeEvent(self, a0:QResizeEvent):
    #     if self.init_resize:
    #         self.init_resize = False   #第一次初始化，不能动pixoffset.
    #     else:
    #         self.pixoffset.setX(int(self.pixoffset.x()*a0.size().width()/a0.oldSize().width()))
    #         self.pixoffset.setY(int(self.pixoffset.y()*a0.size().height()/a0.oldSize().height()))
    #         self.repaint()
    #     return super().resizeEvent(a0)
    # resizeEvent暂时不太靠谱.

    def closeEvent(self, a0:QCloseEvent):
        if self.paint_manager is not None:
            self.paint_manager.onClose()
        return super().closeEvent(a0)

    # 槽函数
    def changeImage(self, index):
        # 根据滑动条的值更换展示图片.
        # 同样有可能被打断.
        # 用作刷新.
        if self.isPainting():
            if self.paint_manager.setIndex(index) != 0:  #在paint_manager中保证了不会重复替换.
                return 1  #打断.
        self.cur_img = index
        self.ori_pixmap = QPixmap.fromImage(self.getQImage(index))
        self.pixmap = self.scale_ori_pixmap()
        self.repaint()
        return 0

    def change_img_fdata(self, new_fdata):
        '''
        更换所显示的flair, t1, 等模态.
        仅更换img_fdata, 不更新seg_mask(显示那个区域的肿瘤)，为了和主界面按钮同步
        '''
        if new_fdata.shape!=self.seg_fdata.shape:
            QMessageBox.critical(self, "ERROR", "文件不匹配", QMessageBox.StandardButton.Ok)
            return
        self.img_fdata = preprocessMRI4Image(new_fdata.copy())
        self.changeImage(self.cur_img)

    def change_seg_fdata(self, new_seg_fdata):
        '''
        更换当前显示显示所用的标签数据。注意canvas不会保存之前的标签数据.
        '''
        if new_seg_fdata.shape!=self.seg_fdata.shape:
            QMessageBox.critical(self, "ERROR", "文件不匹配", QMessageBox.StandardButton.Ok)
            return
        self.seg_fdata = preprocessMRI4Image(new_seg_fdata.copy())
        self.changeImage(self.cur_img)

    def changeThreeView(self, new_perspective):
        '''
        切换三视图。0-x,侧视图，1-y，正视图，2-z，俯视图
        不检查new_perspective范围.
        绘画模式下，有可能被打断!
        '''
        if self.paint_manager is not None and self.paint_manager.isActivated():
            if self.paint_manager.switchThreeView(new_perspective) != 0:
                return 1    # 反悔了，打断.
            self.cur_img = self.paint_manager.index  #如果有paint_manager,和它的index同步.
        elif self.img_fdata.shape[new_perspective] <= self.cur_img:
            self.cur_img = self.img_fdata.shape[new_perspective]-1  #直接放最大值吧..
        self.index = new_perspective
        self.changeImage(self.cur_img)
        return 0  #继续.

    def AddOrRemoveMask(self, code):
        '''
        code: 要修改状态的肿瘤区域编号.
        '''
        if self.seg_mask[code]==1:
            self.seg_mask[code]=0
        else:
            self.seg_mask[code]=1
        #self.seg_mask[code]=~self.seg_mask[code]
        self.changeImage(self.cur_img)

    def getShowingImage(self):
        return self.showing_img
    
    def getView(self):
        '''
        返回三视图编号
        '''
        return self.index


class NiiImageViewer(QWidget):
    '''
    Nii大图查看器, 由工具栏、滑动条、画布构成.
    '''
    def __init__(
            self, img_fdata, t1, t1ce,t2,
            init_modal:str, seg_fdata, index,
            init_seg_mask=[0,0,0,0,0,0],
            parent=None, width=1200, height=800, first_img=0,
            file_name="", savePath:str=None, affine:np.ndarray=None) -> None:
        '''
        img_fdata: numpy array of .nii, 需要提前用img_fdata / np.max(img_fdata) * 255映射到灰度区域.
        img_fdata一定是flair.
        t1, t1ce, t2模态一并传入,注意标准化到[0,255],和img_fdata一样..
        init_modal: 字符串'flair','t1','t1ce',t2',最开始是哪个模态，应该和点开的小窗口同步.
        seg_fdata: 预测出来的肿瘤分布.
        index: 展示哪个方向的图, img_fdata[i,:,:] if index==0
        myicon: MyIcon的一个实例..需要在使用本类之前调用,在Constant.py修改,并修改本类的setIcons方法
        first_img: 点开这个窗口时第一个显示的图片.
        file_name: 显示的图片名字.
        init_seg_mask: 为1表示要显示这个肿瘤. init_seg_mask[1]为ED... **注意空出下标0,或者用字典**
        savePath: 保存修改后mask的路径.如果指定就直接用这个，如果为None会在绘画保存的时候弹窗询问.建议在外部指定，
        否则在这里弹窗询问的结果外部可能不方便知道.
        affine: 用作生成.nii文件的放射变换矩阵.感觉暂时用不上，不做检查.
        '''
        print("init NiiImageViewer")
        super().__init__(parent)
        self.resize(width, height)
        self.setMinimumHeight(800)
        self.setMinimumWidth(1000)
        self.setWindowTitle("ImageViewer | "+file_name)
        self.setStyleSheet('QWidget{background-color:#000000;}')
        self.brainModalDict = {'flair':img_fdata,'t1':t1,'t1ce':t1ce,'t2':t2} #保存所有flair等模态
        #self.img_num = self.brainModalDict[init_modal].shape[index]
        self.cur_modal = init_modal  # 当前正在显示的大脑模态（flair）等.
        self.seg_mask = copy.copy(init_seg_mask)   # 拷贝！拷贝！
        # savePath and affine for painting mode:
        self.savePath = savePath
        if savePath is not None and not savePath.endswith(".nii.gz"):
            # 不合法的指定.
            self.savePath = None
        self.affine = affine
        # 图标
        self.setIcons()
        # 布局.
        #layout = QVBoxLayout()
        # 画布.
        self.canvas = NiiImageCanvas(self.brainModalDict[init_modal], seg_fdata,index,init_seg_mask,first_img,self,height,width)
        # 状态栏 -> label, 为了弥补QT的bug之菜单里不会同时有图标和文字...
        # 暂时没办法实现.
        # 菜单栏、工具栏
        ## 保存图片.
        self.menubar = QMenuBar(self)
        self.menubar.setCursor(Qt.ArrowCursor)
        self.export_act = QAction(self.icons['export_image'], "导出", self)
        self.export_act.triggered.connect(self.exportCurImage)
        self.menubar.addAction(self.export_act)
        ## 设置各种模态、视图的开关.
        modal = self.menubar.addMenu(self.icons['brain'],"模态")
        ### 三视图
        self.three_view_act_list = []
        self.three_view_act_list.append(self.threeViewAction('side',0))
        self.three_view_act_list.append(self.threeViewAction('front',1))
        self.three_view_act_list.append(self.threeViewAction('top',2))
        three_view = modal.addMenu(self.icons['three_view'],'perspectives')
        three_view.addAction(self.three_view_act_list[1])
        three_view.addAction(self.three_view_act_list[0])
        three_view.addAction(self.three_view_act_list[2])
        ### flair等脑模态
        self.modal_act_dict={}
        self.modal_act_dict['flair'] = self.modalAction('flair', 0)
        self.modal_act_dict['t1'] = self.modalAction('t1', 0)
        self.modal_act_dict['t1ce'] = self.modalAction('t1ce', 0)
        self.modal_act_dict['t2'] = self.modalAction('t2', 0)
        bra_modal = modal.addMenu(self.icons['brain_modal'],'brain modality')
        bra_modal.addAction(self.modal_act_dict['flair'])
        bra_modal.addAction(self.modal_act_dict['t1'])
        bra_modal.addAction(self.modal_act_dict['t1ce'])
        bra_modal.addAction(self.modal_act_dict['t2'])
        ### 肿瘤模态
        modal.addAction(self.modalAction("ED", 1))
        modal.addAction(self.modalAction("NCR", 2))
        modal.addAction(self.modalAction("ET", 3))
        modal.addAction(self.modalAction("TC", 4))
        modal.addAction(self.modalAction("WT", 5))

        ## 开关绘画模式的Action开关!
        self.paint_action = QAction(self.icons['brush_off'],'Painting Mode Off', self)
        self.paint_action.triggered.connect(self.PaintingOn)
        self.menubar.addAction(self.paint_action)

        ## 退出按钮
        self.exit_action = QAction(self.icons['exit'], "退出", self)
        self.menubar.addAction(self.exit_action)

        # 控制展示第几张图片的滑动条
        #窗口上来会调用个resizeEvent, 所以init里的move没用..
        img_num = self.brainModalDict[init_modal].shape[index]  #当前图像总数.
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setCursor(Qt.ArrowCursor)
        self.slider.resize(int(0.7*width), 10)
        self.slider.setMinimum(1)
        self.slider.setMaximum(img_num)
        self.slider.setTickPosition(QSlider.TicksAbove)
        self.slider.setSingleStep(1)
        self.slider.move(self.size().width()//2-self.slider.size().width()//2, 40)
        self.slider.valueChanged.connect(self.changeSliderValue)
        # 和slider绑定的微调框.虽然不美，但直接贴在旁边，不再另起一类.
        self.spinbox = QSpinBox(self)
        #self.spinbox.resize(60, 10)
        self.spinbox.setCursor(Qt.ArrowCursor)
        self.spinbox.setMinimum(1)
        self.spinbox.setMaximum(img_num)
        self.spinbox.setSingleStep(1)
        self.spinbox.move(self.size().width()//2-self.slider.size().width()//2-100, 40)
        self.spinbox.setStyleSheet("QWidget{color:#ffffff;}")
        # 后缀.
        self.spinbox.setSuffix(" / %d"%(img_num))
        # 将其和self.slider绑定.
        self.spinbox.valueChanged.connect(self.slider.setValue)   #self.slider.valueChanged绑定了真正的执行函数
        # slider的变值可能会反悔，所以不能直接和spinbox的setValue绑定.

        # 所有组件就绪，先用setValue调整当前展示的图片.
        self.slider.setValue(first_img+1)

        # painting bar. it's None at first.
        self.paint_bar = None
        self.circle_pixmap = QPixmap("../UI/pictures/circle.png")   #用作鼠标的圆形.

    # 辅助初始化的函数.
    def modalAction(self, name, code):
        # 控制显示与否的QAction，顺便根据初始的显示情况初始化图标
        act = QAction(name, self)
        if name in ['flair','t1','t1ce','t2']:   # 是flair等模态.
            if self.cur_modal==name:
                act.setIcon(self.icons['ok_circle'])
            else:
                act.setIcon(self.icons['circle'])
            act.triggered.connect(lambda:self.changeBrain(name))
            return act

        if self.seg_mask[code]:
            act.setIcon(self.icons['open_eye'])  # 睁眼
        else:
            act.setIcon(self.icons['close_eye'])
        act.triggered.connect(lambda:self.changeModal(code, act))
        return act

    def threeViewAction(self, name, code):
        # 初始化切换视角action的函数。0-侧视图，1-正视图，2-俯视图
        act = QAction(name,self)
        if self.canvas.index == code:
            act.setIcon(self.icons['ok_circle'])
        else:
            act.setIcon(self.icons['circle'])
        act.triggered.connect(lambda: self.changeThreeView(code))
        return act

    def setIcons(self):
        myicon = MyIcons()
        self.icons={'close_eye':myicon.close_eye,'open_eye':myicon.open_eye,'brain':myicon.brain,
                    'export_image':myicon.export_image,'ok_circle':myicon.ok_circle,'circle':myicon.circle,
                    'brain_modal':myicon.brain_modal,'brush':myicon.brush,'brush_off':myicon.brush_off,
                    'three_view':myicon.three_view, 'exit':myicon.exit}

    # 开启/关闭paint模式.
    def PaintingOn(self):
        if self.paint_action.text().endswith("On"):
            return    #已经开着了，直接返回.
        if self.paint_bar is None:
            # 分配一个paint_bar.
            width = int(0.9*self.size().width())
            if width > 1000:
                width = 1000
            self.paint_bar = PaintingBar(self,width=width,height=60)
            self.paint_bar.move(self.size().width()-self.paint_bar.size().width(), 60)
            # slider, 三视图等都不需要额外和canvas或者paint_manager绑定.
            paint_manager = self.canvas.PaintingModeOn(self.savePath, self.affine)
            self.paint_bar.region_combo.currentIndexChanged.connect(paint_manager.bind)
            self.paint_bar.width_combo.currentTextChanged.connect(paint_manager.setWidth_str)
            self.paint_bar.width_combo.currentTextChanged.connect(self.setCircleCursor)
            self.paint_bar.paintModeOn.connect(self.canvas.useBrush)
            self.paint_bar.paintModeOn.connect(self.setCircleCursor)
            self.paint_bar.eraseModeOn.connect(self.canvas.useEraser)
            self.paint_bar.eraseModeOn.connect(self.setCircleCursor)
            self.paint_bar.dragModeOn.connect(self.canvas.useDragger)
            self.paint_bar.dragModeOn.connect(self.setHandCursor)
            # 槽函数执行顺序和链接顺序一致.
            self.paint_bar.fallback.clicked.connect(self.canvas.paint_manager.fallBack2SavedSeg3d)
            self.paint_bar.fallback.clicked.connect(self.repaint)
            self.paint_bar.ok.clicked.connect(self.PaintingOff)
            self.paint_bar.save.clicked.connect(self.canvas.paint_manager.saveIntoSeg3d)
            self.paint_bar.setVisible(True)
            # 把菜单里的导出绑定到paint_manager的导出!
        else:
            self.paint_bar.setVisible(True)
            self.paint_bar.restart()
            self.canvas.PaintingModeOn()
        # 设置图标.
        self.paint_action.setIcon(self.icons['brush'])
        self.paint_action.setText("Painting Mode On")  #用这个文字来标志开关状态.
        self.setCircleCursor()
        # 每次都重新绑定!
        self.export_act.triggered.disconnect(self.exportCurImage)
        self.export_act.triggered.connect(self.canvas.paint_manager.exportSeg3d)
        # 就保留原本正在显示的肿瘤mask吧.

    def PaintingOff(self):
        '''
        关闭绘画模式,可能打断,如果关闭之后在PaintingManager指定的文件夹下有保存，就换成新的seg_mask.
        '''
        if self.canvas.PaintingModeOff() != 0:
            # 被打断.
            return
        self.paint_bar.setVisible(False)
        # 设置图标.
        self.paint_action.setIcon(self.icons['brush_off'])
        self.paint_action.setText('Painting Mode Off')
        # 检查有没有新的seg文件，并更新
        # 固定检查保存路径下是否有文件，有就重新加载，感觉有点浪费，因为即使有也可能刚才没做修改.
        if self.canvas.paint_manager.hasSaved3D:
            path = self.canvas.paint_manager.getSavedPath()
            if os.path.exists(path) and path.endswith(".nii.gz"):
                nii = nibabel.load(path)
                self.canvas.change_seg_fdata(nii.get_fdata())
        # 把export_act绑定回去.
        self.export_act.triggered.disconnect(self.canvas.paint_manager.exportSeg3d)
        self.export_act.triggered.connect(self.exportCurImage)
        self.setCursor(Qt.ArrowCursor)

    # 事件函数
    def resizeEvent(self, a0: QResizeEvent) -> None:
        self.slider.move(self.size().width()//2-self.slider.size().width()//2, 40)
        self.spinbox.move(self.size().width()//2-self.slider.size().width()//2-100, 40)
        self.canvas.resize(self.size())
        if self.paint_bar is not None and self.paint_bar.isVisible():
            # 正在显示这个paint_bar.
            width = int(0.9*self.size().width())
            if width > 1000:
                width = 1000
            self.paint_bar.resize(width, self.paint_bar.size().height())
            self.paint_bar.move(self.size().width()-self.paint_bar.size().width(), 60)
        return super().resizeEvent(a0)

    # 槽函数.
    def change_img_fdata(self, new_fdata):
        self.canvas.change_img_fdata(new_fdata)

    def changeBrain(self, name):
        '''
        切换当前正在显示的flair等模态信息.
        '''
        if name!=self.cur_modal:
            self.modal_act_dict[name].setIcon(self.icons['ok_circle'])
            self.modal_act_dict[self.cur_modal].setIcon(self.icons['circle'])
            self.cur_modal = name
            self.canvas.change_img_fdata(self.brainModalDict[name])

    def changeModal(self, code, act:QAction):
        if self.seg_mask[code]==1:
            self.seg_mask[code] = 0
            act.setIcon(self.icons['close_eye'])
        else:
            self.seg_mask[code] = 1
            act.setIcon(self.icons['open_eye'])
        self.canvas.AddOrRemoveMask(code)

    def changeThreeView(self, code):
        '''
        可能会被打断.
        切换三视图.小心，要跟着切换滑动条的范围..并且判断当前是否已经大于最大范围.
        如果当前范围已经超过最大范围，QSlider自动调整slider的值，改变值会发出valueChanged信号.
        '''
        if self.canvas.index==code:
            return
        prev = self.canvas.index    # self.canvas.changeThreeView之后会把self.canvas.index改为code
        if self.canvas.changeThreeView(code) != 0:
            return # 打断结束.
        self.three_view_act_list[prev].setIcon(self.icons['circle'])
        self.three_view_act_list[code].setIcon(self.icons['ok_circle'])
        self.slider.setMaximum(self.canvas.img_fdata.shape[code])
        self.spinbox.setMaximum(self.canvas.img_fdata.shape[code]) #这个似乎会直接把slider值变为最大值
        self.spinbox.setSuffix(" / %d"%(self.canvas.img_fdata.shape[code]))

    def changeSliderValue(self, index):
        '''
        slider的槽函数.因为绘画模式下可能打断，所以需要单独的槽函数.  
        小心一种震荡bug:  
        为了防止slider改变spinbox,spinbox的valueChanged又试图改变slider的震荡，
        认为spinbox为第一级，在从spinbox修改slider之前先检查，是否spinbox当前值已和slider一样.  
        绘画模式中，修改slider询问保存，如果cancel，slider会被set回原值，但是第一次set依旧触发了spinbox  
        的valueChanged，这个valueChanged又试图改变slider值与他一样，来回震荡!  

        由于x,y的时候要从后往前，所以应该处理一下滑动条的值...在滑动条的值传入change_xxx之前处理!
        '''
        view = self.canvas.getView()
        if view == 0 or view == 1:
            change_to = self.canvas.img_fdata.shape[view] - index
            cur_sld_val = self.canvas.img_fdata.shape[view] - self.canvas.cur_img
        else:
            change_to = index - 1
            cur_sld_val = self.canvas.cur_img + 1
        # change_to: 这次指示要切换到的下标，从0开始。  
        # cur_sld_val: 切换之前的滑动条的值. 从1开始
        if change_to == self.canvas.cur_img:
            return
        # 失败了，是slider发起就不动spinbox，是spinbox发起就把spinbox改回去.
        if self.canvas.changeImage(change_to)!=0:
            self.slider.setValue(cur_sld_val)  # 被打断，不改,也不改spinbox的值
            if index == self.spinbox.value():        # 新值和spinbox一样，是spinbox发起的，把spinbox改回去.
                self.spinbox.setValue(cur_sld_val)
            return
        # 成功了，是spinbox发起就不管，是slider发起就改spinbox.
        elif index != self.spinbox.value():          # 新值和spinbox不一样，是slider发起
            self.spinbox.setValue(index)             # 修改成功，同步spinbox过来.

    def changeIndex(self, newIndex):
        '''
        改变正在显示的索引值注意NewIndex是从1开始的!  
        而且这个NewIndex是“原始”的，比如说，按照的当前的图片选取规则，选取0通道上的切片应该
        [shape[0]-newIndex, :, :]这样取.  
        这里不对newIndex做检查  
        '''
        self.slider.setValue(newIndex)

    def setCircleCursor(self, width:str = None):
        #self.setCursor(Qt.ArrowCursor)
        # 如果width是None,就从self...里面读出来.
        # 槽函数会按照链接顺序执行.
        if self.canvas.enableDragging:    #正在拖拽状态，不应该修改光标...
            return
        if width is None:
            w = self.canvas.paint_manager.pen_width
        else:
            try:
                w = int(width)
            except:
                return
        if w>=10:
            scaled_circle = self.circle_pixmap.scaled(w,w)
            cursor = QCursor(scaled_circle)
            self.setCursor(cursor)

    def setHandCursor(self):
        # 防止在dragmode中由width_combo触发setCircleCursor...
        self.setCursor(Qt.OpenHandCursor)

    def exportCurImage(self):
        '''
        将当前展示的图片导出为.png, 由用户指定名称.
        '''
        savename,_ = QFileDialog.getSaveFileName(self, "save as...", "./", "Image files (*.png)")
        # 最后结果确实是包含了完整路径的.
        if savename != "":
            pic = self.canvas.getShowingImage()
            pic = cv2.cvtColor(pic, cv2.COLOR_RGB2BGR)         
            cv2.imencode('.png', pic)[1].tofile(savename)

    def closeEvent(self, a0: QCloseEvent) -> None:
        # 上层窗口的closeEvent似乎不会传递到子窗口...
        self.canvas.closeEvent(a0)
        return super().closeEvent(a0)


class NiiImageAttachment(NiiImageCanvas):
    '''
    废弃.
    NiiImageCanvas(Core)的子类，唯一的区别是多了一个待重写的mouseDoubleClickEvent.
    主窗口中的小视图用这个，双击它切换出NiiImageViewer完整的大窗口.
    留出的槽函数：
    changeImage(index): 留给滑动条，切换为显示第几张图片
    AddOrRemoveMask(code): 显示/隐藏哪个模态，code为模态编码.
    getShowImage(): 获取当前正在显示的图像.
    change_img_fdata(new_fdata): 在外部控制切换flair等模态, 小窗口只会存储当前正在显示的模态数据，所以切换的时候
    需要外部传入.
    在这里加入changeFocusPoint(x,y,z): 用来接收NiiModelCore的rightClickSignal信号,用来"跳转到
    被在3d框中选中的位置，独立的图片查看大窗口中没有添加此槽函数，因为大窗口将覆盖，操作不了3D，故不需要."
    因为这个窗口里面没有继承滑动条，所以一定要在主界面的滑动条上也加一个槽函数，来与3D选中造成的跳转同步!
    '''
    def __init__(self, img_fdata, seg_fdata, index, init_seg_mask=[0, 0, 0, 0, 0, 0], first_img=0, parent=None, height=800, width=800) -> None:
        '''
        img_fdata: 要显示的模态数据，必须先 / np.max(img_fdata) * 255后再传进来.
        seg_fdata: 预测后的肿瘤mask.
        index: 方向.  img_fdata[_,:,:] if index=0
        init_seg_mask: 初始时显示的脑瘤区域。注意空出位置0。
        first_img: 初始时显示第几张图片,**不做越界检查**!
        '''
        super().__init__(img_fdata, seg_fdata, index, init_seg_mask, first_img, parent, height, width)
        self.setStyleSheet('QWidget{background-color:#000000;}')

    def mouseDoubleClickEvent(self, a0:QMouseEvent) -> None:
        '''
        TODO
        根据主窗口的切换逻辑，重写这个事件函数，用来“双击小窗口的时候切换出大窗口”
        如果在小窗口类中直接操作切换不方便，可以定义一个信号，在这里emit.
        '''
        return super().mouseDoubleClickEvent(a0)

    def changeFocusPoint(self, x:int, y:int, z:int):
        #print(x,y,z)
        self.focusPoint.x = x
        self.focusPoint.y = y
        self.focusPoint.z = z
        if x==-1 or y==-1 or z==-1:
            self.repaint()
        else:
            self.cur_img = self.focusPoint[self.index]
            self.changeImage(self.cur_img)   # 这个函数会repaint


def main():
    img_fdata = nibabel.load("../nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task500_BraTS2021/imagesTs/BraTS2021_00022_0003.nii.gz").get_fdata()
    img_fdata = img_fdata / np.max(img_fdata) *255
    seg_fdata = nibabel.load("../nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task500_BraTS2021/ensemble_result/BraTS2021_00022.nii.gz").get_fdata()
    t1 = nibabel.load("../nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task500_BraTS2021/imagesTs/BraTS2021_00022_0000.nii.gz").get_fdata()
    t1 = t1/np.max(t1)*255
    t1ce = nibabel.load("../nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task500_BraTS2021/imagesTs/BraTS2021_00022_0001.nii.gz").get_fdata()
    t1ce = t1ce/np.max(t1ce)*255
    t2 = nibabel.load("../nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task500_BraTS2021/imagesTs/BraTS2021_00022_0002.nii.gz").get_fdata()
    t2 = t2/np.max(t2)*255
    app = QApplication(sys.argv)
    #myicon = MyIcons()
    qm.apply_stylesheet(app, "dark_blue.xml")
    test = NiiImageViewer(img_fdata, t1,t1ce,t2,'flair', seg_fdata, 1,width=1200)
    test.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()