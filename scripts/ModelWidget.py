#############
# author : JhLi
#############

import sys
import numpy as np
import vedo
import nibabel
import copy

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import qt_material as qm
from vtk import vtkVolumePicker

from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from scripts.Constant import Constant,MyIcons
from scripts.util import distance_2

class VolumeWrapper:
    '''
    对vedo.Volume的一个封装,包含一些用得到的参数和回调函数.
    实现：进度条调整透明度（实际上只能看到表面颜色改变）
    按钮切换max_projection——但这并不是完全的切面图,而是透视图.
    是否显示这个Volume在外面的Plotter控,不在这里.
    '''
    def __init__(self, img_fdata, color, mode = 0) -> None:
        #self.showing = showing
        self.data_min = 0
        self.mode = mode
        self.data_max = np.max(img_fdata)
        self.alpha = [(0,0), (self.data_max//2, 0.5), (self.data_max, 1)]
        self.vol = vedo.Volume(img_fdata, color, alpha=self.alpha)

    def alphaSliderCallback(self, value, plt:vedo.Plotter):
        '''
        是控制透明度的slider的回调函数
        '''
        self.alpha = [(0,0), (value, 0.5), (self.data_max, 1)]
        self.vol.alpha(self.alpha)
        plt.render()

    def modeProjCallback(self, plt:vedo.Plotter):
        if self.mode == 0:
            self.mode = 1
        else:
            self.mode = 0
        self.vol.mode(self.mode)
        plt.render()


class NiiModelCore(QWidget):
    '''
    在这里显示了3D界面后实在很难安排其它按钮/滑动条（布局乱七八糟），索性只用来显示3D界面
    以及设置回调函数，当做“Core”，其它控制部件放在另一个窗口，最后组合成一个总窗口.
    '''
    rightClickSignal = pyqtSignal(int,int,int)
    doubleClickSignal = pyqtSignal()

    def __init__(self, img_fdata:np.ndarray, seg_fdata:np.ndarray, seg_mask=[1,0,0,0,0,0], width=1000, height=800, parent=None) -> None:
        super().__init__(parent)
        self.resize(width, height)
        self.img_fdata = img_fdata[::-1,::-1,:]
        self.seg_fdata = seg_fdata[::-1,::-1,:]
        self.seg_mask = copy.copy(seg_mask)
        self.customAxes = 2
        self.frame = QFrame(self)
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)

        self.picker = vtkVolumePicker()    # 相机射线发射器！用来找到点击位置和volume对应的点!!

        # vedo objects and renderer.
        # 暂时不区分各种不同区域的肿瘤！
        self.flair = VolumeWrapper(self.img_fdata, "Greys")
        self.ED = VolumeWrapper((self.seg_fdata==1)*self.img_fdata, "YlOrBr")
        self.NCR = VolumeWrapper((self.seg_fdata==2)*self.img_fdata, "Greens")
        self.ET = VolumeWrapper((self.seg_fdata>2)*self.img_fdata, "Blues")
        self.TC = VolumeWrapper((self.seg_fdata>=2)*self.img_fdata, "Purples")
        self.WT = VolumeWrapper((self.seg_fdata>0)*self.img_fdata, "Reds")
        #self.plt = vedo.Plotter(bg="#031944",qt_widget=self.vtkWidget)
        self.plt = vedo.Plotter(bg="#000000",qt_widget=self.vtkWidget)
        self.plt.add_callback("RightButtonPress",self.rightButtonPress)
        # self.plt.add_callback("MiddleButtonDoubleClickEvent", self.doubleClick)
        # 按照编号顺序顺序的vedo objects
        self.obj = [self.flair, self.ED, self.NCR, self.ET, self.TC, self.WT]
        self.doAddRemoveModel()
        # 是否要显示圆球.
        self.enable_sphere = False
        # hover_legend
        self.plt.add_hover_legend(maxlength=50)
        # 坐标
        self.plt.add_global_axes(self.customAxes)

        # 结束
        layout = QVBoxLayout()
        layout.addWidget(self.vtkWidget)
        self.setLayout(layout)

    # 事件函数.
    def rightButtonPress(self, evt):
        # 二分查找最终是有点问题. 只能用来找全脑上的点.
        # vtkVolumePicker直接实现射线取点.
        if evt.picked3d is not None:
            #print(evt.picked3d)
            self.picker.Pick(evt.picked2d[0],evt.picked2d[1],0.0,self.plt.renderer)
            pos = self.picker.GetPickPosition()
            #print("pos=",pos)
            self.rightClickSignal.emit(int(pos[0]),int(pos[1]),int(pos[2]))
            if self.enable_sphere:
                sphere = vedo.Sphere(pos,r=4,c='yellow',res=12)
                sphere.legend("x=%d,y=%d,z=%d"%(pos[0],pos[1],pos[2]))
                self.plt.add(sphere)
                sphere.name = "(%d, %d, %d) (in mm)"%(pos[0],pos[1],pos[2])
        else:
            self.rightClickSignal.emit(int(-1),int(-1),int(-1))

    def doubleClick(self, evt):
        print("double click NiiModelCore")
        self.doubleClickSignal.emit()

    def closeEvent(self, a0: QCloseEvent) -> None:
        self.vtkWidget.close()
        return super().closeEvent(a0)

    # 槽函数
    def change_seg_data(self, new_seg_data:np.ndarray):
        '''
        切换肿瘤划分数据.
        '''
        if new_seg_data.shape != self.seg_fdata.shape:
            QMessageBox.critical(
                self, "wrong segmentation!",
                "segmentation does not match!",
                QMessageBox.StandardButton.Ok
            )
            return
        rotated_new_seg_data = new_seg_data[::-1,::-1,:]
        del self.ED, self.NCR, self.ET, self.TC, self.WT
        self.ED = VolumeWrapper((rotated_new_seg_data==1)*self.img_fdata, "YlOrBr")
        self.NCR = VolumeWrapper((rotated_new_seg_data==2)*self.img_fdata, "Greens")
        self.ET = VolumeWrapper((rotated_new_seg_data>2)*self.img_fdata, "Blues")
        self.TC = VolumeWrapper((rotated_new_seg_data>=2)*self.img_fdata, "Purples")
        self.WT = VolumeWrapper((rotated_new_seg_data>0)*self.img_fdata, "Reds")
        self.obj = [self.flair, self.ED, self.NCR, self.ET, self.TC, self.WT]
        self.doAddRemoveModel()   # 重新刷一次.

    def AddOrRemoveVolume(self,code):
        #把seg_mask取反即可
        if self.seg_mask[code]==1:
            self.seg_mask[code]=0
        else:
            self.seg_mask[code]=1
        self.doAddRemoveModel()

    def doAddRemoveModel(self):
        # 按照优先级顺序先清理模型、再加入模型...
        exist = self.plt.get_volumes()
        new = []
        for k in [0,5,4,1,3,2]:
            if self.seg_mask[k]==1:
                new.append(self.obj[k].vol)
        self.plt.remove(exist)
        if  new != []:
            self.plt.add(new)
        self.plt.render()

    def changeAlpha(self, value, code):
        self.obj[code].alphaSliderCallback(value, self.plt)

    def changeMode(self, code):
        self.obj[code].modeProjCallback(self.plt)

    def startSphere(self):
        # 在右键点到的地方画上一个带标注的圆球！
        self.enable_sphere = True

    def endSphere(self):
        self.enable_sphere = False
        #并直接删除所有标注，不保存...
        spheres = self.plt.get_meshes()
        self.plt.remove(spheres)
        self.plt.render()

    def show(self):
        self.plt.show()
        super().show()


class NiiModelWidget(QWidget):
    '''
    完整的3D查看器，包含控制与Core
    '''
    def AskOpenPath(window_name = "选择要打开的文件"):
        '''
        询问要打开的.nii.gz的文件路径.
        同样返回(成功接受与否，path)
        '''
        path,_ = QFileDialog.getOpenFileName(None,window_name,"./",'Nifti1 (*.nii.gz)')
        return path!="", path

    def __init__(self, img_fdata, seg_fdata, seg_mask=[1,0,0,0,0,0], width=1000, height=900, parent = None, name="") -> None:
        '''
        img_fdata: 原图.这里不需要标准化为灰度！
        seg_fdata: 预测的肿瘤.
        seg_mask: 初始要显示哪个肿瘤.编号和图片查看器一样，注意下标0表示全脑(比如flair),默认其1。
        width, height: 窗口大小.
        name: 文件名字(会跟在窗口后面)
        这个窗口有最小大小的限制：为(500,500)
        由于右键事件信号是在NiiModelCore中的，所以大窗口其实也有self.core.rightClickSignal信号，如果
        也希望用起来，直接NiiModelAttachment类中所说的连接槽函数changeFocusPoint就可以了（只有图片小窗口才有这个槽函数）。
        详见 NiiModelAttachment下面的注释.
        '''
        super().__init__(parent)
        # 自己大小与标题
        if width < 900:
            width = 900
        if height < 800:
            height = 800
        self.resize(width, height)
        self.setMinimumWidth(900)
        self.setMinimumHeight(800)
        self.setWindowTitle("3D-Model Viewer | "+name)
        # 控制窗体大小与定位
        self.ctrl = QWidget(self)
        self.ctrl.resize(200, height)
        self.ctrl.move(0,0)
        # Core窗体
        self.core = NiiModelCore(img_fdata, seg_fdata,seg_mask, width-self.ctrl.size().width(),height,self)
        self.core.move(200,0)
        # Ctrl布局.
        vlayout = QVBoxLayout()
        vlayout.setSpacing(30)
        # flair控制组件.
        flair_layout = self.singleToolKit("flair",self.core,self.core.flair,0,seg_mask[0])
        ED_layout = self.singleToolKit("ED",self.core,self.core.ED,1,seg_mask[1])
        NCR_layout = self.singleToolKit("NCR",self.core,self.core.NCR,2,seg_mask[2])
        ET_layout = self.singleToolKit("ET", self.core, self.core.ET, 3, seg_mask[3])
        TC_layout = self.singleToolKit("TC", self.core, self.core.TC, 4, seg_mask[4])
        WT_layout = self.singleToolKit("WT",self.core,self.core.WT, 5, seg_mask[5])
        # text，用来显示文本.
        self.text_label = QLabel(" info\n\n",self)
        self.text_label.resize(200,100)
        self.core.rightClickSignal.connect(self.display_info)
        # 要算距离的点，最多两个点，不断刷掉前面的点...
        self.points_for_dist = []

        vlayout.addLayout(self.topCheckBox())
        vlayout.addLayout(flair_layout)
        vlayout.addLayout(ED_layout)
        vlayout.addLayout(NCR_layout)
        vlayout.addLayout(ET_layout)
        vlayout.addLayout(TC_layout)
        vlayout.addLayout(WT_layout)

        self.ctrl.setLayout(vlayout)
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidget(self.ctrl)
        self.scroll_area.resize(self.ctrl.size().width(), self.ctrl.size().height()-100)
        self.scroll_area.move(0,0)
        # ctrl里没有文字框，ctrl加了scroll_area，用scroll_area管理。scroll_area下面接固定大小(200,100)的文字框.
        self.text_label.resize(self.ctrl.size().width(), 100)
        self.text_label.move(0,self.ctrl.size().height()-100)  # 有文字，不要太靠边.
        if __name__!='__main__':
            qm.apply_stylesheet(self.ctrl, 'dark_blue.xml')
            qm.apply_stylesheet(self.scroll_area, 'dark_blue.xml')
            self.text_label.setStyleSheet("QWidget{background-color:#31363b}")

    def topCheckBox(self):
        # 最顶端的单选框：开启/关闭定位球功能；切换标签为用户自己的标签.
        vlayout = QVBoxLayout()
        # 图标
        myicons = MyIcons()
        # 切换肿瘤segmentation功能的按钮
        change_seg_btn = QPushButton(myicons.file,"Segmentation", self)
        change_seg_btn.clicked.connect(self.change_seg_data)
        # 开启/关闭定位球的功能.
        locate_chk = QCheckBox(self)
        locate_chk.setText("Locating Sphere")
        locate_chk.setIcon(myicons.location)
        locate_chk.clicked.connect(self.switchLocation)
        # 切换为用户自己画的标签...
        #...
        vlayout.addWidget(change_seg_btn)
        vlayout.addWidget(locate_chk)
        return vlayout

    def singleToolKit(self, name, core:NiiModelCore, vol:VolumeWrapper, code, show_init=False, mode_init=0):
        '''
        包含一个控制显示与否的勾选框, 使用max projection的勾线框, 一个设置透明度的滑动条.
        返回一个QGridLayout
        '''
        grid_layout = QGridLayout()
        grid_layout.setSpacing(20)
        # 控制showing与否的按钮
        show_btn = QCheckBox(self.ctrl)
        show_btn.setText("show "+name)
        show_btn.setCheckState(show_init)
        show_btn.setTristate(False)
        show_btn.stateChanged.connect(lambda:core.AddOrRemoveVolume(code))
        # 控制max_projection与否的按钮
        proj_btn = QCheckBox(self.ctrl)
        proj_btn.setText(name+" max-projection")
        proj_btn.setCheckState(mode_init)
        proj_btn.setTristate(False)
        proj_btn.stateChanged.connect(lambda:core.changeMode(code))
        # 控制alpha的垂直滑动条
        sld = QSlider(self.ctrl)
        sld.setMaximum(int(vol.data_max)-1)
        sld.setMinimum(int(vol.data_min)+1)
        sld.setValue(int(vol.data_max)//2)
        sld.setInvertedAppearance(True)
        sld.setStyleSheet(
            '''
            QSlider::sub-page:vertical{background:#000000;}
            QSlider::add-page:vertical{background:#83b9ff;}
            '''
        )  
        # 上部被认为是滑动过(sub-page)的，下部被认为没有(add-page)。我希望上部是滑动过的，只能用颜色设置看起来像.
        sld.valueChanged.connect(lambda:core.changeAlpha(sld.value(),code))
        # 布局
        grid_layout.addWidget(show_btn, 0,0)
        grid_layout.addWidget(proj_btn, 1,0)
        grid_layout.addWidget(sld, 0,1,2,1)
        return grid_layout

    def resizeEvent(self, a0: QResizeEvent) -> None:
        #self.ctrl.move(0,0)
        self.scroll_area.move(0,0)
        self.scroll_area.resize(self.ctrl.size().width(), a0.size().height()-100)
        self.text_label.move(0, a0.size().height()-100)
        self.core.resize(self.size().width()-self.ctrl.size().width(), self.size().height())
        self.core.move(self.ctrl.size().width(),0)
        return super().resizeEvent(a0)

    def show(self):
        self.core.show()
        self.ctrl.show()
        return super().show()

    def closeEvent(self, a0: QCloseEvent) -> None:
        self.core.close()
        self.ctrl.close()
        return super().closeEvent(a0)

    # 槽函数，用来接收右键选点信息.
    def display_info(self,x,y,z):
        # 显示右键点中的点的信息.主要是右键选点选中的点，和测量的两点间距离.
        if x==-1 or y==-1 or z==-1:
            # 直接全部清理掉.
            self.text_label.setText(" info\n\n")
            self.points_for_dist.clear()
        else:
            has_dist = False
            if len(self.points_for_dist)==2:
                self.points_for_dist.pop(0)
                has_dist = True
            elif len(self.points_for_dist)==1:
                has_dist = True
            self.points_for_dist.append((x,y,z))
            if has_dist:
                x0,y0,z0 = self.points_for_dist[0]
                self.points_for_dist[1]=(x,y,z)
                dist = np.sqrt((x0-x)**2+(y0-y)**2+(z0-z)**2)
                self.text_label.setText(" x=%d,y=%d,z=%d\n x=%d,y=%d,z=%d\n distance=%.2f (1mm isotropic)"%(x0,y0,z0,x,y,z,dist))
            else:
                self.text_label.setText(" x=%d,y=%d,z=%d(1mm isotropic)\n\n"%(x,y,z))

    def switchLocation(self, state):
        if state==1:
            self.core.startSphere()
        else:
            self.core.endSphere()

    def change_seg_data(self):
        ret, path = NiiModelWidget.AskOpenPath()
        if ret:
            nii = nibabel.load(path)
            self.core.change_seg_data(nii.get_fdata())


class NiiModelAttachment(QWidget):
    def __init__(self, img_fdata, seg, seg_mask=[1,0,0,0,0,0], width=1000,height=800, parent=None) -> None:
        '''
        未使用.
        小窗版本的3D窗口.
        img_fdata, seg, seg_mask不赘述.
        myicon 是Constant.MyIcon的一个实例，必须在QApplication后创建.
        留给外部按钮控制的槽函数（控制肿瘤区域显示与否）：
        self.core.AddOrRemoveVolume(code).  code是控制的肿瘤区域编号.
        信号：
        根据鼠标右键选择的地方发射包含所选点位置信息的信号rightClickSignal,这是NiiModelCore的信号
        在此小窗口中为self.core.rightClickSignal.他发射的信号包含三个整型参数：x,y,z.如果右键没有点中物体，
        就发送三个-1.
        只需要将这个信号和三个主界面小窗口NiiImageAttachment的changeFocusPoint方法连接在一起就可以了（注意，只有图片小窗口有这个槽函数），不过考虑到
        主界面上的进度条也应该和图片的跳转同步，所以应该和相应进度条的setValue函数配合所选位置的参数绑定.
        如果将这个self.core.rightClickSignal信号用起来（完成上面的绑定），效果是点击模型会选出模型最表面的点，在三视图中跳转到
        这个点的位置并用红色十字标出，而没有点到物体，就取消红色十字。
        可能会希望在状态栏或其它位置输出所点处的具体坐标，同样写一个获取rightClickSignal信号并设置显示位置的文本的槽函数即可.
        如果不连接这个槽函数，原有功能也不会受任何影响.
        '''
        super().__init__(parent)
        self.resize(width, height)
        self.core = NiiModelCore(img_fdata,seg,seg_mask,width,height,self)
        self.setIcons()
        self.setStyleSheet("QWidget{background-color:#000000;}")
        layout = QHBoxLayout()
        self.toolbar1 = QToolBar('model',self)

        self.toolbar1.setOrientation(Qt.Vertical)
        # 显示/隐藏flair
        self.flair_act = QAction("show flair",self)
        if seg_mask[0]==1:
            self.flair_act.setIcon(self.icons['open_eye'])
        else:
            self.flair_act.setIcon(self.icons['close_eye'])
        self.flair_act.triggered.connect(lambda:self.core.AddOrRemoveVolume(0))
        self.flair_act.triggered.connect(self.change_flair_act_icon)
        self.toolbar1.addAction(self.flair_act)
        self.toolbar2 = QToolBar("projection",self)
        self.toolbar2.setOrientation(Qt.Vertical)
        self.toolbar2.addAction(self.projActionHelper('flair projection',0))
        self.toolbar2.addAction(self.projActionHelper('ED projection',1))
        self.toolbar2.addAction(self.projActionHelper('NCR projection',2))
        self.toolbar2.addAction(self.projActionHelper('ET projection',3))
        self.toolbar2.addAction(self.projActionHelper('TC projection',4))
        self.toolbar2.addAction(self.projActionHelper('WT projection',5))
        layout.addWidget(self.toolbar1)
        layout.addWidget(self.toolbar2)
        layout.addWidget(self.core)
        self.setLayout(layout)

        self.showingBrain = True
        self.projs = [False,False,False,False,False,False]  # 开始默认全没有投影

    def show(self):
        self.core.show()
        super().show()

    def projActionHelper(self,name ,code):
        act = QAction(self.icons['circle'],name, self)
        act.triggered.connect(lambda:self.core.changeMode(code))
        act.triggered.connect(lambda:self.change_proj_act_icon(act,code))
        return act

    def setIcons(self):
        myicon = MyIcons()
        self.icons = {'open_eye':myicon.open_eye, 'close_eye':myicon.close_eye,'ok_circle':myicon.ok_circle,
                      'circle':myicon.circle}

    def change_flair_act_icon(self):
        if self.showingBrain:
            self.showingBrain=False
            self.flair_act.setIcon(self.icons['close_eye'])
        else:
            self.showingBrain=True
            self.flair_act.setIcon(self.icons['open_eye'])

    def change_proj_act_icon(self, act:QAction, code):
        if self.projs[code]:
            self.projs[code] = False
            act.setIcon(self.icons['circle'])
        else:
            self.projs[code] = True
            act.setIcon(self.icons['ok_circle'])


def main():
    flair_data = nibabel.load("nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task500_BraTS2021/imagesTs/BraTS2021_00022_0003.nii.gz").get_fdata()
    seg_data = nibabel.load("nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task500_BraTS2021/inferTs/BraTS2021_00022.nii.gz").get_fdata()
    app = QApplication(sys.argv)
    qm.apply_stylesheet(app, "dark_blue.xml")
    #myicon = MyIcons()
    test = NiiModelWidget(flair_data, seg_data, name="test.nii.gz")
    #test = NiiModelAttachment(flair_data,seg_data)
    test.show()
    sys.exit(app.exec_())


if __name__=="__main__":
    main()