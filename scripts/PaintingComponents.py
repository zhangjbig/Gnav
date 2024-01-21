#############
# author : JhLi
#############

import numpy as np
import os
from PyQt5.QtGui import (QImage, QPixmap, QPainter, QPen, QColor)
from PyQt5.QtCore import (QPoint, QPointF, Qt)
from PyQt5.QtWidgets import (QMessageBox, QFileDialog)
import cv2
import nibabel as nib
from scripts.util import rotate as do_rotate   #避免和一些本地变量重名...
from scripts.util import QImage2Array_GrayScale8, preprocessMRI4Image, postprocessMRI4Image
from scripts.Constant import Constant

class TumorMaskDrawingBuffer:
    '''
    对QImage的一个封装，装在绘图过程中要跟着画笔设置的mask缓冲区.  
    这是一个8ibts灰度QImage(不用bitmap是因为压缩过的bitmap不方便处理),0表示没有，255表示有.  
    ED, NCR, ET三个区域（对应编号1,2,3）都有这一个这个DrawingBuffer.约束他们互不重叠.
    能做到:  
    1、获取来自上层结构的肿瘤划分信息，基于此初始化mask缓冲区，有则255，无则0.  
    2、输出由自己的缓冲区(np.ndarray)给上层结构，让它更新肿瘤划分3维图以保存，或者让其进一步处理成一个QPixmap以绘制.  
    3、获取设置好的画笔在自己的图上绘制，并自动约束和其它两个区域不重叠.  
    '''
    def __init__(self,seg_2d:np.ndarray) -> None:
        '''
        最早的初始化.  
        seg_2d是0-1肿瘤mask，0表示无，1表示有，必须已经完成了旋转.  
        '''
        self.initialize(seg_2d)
        self.constraintList = []   # 用来约束的其它两个区域的buffer

    def initialize(self, seg_2d:np.ndarray):
        '''
        seg_2d: 经过旋转、转置等同步操作的0-1掩码  
        初始化buffer。当用户在没有保存而想放弃已有更改进行初始化的时候，可以用这个.
        '''
        self.bytes_per_line = seg_2d.shape[1]  #原本每行有多少字节. 
        buf = np.uint8(255*seg_2d) # 转化为0-255“灰度”map  
        self.imageBuffer = QImage(
            buf.tobytes(),
            seg_2d.shape[1], seg_2d.shape[0], seg_2d.shape[1],
            QImage.Format.Format_Grayscale8
        )
    
    def addConstraint(self,*argv):
        # 往里面加入需要考虑的约束对象.  
        for i in argv:
            self.constraintList.append(i)

    def getArray(self):
        '''
        从QImage获取np数组形式的8bits位图，shape保持一样.  
        '''
        return QImage2Array_GrayScale8(self.imageBuffer, self.bytes_per_line)
    
    def getConstArray(self):
        '''
        从QImage获取np数组，这里面的数据是const!  
        '''
        return QImage2Array_GrayScale8(self.imageBuffer, self.bytes_per_line, const=True)

    def paint(self, painter:QPainter, pen:QPen, eraser_mode:bool, start_pos:QPointF, end_pos:QPointF):
        '''
        往自己身上绘制并根据约束删除重复的地方的地方.  
        painter必须已经上层结构设置QPen, 颜色，粗细，外观等，这里只画线+约束.  
        start_pos, end_pos必须是不考虑偏移量、只根据本地坐标来的.注意用QPointF即可  
        之后考虑是否换成QPointF.  
        '''
        painter.begin(self.imageBuffer)
        painter.setPen(pen)    # setPen and setCompositionMode必须要在begin之后setPen才有效!
        if eraser_mode:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        painter.drawLine(start_pos, end_pos)
        painter.end()
        # 如果是画而不是擦除,加约束
        if not eraser_mode:
            buf = self.getArray()  # 根据文档，qimg.bits()来的指针指向这个QImage正在使用的内存，改它即改QImage
            for constraint in self.constraintList:
                constraintBuf = constraint.getConstArray()
                # uint8,且颜色要么0要么255(全1)，所以可以直接像bool那样位运算.  
                buf = buf & (~constraintBuf)
            # 上面那样好像无效，可能那里还是复制了一次内存，再重新初始化一次QImage..
            self.imageBuffer = QImage(
                buf.tobytes(),
                buf.shape[1], buf.shape[0], buf.shape[1],
                QImage.Format.Format_Grayscale8
            )
    
class PaintingManager:
    '''
    管理上面三个buffer的结构。功能包括：  
    1、接收信号设置画笔(粗细等)、改变绘制模式(画、擦)  
    2、管理3个肿瘤区域,用启发于openGL bind的方式绑定“正在绘制”的区域.  
    3、接收起点、终点，进行绘画.  
    4、获取3个肿瘤区域融合成的pixmap，用以绘制到屏幕.  
    5、保存为新的肿瘤seg，并返回或保存.  
    只提供绘制和保存能力.
    '''
    NO_SPECIFIED = 0
    USE_DEFAULT = 1
    CUSTOM_SPECIFIED = 2

    def ExtendChannel(img:np.ndarray):
        '''
        (h,w)->(h,w,4)，并且(h,w,:)中的元素各个相等.  
        cv2.cvtColor会把alpha通道(3)置255,需要处理一下这个.我们把它变为1.  
        需保证img是0-1掩码.   
        返回值是np.uint8
        '''
        ret = cv2.cvtColor(np.uint8(img), cv2.COLOR_GRAY2RGBA)  #(h,w,4)
        ret[:,:,3] = ret[:,:,3] * ret[:,:,0] / np.uint8(255)  #让四个通道一样
        return ret
    
    def AskSave():
        '''
        询问是否保存.
        '''
        ret = QMessageBox.question(
            None,"保存","有未保存的修改，是否保存?",
            QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No|QMessageBox.StandardButton.Cancel
        )
        return ret
    
    def AskSavePath():
        '''
        选择保存路径,返回bool(成功与否), path.
        '''
        path,_ = QFileDialog.getSaveFileName(None, "选择保存路径", './', 'Nifti1 (*.nii.gz)')
        return path!="", path
    
    def doExportNii(img_fdata, filename, dir = None, affine = None):
        '''
        filename可以是路径，但这样dir必须为None.  
        如果dir为None, filename必为路径.  
        把img_fdata以nii的形式保存到dir/filename中，不检查目录是否存在.  
        注意，不做检查.  
        '''
        if affine is None:
            affine = np.eye(4)  #填一个单位矩阵.
        nii = nib.nifti1.Nifti1Image(postprocessMRI4Image(img_fdata), affine)
        if dir is not None:
            path = os.path.join(dir, filename)
        else:
            path = filename
        nib.save(nii, path)

    def __init__(self, 
            seg_3d:np.ndarray, 
            view, index,scale, 
            savePath:str = None,
            affine = None
        ) -> None:
        '''
        seg_3d: 初始的3D seg结构，须是经过复制的，以保护原有文件...
        view: 三视图, 0-侧视[0,:,:], 1-正视，2-俯视.  
        index: 最开始正在看这个通道上的哪张图片.  
        scale: 最开始的时候的缩放.  
        default_dir: 默认的保存文件地址，绘画模式关闭的时候，一份.nii.gz文件会自动保存在这个文件夹下，方便后续调用.  
        filenname: 保存下来的自定义的标记文件名称.  
        affine: nii文件的仿射变换矩阵，目前暂时不考虑它  

        不设置dragMode,这个在外面ImageViewer中检查.  
        '''
        # 文件相关
        if savePath is not None and savePath.endswith(".nii.gz"):
            self.savePath = savePath
            self.PathSelected = PaintingManager.CUSTOM_SPECIFIED
        else:
            self.PathSelected = PaintingManager.NO_SPECIFIED
            self.default_dir = "./assets/"
            self.filename = "CustomSeg.nii.gz"
        self.affine = affine
        self.initialize(seg_3d.copy(), view, index, scale)

    def initialize(self, seg_3d:np.ndarray, view, index,scale):
        # 视图相关..
        self.seg_3d = preprocessMRI4Image(seg_3d.copy())
        self.view = view
        self.index = index        # 当前显示这个通道上的第几张图片，相当于ImageWidget的cur_image
        # 绘画相关..
        self.eraser_mode = False  # if True, setCompositionMode(CompositionMode_Clear), or setColor=0
        self.pen_width = 20        # 某个初始值，记得和窗口那边同步一下.
        self.scale = scale
        # 三个区域.  
        self.ed = TumorMaskDrawingBuffer(self.getSeg2d(region=Constant.tumor_name['ED']))
        self.net = TumorMaskDrawingBuffer(self.getSeg2d(region=Constant.tumor_name['NCR']))
        self.et = TumorMaskDrawingBuffer(self.getSeg2d(region=Constant.tumor_name['ET']))
        self.ed.constraintList += [self.net, self.et]
        self.net.constraintList += [self.ed, self.et]
        self.et.constraintList += [self.ed, self.net]   # 保证里面确实是引用...(应该用函数也可以)
        self.buffers = [None, self.ed, self.net, self.et]  # 空出0,列表里都是引用.
        # 最开始的时候，默认ed为激活的
        self.activatedBuffer = self.buffers[1]  
        self.updatePixmap()

        # 保存标志.
        self.unsaved = False
        # 曾经有过保存操作的标志，有了他说明和之前是有点不一样的，需要导出.  
        self.hasSaved3D = False
        # 激活
        self.activated = True
        
    def getSeg2d(self, region=0, rotate=True):
        '''
        从现有的（最新保存的或本就有的）里获得一个seg2d  
        从现有的seg_3d中选出相应的2d图然后返回，如果region=0，直接返回原本的seg_2d  
        如果rotate=True，像imageWidget中那样旋转了再返回!  
        如果0-ED，1-NCR，2-ET，就返回0-1掩码！
        '''
        if self.view == 0:
            seg_2d = self.seg_3d[self.index,:,:]
            if rotate:
                seg_2d = do_rotate(seg_2d, 0)
        elif self.view == 1:
            seg_2d = self.seg_3d[:,self.index,:]
            if rotate:
                seg_2d = do_rotate(seg_2d, 1)
        else:
            seg_2d = self.seg_3d[:,:,self.index]
            if rotate:
                seg_2d = do_rotate(seg_2d, 2)
        if region!=0:
            seg_2d = (seg_2d==region)   # 转化为0-1数组.
        return seg_2d
    
    def getNewSeg2d(self):
        '''
        从三个buffer中获得新的seg2d图像.
        获取当前视图的三个区域组合成的新的Seg2d，用于接下来的保存.  
        这个seg2d符合ED-1，NCR-2，ET-3
        '''
        ed_mask = self.ed.getConstArray()>0
        net_mask = self.net.getConstArray()>0
        et_mask = self.et.getConstArray()>0
        newSeg2d = ed_mask*1 + net_mask*2 + et_mask*3  #newSeg2d最后应该是用来保存的,而不是展示.  
        # 因为是用来保存的，所以转回原样.
        if self.view==0:
            newSeg2d = do_rotate(newSeg2d, 0, True)
        elif self.view==1:
            newSeg2d = do_rotate(newSeg2d, 1, True)
        else:
            newSeg2d = do_rotate(newSeg2d, 2, True)
        return newSeg2d
    
    def saveIntoSeg3d(self):
        '''
        把当前三个区域的修改保存回save_into_seg3d中!
        '''
        if self.PathSelected == PaintingManager.NO_SPECIFIED:
            ret, path = PaintingManager.AskSavePath()
            if ret:
                self.savePath = path
                self.PathSelected = PaintingManager.CUSTOM_SPECIFIED
            else:
                self.PathSelected = PaintingManager.USE_DEFAULT
        '''
        以上是询问保存地址，如果展示暂时不需要可以注释掉，在前面初始化的时候设置全部使用默认USE_DEFAULT
        '''
        newseg = self.getNewSeg2d()   #这个已经是旋转过、变为和3D的一样的了！
        if self.view==0:
            self.seg_3d[self.index,:,:] = newseg       #copy?  
        elif self.view==1:
            self.seg_3d[:,self.index,:] = newseg
        else:
            self.seg_3d[:,:,self.index] = newseg
        self.unsaved = False
        self.hasSaved3D = True
    
    def updatePixmap(self):
        '''
        从当前视图的三个区域组合成新的QPixmap，用于绘制展示.颜色是argb32.  
        不要旋转.  
        '''
        # 先扩展为(h,w,4)
        ed_mask_argb = PaintingManager.ExtendChannel(self.ed.getConstArray()>0)
        net_mask_argb = PaintingManager.ExtendChannel(self.net.getConstArray()>0)
        et_mask_argb = PaintingManager.ExtendChannel(self.et.getConstArray()>0)
        mat = ed_mask_argb*Constant.color.argb_ED + net_mask_argb*Constant.color.argb_NCR + \
              et_mask_argb*Constant.color.argb_ET
        # 先转QImage保险吧..
        qimg = QImage(
            mat.tobytes(),
            mat.shape[1], mat.shape[0], mat.shape[1]*4, 
            QImage.Format.Format_ARGB32
        )
        self.ori_pixmap = QPixmap.fromImage(qimg)
        self.pixmap = self.scale_ori_pixmap()
        
    def getOriPixmap(self):
        '''
        获取原始大小的原图.ARGB32.  
        '''
        return self.ori_pixmap
    
    def getScaledPixmap(self):
        '''
        获取根据self.scale缩放后的新图，这个是给外部显示的..
        '''
        return self.pixmap
    
    def scale_ori_pixmap(self):
        '''
        重新缩放原pixmap
        '''
        return self.ori_pixmap.scaled(
            int(self.ori_pixmap.size().width()*self.scale),
            int(self.ori_pixmap.size().height()*self.scale)
        )

    def paint(self, start_pos:QPoint, end_pos:QPoint):
        '''
        设置画笔，并执行当前activated的区域的绘画!  
        这里的值在使用之前，都需要缩放回原图大小!  
        '''
        # !!用QColor(255)画灰度QImage是否会出锅?!!
        painter = QPainter()
        width = self.pen_width / self.scale
        start = QPointF(start_pos) / self.scale
        end = QPointF(end_pos) / self.scale
        pen = QPen(
            QColor(0xffffffff),width,
            cap=Qt.PenCapStyle.RoundCap, join=Qt.PenJoinStyle.RoundJoin
        )
        # 只有QColor(0xffffffff)才能保证着色为255
        #painter.setPen(pen)
        self.activatedBuffer.paint(painter,pen,self.eraser_mode, start, end)  # 画了且约束了.
        self.updatePixmap()  # 别忘了更新...
        self.unsaved = True

    # 供外部控制的槽函数.
    def bind(self, code):
        '''
        更换“activated"部分.  
        code: 0-ED, 1-NCR, 2-ET,这是为了和QComboBox的索引对应!
        '''
        self.activatedBuffer = self.buffers[code+1]

    def setWidth(self, width):
        self.pen_width = width

    def setWidth_str(self, width:str):
        try:
            num = int(width)
        except:
            return   #非法输入，不修改，返回.  也许加个提示框.
        if num>=10:
            self.pen_width = num
    
    def setScale(self, scale):
        self.scale = scale
        self.pixmap = self.scale_ori_pixmap()

    def setEraseMode(self):
        self.eraser_mode = True
    
    def setPaintingMode(self):
        self.eraser_mode = False

    def switchThreeView(self, view):
        '''
        要切换三视图，这恰恰是要彻底更换buffer的时候了!  
        view-0,side, 1,front, 2,top  
        return 1，表示取消切换视角的操作，在外面应该break.  
        返回0，表示正常切换.
        '''
        if self.view == view:
            return 0
        if self.unsaved:
            save = PaintingManager.AskSave()
            if save == QMessageBox.StandardButton.Yes:
                self.saveIntoSeg3d()
            elif save == QMessageBox.StandardButton.Cancel:   #取消切换.
                return 1
            self.unsaved = False   # No或者直接叉掉了，丢弃之前的修改，改回已保存然后切换.  
        # 直接叉掉也不会保存.
        # 如果超出范围，在这里修改为新的索引.  
        if self.seg_3d.shape[view] <= self.index:
            #self.index = int(self.index / self.seg_3d.shape[self.view] * self.seg_3d.shape[view])
            self.index = self.seg_3d.shape[view]-1   #越界直接放最大值..
        self.view = view
        self.ed.initialize(self.getSeg2d(1))
        self.net.initialize(self.getSeg2d(2))
        self.et.initialize(self.getSeg2d(3))
        self.updatePixmap()   # 切换完刷新
        return 0   # 正常切换，外面也可以切换了.
    
    def setIndex(self, index):
        '''
        要切换第几张图片，和上面一样，返回1外面应该中断切换操作，并且将slider设置回原来的值!!
        '''
        if self.index==index:
            return 0
        if self.unsaved:
            save = PaintingManager.AskSave()
            if save == QMessageBox.StandardButton.Yes:
                self.saveIntoSeg3d()
            elif save == QMessageBox.StandardButton.Cancel:   #取消切换.
                return 1
            self.unsaved = False   # No或者直接叉掉了，丢弃之前的修改，改回已保存然后切换.  
        self.index = index
        self.ed.initialize(self.getSeg2d(1))
        self.net.initialize(self.getSeg2d(2))
        self.et.initialize(self.getSeg2d(3))
        self.updatePixmap()        # 切换了之后必须更新!!
        return 0   # 正常切换，外面也可以切换了.
    
    def fallBack2SavedSeg3d(self):
        '''
        回退到上一个已经保存的版本.  
        '''
        self.ed.initialize(self.getSeg2d(1))
        self.net.initialize(self.getSeg2d(2))
        self.et.initialize(self.getSeg2d(3))
        self.unsaved = False   # 已经保存.
        self.updatePixmap()

    def onClose(self):
        '''
        在即将结束绘制模式的时候调用，询问是否保存最后的修改，并将结果以.nii输出到默认文件夹.  
        同样，有返回1取消结束的限制!
        '''
        if self.unsaved:
            save = PaintingManager.AskSave()
            if save == QMessageBox.StandardButton.Yes:
                self.saveIntoSeg3d()
            elif save == QMessageBox.StandardButton.Cancel:   #取消切换.
                return 1
            # No或者直接叉掉，丢弃，接下来直接导出.
            self.unsaved = False
        if self.hasSaved3D:
            # 确实修改过，将这个结果以.nii保存到默认文件夹以供后面自动使用.  
            if self.PathSelected == PaintingManager.USE_DEFAULT:
                if not os.path.exists(self.default_dir):
                    os.mkdir(self.default_dir)
                # 将自己这里的seg_3d转化为.nii.gz并保存.  
                PaintingManager.doExportNii(self.seg_3d, self.filename, self.default_dir, self.affine)
            else:
                # 到这里一定调用过saveIntoSeg3d,不会有NO_SPECIFIED的情况.
                PaintingManager.doExportNii(self.seg_3d, self.savePath, affine = self.affine)
        return 0
    
    def exportSeg3d(self):
        '''
        询问保存，并将将当前已有的seg_3d导出到用户选择的文件处.  
        '''
        if self.unsaved:
            save = PaintingManager.AskSave()
            if save == QMessageBox.StandardButton.Yes:
                self.saveIntoSeg3d()
            # 否则不保存，直接导出.  
        path,_ = QFileDialog.getSaveFileName(None, "select saving path", './', 'Nifti1 (*.nii.gz)')
        if path=="":
            return
        PaintingManager.doExportNii(self.seg_3d, path, affine = self.affine)

    def isActivated(self):
        return self.activated
    
    def disActivate(self):
        self.activated = False
    
    def Activate(self):
        self.activated = True

    def getSavedPath(self):
        if self.PathSelected == PaintingManager.CUSTOM_SPECIFIED:
            return self.savePath
        else:
            return os.path.join(self.default_dir, self.filename)

        '''
        测试浮点数起止点、宽度是否会让有些点不完全着色，给一些中间值.  
        确实是全部都是255，没有因浮点数而比例衰减等.
        '''