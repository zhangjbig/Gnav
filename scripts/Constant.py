#############
# author : JhLi
#############

import numpy as np
import qtawesome as qta
class Color:
    ED = np.array([249, 190, 0], dtype=np.uint8)
    NCR = np.array([24, 135, 73], dtype=np.uint8)
    ET = np.array([1, 102, 184], dtype=np.uint8)
    TC = np.array([164, 45, 232], dtype=np.uint8)
    #WT = np.array([187, 245, 123], dtype=np.uint8)
    WT = np.array([233, 77, 44], dtype=np.uint8)   # 镉红
    # 绘图用的半透明颜色
    # argb_ED = np.array([241, 214, 145, 100], dtype=np.uint8)
    # argb_NCR = np.array([128, 174, 128, 100], dtype=np.uint8)
    # argb_ET = np.array([111, 184, 210, 100], dtype=np.uint8)
    argb_ED = np.array([145, 214, 241, 100], dtype=np.uint8)
    argb_NCR = np.array([128, 174, 128, 100], dtype=np.uint8)
    argb_ET = np.array([210, 184, 111, 100], dtype=np.uint8)
    # 顺序是 BGRA


class Constant:
    #maxNiiValue = 3500   # 设一个统一的最大值，避免图片之间无法比较.
    font = "Calibri"
    color = Color()
    tumor_code = {1:'ED', 2:'NCR', 3:'ET', 4:'TC', 5:'WT'}
    tumor_name = {'ED':1, 'NCR':2, 'ET':3, 'TC':4, 'WT':5}   # 如论如何，优先出现小的.

class MyIcons:
    def __init__(self) -> None:
        self.export_image = qta.icon("mdi6.image-move",color='white')  #导出图片
        self.brain = qta.icon("ph.brain-light",color='white')          #大脑图标
        self.close_eye = qta.icon("ph.eye-closed-fill",color='white')  #闭眼图标
        self.open_eye = qta.icon("ph.eye",color='white')               #睁眼图标
        self.ok_circle = qta.icon("ei.ok-circle",color='white')        #圈里一个勾
        self.circle = qta.icon("fa.circle-thin", color='white')        #圈里没有勾
        self.brain_modal = qta.icon("mdi.brain",color='white')         #选择大脑模态的图标.  
        self.location = qta.icon("msc.location",color="white")         #开启3D定位球的按钮图标.
        self.three_view = qta.icon("ri.search-eye-line",color="white") #图片窗口中选择三视图的那个.
        self.brush = qta.icon("mdi6.brush",color="white")              #笔刷图标.
        self.brush_off = qta.icon("mdi6.brush-off",color="white")      #关闭笔刷.
        self.eraser = qta.icon("mdi.eraser", color='white')            #橡皮擦
        self.drag = qta.icon("fa5s.hand-paper", color = 'white')       #拖拽模式图标.
        self.save = qta.icon("fa.save",color = "white")                #保存图标.
        self.ok = qta.icon("ei.ok", color='white')                     #ok,结束绘画.
        self.file = qta.icon("fa.file", color = "white")
        self.exit = qta.icon("mdi.exit-to-app", color="white")  # 退出界面
        self.fallback = qta.icon("ph.arrow-arc-left-bold", color="white") #回退.