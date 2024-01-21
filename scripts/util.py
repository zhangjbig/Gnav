#############
# author : JhLi
#############

import numpy as np
from scripts.Constant import Constant
from PyQt5.QtGui import QImage

def add_mask(img, seg, mask):
    '''
    给图片加不同区域的颜色.  
    优先级5<4<1=2=3.  
    img: 全脑二维RGB图片 (h,w,3)  
    seg: 推理的肿瘤的数组(h,w,3)  
    mask: 五个元素的数组，对应位置为1表示要显示.注意：空出下标为0的位置！
    或者使用字典，总之让mask[5]能确实访问到WT的mask
    5个编号: 1-ED, 2-NCR, 3-ET, 4-TC, 5-WT
    不会改变img的值.  
    '''
    ret = img.copy()
    # 编号5: WT:  
    if mask[5]:
        ret = add_mask2(ret, seg>0, Constant.color.WT)
    # 编号4：TC:  
    if mask[4]:
        ret = add_mask2(ret, seg>=2, Constant.color.TC)
    # 编号3：ET:
    if mask[3]:
        ret = add_mask2(ret, seg>2, Constant.color.ET)
    # 编号2: NCR:
    if mask[2]:
        ret = add_mask2(ret, seg==2, Constant.color.NCR)
    # 编号1：ED  
    if mask[1]:
        ret = add_mask2(ret, seg==1, Constant.color.ED)
    return ret
        

def add_mask2(img, seg_mask, color):
    '''
    img: (h,w,3)  
    seg_mask: (h,w,3)，要改变颜色的位置为1.  
    color: np.array([r,g,b]), 要改变成的颜色.  
    **Warning**: 会直接改变img的值!  
    '''
    assert img.shape == seg_mask.shape
    seg_color_mask_3d = seg_mask * color                 # h,w,3的 带颜色的mask，无颜色的是0  
    img = img * (~seg_mask) + seg_color_mask_3d                # 把要改变颜色的地方清零，加上有颜色的mask
    return img


def standardGrayValue(img_fdata):
    # 把原本特别大的.nii文件中的值映射到[0,255]  
    # 最大值归一化，最大值是经验值.
    return img_fdata / np.max(img_fdata) * 255


def distance_2(a:np.ndarray, b:np.ndarray):
    # 返回距离平方.  
    diff = a-b
    return np.dot(diff, diff)

def QImage2Array_GrayScale8(qimg:QImage,bytes_per_line, const=False):
    '''
    bytes_per_line: 原本每行有多少字节.   
    将QImage转化为同形状的numpy ndarray并返回。  
    只考虑QImage.Format.Format_GrayScale8 8bits灰度类型，其它会直接报错！  
    转化执行深拷贝，但这个拷贝出来的内存就是QImage所使用的的内存，修改它就修改了QImage.  
    如果const is True, 则使用constbits以稍微加速，返回const的内存指针  
    只要经历过操作，最后都会每行四字节对齐.每行最后会多出一点东西，所以会截取原本的shape  
    '''
    assert qimg.format()==QImage.Format.Format_Grayscale8
    if const:
        ptr = qimg.constBits()
    else:
        ptr = qimg.bits()
    ptr.setsize(qimg.byteCount())
    arr = np.frombuffer(ptr,np.uint8,-1).reshape(qimg.size().height(), qimg.bytesPerLine())
    arr = arr[:, :bytes_per_line]
    return arr

def preprocessMRI4Image(data:np.ndarray):
    return data   #x 和 y都反向.

def postprocessMRI4Image(data:np.ndarray):
    return data

def rotate(data: np.ndarray, channal, reverse = False):
    '''
    data: 要做旋转的**二维**数据  
    channal: 0-x, 1-y, 2-z.  
    reverse: 是否反向处理。为true表示把转过的数据转回原样.  
    '''
    if not reverse:
        if channal == 0:
            return np.rot90(data, 1)
        elif channal == 1:
            return np.rot90(data, 1)
        else:
            return np.transpose(data)
    else:
        if channal == 0:
            return np.rot90(data, -1)
        elif channal == 1:
            return np.rot90(data, -1)
        else:
            return np.transpose(data)
    