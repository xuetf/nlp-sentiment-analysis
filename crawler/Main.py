# -*- coding: utf-8 -*-
from CustomThread import *
from Config import *


if __name__ == '__main__':
    #股评数据获取
    thread1 = CustomThread('http://guba.eastmoney.com/list,szzs,f_%d.html', 2, 3555)
    thread1.start()