README:
运行环境：python2.7 Anaconda2
依赖包：见requirements文件


- crawler: 爬虫程序文件夹
 
  
- code：最终最优的模型以及特征处理方法
  - constant.py: 存放全局变量，包括一些常量的定义。 如果运行发现文件找不到，请修改里面的全局变量data_root_path
  - feature_processer.py: 特征预处理和特征表示
  - feature_extraction.py: 特征抽取
  - comment_classcifier.py: 模型类，包括模型的训练与预测/模型的对比/模型可视化，如学习曲线，PR曲线，交叉验证参数选择等。
  - score.py: 评价类，f1,recall,precision,accuracy
  - util.py: 模型保存和加载、中间数据/特征等的保存和加载
  - visiualizer.py: 可视化。文本特征词云可视化/学习曲线等可视化
  - main.py: 测试类。调参、交叉验证、学习曲线、PR曲线、模型对比试验（不同的模型交叉验证性能与学习曲线绘制）



- data:
   - save文件夹：模型中间结果
   - stop.txt:停用词典
   - img文件夹: 可视化结果
   - 其余：请将训练数据和预测数据拷贝到data文件夹，并在main_model/constant.py常量定义中修改相应的文件名称
