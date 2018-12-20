这里分享一下这个项目的数据集来源：
你可以点击[这里](https://btsd.ethz.ch/shareddata/)下载数据集。在下载页面上面有很多的数据集，
但是你只需要下载 BelgiumTS for Classification (cropped images) 目录下面的两个文件：
 
BelgiumTSC_Training (171.3MBytes)
BelgiumTSC_Testing (76.5MBytes)
值得注意的是，原始数据集的图片格式是ppm，这是一种很老的图片保存格式，很多的工具都已经不支持它了。
这也就意味着，我们不能很方便的查看这些文件夹里面的图片。

转换脚本ppm2png.py



