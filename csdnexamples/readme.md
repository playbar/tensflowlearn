vgg16_weights.npz 文件分割与合并

1、分割文件
    zip - ./vgg16_weights.npz | split -b 45000k
 
2、 合并文件
    cat x* >> file.zip