

这个程序文件名为test2.py，主要实现了两个函数：add_alpha_channel和merge_img。

add_alpha_channel函数用于为jpg图像添加alpha通道。它首先将jpg图像的通道分离成蓝色通道、绿色通道和红色通道，然后创建一个与蓝色通道形状相同的alpha通道，像素值全部设置为255。最后将蓝色通道、绿色通道、红色通道和alpha通道合并成一个新的图像，并返回。

merge_img函数用于将png透明图像与jpg图像叠加。它首先判断jpg图像是否已经为4通道，如果不是则调用add_alpha_channel函数为其添加alpha通道。然后根据叠加位置的坐标值，设定一系列叠加位置的限制，以防止png图像超出jpg图像范围导致程序报错。接着获取要覆盖图像的alpha值，并将其除以255，使值保持在0-1之间。最后进行叠加操作，将jpg图像和png图像按照alpha值进行加权融合。最后返回叠加后的jpg图像。

在主程序中，首先定义了图像路径，然后使用cv2.imread函数读取图像。接着设置叠加位置的坐标值，调用merge_img函数进行叠加操作。最后使用cv2.imshow函数显示结果图像，并使用cv2.waitKey函数等待按键操作。

#### 5.3 ui.py

