# 实验报告Convlstm

written by 柴百里



### 相关公式及原理



#### LSTM:



如图所示

<img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/image-20210507202743689.png" alt="image-20210507202743689" style="zoom: 67%;" />



其计算对应的公式为：
$$
I_t = \sigma( X_t W_{xi} + H_{t-1}W_{hi} + b_i  )
$$

$$
F_t = \sigma( X_t W_{xf} + H_{t-1}W_{hf} + b_f  )
$$

$$
O_t = \sigma( X_t W_{xo} + H_{t-1}W_{ho} + b_o  )
$$

$$
C_t = F_t\bigodot C_{t-1} + I_t \bigodot tanh( X_t W_{xc} + H_{t-1}W_{hc} + b_c  )
$$
$$
H_t = O_t\bigodot tanh(C_t)
$$


$$
（注：\bigodot 表示逐个元素相乘）
$$


#### ConvLstm：



<img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/5.png" alt="5" style="zoom:67%;" />



计算公式与LSTM类似，不同的是将其中的矩阵乘法换成卷积操作：
$$
I_t = \sigma( X_t* W_{xi} + H_{t-1}*W_{hi} + b_i  )
$$

$$
F_t = \sigma( X_t *W_{xf} + H_{t-1}*W_{hf} + b_f  )
$$

$$
O_t = \sigma( X_t* W_{xo} + H_{t-1}*W_{ho} + b_o  )
$$

$$
C_t = F_t\bigodot C_{t-1} + I_t \bigodot tanh( X_t *W_{xc} + H_{t-1}*W_{hc} + b_c  )
$$

$$
H_t = O_t\bigodot tanh(C_t)
$$

$$
（注：\bigodot 表示逐个元素相乘 ；*表示卷积）
$$

​				



此外，ConvLSTM 还有一种实现方法，基于Peephole的作法更改，所得到的公式与上面类似
$$
I_t = \sigma( X_t* W_{xi} + H_{t-1}*W_{hi} + W_{ci}\bigodot C_{t-1}+ b_i  )
$$

$$
F_t = \sigma( X_t *W_{xf} + H_{t-1}*W_{hf} + W_{cf}\bigodot C_{t-1} + b_f  )
$$

$$
O_t = \sigma( X_t* W_{xo} + H_{t-1}*W_{ho} + W_{co}\bigodot C_{t}+ b_o  )
$$

$$
C_t = F_t\bigodot C_{t-1} + I_t \bigodot tanh( X_t *W_{xc} + H_{t-1}*W_{hc} + b_c  )
$$

$$
H_t = O_t\bigodot tanh(C_t)
$$

$$
（注：\bigodot 表示逐个元素相乘 ；*表示卷积）
$$

​		

### 实验模型：



<img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/6.png" alt="6" style="zoom:67%;" />



1. 如图所示，采用多层循环神经网络的框架，其中每个Cell 都是采用Convlstm结构实现。
2. 输入为 (batch , timestep , channel , width , height ).
3. 以本次实验为例, 输入即为(4 , 20 , 1 , 140 , 140). 采用输入的0-9帧图片来预测10-19帧图片



### 实验流程：

1.  预处理数据阶段：直接将(1 , 140 , 140)图片拿去训练会因图片过大导致显存溢出。故训练前先将图片转换为(4*4 , 140/4 , 140/4)的样式 。

   

2.  Schedule sampling:  我们的目的是用0-9帧的图片去预测10-19帧图片。但是实际刚开始训练时直接用0-9帧图片去预测10-19帧图片效果会不好，导致模型训练效果不好。故实际训练时会以一定概率融入真实图片信息与预测的图片信息作为输入。

   而在测试时将相应的mask_true置为0，使得只利用预测的信息即可：
   
   ```python
   for t in range(20-1):
   	if t < 10:		#0~9帧用真实的图片信息
       	net = frames[:, t] 
       else:			#10~18帧以一定概率融入真实图片信息和预测的信息
           net = mask_true[:, t -10] * \
           frames[:, t] + (1 - mask_true[:, t - 10]) * x_gen #x_gen表示上以帧预测的图片
   ```



3. 采用递减的学习率







### 实验结果：

​	

#### 训练过程的均方误差：

<img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/loss.png" alt="train_loss" style="zoom: 80%;" />

​	

在第30个epoch之前，为了训练效果更好基本上schedule sampling采样中都用到了0-18帧的真实信息。到了30个epoch后，shedule sampling采样都用的是0-9帧的真实信息去推测后面的信息。所以训练时的均方误差在30个epoch这里会有一个显著的回升，后面又会继续下降



#### 验证过程的均方误差：

<img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/valid_loss.png" alt="valid_loss" style="zoom:80%;" />

此图横轴为 iteration of epoch*2



#### 验证过程的SSIM:

​	<img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/ssim.png" alt="ssim" style="zoom:80%;" />



#### 验证过程的PSNR:

<img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/psnr.png" alt="psnr" style="zoom:80%;" />



#### 验证过程的FMAE:

<img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/fmae.png" alt="fmae" style="zoom:80%;" />



#### 验证过程的SHARPNESS:

<img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/sharpness.png" alt="sharpness" style="zoom:80%;" />



#### 图片显示： 



此为1~10帧的输入图片：

<img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/gt1.png" alt="gt3" style="zoom:80%;" /><img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/gt2.png" alt="gt1" style="zoom: 80%;" /><img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/gt3.png" alt="gt2" style="zoom:80%;" /><img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/gt4.png" alt="gt1" style="zoom: 80%;" /><img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/gt5.png" alt="gt1" style="zoom: 80%;" /><img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/gt6.png" alt="gt1" style="zoom: 80%;" /><img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/gt7.png" alt="gt1" style="zoom: 80%;" /><img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/gt8.png" alt="gt1" style="zoom: 80%;" /><img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/gt9.png" alt="gt1" style="zoom: 80%;" /><img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/gt10.png" alt="gt1" style="zoom: 80%;" />





此为11~20帧的真实图片：

<img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/gt11.png" alt="gt3" style="zoom:80%;" /><img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/gt12.png" alt="gt1" style="zoom: 80%;" /><img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/gt13.png" alt="gt2" style="zoom:80%;" /><img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/gt14.png" alt="gt1" style="zoom: 80%;" /><img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/gt15.png" alt="gt1" style="zoom: 80%;" /><img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/gt16.png" alt="gt1" style="zoom: 80%;" /><img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/gt17.png" alt="gt1" style="zoom: 80%;" /><img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/gt18.png" alt="gt1" style="zoom: 80%;" /><img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/gt19.png" alt="gt1" style="zoom: 80%;" /><img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/gt20.png" alt="gt1" style="zoom: 80%;" />





此为11~20帧的预测图片：

<img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/pd11.png" alt="gt3" style="zoom:80%;" /><img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/pd12.png" alt="gt1" style="zoom: 80%;" /><img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/pd13.png" alt="gt2" style="zoom:80%;" /><img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/pd14.png" alt="gt1" style="zoom: 80%;" /><img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/pd15.png" alt="gt1" style="zoom: 80%;" /><img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/pd16.png" alt="gt1" style="zoom: 80%;" /><img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/pd17.png" alt="gt1" style="zoom: 80%;" /><img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/pd18.png" alt="gt1" style="zoom: 80%;" /><img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/pd19.png" alt="gt1" style="zoom: 80%;" /><img src="https://github.com/Monaco12138/my_convlstm/blob/master/photo/pd20.png" alt="gt1" style="zoom: 80%;" />



