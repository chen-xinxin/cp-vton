# 复现 & 思考

源文件 Readme-raw.md
	
目的：复现实现，感受效果；理解算法过程，分析算法瓶颈；借鉴思路，尝试改进方法；


## 1. 环境 & 配置

**环境**

```
$ pip list | grep torch
torch               1.1.0
torchvision         0.3.0

$ pip install tensorboardX
```

**修改代码 cp_dataset.py**

源代码中对单通道图做transform变换，需要修改。

```
Traceback (most recent call last):
  File "train.py", line 191, in <module>
    main()
  File "train.py", line 176, in main
    train_gmm(opt, train_loader, model, board)
  File "train.py", line 58, in train_gmm
    inputs = train_loader.next_batch()
  File "/vton/cinastanbean-cp-vton/cp_dataset.py", line 166, in next_batch
...
    tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
RuntimeError: output with shape [1, 256, 192] doesn't match the broadcast shape [3, 256, 192]

```

```python
self.transform = transforms.Compose([  \
        transforms.ToTensor(),   \
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
self.transform_1d = transforms.Compose([ \
        transforms.ToTensor(), \
        transforms.Normalize((0.5,), (0.5,))])
```

**--workers 4 --> --workers 0**

```
ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
...
RuntimeError: DataLoader worker (pid 2841) is killed by signal: Bus error.
```


## 2. Train


**STEPs**

```
step1-train-gmm.sh
step2-test-gmm.sh
step3-generate-tom-data.sh
$ mv result/gmm_final.pth/train/* data/train/
step4-train-tom.sh
step5-test-tom.sh
```

显卡内存使用2647MiB(Memory-Usage), 限于生成图像尺寸256x192；训练时长，普通单卡机器1-2天可以完成。

按原作者默认参数训练模型，训练模型存放于[百度网盘](https://pan.baidu.com/s/1h6h9MYswltN4mcp5dfYycg)（链接: https://pan.baidu.com/s/1gJqjGvXQgdoGkCF_YpNAUQ 提取码: b3fk），供下载测试。tensorboard文件大约19G，如果需要有限时间内可联系索取。

```
嗨，按原作者默认参数，我复现了一把实验。
训练模型存放于百度网盘（链接: https://pan.baidu.com/s/1h6h9MYswltN4mcp5dfYycg 提取码: uwgg），供下载测试。
部分分析和拙见写在 Github: https://github.com/cinastanbean/cp-vton ，欢迎同行拍砖讨论。
```


```
$ tree checkpoints/
checkpoints/
├── gmm_train_new
│   ├── gmm_final.pth
│   ├── step_005000.pth
│   ├── ...
│   └── step_200000.pth
└── tom_train_new
    ├── step_005000.pth
    ├── ...
    ├── step_200000.pth
    └── tom_final.pth
```

**TensorBoard**

```
tensorboard/
├── gmm_train_new
│   └── events.out.tfevents.1568110598.tplustf-imagealgo-50529-ever-chief-0
├── gmm_traintest_new
│   └── events.out.tfevents.1568185067.tplustf-imagealgo-50529-ever-chief-0
├── tom_test_new
│   └── events.out.tfevents.1568473618.tplustf-imagealgo-50529-ever-chief-0
└── tom_train_new
    └── events.out.tfevents.1568188644.tplustf-imagealgo-50529-ever-chief-0
    
$ tensorboard --logdir tensorboard/gmm_train_new/
$ tensorboard --logdir tensorboard/gmm_traintest_new/
$ tensorboard --logdir tensorboard/tom_train_new/
$ tensorboard --logdir tensorboard/tom_test_new/
```

web: 

	http://everdemacbook-pro.local:6006/#scalars
	http://everdemacbook-pro.local:6006/#images

scalars / images :

	gmm_train_new

![](pics/gmm-train-sc.png)
![](pics/gmm-train-image.png)

	gmm_traintest_new

![](pics/gmm-traintest-images.png)

	tom_train__new

![](pics/tom-train-sc-1.png)  ![](pics/tom-train-sc-2.png)
![](pics/tom-train-images.png)

	tom_test_new

![](pics/tom-test.png)








## 3. Test

执行前**STEPs**中所列步骤，后执行```python smart_show_test_result.py```， 可以在```result_simple```文件夹下查看生成结果，示例图片如下，从左到右每列图片意思是：

[cloth, cloth-mask, model-image, model-image-parse, cloth-warp, cloth-warp-mask, try-on-result]

![](pics/src_012578_dst_014252.png)
![](pics/src_012849_dst_015439.png)
![](pics/src_012934_dst_010551.png)
![](pics/src_013355_dst_018626.png)
![](pics/src_013583_dst_006296.png)
![](pics/src_013725_dst_005920.png)
![](pics/src_017823_dst_007923.png)
![](pics/src_018876_dst_000192.png)
![](pics/src_019531_dst_015077.png)


## 4. Virtual Try-On 技术路线的瓶颈

虚拟模特图像生成，技术上大致有三条路实现。

“Virtual Try-On”（VTON）是其中一种方式。


**VTON技术有如下考虑：**

1. 规避模特生成问题，模特生成本身比较难以做到，难以做到对模特面孔头发、身材真实性等方面的保真度，VTON技术路线规避该问题；
2. 默认模特已经穿着了和待合成服饰尺寸形状大体一致的服饰，通过对服饰做Warping进而“贴图”，实现Try-On的效果。


**技术产品化VTON思路还有些问题：**

 1. 对指定模特，给他换上另外一套衣服，需要妥善处理版权问题；
 2. 服装和人的搭配问题，如何保持视觉协调；
 3. 服装穿着在人身上产生的自然形变，因为对服饰做Warping没有根本解决对服饰的理解问题；（如下图条纹状服饰）
 4. 模特摆拍姿势多样，肢体和服装之间的遮挡问题;（如下图手臂遮挡服饰）
 5. 当前数据和实验，数据限于上衣短袖类目，图像尺寸256x192, 还属于Toy级别实验；

**条纹状服饰**

![](pics/src_012377_dst_017227_p1.png)
![](pics/src_019001_dst_010473_p1.png)
![](pics/src_013309_dst_002031_p1.png)

**手臂遮挡服饰**
 
![](pics/src_012830_dst_008479_p2.png)
![](pics/src_012975_dst_007423_p2.png)
 
 
## 5. 算法演进方向

**Virtual Try-on**

致敬诸位的创意，这条路还有很多技术点要解决。


```
.
├── 2017-VITON-MalongTech
│   ├── 1705.09368.Pose Guided Person Image Generation.pdf
│   ├── 1711.08447.VITON- An Image-based Virtual Try-on Network.pdf
│   ├── 1902.01096.Compatible and Diverse Fashion Image Inpainting.pdf
│   └── 2002-TPAMI-Shape matching and object recognition using shape contexts.pdf
├── 2018-CP-VTON-SenseTime
│   └── 1807.07688.Toward Characteristic-Preserving Image-based Virtual Try-On Network.pdf
├── 2019-Multi-pose Guided Virtual Try-on Network (MG-VTON)
│   └── 1902.11026.Towards Multi-pose Guided Virtual Try-on Network.pdf
├── 2019-WUTON
│   └── 1906.01347.End-to-End Learning of Geometric Deformations of Feature Maps for Virtual Try-On.pdf 
```

