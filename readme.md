

the reproduction of cp-vton
=======

**cp-vton的复现**

### 通用环境搭建

****

```
conda create -n gen python=3.9
```

```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

```
pip install tensorboardX
```

```
conda install tensorboard
```

```
conda install tqdm
```

```
pip install nvitop	#用来监控显卡
```

```
pip install opencv-python
注:千万不要用conda install opencv,会把pytorch改成cpu版本
```



## Debug日志

报错`RuntimeError: CuDNN error: CUDNN_STATUS_SUCCESS`

**解决方法**：代码中添加`torch.backends.cudnn.benchmark = True`

原来的环境是`pytorch=0.4.1+cuda9.0`，报错`RuntimeError: cublas runtime error :……thcblas.cu:411`

**解决方法**：把cuda升到9.2

`conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=9.2 -c pytorch`

**后续还是不行**：把cuda升到10.2

`conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch`

### 最终解决方案：环境的cuda版本要和机子一致

最终用pytorch1.13+cuda11.6（和机子cuda版本一致）成功运行！



报错`RuntimeError: Caught RuntimeError in DataLoader worker process 0.`

**解决方法**：线程设置而引发的错误，把num_workers设置成0，仅执行主进程



报错`RuntimeError: output with shape [1, 256, 192] doesn't match the broadcast shape [3, 256, 192]`

**解决方法**：pytorch版本的问题，把

```python
self.transform = transforms.Compose([  \ transforms.ToTensor(),   \ transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
```

改成

```python
self.transform = transforms.Compose([  \ transforms.ToTensor(),   \ transforms.Normalize((0.5,), (0.5,))])
```



报错`RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED`

**解决方法**：CUDNN没安装或版本不匹配，重新安装即可



报错`RuntimeError: view size is not compatible with input tensor‘s size and stride`

**解决方法**：这是因为view()需要Tensor中的元素地址是连续的，但可能出现Tensor不连续的情况，所以先用 .contiguous() 将其在内存中变成连续分布将`x = x.view(x.size(0), -1)`改为`x = x.contiguous().view(x.size(0), -1)`



## 运行

依次执行

```
step1-train-gmm.sh
step2-test-gmm.sh
step3-generate-tom-data.sh
mv result/gmm_final.pth/train/* data/train/
mv result/gmm_final.pth/test/* data/test/
step4-train-tom.sh
step5-test-tom.sh
```

然后执行```python smart_show_test_result.py```， 可以在```result_simple```文件夹下查看生成结果，示例图片如下，从左到右每列图片意思是：

[cloth, cloth-mask, model-image, model-image-parse, cloth-warp, cloth-warp-mask, try-on-result]





### CP-VTON跑VTON-HD数据集

图片尺寸要从1024x768变成256x192：用interpolate方法，见`prepocess/resize/resized.py`

分析图要重新跑：用PGN跑，将图片放在`datasets/images`中，然后`python inf_pgn.py`

openpose：尝试用CP-VTON跑VITON-HD数据集，发现pose的pose_keypoints数量不一致，因为openpose版本不一致，VITON-HD用的是1.3，CP-VTON用的是1.0

在`cp_dataset.py`里用转换代码处理

```
# 25 keypoint to 18 keypoint
changed_pose_data = np.zeros(shape=(18,3))
changed_pose_data[0] = pose_data[0] # Nose
changed_pose_data[1] = pose_data[1] # Neck
changed_pose_data[2] = pose_data[2] # RShoulder
changed_pose_data[3] = pose_data[3] # RElbow
changed_pose_data[4] = pose_data[4] # RWrist
changed_pose_data[5] = pose_data[5] # LShoulder
changed_pose_data[6] = pose_data[6] # LElbow
changed_pose_data[7] = pose_data[7] # LWrist
changed_pose_data[8] = pose_data[9] # RHip
changed_pose_data[9] = pose_data[10] # RKnee
changed_pose_data[10] = pose_data[11] # RAnkle
changed_pose_data[11] = pose_data[12] # LHip
changed_pose_data[12] = pose_data[13] # LKnee
changed_pose_data[13] = pose_data[14] # LAnkle
changed_pose_data[14] = pose_data[15] # REye
changed_pose_data[15] = pose_data[16] # LEye
changed_pose_data[16] = pose_data[17] # REar
changed_pose_data[17] = pose_data[18] # LEar
```
