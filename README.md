# BigDL性能测试

---

### 1.Openvino YOLOX

YOLOX单独测试包, 项目参考：[YOLOX Openvino](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/OpenVINO)

使用方法：

```shell
conda create -n yolox python=3.6
conda activate yolox
cd openvino_yolox
#从https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/OpenVINO/python下载YOLOX-L模型放置 models
pip install -r requirements.txt
python yolox.py
```

### 2.Redis Openvino YOLOX

BigDL Cluster Serving redis测试包

使用方式：

```shell
# 1.配置config.yaml  其中模型使用openvino_yolox/models里面的
# 2.启动cluster serving
# 3.运行put_images_data.ipynb 压入数据
# 4.运行get_images_results.ipynb 获取预测结果
```

### 3.Kafka Openvino YOLOX

BigDL Cluster Serving Kafka测试包

使用方式，同redis版本相似，只是需要修改config.yaml。
