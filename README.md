# Prototype of RoCC

本仓库开源RoCC原型的主要组件代码，和论文测量指标时使用的实验数据和测试代码

测试环境为虚拟机Linux Ubuntu 16.04，Conda Python=3.7, 3.8



## Provably Secure Steganography 

Discop文件夹中给出可证明安全隐写组件的代码，实现时使用Cython.Build加速代码的执行效率。

setup: 在Discop目录下build

```bash
pip install -r requirements.txt
python src/setup.py build_ext --build-lib=src/
```

测试单次隐写执行

```bash
python run_single_example.py
```

生成式隐写算法Discop提供了三种隐写功能：图片、音频和文本

```python
def test_stega_text(settings: Settings = text_default_settings):
    model = get_model(settings)
    tokenizer = get_tokenizer(settings)
	...

def test_staga_image(settings: Settings = image_default_settings):
    context_ratio = 0.5
    original_img = Image.open('small.png')
    ...
   
def test_stega_tts(settings: Settings = audio_default_settings):
    text = 'Taylor Alison Swift (born December 13, 1989) is an American singer-songwriter. '
    vocoder, tacotron, cmudict = get_tts_model(settings)
    ...
```

RoCC实验时调用文本隐写功能执行秘密消息的嵌入

```bash
python rocc_discop_text.py enc "This is a test example text" ./secretMsg
python rocc_discop_text.py dec "This is a test example text" ./coverText
```



## Similarity Analysis

SimilarityAnalysis 文件夹给出文本相似度分析的测试代码，实验时需要提供`data.json`文件，文件中的数据格式如下

```json
{
    Entry:{
        "TextID":number,
        "Text":string,
        "Pub-Time":time
    },
    ...
}
```

其中，TextID用于标记地面真相，方便测试指标，但实际情况中不需要该字段；Text为文本数据；Pub-Time为公开数据库中的文本发布时间戳。



## Steganalysis

隐写分析工具采用三种SOTA方案，均被包括在[]()仓库中

- [Linguistic Steganalysis via Densely Connected LSTM with Feature Pyramid (BiLISTM-Dense)](https://dl.acm.org/doi/abs/10.1145/3369412.3395067)
- [A Fast and Efficient Text Steganalysis Method (TS-FCN)](https://ieeexplore.ieee.org/document/8653856)
- [A Hybrid R-BILSTM-C Neural Network Based Text Steganalysis(R-BiLSTM-C)](https://ieeexplore.ieee.org/abstract/document/8903243)

模型训练微调和应用分析具体见`TextSteganalysis`下的`readme.md`文档。



## Robustness Experiment

实验时，按不同网络丢包率0%、1%、2%、5%和10%设置传输通道，模拟音频数据损失。

需要使用`tc`和`netem`工具实现模拟丢包

```bash
# set the loss rate as 5%
sudo tc qdisc add dev eth0 root handle 1: netem loss 5%

# show success
sudo tc qdisc show dev eth0

# after experiment should reset tc rule
sudo tc qdisc del dev eth0 root
```



分别传输100条语音数据，及识别出对应的有损文本，构建数据集，输入到SA模块中进行相似度匹配实验。实验时使用的指标为
$$
Recall@K = \frac{Relevant Documents Retrieved in Top K}{Total Relevant Documents}
$$
利用文本数据集中的TextID构建true pairs，使用SA模块预测结果作为predicted pairs。实验时前K个返回结果中存在true pairs时视为正确匹配。



## Channel Construction

信道A: 使用[TVoIP](https://github.com/T-vK/tvoip)建立VoIP信道

环境依赖安装

```bash
sudo apt-get install libasound2-dev alsa-base alsa-utils
```

第三方软件需要安装 [git](https://git-scm.com/downloads) 、[NodeJS](https://nodejs.org/en/download/) 和 [node-gyp](https://github.com/nodejs/node-gyp)

加载ALSA loopback模块来路由音频流数据 

```bash
sudo modprobe snd-aloop
```

查看loopback设备的卡号和设备号信息 

```bash
aplay -l
arecord -l
```

在VoIP应用中设置ALSA loopback设备作为microphone的输入。假设将音频流路由到的设备为`hw:Loopback,1,0`，使用以下命令将音频文件输入loopback设备

```bash
aplay -D hw:Loopback,1,0 /path/to/audio/file.wav
```



信道B: 使用一个虚拟机存储网络文本，模拟公开文本平台，提供开放端口允许来自其他主机对文本库的访问。

实验时，使用本地共享存储的虚拟硬盘模拟云端的公开数据库，让通信双方的主机可以同步访问相同文本数据，文本数据使用纯文本文件json组织，可以按发布时间检索。

实际应用中，可以使用SQLite数据库保存实时更新的文本数据库，以及Flask作为Web框架来创建一个RESTful API来处理网络上对文本数据的请求。