最近赋闲在家，也没啥事干，偶然间了解到老人跌倒将会造成的巨大伤害：一篇来自新华社的报道显示，每年中国有着高达4000万的老人至少摔倒过一次，其中10%存在骨折情况，15%存在头部受伤情况，其已经成为了目前国内老人因伤至死排名的第一名。联想到外婆之后将会一个人在家，于是遂萌发了给外婆设计一套实时跌倒检测系统的想法，同时还可以附加在家情况的监测。

核心的初步构架是米家云台摄像头+Tesla K20为核心的CUDA检测服务端+飞书所提供的机器人服务进行实时情况报送，以此构建完整的监测链。

米家云台摄像头没什么好说的，摔倒检测的核心是基于ST-GCN图神经网络的复现，主要基于"Human-Falling-Detact-Track"项目，GitHub链接[在此](https://github.com/GajuuzZ/Human-Falling-Detect-Tracks)。这个项目主要采用了AlphaPose获取骨架信息，使用基于时间序列的卷积对视频中的人进行姿态检测。嗯，就用它了。因为有批量通知的需求，所以很自然的会想到社交软件，但是Wechat和QQ等国内主流社交软件并没有提供这样的API让我能操控一个机器人去发信息，Telegram倒是有，但是需要一点上网技术，不对所有人友好，遂放弃。后了解到钉钉可以提供类似的解决方案，看了一眼，不仅有机器人API，还在PyPI上有人已经做好了包，于是采用钉钉。

### 第一部分：米家云台摄像头+在家检测

把米家云台摄像头配好后，在它的设置页面里有个「人物走动提醒」功能，但是我的需求是如果一段时间内没检测到人，就提醒我。所以直接用米家肯定是行不通的，还得套一层。那么这就是第一个方案：直接拿米家的通知作为Heartbeat，每三分钟推送一条通知到我的目标手机上，用adb命令实时检测手机上的推送通知，如果发现指定时间内没有新的米家推送就直接用钉钉发送一条消息到家庭群里。

很明显，第一个方案不是个好方案，这个系统的健壮性有可能直接取决于那条连接手机的USB线是否牢固，考虑我人可能很快就要回成都，如果连接电脑的那台手机炸了，那这个系统直接就GG。好了，那么还有什么办法呢？一开始还真没有，直到我开始配NAS。

米家的摄像头有一个功能可以说是这个系统的基石：上传视频文件到NAS，因为太(shi)过(zai)复(mei)杂(qian)，我把NAS和检测处理服务器合二为一了，这样在读取ST-GCN读取文件的时候就无需再过一道网络了。米家的摄像头文件上传NAS提供了两个选项：选项1:全部上传，选项2:仅上传变动画面。那这样方案二就呼之欲出了：直接用os包检测指定文件夹内有没有新文件，如果有新文件，则将其传入ST-GCN，返回跌倒检测的情况；如果没有新文件，则启动一个计数器，没有新文件达到一定时间阈值则在钉钉家庭群里发「外婆已消失xx分钟」。

方案二把信息来源完全置于了我的服务器下，在成都我也可以ssh进来远程，而且主板上的总线也总比USB线要牢固不少，那我用什么方案就不用我多说了吧。

在配米家云台摄像头的时候，因为需要在服务器上构建smb服务器，于是安装了samba，之前在ubuntu上使用是没问题的，但是到了opensuse这里还有GUI，结果摄像头提示我列不出smb目录。解决方案：米家这个摄像头只支持smb第一代协议，opensuse不知是出于安全性还是其他问题，把smb第一代协议给禁用了，遂修改配置文件打开SMB1，所以这个更好看的GUI有什么用呢？

### 第二部分：跌倒检测

因为家中的台式并没有能支持CUDA计算的显卡，又考虑到当前显卡物价飞涨，只能去小黄鱼上收了个7年前的老Kepler架构Tesla K20来用（因为显存大小、CUDA计算单元数量都还算可以），没想到这个K20直接引爆了三个问题，让我一度怀疑这个项目还能不能做下去（这里要特别特别特别感谢湘江一桥，在我家整了一天才把问题fix掉）。

问题一：Debian系发行版在网卡上打的驱动是r8169而不是官方推荐的r8168，和NVIDIA驱动不兼容，我重装了五次系统，包括Pop!_OS、Ubuntu、Debian的最新LTS与最新发行版，三次NVIDIA驱动导致进不去系统，两次系统进得去，网直接没了。

解决方案：由@湘江一桥亲自上门安(chuan)装(jiao)OpenSUSE，CUDA、CUDNN、NVIDIA驱动直接一条龙安装完，一看还有网，这个问题直接给拿下了。

问题二：K20的FP32浮点算力为3.5，PyTorch不愿意跑在这么拉垮的卡上，跑到激活函数LeakyReLU的时候直接报错CUDA不能计算。

解决方案：Google上发现有人通过改PyTorch源码实现了这一功能，遂修改code后重新编译PyTorch。问题解决。

问题三：K20卡因为设计为全被动散热，前任主人给它加了个风扇，导致整张卡太长，机箱装不下。

解决方案：暴力分离风扇与卡，把风扇反向接入到机箱外部，由外往里吹。虽然这明显会导致机箱内风道严重混乱，但本着也不是又不是不能用的原则，也算解决了😂



好了，所有的硬件问题都解决了，接下来修改一下code：因为这个源代码是需要传parameter进来的，所以主要的修改还是把它的核心部分打包成了一个模块，简化成一个仅需要传入需要检测的文件的函数，同时return回两个列表，第一个为以时间序列检测到的所有动作，包含七种动作：Standing, Walking, Sitting, Lying Down, Stand up, Sit down, Fall Down。第二个列表为Lying Down与Fall Down下的「置信度」。

核心部分代码如下：

```python
from act_dect import model
import os
import time
import torch
import dingding

def transtime(time):
    '''修改时间格式，添加0'''
    if time < 10:
        return '0'+str(time)
    else:
        return str(time)

def last_file():
    '''获取文件夹中的最新文件'''
    localtime = time.localtime(time.time())
    years = str(localtime.tm_year)
    month = transtime(localtime.tm_mon)
    day = transtime(localtime.tm_mday)
    hour = transtime(localtime.tm_hour)
    last_path = years+month+day+hour
    file_list = os.listdir("/home/disk0/xiaomi_camera_videos/607ea438b7ff/"+last_path)
    file_list.sort(key = lambda x:int(x[:2]))
    return "/home/disk0/xiaomi_camera_videos/607ea438b7ff/"+last_path+"/"+ file_list[-1]


times = 0
flag = True
fall_times = 0

var_last_file = ''
while flag:
    #使用try，避免因NAS文件没传完导致的循环中断
    try:
        a = last_file()
        if var_last_file == last_file():
            times += 1
            if times >= 24 and times % 10 == 0:
                text = "外婆未出现已经{}分钟".format(times * 100 / 60)
                dingding.sed_msg(text)
            time.sleep(60)
            continue
        times = 0
        var_last_file = last_file()
        print(var_last_file)
        start = time.time()
        time.sleep(10)
        fallen_list,pos_list = model(var_last_file)#传入ST-GCN模型
        torch.cuda.empty_cache()
        # 如果跌倒的最大置信度超过了30%，且检测到跌倒的比例超过了全部list的25%
        if max(pos_list) > 0.30 and (len(pos_list) / len(fallen_list)) > 0.25: 
            text = "外婆疑似跌倒，置信度{:.2f}{}".format(max(pos_list)*135,"%")
            dingding.sed_msg(text)

        localtime = time.localtime(time.time())

        print(localtime.tm_hour)
        # 仅在工作日白天进行循环
        if (localtime.tm_hour < 19 and localtime.tm_hour >= 8) and (localtime.tm_wday >= 0 and localtime.tm_wday <= 4): 
            flag = True
        else:
            flag = False
    except Exception as e:
        print(e)

```

我把代码的中止逻辑内建在了代码里，启动则主要依靠于corn服务工作日早上8点定时启动。

### 第三部分：异地登录服务器

因为很快就要回到成都，为了这套系统的可维护性，一定要有办法能够让它在公网上accessible。但是我是卑微的家庭宽带用户，因为IPv4地址的有限性，所有的网络地址都要经过NAT进行转换，这意味着我并没有一个公网IP。我初一的时候建Minecraft服务器也是遇到了类似的问题，最后只能寻求于购买服务器。但在六年前，IPv6还没有普及，但今天，我可是有了IPv6了（感谢@CharlesYang的提醒）。IPv6可是号称可以「给地球上的每一粒沙子一个IP地址」。遂从GitHub上下载一个DDNS脚本，通过DNS动态解析把我的v6解析到我的网址上。

### 第四部分：验收

验收结果如下：[![hASFzt.png](https://z3.ax1x.com/2021/08/24/hASFzt.png)](https://imgtu.com/i/hASFzt)

### 已知问题：

1. 跌倒检测过于灵敏，后续还需要根据置信度与检测结果进行进一步调试
2. 没有把它内建成一个服务，仍不算规范