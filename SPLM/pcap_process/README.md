# 用于提取特征，生成特征文件，时间戳和出入包长

# pcap_processor.C

用C读取pcap包，提取流特征，包括时间戳和出入包长。返回多个流特征

在windows64位版本运行如下，生成.dll文件

```bash
x86_64-w64-mingw32-gcc -shared -o pcap_processor.dll pcap_processor.c -I "C:\\Users\\mason\\Downloads\\npcap-sdk-1.15\\Include" -L "C:\\Users\\mason\\Downloads\\npcap-sdk-1.15\\Lib\\x64" -lwpcap -lPacket -lws2_32
```

其中需要指定npcap所在的库，需要安装npcap，在Include中包含pcap.h,以及下载npcap的SDK，在/Lib/x64中包含相应的库。

# NormalExtractor.py

从normal流量中提取特征，由于normal流量数据包很大， 因此用C语言很快。

使用方法：

```bash
python NormalExtractor.py <input_dir> <output_dir>
```

# HSExtractor.py

从隐藏服务pcap包中提取特征。

使用方法：

```bash
python .\HSExtractor.py --input <input_dir> --output <output_dir>
```

1. 删除每个文件夹中较大和较小的数据包
2. 只提取超过100个数据包的TCP流
3. 每个pcap包只提取最长的流
4. 删除特征文件少于100的文件夹

