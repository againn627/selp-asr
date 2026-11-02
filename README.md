# selp-asr

**给桌面应用加语音输入的本地服务。Windows 一条命令，CPU 够快，中文准，用不到时主动让出内存。**

```
双击 start-server.bat  →  自动装好、自动下模型、自动启动
```

不用 Docker，不用 GPU，不用 conda，不装 torch。**也不要求你电脑上已经装了 Python** —— 项目自带一份 Python 运行时，装在自己的目录里，跟你系统里的 Python、conda 互不干扰。第一次约 5 分钟（下 60 MB 运行时 + 941 MB 模型），之后双击 3 秒起来。

|  |  |
| --- | --- |
| 识别模型 | Qwen3-ASR-0.6B（sherpa-onnx int8） |
| 中文表现 | 明显好于 whisper-large-v3，支持粤语、中英混读 |
| 自带运行时 | Python 3.12（解开 60 MB，首次自动下载） |
| 依赖体积 | 130 MB（无 torch、无 CUDA） |
| 模型体积 | 941 MB（首次自动下载） |
| 内存 | 加载后约 1.1 GB，空闲 5 分钟自动回落到 87 MB |
| 速度 | 2 秒语音约 1.1 秒出字（i9-13900H，纯 CPU） |
| 接口 | OpenAI Audio API 兼容，接进来只改 `base_url` |

---

## 一、30 秒跑起来

**Windows**：下载本仓库，双击 `start-server.bat`。

窗口会依次显示：

```
[1/3] 检查运行环境
  项目自带 Python 3.12.16，装到 E:\selp-asr\runtime\python
  下载 Python（21.0 MB）
  安装依赖（约 130 MB，只装进项目自己的 runtime）
[2/3] 检查模型文件
  需要下载模型，共约 941.3 MB
    出口 IP ...，地区 CN → 国内镜像优先
  [1/6] conv_frontend.onnx（42.1 MB）
      [####------------------] 12.3 MB/42.1 MB  29%  1.8 MB/s
[3/3] 启动服务
==============================================
  语音识别服务已启动
==============================================
  地址    http://127.0.0.1:8765
  模型    qwen3-asr-0.6b
  内存    用时约 1.1 GB，空闲 5 分钟后自动释放
  说话测试  另开一个窗口运行 test-server.bat
  停止服务  Ctrl+C，或直接关掉这个窗口
==============================================
```

**Linux / macOS**：同样的事情用一条命令（系统自带的 python3 只当启动器，装完会切到项目自己的运行时）。

```bash
python3 bootstrap.py        # 装运行时 + 装依赖 + 下模型 + 起服务
```

> **不需要先装 Python**。`start-server.bat` 发现 `runtime\python` 不在时，会用 Windows 自带的 PowerShell 下载官方 python-build-standalone（和 uv 用的是同一家），解压到 `runtime\python`，之后所有依赖都装进这份运行时。项目不读 PATH、不读注册表、不找 conda、不建 venv；下载失败就明确报错并给出手动放包的位置，**不会退回去用你系统里的 Python**。
>
> 运行时装好后，`runtime/` 整个目录可以随项目一起拷到其他机器（同一操作系统与架构）直接用，不需要重新下。
>
> 想换模型位置：`set ASR_DATA_DIR=D:\asr-data`（运行时始终跟着项目走）。

---

## 二、验证它能用

另开一个窗口，双击 `test-server.bat`：

```
  服务正常（模型已加载）
  麦克风：麦克风阵列 (适用于数字麦克风的英特尔® 智音技术)

  回车开始说话，说完再按回车出字；输入 q 退出。
  按回车开始（q 退出）：
  录音 [##########----------] 0.52  （说完按回车停止）
  识别中 ...
  音频 2.94s | 出字 1.50s（服务端 1.40s）
  >> 今天天气不错，我们下午三点开会
```

录音时会显示音量条，没声音时是空的——能一眼看出麦克风有没有在工作。

---

## 三、接入自己的程序

接口是 OpenAI Audio API，**已经用过 OpenAI 的代码只改 `base_url`**。

### curl

```bash
curl -X POST http://127.0.0.1:8765/v1/audio/transcriptions \
  -F "file=@你的录音.wav"
# => {"text":"今天天气不错","language":"zh","duration":2.94,"elapsed":1.40}
```

### Python（OpenAI 官方 SDK）

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="not-needed")
print(client.audio.transcriptions.create(
    model="qwen3-asr-0.6b",
    file=open("你的录音.wav", "rb"),
).text)
```

### 其他语言

服务自带交互式接口文档，浏览器打开 <http://127.0.0.1:8765/docs> 就能直接试。

### 支持的接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/v1/audio/transcriptions` | 转录。multipart 字段 `file` 必填，可选 `model`、`response_format`（json / text / verbose_json） |
| GET | `/v1/models` | 模型列表（OpenAI 同构） |
| GET | `/health` | 是否已加载、空闲多久、线程数 |
| POST | `/v1/audio/warmup` | *可选* 提前加载模型（与录音并行，省掉第一句的等待） |
| POST | `/v1/audio/unload` | *可选* 卸掉模型释放内存，进程继续服务 |
| POST | `/v1/audio/shutdown` | *可选* 优雅退出，给停止脚本用 |

`warmup` / `unload` / `shutdown` 是加速与运维用的**可选**端点，不调也完全能用。代价是：不 warmup，第一次请求要多等约 3.5 秒（加载模型）；不 unload，空闲 5 分钟后服务自己会卸。

支持的音频格式：wav / flac / ogg / mp3。m4a、amr 等需要先转成 wav。

---

## 四、省内存

模型加载后约 1.1 GB。桌面应用不需要一直占着这块内存：

- **自动**：空闲 300 秒后服务自己卸载，回落到 87 MB，下次说话时再加载（约 3 秒，通常与用户开口的时间重叠）。
- **手动**：`POST /v1/audio/unload`。
- **调整**：`config.toml` 里 `idle_unload_seconds`，设 0 表示常驻不卸。

调用方想更快出字，可以在用户按下麦克风的那一刻调 `POST /v1/audio/warmup`，与录音并行加载，说完正好就绪。

**实测数据**（同一段 2 秒语音，i9-13900H）：

| 场景 | 端到端 |
| --- | --- |
| 服务已加载 | 约 1.1 s |
| 空闲卸载后第一句（含 3.5 s 加载） | 约 4.6 s |

---

## 五、配置

改 `config.toml`，重启生效。仓库里没有这个文件，需要时从 `config.example.toml` 复制一份改名：

```bash
copy config.example.toml config.toml    # Linux/macOS: cp config.example.toml config.toml
```

不建也能跑——所有项都有默认值，下面列的就是默认值。

| 项 | 默认 | 说明 |
| --- | --- | --- |
| `host` | `127.0.0.1` | 监听地址。改 `0.0.0.0` 允许局域网访问，**这时必须设 token**，否则拒绝启动 |
| `port` | `8765` | 端口 |
| `token` | 空 | 请求鉴权 `Authorization: Bearer <token>`。留空 = 不校验，只适合本机自用 |
| `num_threads` | `6` | CPU 线程数。实测 4/6/8/12 在短句上差别不大，6 最快，长段只比 8 慢 2% |
| `max_total_len` | `2048` | KV cache 长度。**别改小**，512 时超过 30 秒的音频会输出乱码 |
| `idle_unload_seconds` | `300` | 空闲多少秒卸载模型，0 = 不卸 |
| `hotwords` | 空 | 热词，逗号分隔。常说的专有名词写进来能明显提升识别率 |
| `preload` | `false` | 启动时是否同步加载模型。`false` 时 0.5 秒即可接收请求 |

也可以用环境变量覆盖，适合 Docker / systemd：

```bash
ASR_HOST=0.0.0.0 ASR_PORT=9000 ASR_TOKEN=mysecret python3 -m asr_server
```

---

## 六、性能实测

环境：i9-13900H / 31.6 GB / Intel Iris Xe 核显（**无独显**），6 线程，纯 CPU。

| 音频长度 | 出字耗时 | RTF |
| --- | --- | --- |
| 2 s | 1.09 s | 0.55 |
| 3 s | 1.56 s | 0.52 |
| 5 s | 2.35 s | 0.47 |
| 11 s | 5.38 s | 0.47 |
| 首次加载模型 | 3.6 s | — |
| 之后 warmup | 0.06 s | — |

**线程数对比**（11.38 秒中文音频，取最快一次）：

| 线程 | 2 s | 3 s | 5 s | 11 s |
| --- | --- | --- | --- | --- |
| 4 | 1.17 | 1.56 | 2.46 | 5.71 |
| **6** | **1.09** | **1.56** | **2.35** | 5.39 |
| 8 | 1.34 | 1.72 | 2.49 | **5.38** |
| 12 | 1.65 | 2.19 | 2.70 | 5.45 |

短句是主要场景，6 线程最优；长段 8 线程只快 2%，不值得。换机器可以跑 `python -m scripts.bench` 重测。

**识别效果**（官方测试音频，带标准答案）：

| 音频 | 结果 |
| --- | --- |
| `cantonese.wav`（粤语） | 正确，含 `my princess` 中英混读 |
| `fast1.wav`（极快语速） | 主语正确，个别词序有偏差 |
| `noise2.wav`（带噪声电话录音） | 主体正确，数字串部分有误 |
| `raokouling.wav`（绕口令） | 疑难词有错，常规语句正确 |
| `qiqiu1.wav`（50 秒唱歌） | 文本连贯，歌词意境类词汇有偏差 |

日常说话、会议、口述效果良好；绕口令、专业术语、歌曲建议配 `hotwords`。

---

## 七、常见问题

**端口被占用**

```
[启动失败] 端口 8765 已被占用。
  1) 可能服务已经在跑了：浏览器打开 http://127.0.0.1:8765/health
  2) 想换端口：改 config.toml 的 port，或设环境变量 ASR_PORT=9000
```

**下载模型慢或失败**
程序会实测两个源（hf-mirror 国内镜像 / huggingface 官方）的速度，自动选快的；下载中途变慢会自动换源并续传。全部失败时会打印手动下载地址和放到哪个目录。

**Python 运行时下载失败（`Could not download the Python runtime`）**
项目要靠这份运行时才能跑，不会改用你系统里的 Python。按提示手动下载后放进 `runtime\_cache\`，再双击 `start-server.bat`（会自动校验 sha256 并解压）：

```
https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone/20261026/
https://github.com/astral-sh/python-build-standalone/releases/download/20261026/
```

具体文件名看 `runtime.lock.json` 里对应你平台的那一条。这两个地址和校验值都写在那个文件里，升级版本也只改那一个地方。

**想确认它真的没碰我的 Python**

```
runtime\python\python.exe -c "import sys;print(sys.prefix)"
# => E:\selp-asr\runtime\python       （不是你的 conda / Python312）
```

`sys.prefix` 和 `sys.base_prefix` 相同，说明它不借外部解释器——你卸载、升级自己的 Python 都不影响它。

**几乎没收到声音**
检查系统默认输入设备，`test-server.bat --devices` 列出所有录音设备。Windows 设置 → 系统 → 声音 → 输入里确认麦克风没被静音。

**识别结果不准**
在 `config.toml` 里加 `hotwords`，把常说的专有名词、人名写进去，逗号分隔。

**想开机自启**
把 `start-server.bat` 的快捷方式放进 `shell:startup` 目录（Win+R 输入这个就能打开）。

---

## 八、开发

```bash
.\runtime\python\python.exe -m pip install -e ".[dev]"
.\runtime\python\python.exe -m pytest tests -v     # 20 项，含真实识别回归
.\runtime\python\python.exe -m ruff check .
```

只有一个解释器：`runtime\python\python.exe`。没有 venv 要激活，也没有「用哪个 python」的选择题。跑之前先执行一次 `start-server.bat`（或 `runtime\python\python.exe bootstrap.py --setup`）把运行时和依赖装好。

测试分两组：不依赖模型的（配置、multipart、音频解码容错、运行时锁文件一致性）和依赖模型的（真实音频断言关键短语，防止参数改错导致识别退化）。没有模型时自动跳过后者。

目录：

```
asr_server/
  __main__.py    启动入口（端口检查、优雅退出）
  server.py      FastAPI 路由
  engine.py      模型封装（懒加载、空闲卸载、串行解码）
  runtime.py     项目自带 Python 运行时（下载、sha256 校验、解压）
  download.py    模型下载（选源、测速、续传、大小校验）
  client.py      HTTP 客户端（cli 与麦克风测试共用）
  mic.py         麦克风说话测试
  config.py      配置加载（文件 + 环境变量）
  paths.py       数据目录定位（便携模式 / pip 安装）
  audio.py       音频解码
bootstrap.py     一键启动（装运行时、装依赖、下模型、起服务）
runtime.lock.json 运行时版本与校验值的唯一定义处
scripts/
  install-runtime.ps1  无 Python 电脑上的引导（Windows 自带 PowerShell）
  bench.py             线程数与耗时对比
tests/           回归测试
```

---

## 九、为什么不直接用别的

同类开源项目（2026-11 实测 star）：

| 项目 | 形态 | Windows 一条命令 | 不装 Docker | 免装 Python | CPU 可用 | 中文 |
| --- | --- | --- | --- | --- | --- | --- |
| speaches (3.8k) | 服务端 | ✗ | ✗ | ✗ | ✓ | 一般 |
| QwenLM/Qwen3-ASR (3.8k) | 模型仓库 | ✗ | — | ✗ | ✗ 需 GPU | ✓ |
| antirez/qwen-asr (635) | C 实现 | ✗ 需编译 | ✓ | ✓ | ✓ | ✗ 偏英文 |
| achetronic/parakeet (338) | 服务端 | ✗ | ✗ | ✗ | ✓ | ✗ |
| nyosegawa/speech.cpp (288) | 单文件 | ✗ 需编译 | ✓ | ✓ | ✓ | ✗ |
| gpustack/vox-box (216) | 服务端 | ✗ | ✗ | ✗ | ✗ 需 GPU | ✓ |

本项目补的是这个空档：**Windows 上双击就装好、不装任何环境（连 Python 都不用先装）、CPU 就能跑、中文准、内存可主动归还**。运行时 60 MB + 依赖 130 MB，无 torch；模型 941 MB int8。这套组合目前没有第二家。

它不是标准制定者，对外只用 OpenAI 现成接口，不引入任何自定义协议。

---

## 许可

[Apache-2.0](LICENSE)

模型来自 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)（Apache-2.0）提供的 Qwen3-ASR-0.6B int8 权重，遵循其原始许可。
