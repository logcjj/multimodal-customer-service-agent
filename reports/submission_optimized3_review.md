# submission_optimized(3).csv 审查报告

## 结论
- 结构上：400 行、400 个唯一 id，与 question_public.csv 完全对应；ret 字段可解析；`<PIC>` 数量与图片数组数量一致；图片文件均存在。
- 质量上：发现 182 条需要复核，其中严重问题 56 条，中等问题 126 条。
- 主要风险不是 CSV 格式，而是检索/重排质量：英文汇总手册题存在跨产品错召回、答案只返回章节标题、从段落中间截断、手册 OCR/LaTeX 噪声未清洗等问题。

## 统计
- 含 `<PIC>` 的答案：267 条。
- 图片引用过多（`<PIC>` >= 5）：40 条。
- 答案过长（>1800 字符）：6 条。
- 说明书题答案过短（<100 字符）：12 条。
- 只返回标题/章节名：5 条。
- 疑似严重错召回或答非所问：48 条。
- 包含未清洗手册/OCR/LaTeX/目录噪声：110 条。
- 英文问题返回大量中文：3 条。
- 相对已有 `submission_optimized.csv` 少了图片引用：39 条。

## 典型问题
### ID 70
- 问题："空调的重要组成部件有哪些？"
- 问题标签：说明书题答案过短，信息不足
- 答案开头：您好，可以这样处理： / 1. 部件介绍 / 室内机 / <PIC> / * 功能会根据机型有所调整。 / 2. 重要安全说明 / 3. 3 重要安全说明

### ID 112
- 问题："空气净化器需要长期存放时该怎么做？"
- 问题标签：说明书题答案过短，信息不足；相对已有 submission_optimized.csv 少了图片引用，需核验是否漏图
- 答案开头：您好，可以这样处理： / 1. 长期存放 1 晴天运行设备 1 小时，拔下电源插头，晾干设备内部。2 遮盖设备并存放在干燥、无阳光直射的地方。

### ID 124
- 问题："我的DCB101型号电钻指示灯闪烁时，这些闪烁标识代表什么含义？"
- 问题标签：相对已有 submission_optimized.csv 少了图片引用，需核验是否漏图
- 答案开头：您好，可以这样处理： / 1. 电池组留在充电器中 充电器和电池组可保持连接，指示灯显示已充满。 弱电池组：弱电池组仍可工作，但预期工作量会下降。 故障电池组：本充电器无法为故障电池组充电。充电器会通过不亮灯或显示电池组/充电器故障来提示。 注：这也可能表示充电器存在故障。 / 2. 电源故障（DCB101） 部分充电器带有电源故障指示灯。当充电器与发电机等便携式电源或直流转交流电源配合使用时，充电器可能暂时停止工作，红色指示灯快速闪烁两次后暂停。这表示电源超出允许范围。 / 3. 充电指示灯 本充电器可检测可能出现的某些问题。问题通过红色指示灯快速闪烁指示。出现此情况时，重新将电池组插入充电器。如果问题仍然存在，尝试更换另一块电池组，判断充电器是否正常工作。如果新电池组充电正常，则原电池组有故障，应送回服务中心或其他回收点回收。如果新电池组出现与原电池组相同的故障提示，请将充电器和电池组送至授权服务中心检测。

### ID 241
- 问题："If this is the first time to use airfryer, What should I do before first use?"
- 问题标签：包含未清洗的手册/OCR/LaTeX/目录噪声
- 答案开头：Here is what you can do: / 1. Before first use 1Remove all packing material 2Remove any stickers or labels (if available) from the appliance 3Thoroughly clean the appliance before first use, as indicated in the cleaning chapter. / 2. ndset. When the handset is placed correctly on the base station, you hear a docking sound. $\hookrightarrow$ The handset starts charging. <PIC> •• Charge the batteries for 8 hours before fir

### ID 265
- 问题："How to use the energy saving mode of a coffee machine?"
- 问题标签：疑似严重错召回或答非所问；答案从段落中间截取，语义不完整；相对已有 submission_optimized.csv 少了图片引用，需核验是否漏图
- 答案开头：Here is what you can do: / 1. s,nicks,minordents,and cosmetic damages on external surfaces and exposed parts Products on which the serial numbers have been altered,defaced,orremoved;service visits to teach you how to use the Product,or visits where there is nothing wrong with the Product;correction of installation problems(you are solely responsible for any structure and setting for the Product,including all electrical

### ID 278
- 问题："How should fire extinguishers be stored on board and where should they be placed?"
- 问题标签：疑似严重错召回或答非所问；答案从段落中间截取，语义不完整；相对已有 submission_optimized.csv 少了图片引用，需核验是否漏图
- 答案开头：Here is what you can do: / 1. nds Always disconnect vacuum from the vacuum before cleaning or maintaining it. Ensure voltage rating for enclosed vacuum matches standard outlet voltage Used battery packs should be placed in a sealed plastic bag and disposed of safely according to locale nv ironmen fal regulations. Before every use,check the battery pack for any sign of damage or leakage.Do not charge damaged or leaking 

### ID 287
- 问题："How to set an off-center subject mode of a camera?"
- 问题标签：疑似严重错召回或答非所问；英文问题返回了大量中文内容
- 答案开头：Here is what you can do: / 1. 装入电池 <PIC> / 滑动电池盖将其打开。 <PIC> / 按照电池上的箭头指示装入电池。 确保电池上的黄线与相机上的标记对齐。• 取出电池时，将电池锁扣推向一侧，然后将电池滑出相机。 <PIC> / 关闭电池盖。 电池充电 注意安装方向。 / <PIC> 使用随附的USB线将相机与智能手机专用交流电源适配器连接，然后将交流电源适配器插入室内电源插座。 / • 使用符合以下额定输出的交流电源适配器：直流5.0伏/1000毫安 • 充电过程中可以拍摄或打印图像。 • 充电时间约为三到四小时。

### ID 296
- 问题："After the earphones is in my hand, what are the components I should have?"
- 问题标签：疑似严重错召回或答非所问；相对已有 submission_optimized.csv 少了图片引用，需核验是否漏图
- 答案开头：Here is what you can do: / 1. No docking tone. • • The handset is not placed properly on the base station/charger. • • The charging contacts are dirty. Disconnect the power supply first and clean the contacts with a damp cloth.

### ID 308
- 问题："I can't find the way to play video using this eReader, can you tell me how to do that?"
- 问题标签：疑似严重错召回或答非所问；相对已有 submission_optimized.csv 少了图片引用，需核验是否漏图
- 答案开头：Here is what you can do: / 1. 2.How to use ·Turn the set on and press the TV/viDEO button on the remote control or TV/VIDEO button on thefront panel to select CoMPoNENT. ·Try this after turning on the DVD set. / 2. Videocable. .110 / Video OUT. .110 / Viewfinder.. .13 / 3. DOWNLOAD THE APP AND CONNECT TO WI-FI ·Waf chan overview video with instructions on how fo setup and use your vacuum. Set an automatic cleaning schedule(upf

### ID 352
- 问题："How can you connect the base station of a landline?"
- 问题标签：说明书题答案过短，信息不足；只返回标题/章节名，没有具体步骤或说明；相对已有 submission_optimized.csv 少了图片引用，需核验是否漏图
- 答案开头：Here is what you can do: / 1. Connect the base station

### ID 365
- 问题："How can you replace the mower belt of a lawn mower?"
- 问题标签：说明书题答案过短，信息不足；只返回标题/章节名，没有具体步骤或说明；相对已有 submission_optimized.csv 少了图片引用，需核验是否漏图
- 答案开头：Here is what you can do: / 1. ESS66300 DRIVEV-BELT REPLACEMENT(ET410TR/VK540E)

### ID 383
- 问题："How do you know about system memory of a motherboard?"
- 问题标签：说明书题答案过短，信息不足；疑似严重错召回或答非所问；相对已有 submission_optimized.csv 少了图片引用，需核验是否漏图
- 答案开头：Here is what you can do: / 1. 1.4 System memory

### ID 401
- 问题："What is the robot anatomy of the vacuum?"
- 问题标签：说明书题答案过短，信息不足；只返回标题/章节名，没有具体步骤或说明；疑似严重错召回或答非所问
- 答案开头：Here is what you can do: / 1. VACUUM <PIC>

### ID 402
- 问题："What are the two primary modes available on a vacuum cleaner that can be used to tailor its performance to meet different home cleaning needs?"
- 问题标签：疑似严重错召回或答非所问；包含未清洗的手册/OCR/LaTeX/目录噪声；图片引用过多，可能图文不聚焦
- 答案开头：Here is what you can do: / 1. Key Lock TheTV canbeset so that the remote control is needed to control it This feature can be used to prevent unauthorized viewing. 1. Press the MENU button and then $\pmb{\triangle}/\pmb{\nabla}$ button the until the menu is displayed as shown right 2.Press the $\blacktriangleright$ and then $\bigstar/\bigstar$ button to select Keylock 3. Press the button and then $\pmb{\triangle}/\pmb{\

### ID 416
- 问题："What are the essential preparation checks to perform before using a snowmobile?"
- 问题标签：疑似严重错召回或答非所问
- 答案开头：Here is what you can do: / 1. Post-launch checks Perform the post-launch checks in the preoperation checklist while the boat is in the water and the engines are running.

### ID 426
- 问题："What are the steps to clean a snowmobile?"
- 问题标签：疑似严重错召回或答非所问；包含未清洗的手册/OCR/LaTeX/目录噪声；相对已有 submission_optimized.csv 少了图片引用，需核验是否漏图
- 答案开头：Here is what you can do: / 1. Stop the cleaning. ·Turn the $<\widehat{\mathbb{Q}^{\flat}}\gg$ switchto ${\tt{<}}0{\tt{F F}}{\tt{>}}$ ，The camera will turn off, the shutter will close, and the mirror will go back down. ）Set the $<\widehat{\mathbb{Q}^{\flat}}>$ switch to ${\tt{<}}0{\sf{N}}{\tt{>}}$ The camera will then be ready to shoot. · During the sensor cleaning, never do any of the following that would turn off the 

### ID 429
- 问题："What are the steps to enable and customize captions and on-screen text settings on a television?"
- 问题标签：疑似严重错召回或答非所问；相对已有 submission_optimized.csv 少了图片引用，需核验是否漏图
- 答案开头：Here is what you can do: / 1. 10 Phone settings You can customize the settings to make it your own phone.

### ID 431
- 问题："What are the steps to properly connect a DVD player to a television or audio system for optimal performance?"
- 问题标签：疑似严重错召回或答非所问
- 答案开头：Here is what you can do: / 1. 10. System panel connector (20-5 pin PANEL) This connector supports several chassis-mounted functions. <PIC> / System panel connector

## 重点复核 ID
- 严重错召回/答非所问高风险：244、245、246、248、249、256、263、265、268、270、274、278、280、283、287、290、295、296、297、298、307、308、309、311、312、313、314、321、361、364、369、373、375、383、401、402、408、412、415、416、418、419、420、426、428、429、431、432
- 低信息/标题型答案：70、112、175、179、183、352、365、383、401、418、419、424
- 相对上一版疑似漏图：112、124、191、202、256、261、265、268、278、291、292、293、296、297、299、308、315、348、350、352、355、361、362、365、366、373、376、377、383、412、417、418、419、421、424、425、426、429、432

## 明细文件
- `reports/submission_optimized3_audit.csv`