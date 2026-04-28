# submission_example.csv 审查报告

## 总体结论
- question_public.csv 共 400 题；submission_example.csv 共 400 条答案，id 能对应。
- 示例答案只有 6 种重复占位话术，没有任何一条给出手册步骤、参数、政策条件、图片引用或可执行处理方案。
- 按题号与内容粗分：通用客服政策/售后题 50 条，产品手册/说明书题 350 条。
- 英文题 187 条，示例答案均为中文泛化话术，存在语言不匹配。
- 参考资料型答案中有 288 条包含 <PIC>，示例答案 0 条包含 <PIC> 或图片数组。
- 下方“资料型回答方向”取自项目已有 `submissions/submission_optimized.csv`，用于说明示例答案缺失的信息类型；正式答案仍建议逐题回到手册原文与插图核验。

## 重复话术分布
- 79 条：正在为您核实，由于咨询人数较多，请您稍后再来咨询。
- 74 条：好的，没问题的，我们会尽快给您答复的。
- 70 条：感谢您的咨询，相关问题正在后台核实中，请随时关注后续动态。
- 66 条：您好，您的问题已收到，请您耐心等待处理结果，谢谢。
- 57 条：收到相关反馈，我们正在加急处理，请稍候。
- 54 条：很抱歉给您带来不便，我们已记录您的反馈，祝您生活愉快。

## 典型问题示例
### ID 1
- 问题："请问你们家的商品支持7天无理由退换货吗？", / "需要自己承担运费吗？"
- 示例答案：您好，您的问题已收到，请您耐心等待处理结果，谢谢。
- 主要问题：答案是占位式话术，未直接回答用户问题；缺少政策条件、费用承担、处理流程或时效说明；未逐项回应多个子问题
- 资料型回答方向："您好，符合条件的商品支持7天无理由退换货。商品需保持不影响二次销售，配件、赠品、包装和发票等尽量完整。非质量问题退换货时，寄回运费通常由买家承担；若是质量问题、错发、漏发或运输破损，经核实后由商家承担相应运费。"

### ID 64
- 问题："使用吹风机时，人员需要佩戴哪些防护装备？"
- 示例答案：正在为您核实，由于咨询人数较多，请您稍后再来咨询。
- 主要问题：答案是占位式话术，未直接回答用户问题；没有基于对应手册给出步骤、参数、注意事项或故障排查依据；参考手册答案涉及插图，示例缺少<PIC>占位符和图片数组
- 资料型回答方向："您好，相关说明如下：\n1. 个人防护装备 使用吹风机的人员需佩戴以下防护装备： 1. 合格的听力防护装备 2. 合格的眼部防护装备。 3. 在多尘环境中操作时佩戴面罩。 4. 防滑鞋底的工作靴或工作鞋 5. 急救箱。 <PIC>", ["Manual04_3"]

### ID 132
- 问题："我刚发现健身追踪器电量低，该如何正确为其充电？"
- 示例答案：正在为您核实，由于咨询人数较多，请您稍后再来咨询。
- 主要问题：答案是占位式话术，未直接回答用户问题；没有基于对应手册给出步骤、参数、注意事项或故障排查依据；参考手册答案涉及插图，示例缺少<PIC>占位符和图片数组
- 资料型回答方向："您好，相关说明如下：\n1. 为手表充电 充满电的健身追踪器续航时间可达6天以上。电池续航和充电周期随使用情况及其他因素而异；实际结果可能有所不同。\n为健身追踪器充电： 1. 将充电线插入电脑的USB端口、UL认证的USB墙充充电器或其他低功耗充电设备。\n2. 将充电线的另一端靠近手表背面的充电接口，直至磁吸吸附。确保充电线上的金属触点与手表背面的接口对齐。\n<PIC> 为健身追踪器充电12分钟即可续航24小时。手表充电时，轻点屏幕两次或按下按钮即可点亮屏幕。完全充满电大约需要1-2小时。\n<PIC>", ["Manual16_1", "Manual16_2"]

### ID 241
- 问题："If this is the first time to use airfryer, What should I do before first use?"
- 示例答案：很抱歉给您带来不便，我们已记录您的反馈，祝您生活愉快。
- 主要问题：答案是占位式话术，未直接回答用户问题；没有基于对应手册给出步骤、参数、注意事项或故障排查依据；用户使用英文提问，但示例用中文泛化回复，语言不匹配
- 资料型回答方向："The relevant instructions are:\n1. Before first use 1Remove all packing material 2Remove any stickers or labels (if available) from the appliance 3Thoroughly clean the appliance before first use, as indicated in the cleaning chapter."

### ID 289
- 问题："How do I select the AF Mode when before taking a picture?"
- 示例答案：您好，您的问题已收到，请您耐心等待处理结果，谢谢。
- 主要问题：答案是占位式话术，未直接回答用户问题；没有基于对应手册给出步骤、参数、注意事项或故障排查依据；用户使用英文提问，但示例用中文泛化回复，语言不匹配；参考手册答案涉及插图，示例缺少<PIC>占位符和图片数组
- 资料型回答方向："The relevant instructions are:\n1. Auto Picture Control You can select picture modes programmed at factory as you prefer Using the APC button 1. Press theAPC button. 2. Press the APC button to select Magic eye(some models), Clear Optimum, Soft or User. 3.Press theENTER()button to exit <PIC>\n2. Setting the AF, Metering, and Drive Modes 63 Selecting the AF Mode 64\nSelecting the AF Point.. 67\nFocusing an Off-Center Subject. .69\nWhen Autofocus F...

## 逐题明细
详见 `reports/submission_example_audit.csv`。
