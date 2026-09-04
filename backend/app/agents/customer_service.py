from __future__ import annotations

from time import perf_counter

from app.contracts.models import AgentRequest, AgentResult, Claim, Evidence, ModelKind
from app.models.llm_gateway import LLMGateway


_AFTER_SALES_POLICY = """售后处理通用边界：
1. 试用与退换货：是否提供单独试用、无理由退货期限及商品完好标准，以商品页面、订单和平台当前政策为准。质量问题、错发或与页面描述实质不符时可发起退货退款或换货申请并提交证据；个人原因退货的运费承担按订单政策确认，不能在核实前承诺。
2. 购买凭证：发票遗失通常不等于失去售后资格，可用平台订单、支付记录、电子发票或序列号等核验购买关系；线下或非本人订单可能需要补充其他凭证。
3. 物流加急：是否支持加急、额外费用和送达时间取决于库存、收货地区、订单状态及承运商，以下单页或承运商确认结果为准，不能保证具体时效。
4. 支付方式：未付款订单可取消后重新下单并选择支付方式；已付款订单通常不能直接变更支付渠道，退款或重新下单还需考虑库存、价格和优惠变化。
5. 收货地址：订单出库前可尽快联系平台尝试修改；已出库后通常只能申请承运商改址、拦截或拒收，能否成功及费用以承运商和订单状态为准。
6. 质量、描述不符与错发：保留商品页面、开箱照片或视频、型号标签、故障现象和沟通记录，可申请平台核验责任并选择适用的退货退款、换货或维修路径。赔偿不是自动成立，需依据核验结果、实际损失、平台政策和适用法律处理。
7. 重复质量问题：换货后再次出现同类问题，应关联原售后单升级复核，可提出再次换货、维修或退货退款诉求，最终方案以检测和政策为准。
8. 说明书、合格信息、日期或外观异常：先拍摄包装、铭牌、标签和异常部位，核对是否存在电子说明书或电子合格信息；缺失、涂改、污损或无法正常使用时可提交售后申请。是否属于违规商品及是否赔偿需经平台或监管核验，客服不得直接定性。
9. 争议处理：商家拒绝受理时，保存订单和沟通证据，通过平台申诉或当地消费者争议渠道复核，不作未经证据支持的结果承诺。"""


class CustomerServiceAgent:
    id = "customer-service"

    def __init__(self, llm_gateway: LLMGateway | None = None) -> None:
        self.llm_gateway = llm_gateway

    def run(self, request: AgentRequest) -> AgentResult:
        started = perf_counter()
        evidence = Evidence(
            evidence_id="policy:after-sales-v1",
            source_type="policy",
            title="售后处理原则 v1",
            text=_AFTER_SALES_POLICY,
            document_id="after-sales-policy",
            document_version="v1",
        )
        deterministic_answer = (
            "退货、换货或退款能否通过，需要结合订单当前状态、问题责任和平台政策处理。"
            "请先在订单售后入口提交申请并保留相关凭证；在责任和适用政策核实前，"
            "不能直接承诺退款、赔偿、运费承担或具体时效。"
        )
        llm_output = None
        if self.llm_gateway and self.llm_gateway.available(ModelKind.LLM):
            llm_output = self.llm_gateway.generate(
                kind=ModelKind.LLM,
                system_prompt=(
                    "你是客服业务智能体。只能依据给定政策回答。先直接回答用户每个问题，再说明适用条件、"
                    "可执行步骤和真正缺少的信息。不得擅自承诺退款、赔付、免费维修、运费承担或具体时效；"
                    "不得把‘可以申请’写成‘一定通过’，也不要机械索取与本问题无关的字段。回答清楚、克制。"
                ),
                user_prompt=f"用户问题：{request.question}\n\n政策证据：[{evidence.evidence_id}] {evidence.text}",
                temperature=0,
            )
        answer = llm_output.text if llm_output else deterministic_answer
        return AgentResult(
            task_id="service-1",
            agent_id=self.id,
            status="completed",
            answer_fragment=answer,
            claims=[Claim(text=answer, evidence_ids=[evidence.evidence_id], risk_level="high")],
            evidence=[evidence],
            confidence=0.82,
            missing_information=self._missing_information(request.question),
            recommended_next_action="collect-order-context",
            latency_ms=round((perf_counter() - started) * 1000),
            llm_generated=llm_output is not None,
            model_used=llm_output.model if llm_output else None,
        )

    @staticmethod
    def _missing_information(question: str) -> list[str]:
        normalized = question.lower()
        if any(term in normalized for term in ("付款", "支付方式")):
            return ["订单号", "订单状态（未付款/已付款/已发货）"]
        if any(term in normalized for term in ("收货地址", "改地址")):
            return ["订单号", "订单物流状态", "新的收货地址"]
        if any(term in normalized for term in ("物流", "配送", "送达", "加急")):
            return ["订单号", "收货地区", "当前物流状态"]
        if "发票" in normalized:
            return ["订单号或其他购买凭证", "购买渠道"]
        return ["订单号", "签收时间", "商品使用情况", "问题证据"]
