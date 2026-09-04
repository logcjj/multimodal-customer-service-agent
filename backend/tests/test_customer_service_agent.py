from __future__ import annotations

from app.agents.customer_service import CustomerServiceAgent
from app.contracts.models import AgentRequest, ModelKind
from app.models.llm_gateway import LLMOutput


def test_payment_method_answer_uses_relevant_policy_and_requests_order_state_only() -> None:
    class PolicyGateway:
        def available(self, kind: ModelKind) -> bool:
            return kind == ModelKind.LLM

        def generate(self, **kwargs) -> LLMOutput:
            prompt = kwargs["user_prompt"]
            assert "已付款订单通常不能直接变更支付渠道" in prompt
            assert "未付款订单" in prompt
            assert kwargs["temperature"] == 0
            return LLMOutput(
                text="未付款时可取消后重新下单；已付款后通常不能直接修改支付方式。",
                provider="fake",
                model="fake-llm",
                latency_ms=5,
            )

    result = CustomerServiceAgent(PolicyGateway()).run(
        AgentRequest(question="我想修改订单的付款方式，能修改吗？")
    )

    assert result.missing_information == ["订单号", "订单状态（未付款/已付款/已发货）"]
    assert "签收时间" not in result.missing_information
    assert result.llm_generated is True
