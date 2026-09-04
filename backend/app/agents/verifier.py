from __future__ import annotations

import re

from app.contracts.models import (
    AgentRequest,
    AgentResult,
    Evidence,
    VerificationIssue,
    VerificationReport,
)
from app.runtime.dynamic_routing import general_request_requires_domain


class VerifierAgent:
    id = "verifier"

    _PROHIBITION_TERMS = ("严禁", "禁止", "不得", "切勿", "不要", "不可", "不能")
    _POSITIVE_CONTINUATION = re.compile(r"(?:仍然?|依然?)?(?:可以|可|能够|能)继续|(?:可以|可)正常使用")
    _COMMITMENT_TERMS = (
        "免费维修",
        "免费更换",
        "全额退款",
        "无条件退货",
        "无理由退货",
        "赔付",
        "赔偿",
        "上门维修",
    )
    _CERTAINTY_TERMS = ("确认", "确定", "就是", "证明", "已经", "表明")
    _OBSERVATION_QUALIFIERS = ("图片中", "图片显示", "画面中", "可见", "疑似", "可能", "看起来", "识别到")
    _GENERAL_BOUNDARY_TERMS = (
        "拆机",
        "拆开维修",
        "自行维修",
        "自己维修",
        "全额退款",
        "无条件退货",
        "保证退款",
        "保证赔偿",
        "免费维修",
        "官方说明书指出",
        "根据官方手册",
    )

    def verify_general(
        self,
        request: AgentRequest,
        result: AgentResult,
    ) -> VerificationReport:
        answer = result.answer_fragment.strip()
        issues: list[VerificationIssue] = []
        if general_request_requires_domain(request.question):
            issues.append(
                VerificationIssue(
                    code="general-request-out-of-domain",
                    message="原始请求涉及产品事实、维修或安全操作，不属于通用回答范围",
                    severity="error",
                )
            )
        elif result.status != "completed" or not answer:
            issues.append(
                VerificationIssue(
                    code="general-model-unavailable",
                    message="通用模型没有返回可用答案",
                    severity="error",
                )
            )
        elif any(term in answer for term in self._GENERAL_BOUNDARY_TERMS):
            issues.append(
                VerificationIssue(
                    code="general-route-safety-boundary",
                    message="通用回答越过产品维修、安全或官方承诺边界",
                    severity="error",
                )
            )
        passed = not issues
        return VerificationReport(
            passed=passed,
            action="accept" if passed else "handoff",
            issues=issues,
            confidence=result.confidence if passed else 0,
        )

    def verify(
        self,
        results: list[AgentResult],
        expected_domain_agents: set[str],
        *,
        enhanced: bool = True,
    ) -> VerificationReport:
        completed_agents = {item.agent_id for item in results if item.status == "completed"}
        evidence_ids = {evidence.evidence_id for item in results for evidence in item.evidence}
        evidence_records = {
            evidence.evidence_id: evidence for item in results for evidence in item.evidence
        }
        evidence_by_id = {
            evidence_id: self._support_text(evidence).lower()
            for evidence_id, evidence in evidence_records.items()
        }
        claims = [claim for item in results for claim in item.claims if item.agent_id != "multimodal"]
        issues: list[VerificationIssue] = []

        missing_agents = expected_domain_agents - completed_agents
        if missing_agents:
            issues.append(
                VerificationIssue(
                    code="missing-domain-result",
                    message=f"缺少专业智能体结果：{', '.join(sorted(missing_agents))}",
                    severity="error",
                )
            )
        unsupported = [claim.text for claim in claims if set(claim.evidence_ids) - evidence_ids]
        if unsupported:
            issues.append(
                VerificationIssue(
                    code="unsupported-claim",
                    message="存在未绑定有效证据的结论",
                    severity="error",
                )
            )
        if not claims:
            issues.append(
                VerificationIssue(code="no-claims", message="没有可验证的回答结论", severity="error")
            )
        if not enhanced:
            return self._report(results, expected_domain_agents, claims, issues)
        unsupported_terms: list[str] = []
        for claim in claims:
            support = "".join("\n".join(evidence_by_id.get(item, "") for item in claim.evidence_ids).split())
            codes = re.findall(r"\b(?=[a-z0-9-]*[a-z])(?=[a-z0-9-]*\d)[a-z0-9-]+\b", claim.text.lower())
            measured = re.findall(
                r"\b\d+(?:\.\d+)?\s*(?:%|v|w|a|hz|mm|cm|分钟|小时|天|次|年|月|日|℃)",
                claim.text.lower(),
            )
            code_terms = {"".join(item.split()) for item in codes}
            measured_terms = {"".join(item.split()) for item in measured}
            unsupported_terms.extend(sorted(term for term in code_terms if term not in support))
            unsupported_terms.extend(
                sorted(term for term in measured_terms if not self._measurement_supported(term, support))
            )
        if unsupported_terms:
            issues.append(
                VerificationIssue(
                    code="unsupported-number-or-model",
                    message=f"回答中的数字或型号缺少证据支持：{', '.join(dict.fromkeys(unsupported_terms))}",
                    severity="error",
                    evidence_ids=list(dict.fromkeys(
                        evidence_id for claim in claims for evidence_id in claim.evidence_ids
                    )),
                )
            )
        for claim in claims:
            bound_evidence = [
                evidence_records[item]
                for item in claim.evidence_ids
                if item in evidence_records
            ]
            ordered_sources = [
                evidence.evidence_id
                for evidence in bound_evidence
                if self._steps_out_of_order(claim.text, self._support_text(evidence))
            ]
            if ordered_sources:
                issues.append(
                    VerificationIssue(
                        code="safety-step-order-mismatch",
                        message="回答改变了证据中的安全操作顺序",
                        severity="error",
                        evidence_ids=ordered_sources,
                    )
                )

            prohibition_sources = [
                evidence.evidence_id
                for evidence in bound_evidence
                if self._contradicts_prohibition(claim.text, self._support_text(evidence))
            ]
            if prohibition_sources:
                issues.append(
                    VerificationIssue(
                        code="contradicts-prohibition",
                        message="回答与说明书中的禁止项或安全警告冲突",
                        severity="error",
                        evidence_ids=prohibition_sources,
                    )
                )

            unsupported_commitments = self._unsupported_commitments(
                claim.text,
                "\n".join(self._support_text(evidence) for evidence in bound_evidence),
            )
            if unsupported_commitments:
                issues.append(
                    VerificationIssue(
                        code="unsupported-service-commitment",
                        message=f"售后承诺缺少政策证据支持：{', '.join(unsupported_commitments)}",
                        severity="error",
                        evidence_ids=[item.evidence_id for item in bound_evidence],
                    )
                )

            if self._promotes_visual_observation(claim.text, bound_evidence):
                issues.append(
                    VerificationIssue(
                        code="visual-observation-promoted",
                        message="图片观察被表述为已确认事实，应保留‘图片显示’或‘疑似’等限定",
                        severity="error",
                        evidence_ids=[item.evidence_id for item in bound_evidence],
                    )
                )
        return self._report(results, expected_domain_agents, claims, issues)

    @staticmethod
    def _report(
        results: list[AgentResult],
        expected_domain_agents: set[str],
        claims,
        issues: list[VerificationIssue],
    ) -> VerificationReport:
        passed = not any(item.severity == "error" for item in issues)
        evidence_ids = {evidence.evidence_id for item in results for evidence in item.evidence}
        confidence = VerifierAgent._confidence(
            results=results,
            expected_domain_agents=expected_domain_agents,
            issues=issues,
            passed=passed,
        )
        action = "accept" if passed else ("revise" if claims and evidence_ids else "fallback")
        return VerificationReport(
            passed=passed,
            action=action,
            verified_claims=claims if passed else [],
            issues=issues,
            confidence=confidence,
        )

    @staticmethod
    def _confidence(
        *,
        results: list[AgentResult],
        expected_domain_agents: set[str],
        issues: list[VerificationIssue],
        passed: bool,
    ) -> float:
        """Calibrate verification confidence from result, evidence, and coverage."""

        if not passed:
            return 0.0
        agent_confidences = [
            item.confidence
            for item in results
            if item.agent_id in expected_domain_agents and item.status == "completed"
        ]
        if not agent_confidences:
            return 0.0

        evidence = [item for result in results for item in result.evidence]
        evidence_confidences = [
            item.evidence_confidence
            if item.evidence_confidence is not None
            else min(0.85, max(0.35, float(item.score or 0.0)))
            for item in evidence
        ]
        evidence_support = (
            sum(evidence_confidences) / len(evidence_confidences)
            if evidence_confidences
            else 0.0
        )
        coverage = min(1.0, len({item.evidence_id for item in evidence}) / 5)
        warnings = sum(item.severity == "warning" for item in issues)
        confidence = (
            min(agent_confidences) * 0.46
            + evidence_support * 0.3
            + coverage * 0.16
            + 0.08
            - warnings * 0.04
        )
        return round(min(0.98, max(0.0, confidence)), 4)

    @staticmethod
    def _support_text(evidence: Evidence) -> str:
        return f"{evidence.document_name or ''}\n{evidence.title}\n{evidence.text}"

    @staticmethod
    def _numbered_steps(text: str) -> list[str]:
        markers = list(re.finditer(r"(?:^|[\n。；;])\s*(?:步骤\s*)?(\d+)[.、:：]\s*", text))
        if len(markers) < 2:
            return []
        steps: list[str] = []
        for index, marker in enumerate(markers):
            end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
            value = text[marker.end() : end].strip(" \n。；;，,")
            if value:
                steps.append(value)
        return steps

    @classmethod
    def _steps_out_of_order(cls, claim: str, support: str) -> bool:
        steps = cls._numbered_steps(support)
        if len(steps) < 2:
            return False
        positions = [claim.find(step) for step in steps]
        matched = [position for position in positions if position >= 0]
        return len(matched) >= 2 and matched != sorted(matched)

    @classmethod
    def _contradicts_prohibition(cls, claim: str, support: str) -> bool:
        if not any(term in support for term in cls._PROHIBITION_TERMS):
            return False
        positive = cls._POSITIVE_CONTINUATION.search(claim)
        if positive is None:
            return False
        prefix = claim[max(0, positive.start() - 3) : positive.start()]
        return not any(term in prefix for term in ("不", "不能", "不可", "禁止", "严禁"))

    @classmethod
    def _unsupported_commitments(cls, claim: str, support: str) -> list[str]:
        normalized_support = "".join(support.split())
        commitments = [term for term in cls._COMMITMENT_TERMS if term in claim]
        deadlines = re.findall(r"\d+(?:\.\d+)?(?:分钟|小时|天|工作日)内", claim)
        candidates = list(dict.fromkeys([*commitments, *deadlines]))
        return [item for item in candidates if "".join(item.split()) not in normalized_support]

    @classmethod
    def _promotes_visual_observation(cls, claim: str, evidence: list[Evidence]) -> bool:
        if not evidence or any(item.source_type not in {"vision", "ocr"} for item in evidence):
            return False
        if any(term in claim for term in cls._OBSERVATION_QUALIFIERS):
            return False
        return any(term in claim for term in cls._CERTAINTY_TERMS)

    @staticmethod
    def _measurement_supported(term: str, support: str) -> bool:
        support = support.replace("\\~", "~")
        if term in support:
            return True
        match = re.fullmatch(r"(\d+(?:\.\d+)?)(%|v|w|a|hz|mm|cm|分钟|小时|天|次|年|月|日|℃)", term)
        if not match:
            return False
        value = float(match.group(1))
        unit = match.group(2)
        date_parts = re.findall(
            r"(?<!\d)(?:(\d{4})[-/年])?(\d{1,2})[-/月](\d{1,2})(?:日)?",
            support,
        )
        for year, month, day in date_parts:
            if unit == "年" and year and float(year) == value:
                return True
            if unit == "月" and float(month) == value:
                return True
            if unit == "日" and float(day) == value:
                return True
        aliases = {
            "分钟": r"(?:分钟|minutes?|mins?)",
            "小时": r"(?:小时|hours?|hrs?)",
            "天": r"(?:天|days?)",
            "次": r"(?:次|times?|cycles?)",
            "年": r"(?:年|years?)",
            "月": r"(?:月|months?)",
            "日": r"(?:日|days?)",
            "℃": r"(?:℃|°c|degrees?c)",
            "%": r"%",
            "v": r"v",
            "w": r"w",
            "a": r"a",
            "hz": r"hz",
            "mm": r"mm",
            "cm": r"cm",
        }
        pattern = rf"(\d+(?:\.\d+)?)(?:[-–~至](\d+(?:\.\d+)?))?{aliases[unit]}"
        for found in re.finditer(pattern, support, flags=re.IGNORECASE):
            lower = float(found.group(1))
            upper = float(found.group(2)) if found.group(2) else lower
            if min(lower, upper) <= value <= max(lower, upper):
                return True
        return False
