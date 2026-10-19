"""
医疗安全规则模块

职责:
    - 定义医疗问答系统的安全边界
    - 问题分类与拒答策略
    - 医疗免责声明
    - 回答内容安全检查

安全级别:
    SAFE       — 可通过，正常回答
    CAUTION    — 可通过，附加免责声明
    RESTRICTED — 限制回答，强调就医建议
    BLOCKED    — 拒绝回答，返回安全提示

设计原则:
    - 宁可拒答，不可误导
    - 涉及诊断、用药剂量必须强调就医
    - 紧急情况建议立即就医
    - 心理危机提供热线信息

用法:
    from src.medical_core.safety import SafetyChecker

    checker = SafetyChecker()
    result = checker.check("我头痛怎么办?")
    if result.level == SafetyLevel.BLOCKED:
        print(result.rejection_message)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# 安全级别
# ============================================================

class SafetyLevel(Enum):
    """安全级别"""
    SAFE = "safe"            # 可通过
    CAUTION = "caution"      # 可回答但需附加免责声明
    RESTRICTED = "restricted"  # 限制回答
    BLOCKED = "blocked"      # 拒绝回答


@dataclass
class SafetyCheckResult:
    """
    安全检查结果

    Attributes:
        level: 安全级别
        reason: 分类原因
        disclaimer_needed: 是否需要附加免责声明
        rejection_message: 拒答消息（BLOCKED 级别使用）
        modified_prompt: 修改后的 prompt（CAUTION/RESTRICTED 级别使用）
    """
    level: SafetyLevel
    reason: str = ""
    disclaimer_needed: bool = False
    rejection_message: str = ""
    modified_prompt: str = ""

    @property
    def is_allowed(self) -> bool:
        return self.level != SafetyLevel.BLOCKED

    @property
    def is_restricted(self) -> bool:
        return self.level in (SafetyLevel.RESTRICTED, SafetyLevel.BLOCKED)


# ============================================================
# 医疗免责声明
# ============================================================

MEDICAL_DISCLAIMER = (
    "⚠️ **医疗免责声明**\n\n"
    "本系统提供的医疗信息仅供参考，不能替代专业医生的诊断和治疗建议。"
    "如有具体健康问题，请及时咨询持证医师或前往正规医疗机构就诊。"
    "在紧急情况下，请立即拨打急救电话（中国：120）。"
)

URGENT_DISCLAIMER = (
    "🚨 **紧急提醒**\n\n"
    "您描述的情况可能需要紧急医疗救助。"
    "请立即拨打急救电话（中国：120）或前往最近的医院急诊科。"
    "本系统无法提供紧急医疗建议。"
)

MENTAL_HEALTH_DISCLAIMER = (
    "❤️ **心理健康提醒**\n\n"
    "如果您正在经历心理困扰，请知道您并不孤单。\n"
    "- 全国24小时心理援助热线：400-161-9995\n"
    "- 北京心理危机研究与干预中心：010-82951332\n"
    "- 生命热线：400-821-1215\n\n"
    "请尽快联系专业心理咨询师或精神科医生。"
)

SYSTEM_PROMPT_SAFE = (
    "你是一个医疗知识辅助助手。\n"
    "重要规则:\n"
    "1. 你的回答不能替代医生诊断和治疗建议\n"
    "2. 如果问题涉及个人病情诊断，请建议用户咨询专业医生\n"
    "3. 对于不确定的医学问题，请明确说明不确定性\n"
    "4. 回答应基于提供的医学知识库，并注明来源\n"
    "5. 不得主动提供具体药物剂量、服药频次或个体化用药方案；"
    "即使用户问题不是剂量请求，也不要在回答中复述检索资料里的具体剂量数值\n"
    "6. 对不了解的医学问题请直接说明，不要编造信息"
)


# ============================================================
# 安全检查器
# ============================================================

class SafetyChecker:
    """
    医疗问答安全检查器

    负责对用户问题进行安全分类，决定回答策略。

    分类规则:
        - BLOCKED:    心理健康/自杀/自伤 → 提供热线
        - RESTRICTED: 个人诊断/用药剂量 → 仅通用信息
        - CAUTION:    症状咨询/药物咨询 → 附加免责
        - SAFE:       概念解释/健康建议 → 正常回答

    用法:
        checker = SafetyChecker()
        result = checker.check(user_input)
        if not result.is_allowed:
            return result.rejection_message
    """

    # --- 关键词触发规则 ---

    # 需要 BLOCK 的高危关键词（心理健康危机/自伤/暴力）
    BLOCKED_PATTERNS: list[str] = [
        "自杀", "自残", "自伤", "想死", "不想活",
        "结束生命", "怎么死", "安眠药自杀",
    ]

    # 需要 RESTRICT 的敏感模式（个人诊断/具体用药方案）
    RESTRICTED_PATTERNS: list[str] = [
        "我得了什么病", "我是什么病", "帮我诊断",
        "我的症状是", "我是不是得了",
        "给我开", "开什么药", "用什么药", "吃什么药好",
        "药量", "用量", "吃几片", "吃多少", "一天几次",
        "治得好吗", "能治好吗", "会死吗",
    ]

    # 需要 CAUTION 的模式（一般性医疗咨询）
    CAUTION_PATTERNS: list[str] = [
        "症状", "怎么治", "治疗方法", "用什么药",
        "副作用", "不良反应", "危害", "饮食",
        "注意什么", "怎么预防", "怎么办",
    ]

    def __init__(self):
        """初始化安全检查器"""
        pass

    def check(self, user_input: str) -> SafetyCheckResult:
        """
        对用户输入进行安全检查分类。

        Args:
            user_input: 用户原始输入

        Returns:
            SafetyCheckResult 包含安全级别和处理建议
        """
        text = user_input.strip()

        # 1. 检查 BLOCKED 模式（优先级最高）
        for pattern in self.BLOCKED_PATTERNS:
            if pattern in text:
                return SafetyCheckResult(
                    level=SafetyLevel.BLOCKED,
                    reason=f"包含高危关键词: '{pattern}'",
                    rejection_message=(
                        f"检测到您的问题涉及心理健康危机。\n\n"
                        f"{MENTAL_HEALTH_DISCLAIMER}"
                    ),
                )

        # 2. 检查 RESTRICTED 模式
        for pattern in self.RESTRICTED_PATTERNS:
            if pattern in text:
                return SafetyCheckResult(
                    level=SafetyLevel.RESTRICTED,
                    reason=f"包含诊断/用药限制关键词: '{pattern}'",
                    disclaimer_needed=True,
                    modified_prompt=(
                        f"{SYSTEM_PROMPT_SAFE}\n\n"
                        f"特别提醒: 用户可能正在描述个人症状，请在回答中强调:\n"
                        f"  - 你无法进行诊断，仅能提供通用医学知识\n"
                        f"  - 建议用户前往正规医院就诊\n"
                        f"  - 切勿提供具体的用药方案或剂量建议"
                    ),
                )

        # 3. 检查 CAUTION 模式
        for pattern in self.CAUTION_PATTERNS:
            if pattern in text:
                return SafetyCheckResult(
                    level=SafetyLevel.CAUTION,
                    reason=f"医疗相关咨询: '{pattern}'",
                    disclaimer_needed=True,
                )

        # 4. 默认 SAFE（解释型问题、健康知识等）
        return SafetyCheckResult(
            level=SafetyLevel.SAFE,
            reason="一般性问题",
        )

    def get_safe_system_prompt(self, safety_result: SafetyCheckResult) -> str:
        """
        根据安全检查结果生成安全的 System Prompt。

        Args:
            safety_result: 安全检查结果

        Returns:
            适合的 system_prompt 文本
        """
        if safety_result.level == SafetyLevel.RESTRICTED:
            return safety_result.modified_prompt or SYSTEM_PROMPT_SAFE
        else:
            return SYSTEM_PROMPT_SAFE

    def format_answer(
        self,
        raw_answer: str,
        safety_result: SafetyCheckResult,
    ) -> str:
        """
        根据安全级别格式化最终回答。

        Args:
            raw_answer: 原始回答文本
            safety_result: 安全检查结果

        Returns:
            格式化后的回答（可能附加免责声明）
        """
        if safety_result.level == SafetyLevel.BLOCKED:
            return safety_result.rejection_message

        if safety_result.disclaimer_needed:
            return f"{raw_answer}\n\n---\n{MEDICAL_DISCLAIMER}"

        return raw_answer


# ============================================================
# 紧急关键词检测（独立函数，可用于任何阶段）
# ============================================================

def detect_emergency(text: str) -> bool:
    """
    检测是否为紧急医疗情况。

    Args:
        text: 用户输入文本

    Returns:
        是否为紧急情况
    """
    emergency_keywords = [
        "胸痛", "呼吸困难", "大出血", "昏迷", "抽搐",
        "中毒", "严重过敏", "休克", "心脏骤停",
    ]
    return any(kw in text for kw in emergency_keywords)


def get_emergency_response() -> str:
    """获取紧急情况的标准回复"""
    return (
        "🚨 根据您描述的症状，这可能是紧急医疗情况。\n\n"
        "**请立即采取以下措施：**\n"
        "1. 拨打急救电话 **120**（中国）\n"
        "2. 如有人在旁，请大声呼救\n"
        "3. 不要自己驾车前往医院\n"
        "4. 保持电话畅通，听从急救调度员指导\n\n"
        "本系统无法处理医疗紧急情况。"
    )
