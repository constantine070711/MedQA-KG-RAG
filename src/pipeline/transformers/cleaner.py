"""
文本清洗器

职责:
    - 清理文本中的噪声（控制字符、多余空白、HTML 实体等）
    - 规范化 Unicode 字符
    - 不修改领域语义内容（不改写、不翻译）

清洗规则:
    1. 移除控制字符（保留换行和制表符）
    2. 规范化空白（连续空白合并为单个空格）
    3. 解码常见 HTML 实体（&amp; → &）
    4. 统一全角/半角标点（可选，默认关闭以保留医学文本特征）
    5. 去除首尾空白

用法:
    from src.pipeline.transformers import TextCleaner
    cleaner = TextCleaner()
    cleaned = cleaner.clean("文本  内容  ")
"""

import re
import logging
import unicodedata

logger = logging.getLogger(__name__)


class TextCleaner:
    """
    文本清洗器。

    处理 Pipeline 中所有文本字段（name / description / raw_text 等）。
    保持医学语义不变，仅做格式清理。
    """

    # 保留的空白字符
    PRESERVE_CHARS = {"\n", "\r", "\t"}

    # HTML 实体映射（常见的医学文本中可能出现）
    HTML_ENTITIES = {
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&apos;": "'",
        "&nbsp;": " ",
        "&mdash;": "—",
        "&ndash;": "–",
        "&ldquo;": "“",
        "&rdquo;": "”",
    }

    def __init__(
        self,
        normalize_whitespace: bool = True,
        remove_control_chars: bool = True,
        decode_html_entities: bool = True,
        strip: bool = True,
    ):
        """
        Args:
            normalize_whitespace: 规范化空白
            remove_control_chars: 移除控制字符
            decode_html_entities: 解码 HTML 实体
            strip: 去除首尾空白
        """
        self.normalize_whitespace = normalize_whitespace
        self.remove_control_chars = remove_control_chars
        self.decode_html_entities = decode_html_entities
        self.strip = strip

    def clean(self, text: str) -> str:
        """
        清洗文本。

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        if not text or not isinstance(text, str):
            return ""

        result = text

        # 1. 移除控制字符（保留换行和制表符）
        if self.remove_control_chars:
            result = self._remove_control_chars(result)

        # 2. 解码 HTML 实体
        if self.decode_html_entities:
            result = self._decode_html_entities(result)

        # 3. 规范化空白
        if self.normalize_whitespace:
            result = self._normalize_whitespace(result)

        # 4. 去除首尾空白
        if self.strip:
            result = result.strip()

        return result

    def clean_entity(self, raw_entity: dict) -> dict:
        """
        清洗整个 RawEntity 的所有文本字段。

        处理字段: name / description / raw_text / category / aliases

        Args:
            raw_entity: RawEntity dict

        Returns:
            清洗后的 dict（不修改原始 dict，返回新 dict）
        """
        cleaned = dict(raw_entity)  # 浅拷贝

        # 清洗文本字段
        text_fields = ["name", "description", "raw_text", "category"]
        for field in text_fields:
            if field in cleaned and isinstance(cleaned[field], str):
                cleaned[field] = self.clean(cleaned[field])

        # 清洗别名列表
        if "aliases" in cleaned and isinstance(cleaned["aliases"], list):
            cleaned["aliases"] = [
                self.clean(a) for a in cleaned["aliases"]
                if isinstance(a, str) and self.clean(a)
            ]

        return cleaned

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------

    @staticmethod
    def _remove_control_chars(text: str) -> str:
        """移除控制字符（保留 \\n \\r \\t）"""
        result = []
        for ch in text:
            cp = ord(ch)
            if cp < 0x20:
                if ch in TextCleaner.PRESERVE_CHARS:
                    result.append(ch)
                # else: 跳过其他控制字符
            elif cp == 0x7F:
                continue  # DEL
            else:
                result.append(ch)
        return "".join(result)

    @staticmethod
    def _decode_html_entities(text: str) -> str:
        """解码常见 HTML 实体"""
        result = text
        for entity, char in TextCleaner.HTML_ENTITIES.items():
            result = result.replace(entity, char)
        return result

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """规范化空白：连续空白 → 单个空格"""
        # 将制表符替换为空格
        text = text.replace("\t", "    ")
        # 规范化行内空白（保留换行）
        lines = text.split("\n")
        cleaned_lines = [re.sub(r"[^\S\n]+", " ", line) for line in lines]
        return "\n".join(cleaned_lines)
