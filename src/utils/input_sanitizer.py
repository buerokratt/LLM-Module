"""Input sanitization utilities for preventing XSS and normalizing content."""

import re
import html
from typing import Optional, List, Dict, Any
from src.loki_logger import LokiLogger

# Initialize Loki logger
logger = LokiLogger(service_name="input-sanitizer")


class InputSanitizer:
    """Utilities for sanitizing user input to prevent XSS and normalize content."""

    # HTML tags that should always be stripped
    DANGEROUS_TAGS = [
        "script",
        "iframe",
        "object",
        "embed",
        "link",
        "style",
        "meta",
        "base",
        "form",
        "input",
        "button",
        "textarea",
    ]

    # Event handlers that can execute JavaScript
    EVENT_HANDLERS = [
        "onclick",
        "onload",
        "onerror",
        "onmouseover",
        "onmouseout",
        "onfocus",
        "onblur",
        "onchange",
        "onsubmit",
        "onkeydown",
        "onkeyup",
        "onkeypress",
        "ondblclick",
        "oncontextmenu",
    ]

    @staticmethod
    def strip_html_tags(text: str) -> str:
        """
        Remove all HTML tags from text, including dangerous ones.

        Args:
            text: Input text that may contain HTML

        Returns:
            Text with HTML tags removed
        """
        if not text:
            return text

        text = html.unescape(text)

        # First pass: Remove dangerous tags and their content
        for tag in InputSanitizer.DANGEROUS_TAGS:
            # Remove opening tag, content, and closing tag
            pattern = rf"<{tag}[^>]*>.*?</{tag}>"
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
            # Remove self-closing tags
            pattern = rf"<{tag}[^>]*/>"
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Second pass: Remove event handlers (e.g., onclick="...")
        for handler in InputSanitizer.EVENT_HANDLERS:
            pattern = rf'{handler}\s*=\s*["\'][^"\']*["\']'
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Third pass: Remove all remaining HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        return text

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        Normalize whitespace: collapse multiple spaces, remove leading/trailing.

        Args:
            text: Input text with potentially excessive whitespace

        Returns:
            Text with normalized whitespace
        """
        if not text:
            return text

        # Replace multiple spaces with single space
        text = re.sub(r" +", " ", text)

        # Replace multiple newlines with double newline (preserve paragraph breaks)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

        # Replace tabs with spaces
        text = text.replace("\t", " ")

        # Remove trailing whitespace from each line
        text = "\n".join(line.rstrip() for line in text.split("\n"))

        # Strip leading and trailing whitespace
        text = text.strip()

        return text

    @staticmethod
    def sanitize_message(message: str, chat_id: Optional[str] = None) -> str:
        """
        Sanitize user message: strip HTML, normalize whitespace.

        Args:
            message: User message to sanitize
            chat_id: Optional chat ID for logging

        Returns:
            Sanitized message
        """
        if not message:
            return message

        original_length = len(message)

        # Strip HTML tags
        message = InputSanitizer.strip_html_tags(message)

        # Normalize whitespace
        message = InputSanitizer.normalize_whitespace(message)

        sanitized_length = len(message)

        # Log if significant content was removed (potential attack)
        if original_length > 0 and sanitized_length < original_length * 0.8:
            logger.warning(
                f"Significant content removed during sanitization: "
                f"{original_length} -> {sanitized_length} chars "
                f"(chat_id={chat_id})"
            )

        return message

    @staticmethod
    def sanitize_conversation_history(
        history: List[Dict[str, Any]], chat_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Sanitize conversation history items.

        Args:
            history: List of conversation items (dicts with 'content' field)
            chat_id: Optional chat ID for logging

        Returns:
            Sanitized conversation history
        """
        if not history:
            return history

        sanitized: List[Dict[str, Any]] = []
        for item in history:
            # Item should be a dict (already typed in function signature)
            sanitized_item = item.copy()

            # Sanitize content field if present
            if "content" in sanitized_item:
                sanitized_item["content"] = InputSanitizer.sanitize_message(
                    sanitized_item["content"], chat_id=chat_id
                )

            sanitized.append(sanitized_item)

        return sanitized
