"""
Secure Error Handler for Contextual Retrieval

Provides secure error handling, sanitization, and logging to prevent
information disclosure while maintaining useful debugging capabilities.
"""

import re
from typing import Dict, Any, Optional, Union
from urllib.parse import urlparse, urlunparse
from src.loki_logger import LokiLogger
import httpx

# Initialize Loki logger
logger = LokiLogger(service_name="error-handler")


class SecureErrorHandler:
    """
    Handles error sanitization and secure logging for contextual retrieval components.

    Prevents sensitive information disclosure while maintaining debugging capabilities.
    """

    # Sensitive header patterns (case-insensitive)
    SENSITIVE_HEADERS = {
        "authorization",
        "x-api-key",
        "api-key",
        "apikey",
        "x-auth-token",
        "auth-token",
        "bearer",
        "token",
        "x-access-token",
        "access-token",
        "x-secret",
        "secret",
        "password",
        "x-password",
        "passwd",
        "credentials",
        "x-credentials",
    }

    # URL patterns that might contain sensitive info
    SENSITIVE_URL_PATTERNS = [
        r"password=([^&\s]+)",
        r"token=([^&\s]+)",
        r"key=([^&\s]+)",
        r"secret=([^&\s]+)",
        r"auth=([^&\s]+)",
        r"api_key=([^&\s]+)",
        r"access_token=([^&\s]+)",
    ]

    @staticmethod
    def sanitize_url(url: str) -> str:
        """
        Remove sensitive information from URLs.

        Args:
            url: URL that may contain sensitive information

        Returns:
            Sanitized URL with sensitive parts replaced with [REDACTED]
        """
        if not url:
            return url

        try:
            # Parse URL components
            parsed = urlparse(url)

            # Sanitize password in netloc (user:password@host)
            if parsed.password:
                netloc = parsed.netloc.replace(f":{parsed.password}@", ":[REDACTED]@")
            else:
                netloc = parsed.netloc

            # Sanitize query parameters
            query = parsed.query
            if query:
                for pattern in SecureErrorHandler.SENSITIVE_URL_PATTERNS:
                    query = re.sub(
                        pattern, r"\1=[REDACTED]", query, flags=re.IGNORECASE
                    )

            # Reconstruct URL
            sanitized_parsed = parsed._replace(netloc=netloc, query=query)
            return urlunparse(sanitized_parsed)

        except Exception:
            # If URL parsing fails, do basic pattern replacement
            sanitized = url
            for pattern in SecureErrorHandler.SENSITIVE_URL_PATTERNS:
                sanitized = re.sub(
                    pattern, r"\1=[REDACTED]", sanitized, flags=re.IGNORECASE
                )
            return sanitized

    @staticmethod
    def sanitize_headers(headers: Union[Dict[str, Any], None]) -> Dict[str, Any]:
        """
        Remove sensitive headers from header dictionary.

        Args:
            headers: HTTP headers dictionary

        Returns:
            Sanitized headers with sensitive values replaced
        """
        if not headers:
            return {}

        sanitized: Dict[str, Any] = {}
        for key, value in headers.items():
            if key.lower() in SecureErrorHandler.SENSITIVE_HEADERS:
                # Check if it's a bearer token or similar
                if isinstance(value, str) and value.lower().startswith("bearer "):
                    sanitized[key] = "Bearer [REDACTED]"
                else:
                    sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = value

        return sanitized

    @staticmethod
    def sanitize_error_message(error: Exception, context: str = "") -> str:
        """
        Create safe error messages for user consumption.

        Args:
            error: Exception that occurred
            context: Additional context about where error occurred

        Returns:
            Sanitized error message safe for user consumption
        """
        error_type = type(error).__name__

        # Handle specific error types with appropriate sanitization
        if isinstance(error, httpx.HTTPError):
            return SecureErrorHandler._sanitize_http_error(error, context)
        elif isinstance(error, ConnectionError):
            return f"Connection error in {context}: Unable to connect to service"
        elif isinstance(error, TimeoutError):
            return f"Timeout error in {context}: Operation timed out"
        elif isinstance(error, ValueError):
            # ValueError might contain sensitive data, be generic
            return f"Invalid data error in {context}: Please check input parameters"
        else:
            # Generic error - don't expose internal details
            return f"{error_type} in {context}: An internal error occurred"

    @staticmethod
    def _sanitize_http_error(error: httpx.HTTPError, context: str) -> str:
        """Sanitize HTTP-specific errors."""
        if isinstance(error, httpx.ConnectError):
            return f"Connection error in {context}: Unable to connect to server"
        elif isinstance(error, httpx.TimeoutException):
            return f"Timeout error in {context}: Request timed out"
        elif isinstance(error, httpx.HTTPStatusError):
            # Don't expose response content, just status
            return f"HTTP error in {context}: Server returned status {error.response.status_code}"
        else:
            return f"HTTP error in {context}: Network communication failed"

    @staticmethod
    def log_secure_error(
        error: Exception,
        context: str,
        request_url: Optional[str] = None,
        request_headers: Optional[Dict[str, Any]] = None,
        level: str = "error",
    ) -> None:
        """
        Log errors securely without exposing sensitive data.

        Args:
            error: Exception that occurred
            context: Context where error occurred
            request_url: URL being accessed (will be sanitized)
            request_headers: Request headers (will be sanitized)
            level: Log level (error, warning, debug)
        """
        # Create base log data
        log_data: Dict[str, Any] = {
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

        # Add sanitized request information if provided
        if request_url:
            log_data["url"] = SecureErrorHandler.sanitize_url(request_url)

        if request_headers:
            log_data["headers"] = SecureErrorHandler.sanitize_headers(request_headers)

        # Add HTTP-specific details for HTTP errors
        if isinstance(error, httpx.HTTPStatusError):
            # HTTPStatusError has response attribute
            log_data["status_code"] = error.response.status_code
            # Don't log response content as it might contain sensitive data

        # Log at appropriate level
        log_message = f"Secure error in {context}: {type(error).__name__}"

        if level == "debug":
            logger.debug(log_message, **log_data)
        elif level == "warning":
            logger.warning(log_message, **log_data)
        else:
            logger.error(log_message, **log_data)

    @staticmethod
    def create_user_safe_response(error: Exception, operation: str) -> Dict[str, Any]:
        """
        Create a user-safe error response dictionary.

        Args:
            error: Exception that occurred
            operation: Operation being performed

        Returns:
            Dictionary with safe error information for API responses
        """
        return {
            "success": False,
            "error": {
                "type": "operation_failed",
                "message": SecureErrorHandler.sanitize_error_message(error, operation),
                "operation": operation,
                "timestamp": None,  # Will be added by calling code if needed
            },
        }

    @staticmethod
    def is_user_error(error: Exception) -> bool:
        """
        Determine if error is likely a user error vs system error.

        Args:
            error: Exception to classify

        Returns:
            True if likely a user error, False if system error
        """
        # User errors - safe to provide more specific feedback
        user_error_types = (ValueError, TypeError, KeyError, httpx.HTTPStatusError)

        if isinstance(error, user_error_types):
            # Additional checks for HTTP errors
            if isinstance(error, httpx.HTTPStatusError):
                # 4xx errors are typically user errors
                return 400 <= error.response.status_code < 500
            return True

        return False
