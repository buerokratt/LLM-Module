#!/usr/bin/env python3
"""
RSA-OAEP Decryption Utility for Vault Secrets

This utility provides in-memory decryption of RSA-OAEP encrypted values
without writing sensitive data to disk.
"""

import sys
import base64
from typing import NoReturn
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.backends import default_backend

try:
    # In cron-manager: loki_logger.py is mounted into /app/src/vector_indexer/
    from loki_logger import LokiLogger
except ModuleNotFoundError:
    # In the main llm-service: import via the full src package path.
    from src.loki_logger import LokiLogger

logger = LokiLogger(service_name="vault-secrets-decryptor")


def base64url_to_bytes(base64url_string: str) -> bytes:
    """
    Convert Base64URL encoded string to bytes.

    Args:
        base64url_string: Base64URL encoded string (with - and _ instead of + and /)

    Returns:
        Decoded bytes
    """
    # Replace URL-safe characters with standard base64 characters
    base64_string = base64url_string.replace("-", "+").replace("_", "/")

    # Add padding if needed (base64 requires length to be multiple of 4)
    padding_needed = 4 - (len(base64_string) % 4)
    if padding_needed != 4:
        base64_string += "=" * padding_needed

    # Decode base64 to bytes
    return base64.b64decode(base64_string)


def decrypt_rsa_oaep(encrypted_base64url: str, private_key_pem: str) -> str:
    """
    Decrypt RSA-OAEP encrypted value using private key.

    All operations are performed in memory - no disk I/O.

    Args:
        encrypted_base64url: Base64URL encoded encrypted data
        private_key_pem: RSA private key in PEM format

    Returns:
        Decrypted plaintext string

    Raises:
        ValueError: If decryption fails (invalid key, padding, or corrupted data)
        Exception: For other errors
    """
    try:
        # Load private key from PEM string (in memory, no file)
        loaded_key = serialization.load_pem_private_key(
            private_key_pem.encode("utf-8"), password=None, backend=default_backend()
        )

        # Type check and cast to RSA private key
        if not isinstance(loaded_key, rsa.RSAPrivateKey):
            raise TypeError("Private key must be an RSA key")

        private_key: rsa.RSAPrivateKey = loaded_key

        # Decode Base64URL to bytes (in memory)
        encrypted_bytes = base64url_to_bytes(encrypted_base64url)

        # Decrypt using RSA-OAEP with SHA-256 (in memory)
        plaintext_bytes = private_key.decrypt(
            encrypted_bytes,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

        # Convert bytes to string
        plaintext = plaintext_bytes.decode("utf-8")

        logger.debug(
            f"Successfully decrypted {len(encrypted_base64url)} bytes of encrypted data"
        )

        return plaintext

    except ValueError as e:
        # Invalid padding or decryption failure
        raise ValueError(f"Decryption failed: {e}") from e
    except TypeError as e:
        # Wrong key type
        raise TypeError(f"Invalid key type: {e}") from e
    except Exception as e:
        # Other errors (invalid key format, corrupted data, etc.)
        raise RuntimeError(f"Decryption error: {e}") from e


def main() -> NoReturn:
    """
    CLI interface for decryption.

    Usage:
        python decrypt_vault_secrets.py <encrypted_base64url> <private_key_pem>

    The private key PEM should be a single string with \n for newlines.
    """
    if len(sys.argv) != 3:
        logger.error(
            "Usage: python decrypt_vault_secrets.py <encrypted_base64url> <private_key_pem>"
        )
        logger.info(
            "Example: python decrypt_vault_secrets.py 'A8xF2nQ7...' '-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----'"
        )
        sys.exit(1)

    encrypted_value = sys.argv[1]
    private_key_pem = sys.argv[2]

    logger.debug(f"Received encrypted value of length: {len(encrypted_value)}")
    logger.debug("Private key PEM received")

    # Handle literal \n in shell arguments
    private_key_pem = private_key_pem.replace("\\n", "\n")

    try:
        plaintext = decrypt_rsa_oaep(encrypted_value, private_key_pem)

        logger.debug("Decryption successful, outputting plaintext to stdout")

        # Output to stdout only (for shell script capture)
        # Logger writes to stderr, print to stdout - intentional separation for piping
        print(plaintext)  # noqa: T201

        sys.exit(0)

    except ValueError as e:
        logger.error(f"Decryption failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
