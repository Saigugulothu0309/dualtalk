import argparse

try:
    from cryptography.fernet import Fernet
except ImportError as exc:
    Fernet = None
    _CRYPTO_IMPORT_ERROR = exc
else:
    _CRYPTO_IMPORT_ERROR = None


class MessageCrypto:
    """
    Symmetric Fernet encryption for WebSocket messages.
    Both sender and receiver must use the same key.
    """

    def __init__(self, key: bytes | None = None):
        if Fernet is None:
            raise ImportError(
                "Missing dependency: cryptography. Install using 'pip install cryptography'"
            ) from _CRYPTO_IMPORT_ERROR
        if key is None:
            key = Fernet.generate_key()
        self._fernet = Fernet(key)
        self.key = key

    def encrypt(self, text: str) -> bytes:
        return self._fernet.encrypt(text.encode("utf-8"))

    def decrypt(self, token: bytes) -> str:
        return self._fernet.decrypt(token).decode("utf-8")

    @staticmethod
    def generate_key() -> bytes:
        if Fernet is None:
            raise ImportError(
                "Missing dependency: cryptography. Install using 'pip install cryptography'"
            ) from _CRYPTO_IMPORT_ERROR
        return Fernet.generate_key()

    @staticmethod
    def key_to_string(key: bytes) -> str:
        return key.decode("utf-8")

    @staticmethod
    def key_from_string(key_str: str) -> bytes:
        key_value = key_str.strip()
        if key_value.startswith("DUALTALK_KEY="):
            key_value = key_value.split("=", 1)[1]
        return key_value.encode("utf-8")

    @classmethod
    def from_string(cls, key_str: str):
        return cls(cls.key_from_string(key_str))


def resolve_crypto(
    enabled: bool,
    key_str: str | None = None,
    default_key: str | None = None,
    generate_if_missing: bool = False,
):
    if not enabled:
        return None

    if Fernet is None:
        raise ImportError(
            "Missing dependency: cryptography. Install using 'pip install cryptography'"
        ) from _CRYPTO_IMPORT_ERROR

    selected_key = key_str or default_key
    if selected_key:
        return MessageCrypto.from_string(selected_key)

    if generate_if_missing:
        crypto = MessageCrypto()
        print(f"DUALTALK_KEY={MessageCrypto.key_to_string(crypto.key)}")
        return crypto

    raise SystemExit(
        "Encryption enabled but no key provided. Pass --key DUALTALK_KEY=<value>."
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate or parse DualTalk encryption keys.")
    parser.add_argument(
        "--generate-key",
        action="store_true",
        help="Generate a new Fernet key and print it as DUALTALK_KEY=<value>.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.generate_key:
        key = MessageCrypto.generate_key()
        print(f"DUALTALK_KEY={MessageCrypto.key_to_string(key)}")


if __name__ == "__main__":
    main()
