from dataclasses import dataclass, field
from typing import Any, Dict, Protocol


@dataclass
class OCRResult:
    provider: str
    text: str
    elapsed_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class OCRProvider(Protocol):
    name: str

    def recognize(self, image_path: str, config: dict) -> OCRResult:
        ...
