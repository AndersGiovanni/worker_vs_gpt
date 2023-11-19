from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import List
from typing import Optional


@dataclass
class TextPair:
    original_label: str
    original_text: str
    augmented_label: str
    augmented_text: str
    aug_from_ori: bool

    prompt: Optional[List[Dict[str, str]]] = field(init=False)
    prompt_reponse: str = field(default_factory=str, init=False)


if __name__ == "__main__":
    tp = TextPair(
        original_label="knowledge",
        original_text="original_text",
        augmented_label="augmented_label",
        augmented_text="augmented_text",
        aug_from_ori=True,
    )

    tp.prompt = [{"virker det?": "ja"}]
    tp.prompt_reponse = "ja"
