"""wgate: confidence-weighted memory writes for a frozen CUT3R (WRITE_GATE_VARIANTS.md).

Heads and pure functions live in ``heads.py``; the model hooks are in ``dust3r/model.py``
(``attach_frame_gate`` / ``attach_mem_gate`` and the ``frame_gate_*`` / ``mem_gate_*`` view keys).
"""

from .heads import *  # noqa: F401,F403
from .heads import __all__  # noqa: F401
