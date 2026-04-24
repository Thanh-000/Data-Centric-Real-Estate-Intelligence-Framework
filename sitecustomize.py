"""Repository-local Python startup hooks.

This repo includes a narrow Windows-only workaround for Jupyter kernel startup
under OneDrive-managed directories, where the default ACL tightening in
``jupyter_core.paths.win32_restrict_file_to_user`` can fail with
``SetFileSecurity: Access is denied``. The patch is intentionally limited to
this specific Jupyter helper so local notebook validation commands can run.
"""

from __future__ import annotations

import os
import sys


if os.name == "nt" and not os.environ.get("DC_REIF_DISABLE_JUPYTER_ACL_PATCH"):
    try:
        import jupyter_core.paths
    except Exception:
        pass
    else:
        def _noop_win32_restrict_file_to_user(_fname: str) -> None:
            return None

        jupyter_core.paths.win32_restrict_file_to_user = _noop_win32_restrict_file_to_user
