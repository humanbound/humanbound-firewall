# hb-firewall — DEPRECATED

This package has been **renamed** to [`humanbound-firewall`](https://pypi.org/project/humanbound-firewall/).

Please migrate:

```bash
pip uninstall hb-firewall
pip install humanbound-firewall
```

```python
# Old
from hb_firewall import Firewall

# New
from humanbound_firewall import Firewall
```

The `hb-firewall` package on PyPI now resolves to a transitional stub that
depends on `humanbound-firewall` and emits a `DeprecationWarning` on import.
This stub will be yanked on or after **2026-06-20**. Please migrate before then.

See the [CHANGELOG](https://github.com/humanbound/humanbound-firewall/blob/main/CHANGELOG.md)
for details on the rename.
