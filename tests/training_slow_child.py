"""Test child that remains alive until cancelled or its parent disappears."""

import sys
import threading
import time

from dialogentis.training.child import _watch_parent

sys.stdin.buffer.readline()
threading.Thread(target=_watch_parent, daemon=True).start()
print("validation", flush=True)
while True:
    time.sleep(1)
