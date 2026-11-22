# Started by the desktop shortcut with pythonw.exe (no console window).
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
os.chdir(here)

from pcore.app import main  # noqa: E402

main()
