"""Portable verification and workspace preparation for this study."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/"tools"))
from control_study import main
if __name__=="__main__": main(ROOT)
