"""一键生成全部图表"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from obscodec.visualize import generate_all
generate_all()
