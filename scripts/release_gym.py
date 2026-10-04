import json
import sys
from pathlib import Path

from pramaan.gym.release import release_gym

print(json.dumps(release_gym(Path(sys.argv[1] if len(sys.argv) > 1 else "release/gym")), indent=1, default=str))
