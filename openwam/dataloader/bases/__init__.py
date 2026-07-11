"""Base / scaffold Dataset classes used by concrete readers and wrappers.

Three classes live here:

- ``BaseDataset`` — abstract root of every reader. Every dataset
  in this package inherits from it directly or transitively.

- ``LeRobotV3Reader`` — unified single-bucket LeRobot v3 reader base.
  Direct parent of all 4 OXE readers (BC-Z / Bridge / Fractal / DROID),
  :class:`~openwam.dataloader.robocoin.RoboCOINDataset`, and
  :class:`~openwam.dataloader.egodex.EgoDexDataset` — equal siblings, no
  per-family intermediate base.

- ``MultiLeRobotV3Reader`` — concrete base for the "aggregate N homogeneous
  LeRobot v3 buckets into one Dataset" pattern. Subclassed by
  :class:`~openwam.dataloader.robocoin.MultiRobotCOINDataset` and
  :class:`~openwam.dataloader.egodex.MultiBucketEgoDexDataset`.

All are re-exported here so callers can ``from openwam.dataloader.bases
import BaseDataset, LeRobotV3Reader, MultiLeRobotV3Reader`` without caring
which file they live in.
"""

from openwam.dataloader.bases.dataset import BaseDataset
from openwam.dataloader.bases.lerobot_v3_reader import LeRobotV3Reader
from openwam.dataloader.bases.multi_lerobot_v3_reader import MultiLeRobotV3Reader

__all__ = ["BaseDataset", "LeRobotV3Reader", "MultiLeRobotV3Reader"]
