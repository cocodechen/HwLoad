from enum import Enum, auto

class ProfileType(Enum):
    Compute = auto()   # 计算密集
    Memory = auto()    # 内存/带宽密集
    Data = auto()      # 混合
    IO = auto()        # IO
    Random = auto()    # [NEW] 随机模拟真实负载

class LoadLevel(Enum):
    Idle = auto()
    Low = auto()
    Medium = auto()
    High = auto()
    Saturated = auto()

class LoadTask:
    def __init__(self, device: str, profile: ProfileType, level: LoadLevel):
        self.device = device
        self.profile = profile
        self.level = level