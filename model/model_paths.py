from pathlib import Path

class ModelPaths:
    def __init__(self):
        self.model_path  = Path(rf'./model_instance').resolve()
        self.chal_path  = Path(rf'./model/challenge 2024.wav').resolve()
        self.create_dirs()

    def create_dirs(self):
        self.model_path.mkdir(exist_ok=True, parents=True)

