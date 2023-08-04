import argparse


class VBBParser(argparse.ArgumentParser):
    def __init__(self) -> None:
        super().__init__()

        # Add VBB envrionment arguments.
        self.add_argument("--task", type=str, choices=["VisualBlockBuilder", "ReachSpecificTarget"], default="VisualBlockBuilder",
                          help="The task to look at.")
        self.add_argument("--case", type=str, choices=["Singletower", "Pyramid", "Multitower", "All"], default="Singletower",
                          help="The case to run. Must be one of 'Singletower', 'Pyramid', 'Multitower', or 'All'.")
        self.add_argument("--num_blocks", type=int, default=2,
                          help="The number of blocks in the environment. Must be between 1 and 17.")
        self.add_argument("--robot", type=str, choices=["default", "simplified"], default="simplified",
                          help="Robot model to use.")
        self.add_argument("--stack_only", action="store_true")
        self.add_argument("--viewpoint", type=str, choices=["frontview", "topview", "external_camera_0"], default="frontview",
                          help="The viewpoint/camera to use for rendering.")
