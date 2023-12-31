from setuptools import setup, find_packages


setup(
        name="visual-block-builder",
        version="0.1",
        packages=find_packages(),
        install_requires=[
            "cython<3",
            "fetch-block-construction",
            "gym<=0.17.3",
            "mujoco-py", 
            "opencv-python",
            "pynput",
            "tabulate",
            "tqdm"
        ],
    )
