from setuptools import setup, find_packages

setup(
    name="mr_qwen3tts",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "websockets>=12.0",
        "python-dotenv",
    ],
)
