from setuptools import setup, find_packages

setup(
    name='LearnableActivationFunction',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'Pillow',
        'numpy',
        'scikit-learn'
    ],
    entry_points={
        'console_scripts': [
            'train_activation_model=example.simple_model.train:main'
        ]
    }
)
