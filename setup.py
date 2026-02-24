# setup.py
setup(
    # ... other parameters ...
    package_data={
        'your_package': ['config.yaml'],
    },
    include_package_data=True,
)