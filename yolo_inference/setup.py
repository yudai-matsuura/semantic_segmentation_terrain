from setuptools import find_packages, setup

package_name = 'yolo_inference'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='go2laptop',
    maintainer_email='matsuura.yudai.q5@dc.tohoku.ac.jp',
    description='YOLO8 inference node',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'inference_node = yolo_inference.inference_node:main',
        ],
    },
)
