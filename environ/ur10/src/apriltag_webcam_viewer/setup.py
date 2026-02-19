from setuptools import setup
import os
from glob import glob

package_name = 'apriltag_webcam_viewer'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'cfg'), glob('cfg/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@todo.todo',
    description='AprilTag webcam TF to RViz',
    license='MIT',
    tests_require=['pytest'],
    entry_points={'console_scripts': [
        'tag_tf_chain_node = apriltag_webcam_viewer.tag_tf_chain_node:main',
        'tag_world_chain_node = apriltag_webcam_viewer.tag_tf_chain_node:main',
    ]},
)
