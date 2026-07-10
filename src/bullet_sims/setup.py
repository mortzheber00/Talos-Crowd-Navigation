from glob import glob
from setuptools import find_packages, setup

package_name = 'bullet_sims'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (f'share/{package_name}/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='devel',
    maintainer_email='nicolas.guenther@tum.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            't2_temp = bullet_sims.t2_temp:main',
            't21= bullet_sims.t21:main',
            't22= bullet_sims.t22:main',
            't23= bullet_sims.t23:main',
            't31= bullet_sims.t31:main',
            'interactive_marker= bullet_sims.interactive_marker:main',
            't41= bullet_sims.t41:main',
            't42= bullet_sims.t42:main',
            't43= bullet_sims.t43:main',
            't51= bullet_sims.t51:main',
            't52= bullet_sims.t52:main',
            'walking= bullet_sims.walking:main',
            'stand_and_perceive= bullet_sims.stand_and_perceive:main',
        ],
    },
)
