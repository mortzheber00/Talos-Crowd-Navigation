from setuptools import find_packages, setup

package_name = 'drake_sim'

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
    maintainer='nick',
    maintainer_email='nicolas.guenther@tum.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'ex1= drake_sim.example_1_pydrake:main',
            'ex2= drake_sim.example_2_pydrake:main',
            'mpc_lipm= drake_sim.mpc_lipm:main',
            'ocp_lipm= drake_sim.ocp_lipm:main',


        ],
    },
)
