from setuptools import find_packages, setup
from typing import List
HYPHEN_E_DOT = '-e .'
# Function to get the requirements from the requirements.txt file
def get_requirements(file_path: str) -> List[str]:  #List return type of requirements
    requirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        [req.replace('\,', "") for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements



setup(
    name='datascience-project',
    version='0.0.1',
    author='arjun',
    author_email='arjunrai2214@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)