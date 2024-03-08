from setuptools import find_packages, setup
from typing import List


hyphen = '-e .'
def get_requirements(file_path:str) -> List[str]:
    
    '''
    this function will return the list of requirements
    '''
    requirements = []
    with open('requirements.txt') as file_obj:
        requirements = file_obj.readlines() 
        requirements = [req.replace("\n", "") for req in requirements]

        if hyphen in requirements:
            requirements.remove(hyphen)
    return requirements

setup(
name='Resturant Chatbot',
version='0.1',
author='Timmy',
author_email='timmyafolami8469@gmail.com',
packages=find_packages(), 
install_requires = get_requirements('requirements.txt')

)