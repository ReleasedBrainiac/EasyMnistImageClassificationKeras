"""
Resource: 

1. https://stackoverflow.com/questions/44638687/how-to-import-functions-from-a-submodule-in-a-python-egg
2. https://forums.databricks.com/questions/10206/importing-a-module-from-a-custom-python-egg.html
3. https://stackoverflow.com/questions/2051192/what-is-a-python-egg
4. https://forums.databricks.com/questions/8203/how-can-i-use-custom-modules-and-packages-i-write.html
5. https://docs.microsoft.com/en-us/azure/databricks/libraries/cluster-libraries#cluster-installed-library


"""


# Egg-Files is python are like libs in Java or C-Languages

#Example: To create an .egg file (like *.zip or *.jar) for a directory say Dataset which itself may have several python scripts, do the following step:

from setuptools import setup, find_packages
setup(
    name = "Dataset",
    version = "v1.3",
    packages = find_packages(exclude=('tests', 'docs')),
    test_suite='nose.collector',
    tests_require=['nose'],
    zip_safe=False
    ) 
# name is the folder root
# version is your version of the *.egg
# find_packages looks for all packages under the current folder

# To Execute the *.egg creation do the following in terminal

# $ python setup.py bdist_egg

# Finaly you got the *.egg under new folder dist

# To import your EGG-File as lib into Databricks just go to compute > choose your running cluster > libraries > Install New > Python Egg