
import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='lpips-jax',  
     version='0.0.1',
     author="Wilson Yan",
     author_email="wilson1.yan@berkeley.edu",
     description="LPIPS Similarity metric for Jax",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/wilson1yan/lpips-jax",
     packages=['lpips-jax'],
 )
