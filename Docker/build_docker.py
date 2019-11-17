import os
import sys
import subprocess
from subprocess import call

out = subprocess.check_output("git diff --name-only HEAD~1 HEAD".split())
out = out.decode('ascii').split('\n')[:-1]
dockerfiles = [files for files in out if "Dockerfile" in files]

DOCKER_REPO = "sachinruk/"
DOCKER_IMAGE = DOCKER_REPO + "profetorch"
TAG = str(sys.argv[1])

def call_cmd(cmd):
    ret_code = call(cmd.split())
    if ret_code != 0:
        print("The following command failed: " + cmd)
        sys.exit(ret_code)
        
if dockerfiles:
    tagged_version = DOCKER_IMAGE + ":" + TAG
    call_cmd("docker build -t " + DOCKER_IMAGE + " ./Docker/")
    call_cmd("docker tag " + DOCKER_IMAGE + " " + tagged_version)
    call_cmd("docker push " + DOCKER_IMAGE)
    print("="*50)