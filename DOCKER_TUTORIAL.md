# DOCKER TUTORIAL: CREATE DOCKER IMAGE TO requirements.txt

## Installing and Enabling the WSL 2 Based Engine
1. Install Docker Desktop for Windows
2. Ensure the checkbox for *Use the WSL 2 based engine* is selected (Setting -> General -> Use the WSL 2 based engine)
3. Settings -> Resources -> WSL integration -> Check the box for *Enable integration with my default WSL distro*
4. Below that, explicitly toggle the switch for your installed distribution (e.g., Ubuntu) to the ON position.
5. Apply & restart

## Check installation
1. Execute the version command on Ubuntu WSL terminal to verify the CLI can communicate with the daemon
```
    docker versions
```

2. Run test contianer
```
    docker run --rm hello-world
```

## Create files
1. In the root of your project, create a file named `.dockerignore` and add the following:
```
    venv/
    env/
    __pycache__/
    *.pyc
    .git/
    .env
```
Note: If you use environment variables via a `.env` file, do not hardcode them into the image. You will pass them at runtime
2. Create a file named `Dockerfile` (no extension) in your project root. This file dictates the environment setup.
**NOTE:** last row of `Dockerfile` defines the default command to run, modify if needed
```
    CMD ["python", "main.py"]
```
**TODO**: decide what to do for last row at the end of the project

