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
    docker version
```

2. Run test contianer
```
    docker run --rm hello-world
```

## Create files
1. In the root of your project, create a file named `.dockerignore`

Note: If you use environment variables via a `.env` file, do not hardcode them into the image. You will pass them at runtime
2. Create a file named `Dockerfile` (no extension) in your project root. This file dictates the environment setup.

## Build the Docker Image
```
    docker build -t image-name:latest .
```
where the `-t` flag tags the image with a name, and the `.` specifies the current directory as the build context.

Watch the output. Docker will pull the base image, create the `/app` directory, install your dependencies from `requirements.txt`, and package your code.

## Test the Container Locally
```
    docker run --rm -it my-python-app:latest
```

- `--rm`: Automatically removes the container instance after it stops.
- `-it`: Runs the container interactively (useful if your script requires terminal input or if you need to see real-time output).

## OPTION A: Export the Image as tar file
Run the docker save command to create an archive of your built image
```
    docker save -o my-python-app.tar my-python-app:latest
```
**TODO**: choose if to push `my-python-app.tar` file or to send it to people

To Load image on another device
1. Load the image: `docker load -i my-python-app.tar`
2. Run the application: `docker run --rm my-python-app:latest`

## OPTION B: Push image on Docker Hub
Enter your Docker Hub username and password (or access token) when prompted.
```
    docker login
```

Tag image
```
    docker tag jax-hypernetworks:latest your_dockerhub_username/jax-hypernetworks:latest
```

Push image
```
    docker push your_dockerhub_username/jax-hypernetworks:latest
```

Now external user just needs to do:

```
    docker run --rm -v "$(pwd):/app" your_dockerhub_username/jax-hypernetworks:latest python main.py
```

**NOTE**: if you run with docker, folders created belongs to root. When you later try to run it locally, you do not have permissions to create subfolders inside it (local user is different from Docker root user). Run following command to gain ownership.
```
    sudo chown -R $(whoami) results
```