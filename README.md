# PACSproject TODO for replication

1. Install Docker Desktop and open the Docker Desktop. Check if Docker is ready with:
```
    docker --version
```
2. Clone repository
3. Create dataset/ folder (TODO: need to check if strictly necessary)
4. Make .sh file executable:
```
    chmod +x run_toy.sh scripts/run_turbulence.sh
```
5. Run the code, either toy or turbulence with:
```
    ./scripts/run_toy.sh
```
or
```
    ./scripts/run_turbulence.sh
```

**BE AWARE**: User is cloning entire repository: they see all library folders in their local machine. However, bash script files use folders inside Docker container, which relies on its own baked-in copies of those library folders.
