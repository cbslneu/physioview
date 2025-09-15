# Beat Editor 📈

**❗️Please ensure Node (v22.x.x +) is installed on your system before 
proceeding with installation below.**
<br>
<br> *Check if Node is installed on your machine:*
```
node --version
```
If an error occurs, please refer to this link to install Node: https://nodejs.org/en/download/package-manager

### Installation
1. Go to the `beat-editor/frontend` directory.
```
cd beat-editor/frontend
```
2. Install the required modules for the Beat Editor.
```
npm install
```
3. Go to the `beat-editor/backend` subdirectory.
```
cd ../backend
```
4. Install the required modules for the Beat Editor's backend.
```
npm install
```

### Startup
1. Within the `beat-editor/backend` subdirectory, run:
```
npm start
```
2. Open another Terminal tab or window, navigate back to the 
   `beat-editor/frontend` directory, and run:
```
npm run dev
```

### Using Docker (For Developers) 🐳
If you are developing the Beat Editor and want an isolated environment 
without installing Node and `npm` locally, you can use Docker.

1. Make sure [Docker Desktop](https://www.docker.com/products/docker-desktop/) 
   is installed on your system.

2. From the `beat-editor` directory, run:
```
docker compose up -d
```
  If the Docker services start successfully, you should see output like:
```
[+] Running 4/4
 ✔ backend                           Built                                                                                                                                                                                                                                                                                                                  0.0s 
 ✔ frontend                          Built                                                                                                                                                                                                                                                                                                                  0.0s 
 ✔ Container beat-editor-backend-1   Started                                                                                                                                                                                                                                                                                                                0.1s 
 ✔ Container beat-editor-frontend-1  Started  
```
3. To stop the services, press `Ctrl`+`C` in your terminal or run:
```
docker compose down
```