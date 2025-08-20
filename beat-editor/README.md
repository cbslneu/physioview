# Beat Editor 📈

**❗️Please ensure Node (v22.x.x +) is installed on your system before 
proceeding with installation below.**
<br>
<br> *Check if Node is installed on your machine:*
```
node --version
```
If an error occurs, please refer to this link to install Node on your machine: https://nodejs.org/en/download/package-manager

### Installation
1. Go to the `beat-editor` directory.
```
cd beat-editor
```
2. Install the required modules for the Beat Editor.
```
npm install
```
3. Go to the `beat-editor/server` subdirectory.
```
cd server
```
4. Install the required modules for the Beat Editor's backend.
```
npm install
```

### Startup
1. Navigate to the `beat-editor/server` subdirectory.
```
cd server
```
2. Run the following line to start the backend.
```
npm start
```
3. Open another terminal tab or window, navigate back to the root 
   `beat-editor` directory, and run:
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
docker compose up
```
3. To stop the services, press `Ctrl`+`C` in your terminal or run:
```
docker compose down
```