const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const bodyParser = require('body-parser');
const app = express();

// Middleware to parse JSON bodies
app.use(bodyParser.json());

// Enable CORS for all origins
app.use(cors());

// Endpoint for fetching JSON files
app.get('/fetch-file', async (req, res) => {
    const dataDir = path.join(__dirname, '..', 'data');
    const savedDir = path.join(__dirname, '..', 'saved');
    try {
        const dataFiles = await fs.promises.readdir(dataDir);
        const fileDirJsonFiles = dataFiles.filter(file => file.endsWith('_edit.json'));
        
        // Skip the /saved folder if it doesn't exist
        const savedFiles = fs.existsSync(savedDir) ? await fs.promises.readdir(savedDir): [];
        const savedDirJsonFiles = fs.existsSync(savedDir)
          ? savedFiles.filter((file) => file.endsWith("_edited.json"))
          : [];

        if (dataFiles.length === 0) {
            return res.status(404).send("No JSON files found.");
        }

        const allFileData = await Promise.all(fileDirJsonFiles.map(async (file) => {
            const filePath = path.join(dataDir, file);
            const fileContent = await fs.promises.readFile(filePath, 'utf-8');
            return {
                fileName: file.replace('_edit.json', ''),
                data: JSON.parse(fileContent)
            };
        }));

        // If there's a '_edited.json' file, take the data in there too to replot
        const allSavedData = savedFiles.length !== 0 ? await Promise.all(savedDirJsonFiles.map(async (file) => {
            const filePath = path.join(savedDir, file);
            const fileContent = await fs.promises.readFile(filePath, 'utf-8');
            return {
                fileName: file,
                data: JSON.parse(fileContent)
            };
        })) : null;

        // Extract unique segment options
        const segmentOptions = [...new Set(allFileData.flatMap(file => file.data.map(o => o.Segment.toString())))];

        res.json({ allFileData, allSavedData, segmentOptions });
    } catch (err) {
        console.error("Error fetching data: ", err);
        res.status(500).send("Error fetching data.");
    }
});

// Endpoint for saving JSON file
app.post('/saved', async (req, res) => {
    const { fileName, data } = req.body;
    const savedFilePath = path.join(__dirname, "..", "saved");
    // Checks if /saved exists, if not make it
    if (!fs.existsSync(savedFilePath)) {
        fs.mkdirSync(savedFilePath);
    }

    const filePath = path.join(__dirname, '..', 'saved', `${fileName}_edited.json`);

    try {
        const { addModeCoordinates, deleteModeCoordinates, unusableSegments } = data;

        // Check if all arrays are empty
        if (addModeCoordinates.length === 0 && deleteModeCoordinates.length === 0 && unusableSegments.length === 0) {
            // Delete the file instead of saving an empty file
            if (fs.existsSync(filePath)) {
                await fs.promises.unlink(filePath);
                console.log('File deleted due to empty data:', fileName);
                res.status(200).send('File deleted due to empty data.');
            } else {
                res.status(200).send('No file to delete.');
            }
            return;
        }

        // Combining both arrays into one for the save
        const combinedCoordinates = [...addModeCoordinates, ...deleteModeCoordinates, ...unusableSegments];

        // Convert JSON to string
        const jsonContent = JSON.stringify(combinedCoordinates, null, 2);

        // Write to file
        await fs.promises.writeFile(filePath, jsonContent);
        console.log('File saved successfully:', fileName);
        res.status(200).send('File saved successfully.');
    } catch (err) {
        console.error('Error saving file:', err);
        res.status(500).send('Error saving file.');
    }
});

// Start the server
const port = process.env.PORT || 3001;
app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});
