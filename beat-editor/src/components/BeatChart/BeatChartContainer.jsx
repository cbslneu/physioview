import { useState, useEffect, useCallback } from "react";
import BeatChart from "./BeatChart";
import {
  EDIT_TYPE_ADD,
  EDIT_TYPE_DELETE,
  EDIT_TYPE_UNUSABLE,
} from "../../constants/constants";
import axios from "axios";

function BeatChartContainer() {
  const [fileData, setFileData] = useState([]);
  const [segmentOptions, setSegmentOptions] = useState([]);
  const [fileName, setFileName] = useState("");
  const [addBeatCoordinates, setAddBeatCoordinates] = useState([]);
  const [deleteBeatCoordinates, setDeleteBeatCoordinates] = useState([]);
  const [unusableBeats, setUnusableBeats] = useState([]);

  const fetchFile = useCallback(async () => {
    try {
      const response = await axios.get("http://localhost:3001/fetch-file");
      const { allFileData, allSavedData, segmentOptions } = response.data;

      if (!allFileData) throw new Error("No file data found.");

      let addBeats = [];
      let deleteBeats = [];
      let unusableBeats = [];

      if (allSavedData) {
        const jsonData = allSavedData[0].data;

        addBeats = jsonData.filter((beat) => beat.editType === EDIT_TYPE_ADD);
        deleteBeats = jsonData.filter(
          (beat) => beat.editType === EDIT_TYPE_DELETE
        );
        unusableBeats = jsonData.filter(
          (beat) => beat.editType === EDIT_TYPE_UNUSABLE
        );
      }

      setFileData(allFileData[0].data);
      setFileName(allFileData[0].fileName);
      setSegmentOptions(segmentOptions);
      setAddBeatCoordinates(addBeats);
      setDeleteBeatCoordinates(deleteBeats);
      setUnusableBeats(unusableBeats);
    } catch (err) {
      throw new Error(`Error fetching JSON file: ${err.message}`);
    }
  }, []);

  useEffect(() => {
    fetchFile();
  }, [fetchFile]);

  return (
    <div className="plot-beat-segment">
      <div className="chart-buttons"></div>
      <BeatChart
        fileData={fileData}
        fileName={fileName}
        segmentOptions={segmentOptions}
        addBeats={addBeatCoordinates}
        deleteBeats={deleteBeatCoordinates}
        unusableBeats={unusableBeats}
      />
    </div>
  );
}

export default BeatChartContainer;
