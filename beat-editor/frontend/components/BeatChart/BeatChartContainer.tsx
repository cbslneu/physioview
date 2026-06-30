import { useState, useEffect, useCallback } from "react";
import axios from "axios";

import { Beat, SavedBeat } from "../../types/types";

import BeatChart from "./BeatChart";

function BeatChartContainer() {
  const [fileData, setFileData] = useState<Beat[]>([]);
  const [segmentOptions, setSegmentOptions] = useState<string[]>([]);
  const [fileName, setFileName] = useState("");
  const [allEdits, setAllEdits] = useState<SavedBeat[]>([]);

  const fetchFile = useCallback(async () => {
    try {
      const response = await axios.get("http://localhost:3001/fetch-file");
      const { allFileData, allSavedData, segmentOptions } = response.data;

      if (!allFileData) throw new Error("No file data found.");

      if (allSavedData && allSavedData.length > 0) {
        const jsonData = allSavedData[0].data;
        setAllEdits(jsonData);
      }

      setFileData(allFileData[0].data);
      setFileName(allFileData[0].fileName);
      setSegmentOptions(segmentOptions);
    } catch (err: any) {
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
        allEdits={allEdits}
        onRefresh={fetchFile}
      />
    </div>
  );
}

export default BeatChartContainer;
