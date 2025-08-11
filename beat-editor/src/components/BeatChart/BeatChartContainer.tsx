import { useState, useEffect, useCallback } from "react";
import axios from "axios";

import {
  EDIT_TYPE_ADD,
  EDIT_TYPE_DELETE,
  EDIT_TYPE_UNUSABLE,
} from "../../constants/constants";
import { Beat, SavedBeat, SegmentObj } from "../../types/types";

import BeatChart from "./BeatChart";



function BeatChartContainer() {
  const [fileData, setFileData] = useState<Beat[]>([]);
  const [segmentOptions, setSegmentOptions] = useState<string[]>([]);
  const [fileName, setFileName] = useState("");
  const [addBeatCoordinates, setAddBeatCoordinates] = useState<SavedBeat[]>([]);
  const [deleteBeatCoordinates, setDeleteBeatCoordinates] = useState<SavedBeat[]>([]);
  const [unusableBeats, setUnusableBeats] = useState<SegmentObj[]>([]);

  const fetchFile = useCallback(async () => {
    try {
      const response = await axios.get("http://localhost:3001/fetch-file");
      const { allFileData, allSavedData, segmentOptions } = response.data;

      if (!allFileData) throw new Error("No file data found.");

      if (allSavedData) {
        const jsonData = allSavedData[0].data;

        setAddBeatCoordinates(jsonData.filter((beat: SavedBeat) => beat.editType === EDIT_TYPE_ADD));
        setDeleteBeatCoordinates(jsonData.filter((beat: SavedBeat) => beat.editType === EDIT_TYPE_DELETE));
        setUnusableBeats(jsonData.filter((beat: SegmentObj) => beat.editType === EDIT_TYPE_UNUSABLE));
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
        addBeats={addBeatCoordinates}
        deleteBeats={deleteBeatCoordinates}
        unusableBeats={unusableBeats}
      />
    </div>
  );
}

export default BeatChartContainer;
