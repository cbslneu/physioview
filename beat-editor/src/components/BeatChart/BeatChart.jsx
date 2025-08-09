import { useEffect, useState, useRef } from "react";
import Highcharts from "highcharts";
import HighchartsMore from "highcharts/highcharts-more";
import HighchartsReact from "highcharts-react-official";
import _ from "lodash";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import "@fortawesome/fontawesome-free/css/all.min.css";
import KeyboardShortcuts from '../KeyboardShortcuts/KeyboardShortcuts';

import SaveButton from "../SaveButton/SaveButton";
import createChartOptions from "../../utils/CreateChartOptions";
import { EDIT_TYPE_ADD, EDIT_TYPE_DELETE } from "../../constants/constants";
import useChartZoom from "../../hooks/useChartZoom";
import useMarkingUnusableMode from "../../hooks/useMarkingUnusableMode";
import useKeyboardShortcuts from "../../utils/key-input-utils";

Highcharts.SVGRenderer.prototype.symbols.cross = function (x, y, w, h) {
  return ["M", x, y, "L", x + w, y + h, "M", x + w, y, "L", x, y + h, "z"];
};

HighchartsMore(Highcharts);

const BeatChart = ({
  fileData,
  fileName,
  segmentOptions,
  addBeats = [],
  deleteBeats = [],
  unusableBeats = [],
}) => {
  const [chartOptions, setChartOptions] = useState(null);
  const [cardiacData, setCardiacData] = useState([]);
  const [beatData, setBeatData] = useState([]);
  const [beatArtifactData, setBeatArtifactData] = useState([]);
  const [isAddMode, setIsAddMode] = useState(false);
  const [isDeleteMode, setIsDeleteMode] = useState(false);
  const [isMarkingUnusableMode, setIsMarkingUnusableMode] = useState(false);
  const [addModeCoordinates, setAddModeCoordinates] = useState([]);
  const [deleteModeCoordinates, setDeleteModeCoordinates] = useState([]);
  const [unusableSegments, setUnusableSegments] = useState([]);
  const [selectedSegment, setSelectedSegment] = useState("1");

  const chartRef = useRef(null);
  const dragStartRef = useRef(null);
  const dragPlotBandRef = useRef(null);
  const isDraggingRef = useRef(false); // Tracks drag during panning
  const isPanningRef = useRef(false); // Tracks if the user is panning the chart
  const lastValidDragEnd = useRef(null);

  let segmentBoundaries = {
    from: null,
    to: null,
  };

  useEffect(() => {
    if (fileData) {
      const xAxisKeys = ["Timestamp", "Sample"];
      const yAxisKeys = ["Filtered", "Signal"];
      const checkDataType = (fileData, data) =>
        fileData.some((o) => o.hasOwnProperty(data));

      const dataTypeX = xAxisKeys.filter(
        (data) => checkDataType(fileData, data) === true
      );
      const dataTypeY = yAxisKeys.filter(
        (data) => checkDataType(fileData, data) === true
      );

      // Filter the data by the selected segment from the dropdown
      const segmentFilteredData = selectedSegment
        ? _.filter(fileData, (o) => o.Segment.toString() === selectedSegment)
        : fileData;
      const beatAnnotatedData = _.filter(
        segmentFilteredData,
        (o) => o.Beat === 1
      );
      const correctedAnnotatedData = _.filter(
        segmentFilteredData,
        (o) => o.Corrected === 1
      );
      const artifactData = _.filter(
        segmentFilteredData,
        (o) => o.Artifact === 1
      );

      const xAxisData = segmentFilteredData.map((o) => o[dataTypeX]);
      const yAxisData = segmentFilteredData.map((o) => o[dataTypeY]);

      const artifactX = artifactData.map((o) => o[dataTypeX]);
      const artifactY = artifactData.map((o) => o[dataTypeY]);

      const initCardiacData = xAxisData.map((dataType, index) => ({
        x: dataType,
        y: yAxisData[index],
      }));

      // Checks if correctedAnnotatedData or beatAnnotatedData has the key 'Filtered'
      const hasFilteredKey = (data) => "Filtered" in data;

      const initBeats =
        correctedAnnotatedData.length > 0
          ? correctedAnnotatedData.map((o) => ({
              x: o.Timestamp,
              y: hasFilteredKey(o) ? o.Filtered : o.Signal,
            }))
          : beatAnnotatedData.map((o) => ({
              x: o.Timestamp,
              y: hasFilteredKey(o) ? o.Filtered : o.Signal,
            }));

      const initArtifacts = artifactX.map((artifactX, index) => ({
        x: artifactX,
        y: artifactY[index],
      }));

      const chartParams = createChartOptions({
        xAxisData: segmentFilteredData.map((o) => o.Timestamp),
        initCardiacData,
        initBeats,
        initArtifacts,
        addModeCoordinates,
        deleteModeCoordinates,
        selectedSegment,
        unusableSegments,
        isAddMode,
        isDeleteMode,
        isMarkingUnusableMode,
        handleChartClick,
        dataTypeX,
        isPanningRef,
      });

      setChartOptions(chartParams);

      setCardiacData(initCardiacData);
      setBeatData(initBeats);
      setBeatArtifactData(initArtifacts);
    }
  }, [
    fileData,
    isAddMode,
    isDeleteMode,
    addModeCoordinates,
    deleteModeCoordinates,
    selectedSegment,
    unusableSegments,
    isMarkingUnusableMode,
  ]);

  segmentBoundaries.from = cardiacData[0];
  segmentBoundaries.to = cardiacData[cardiacData.length - 1];

  const handleChartClick = (event) => {
    // Prevents coordinates from plotting when hitting `Reset Zoom`
    if (
      isPanningRef.current ||
      (event.target &&
        (event.target.classList.contains("highcharts-button-box") ||
          event.target.innerHTML === "Reset zoom"))
    ) {
      return; // Ignore clicks on Reset Zoom
    }
    const newX = !_.isUndefined(event.point)
      ? event.point.x
      : event.xAxis[0].value;
    const newY = !_.isUndefined(event.point)
      ? event.point.y
      : event.yAxis[0].value;

    // Check if the point already exists in cardiacData (for Add Mode) or beatData (for Delete Mode)
    const isSignal = cardiacData.some(
      (point) => point.x === newX && point.y === newY
    );
    const isBeatCoordinate = beatData.some(
      (point) => point.x === newX && point.y === newY
    );
    const isArtifactCoordinate = beatArtifactData.some(
      (point) => point.x === newX && point.y === newY
    );

    // In Add Mode, prevent adding points that already exist in cardiacData
    if (isPanningRef.current && isAddMode && isSignal) {
      return;
    }
    // In Delete Mode, prevent deleting points that don't exist in beatData or are artifacts
    if (
      isPanningRef.current &&
      isDeleteMode &&
      !isBeatCoordinate &&
      !isArtifactCoordinate
    ) {
      return;
    }

    const updatedCardiacData = [...cardiacData, { x: newX, y: newY }];
    const updatedBeatData = [...beatData];
    const updateArtifactData = [...beatArtifactData];

    if (isAddMode) {
      setAddModeCoordinates((prevCoordinates) => {
        const updateCoordinates = [
          ...prevCoordinates,
          {
            x: newX,
            y: newY,
            segment: selectedSegment,
            editType: EDIT_TYPE_ADD,
          },
        ];
        return updateCoordinates;
      });
    } else if (isDeleteMode) {
      if (isBeatCoordinate || isArtifactCoordinate) {
        setDeleteModeCoordinates((prevCoordinates) => {
          const updateCoordinates = [
            ...prevCoordinates,
            {
              x: newX,
              y: newY,
              segment: selectedSegment,
              editType: EDIT_TYPE_DELETE,
            },
          ];
          return updateCoordinates;
        });
      } else {
        toast.error("This is not a beat");
      }
    }

    setCardiacData(updatedCardiacData);
    setBeatData(updatedBeatData);
    setBeatArtifactData(updateArtifactData);
  };

  const undoLastCoordinate = () => {
    if (isAddMode && addModeCoordinates.length > 0) {
      setAddModeCoordinates((prevCoordinates) => {
        const updatedCoordinates = prevCoordinates.slice(0, -1);
        return updatedCoordinates;
      });
    } else if (isDeleteMode && deleteModeCoordinates.length > 0) {
      setDeleteModeCoordinates((prevCoordinates) => {
        const updatedCoordinates = prevCoordinates.slice(0, -1);
        return updatedCoordinates;
      });
    } else if (isMarkingUnusableMode && unusableSegments.length > 0) {
      setUnusableSegments((prevCoordinates) => {
        const updateCoordinates = prevCoordinates.slice(0, -1);
        return updateCoordinates;
      });
    }
  };

  const toggleAddMode = () => {
    resetInteractionState();
    setIsAddMode((prev) => !prev);
    setIsDeleteMode(false);
    setIsMarkingUnusableMode(false);
  };

  const toggleDeleteMode = () => {
    resetInteractionState();
    setIsAddMode(false);
    setIsDeleteMode((prev) => !prev);
    setIsMarkingUnusableMode(false);
  };

  const toggleMarkUnusableMode = () => {
    resetInteractionState();
    setIsMarkingUnusableMode((prev) => !prev);
    setIsAddMode(false);
    setIsDeleteMode(false);
  };

  // Reset all drag and interaction states when toggling modes
  const resetInteractionState = () => {
    dragStartRef.current = null;
    isDraggingRef.current = false;
    isPanningRef.current = false;
    lastValidDragEnd.current = null;
  };

  useEffect(() => {
    setAddModeCoordinates(addBeats);
    setDeleteModeCoordinates(deleteBeats);
    setUnusableSegments(unusableBeats);
  }, [addBeats, deleteBeats, unusableBeats]);

  useKeyboardShortcuts({
    toggleAddMode,
    toggleDeleteMode,
    toggleMarkUnusableMode,
    undoLastCoordinate,
  });

  useMarkingUnusableMode(
    isMarkingUnusableMode,
    chartRef,
    setUnusableSegments,
    selectedSegment,
    dragStartRef,
    isDraggingRef,
    dragPlotBandRef,
    lastValidDragEnd,
    segmentBoundaries
  );

  useChartZoom(chartRef, chartOptions);

  return (
    <div className="beat-chart-container">
      <div className="chart-buttons-wrapper">
        <div className="chart-buttons">
          <select
            className="dropdown"
            value={selectedSegment}
            onChange={(e) => {
              setSelectedSegment(e.target.value);
              resetInteractionState();

              if (chartRef.current && chartRef.current.chart) {
                if (isAddMode || isDeleteMode) {
                  setIsAddMode(false);
                  setIsDeleteMode(false);
                  setIsMarkingUnusableMode(false);
                }
                chartRef.current.chart.zoomOut();
              }
            }}
          >
            <option value="" disabled>
              Segment
            </option>
            {segmentOptions.map((segment) => (
              <option key={segment} value={segment}>
                {segment}
              </option>
            ))}
          </select>
          <button
            className={`${isAddMode ? "add-beat-active" : ""}`}
            onClick={toggleAddMode}
          >
            <i className="fa-solid fa-plus"></i>Add Beat
          </button>
          <button
            className={`${isDeleteMode ? "delete-beat-active" : ""}`}
            onClick={toggleDeleteMode}
          >
            <i className="fa-solid fa-minus"></i>Delete Beat
          </button>
          <button
            className={`${isMarkingUnusableMode ? "mark-unusable-active" : ""}`}
            onClick={toggleMarkUnusableMode}
          >
            <i className="fa-solid fa-marker" />
            Mark Unusable
          </button>
          <button className="undo-beat-entry" onClick={undoLastCoordinate}>
            <i className="fa-solid fa-rotate-left"></i>Undo
          </button>
          <SaveButton
            fileName={fileName}
            addModeCoordinates={addModeCoordinates}
            deleteModeCoordinates={deleteModeCoordinates}
            unusableSegments={unusableSegments}
          />
        
        <KeyboardShortcuts />
        </div>
        <div className="chart-info">
          <h3>Current Segment: {selectedSegment}</h3>
          <h3>Number of Beats: {beatData.length}</h3>
        </div>
      </div>

      <h3 className="chart-name">{fileName}</h3>

      {chartOptions && (
        <HighchartsReact
          highcharts={Highcharts}
          options={chartOptions}
          ref={chartRef}
        />
      )}

      <ToastContainer />
    </div>
  );
};

export default BeatChart;
