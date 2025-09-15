import { useEffect, useState, useRef, useMemo } from "react";
import _ from "lodash";
import Highcharts from "highcharts";
import HighchartsMore from "highcharts/highcharts-more";
import HighchartsReact from "highcharts-react-official";
import mouseWheelZoom from "highcharts/modules/mouse-wheel-zoom";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import "@fortawesome/fontawesome-free/css/all.min.css";

import createChartOptions from "../../utils/CreateChartOptions";
import { EDIT_TYPE_ADD, EDIT_TYPE_DELETE } from "../../constants/constants";
import KeyboardShortcuts from '../KeyboardShortcuts/KeyboardShortcuts';
import useMarkingUnusableMode from "../../hooks/useMarkingUnusableMode";
import useKeyboardShortcuts from "../../utils/key-input-utils";
import { Beat, SavedBeat, ChartCoordinates, ChartClickEvent, SegmentObj } from "../../types/types";
import SaveButton from "../SaveButton/SaveButton";

Highcharts.SVGRenderer.prototype.symbols.cross = function (
  x: number,
  y: number,
  w: number,
  h: number
) {
  return ["M", x, y, "L", x + w, y + h, "M", x + w, y, "L", x, y + h, "z"];
};

HighchartsMore(Highcharts);
mouseWheelZoom(Highcharts);

interface BeatChartProps {
  fileData: Beat[];
  fileName: string;
  segmentOptions: string[];
  addBeats: SavedBeat[];
  deleteBeats: SavedBeat[];
  unusableBeats: SegmentObj[];
}

interface HasDataTypeParams {
  fileData: Beat[];
  data: string;
}

interface transformCoordinatesParams {
  data: Beat[];
  xAxisLabel?: string;
  yAxisLabel?: string;
}

const X_AXIS_KEYS = ["Timestamp", "Sample"];
const Y_AXIS_KEYS = ["Filtered", "Signal"];

const BeatChart = ({
  fileData,
  fileName,
  segmentOptions,
  addBeats,
  deleteBeats,
  unusableBeats,
}: BeatChartProps) => {
  const [chartOptions, setChartOptions] = useState<Highcharts.Options | null>(null);
  const [cardiacData, setCardiacData] = useState<ChartCoordinates[]>([]);
  const [beatData, setBeatData] = useState<ChartCoordinates[]>([]);
  const [beatArtifactData, setBeatArtifactData] = useState<ChartCoordinates[]>([]);
  const [isAddMode, setIsAddMode] = useState(false);
  const [isDeleteMode, setIsDeleteMode] = useState(false);
  const [isPanning, setIsPanning] = useState(false);
  const [isMarkingUnusableMode, setIsMarkingUnusableMode] = useState(false);
  const [addModeCoordinates, setAddModeCoordinates] = useState<SavedBeat[]>([]);
  const [deleteModeCoordinates, setDeleteModeCoordinates] = useState<SavedBeat[]>([]);
  const [unusableSegments, setUnusableSegments] = useState<SegmentObj[]>([]);
  const [selectedSegment, setSelectedSegment] = useState("1");

  const chartRef = useRef<HighchartsReact.RefObject>(null);
  const dragStartRef = useRef(null);
  const dragPlotBandRef = useRef<Highcharts.XAxisPlotBandsOptions | null>(null);
  const isDraggingRef = useRef(false); // Tracks drag during panning
  const lastValidDragEnd = useRef(null);
  const segmentBoundaries = useMemo(() => {
    return {
      from: cardiacData[0],
      to: cardiacData[cardiacData.length - 1],
    };
  }, [cardiacData]);

  useEffect(() => {
    const dataTypeX = X_AXIS_KEYS.find((data) =>
      hasDataType({ fileData, data })
    );
    const dataTypeY = Y_AXIS_KEYS.find((data) =>
      hasDataType({ fileData, data })
    );

    // Filter the data by the selected segment from the dropdown
    const segmentFilteredData = selectedSegment
      ? fileData.filter((data) => data.Segment == selectedSegment)
      : fileData;
    const beatAnnotatedData = segmentFilteredData.filter(
      (data) => data.Beat === 1
    );
    const correctedAnnotatedData = segmentFilteredData.filter(
      (data) => data.Corrected === 1
    );
    const artifactData = segmentFilteredData.filter(
      (data) => data.Artifact === 1
    );

    const initCardiacData = transformCoordinates({
      data: segmentFilteredData,
      xAxisLabel: dataTypeX,
      yAxisLabel: dataTypeY,
    });
    const initArtifacts = transformCoordinates({
      data: artifactData,
      xAxisLabel: dataTypeX,
      yAxisLabel: dataTypeY,
    });

    const initBeats =
      correctedAnnotatedData.length > 0
        ? correctedAnnotatedData.map((o) => ({
            x: (o.Timestamp || o.Sample) as number,
            y: o.Signal,
          }))
        : beatAnnotatedData.map((o) => ({
            x: (o.Timestamp || o.Sample) as number,
            y: o.Signal,
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
    });

    setChartOptions(chartParams);
    setCardiacData(initCardiacData);
    setBeatData(initBeats);
    setBeatArtifactData(initArtifacts);

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

  const handleChartClick = (event: ChartClickEvent) => {
    // Prevents coordinates from plotting when hitting `Reset Zoom`
    if (
      isPanning ||
      (event.target &&
        event.target instanceof Element &&
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
    if (isPanning && isAddMode && isSignal) {
      return;
    }
    // In Delete Mode, prevent deleting points that don't exist in beatData or are artifacts
    if (
      isPanning &&
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

  function hasDataType({ fileData, data }: HasDataTypeParams) {
    return fileData.some((o) => o.hasOwnProperty(data));
  }

  function transformCoordinates({
    data,
    xAxisLabel,
    yAxisLabel,
  }: transformCoordinatesParams) {
    return data.map((item) => ({
      x: item[xAxisLabel as keyof Beat] as number,
      y: item[yAxisLabel as keyof Beat] as number,
    }));
  }

  function undoLastCoordinate() {
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
  }

  function toggleAddMode() {
    resetInteractionState();
    setIsAddMode((prev) => !prev);
    setIsDeleteMode(false);
    setIsMarkingUnusableMode(false);
  }

  function toggleDeleteMode() {
    resetInteractionState();
    setIsAddMode(false);
    setIsDeleteMode((prev) => !prev);
    setIsMarkingUnusableMode(false);
  }

  function toggleMarkUnusableMode() {
    resetInteractionState();
    setIsMarkingUnusableMode((prev) => !prev);
    setIsAddMode(false);
    setIsDeleteMode(false);
  }

  // Reset all drag and interaction states when toggling modes
  function resetInteractionState() {
    dragStartRef.current = null;
    isDraggingRef.current = false;
    lastValidDragEnd.current = null;
    setIsPanning(false);
  }

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

  useMarkingUnusableMode({
    isMarkingUnusableMode,
    chartRef,
    setUnusableSegments,
    selectedSegment,
    dragStartRef,
    isDraggingRef,
    dragPlotBandRef,
    lastValidDragEnd,
    segmentBoundaries,
  });

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
      </div>

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
