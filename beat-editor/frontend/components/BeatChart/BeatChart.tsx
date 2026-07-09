import { useEffect, useState, useRef, useMemo } from "react";
import _ from "lodash";
import Highcharts from "highcharts";
import HighchartsMore from "highcharts/highcharts-more";
import HighchartsReact from "highcharts-react-official";
import mouseWheelZoom from "highcharts/modules/mouse-wheel-zoom";
import { ToastContainer, toast } from "react-toastify";

import createChartOptions from "../../utils/CreateChartOptions";
import {
  EDIT_TYPE_ADD,
  EDIT_TYPE_DELETE,
  EDIT_TYPE_UNUSABLE,
  EDIT_TYPE_VALID,
} from "../../constants/constants";
import KeyboardShortcuts from "../KeyboardShortcuts/KeyboardShortcuts";
import useMarkingUnusableMode from "../../hooks/useMarkingUnusableMode";
import useKeyboardShortcuts from "../../utils/key-input-utils";
import {
  Beat,
  SavedBeat,
  ChartCoordinates,
  ChartClickEvent,
  SegmentObj,
} from "../../types/types";
import SaveButton from "../SaveButton/SaveButton";
import ExportButton from "../ExportButton/ExportButton";
import ImportEditsButton from "../ImportEditsButton/ImportEditsButton";
import LabelToggle from "../LabelToggle/LabelToggle";
import ToolbarDivider from "../ToolbarDivider/ToolbarDivider";

Highcharts.SVGRenderer.prototype.symbols.cross = function (
  x: number,
  y: number,
  w: number,
  h: number,
) {
  return ["M", x, y, "L", x + w, y + h, "M", x + w, y, "L", x, y + h, "z"];
};

HighchartsMore(Highcharts);
mouseWheelZoom(Highcharts);

interface BeatChartProps {
  fileData: Beat[];
  fileName: string;
  segmentOptions: string[];
  allEdits: SavedBeat[];
  onRefresh: () => void;
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
  allEdits,
  onRefresh,
}: BeatChartProps) => {
  const [chartOptions, setChartOptions] = useState<Highcharts.Options | null>(
    null,
  );
  const [cardiacData, setCardiacData] = useState<ChartCoordinates[]>([]);
  const [beatData, setBeatData] = useState<ChartCoordinates[]>([]);
  const [beatArtifactData, setBeatArtifactData] = useState<ChartCoordinates[]>(
    [],
  );
  const [isAddMode, setIsAddMode] = useState(false);
  const [isDeleteMode, setIsDeleteMode] = useState(false);
  const [isPanning, setIsPanning] = useState(false);
  const [isMarkingUnusableMode, setIsMarkingUnusableMode] = useState(false);
  const [isRemoveEditMode, setIsRemoveEditMode] = useState(false);
  const [isMarkValidMode, setIsMarkValidMode] = useState(false);
  const [isLabelOn, setIsLabelOn] = useState(true);
  const [allUserEdits, setAllUserEdits] = useState<(SavedBeat | SegmentObj)[]>(
    [],
  );
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
    setAllUserEdits(allEdits);
  }, [allEdits]);

  useEffect(() => {
    const dataTypeX = X_AXIS_KEYS.find((data) =>
      hasDataType({ fileData, data }),
    );
    const dataTypeY = Y_AXIS_KEYS.find((data) =>
      hasDataType({ fileData, data }),
    );

    // Filter the data by the selected segment from the dropdown
    const segmentFilteredData = selectedSegment
      ? fileData.filter((data) => data.Segment == selectedSegment)
      : fileData;
    const beatAnnotatedData = segmentFilteredData.filter(
      (data) => data.Beat === 1,
    );
    const correctedAnnotatedData = segmentFilteredData.filter(
      (data) => data.Corrected === 1,
    );
    const artifactData = segmentFilteredData.filter(
      (data) => data.Artifact === 1,
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
      selectedSegment,
      allUserEdits,
      isAddMode,
      isDeleteMode,
      isMarkingUnusableMode,
      isMarkValidMode,
      isRemoveEditMode,
      isLabelOn,
      handleChartClick,
      removeEdit,
      dataTypeX,
    });

    setChartOptions(chartParams);
    setCardiacData(initCardiacData);
    setBeatData(initBeats);
    setBeatArtifactData(initArtifacts);
  }, [
    fileData,
    allUserEdits,
    isAddMode,
    isDeleteMode,
    isRemoveEditMode,
    selectedSegment,
    isMarkingUnusableMode,
    isMarkValidMode,
    isLabelOn,
  ]);

  const handleChartClick = (event: ChartClickEvent | SavedBeat) => {
    // Prevents coordinates from plotting when hitting `Reset Zoom`
    if (
      isPanning ||
      ("target" in event &&
        event.target &&
        event.target instanceof Element &&
        (event.target.classList.contains("highcharts-button-box") ||
          event.target.innerHTML === "Reset zoom"))
    ) {
      return; // Ignore clicks on Reset Zoom
    }

    const newX =
      "point" in event && !_.isUndefined(event.point)
        ? event.point.x
        : "xAxis" in event
          ? event.xAxis[0].value
          : event.x;
    const newY =
      "point" in event && !_.isUndefined(event.point)
        ? event.point.y
        : "yAxis" in event
          ? event.yAxis[0].value
          : event.y;

    // Check if the point already exists in cardiacData (for Add Mode) or beatData (for Delete Mode)
    const isSignal = cardiacData.some(
      (point) => point.x === newX && point.y === newY,
    );
    const isBeatCoordinate = beatData.some(
      (point) => point.x === newX && point.y === newY,
    );
    const isArtifactCoordinate = beatArtifactData.some(
      (point) => point.x === newX && point.y === newY,
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
    // In Mark Valid mode, ignore clicks on non-artifact points while panning
    if (isPanning && isMarkValidMode && !isArtifactCoordinate) {
      return;
    }

    const updatedCardiacData = [...cardiacData, { x: newX, y: newY }];
    const updatedBeatData = [...beatData];
    const updateArtifactData = [...beatArtifactData];

    setAllUserEdits((prev) => {
      if (isDeleteMode && !(isBeatCoordinate || isArtifactCoordinate)) {
        toast.error("This is not a beat");
        return prev;
      }

      if (isMarkValidMode && !isArtifactCoordinate) {
        toast.error("This is not an artifactual beat");
        return prev;
      }

      if (isDeleteMode) {
        return [
          ...prev,
          {
            x: newX,
            y: newY,
            segment: selectedSegment,
            editType: EDIT_TYPE_DELETE,
          },
        ];
      } else if (isAddMode) {
        return [
          ...prev,
          {
            x: newX,
            y: newY,
            segment: selectedSegment,
            editType: EDIT_TYPE_ADD,
          },
        ];
      } else if (isMarkValidMode) {
        return [
          ...prev,
          {
            x: newX,
            y: newY,
            segment: selectedSegment,
            editType: EDIT_TYPE_VALID,
          },
        ];
      } else if (isRemoveEditMode) {
        removeEdit(event as SavedBeat);
        return prev;
      }
      // If neither add nor delete mode, return previous state to satisfy setter signature
      return prev;
    });

    setCardiacData(updatedCardiacData);
    setBeatData(updatedBeatData);
    setBeatArtifactData(updateArtifactData);
  };

  const hasDataType = ({ fileData, data }: HasDataTypeParams) => {
    return fileData.some((o) => o.hasOwnProperty(data));
  };

  const transformCoordinates = ({
    data,
    xAxisLabel,
    yAxisLabel,
  }: transformCoordinatesParams) => {
    return data.map((item) => ({
      x: item[xAxisLabel as keyof Beat] as number,
      y: item[yAxisLabel as keyof Beat] as number,
    }));
  };

  const removeEdit = (edit: SavedBeat | SegmentObj) => {
    setAllUserEdits((prev) =>
      prev.filter((curr) => {
        if (edit.editType === EDIT_TYPE_UNUSABLE) {
          if (!("from" in edit && "to" in edit)) {
            return true;
          }

          return !(
            curr.editType === EDIT_TYPE_UNUSABLE &&
            "from" in curr &&
            "to" in curr &&
            curr.segment === edit.segment &&
            curr.from === edit.from &&
            curr.to === edit.to
          );
        }

        return !(
          curr.editType === edit.editType &&
          curr.segment === edit.segment &&
          curr.x === edit.x &&
          curr.y === edit.y
        );
      }),
    );
  };

  const toggleAddMode = () => {
    resetInteractionState();
    setIsAddMode((prev) => !prev);
    setIsDeleteMode(false);
    setIsMarkingUnusableMode(false);
    setIsRemoveEditMode(false);
    setIsMarkValidMode(false);
  };

  const toggleDeleteMode = () => {
    resetInteractionState();
    setIsAddMode(false);
    setIsDeleteMode((prev) => !prev);
    setIsMarkingUnusableMode(false);
    setIsRemoveEditMode(false);
    setIsMarkValidMode(false);
  };

  const toggleMarkUnusableMode = () => {
    resetInteractionState();
    setIsMarkingUnusableMode((prev) => !prev);
    setIsAddMode(false);
    setIsDeleteMode(false);
    setIsRemoveEditMode(false);
    setIsMarkValidMode(false);
  };

  const toggleMarkValidMode = () => {
    resetInteractionState();
    setIsAddMode(false);
    setIsDeleteMode(false);
    setIsMarkingUnusableMode(false);
    setIsRemoveEditMode(false);
    setIsMarkValidMode((prev) => !prev);
  };

  const toggleRemoveEditMode = () => {
    resetInteractionState();
    setIsAddMode(false);
    setIsDeleteMode(false);
    setIsMarkingUnusableMode(false);
    setIsMarkValidMode(false);
    setIsRemoveEditMode((prev) => !prev);
  };

  // Reset all drag and interaction states when toggling modes
  const resetInteractionState = () => {
    dragStartRef.current = null;
    isDraggingRef.current = false;
    lastValidDragEnd.current = null;
    setIsPanning(false);
  };

  useKeyboardShortcuts({
    toggleAddMode,
    toggleDeleteMode,
    toggleMarkUnusableMode,
    toggleMarkValidMode,
    toggleRemoveEditMode,
  });

  useMarkingUnusableMode({
    isMarkingUnusableMode,
    chartRef,
    setAllUserEdits,
    selectedSegment,
    dragStartRef,
    isDraggingRef,
    dragPlotBandRef,
    lastValidDragEnd,
    segmentBoundaries,
  });

  const buttonObjParams = useMemo(
    () => [
      {
        id: "add-beat",
        icon: "fa-solid fa-plus",
        label: "Add Beat",
        className: isAddMode ? "add-beat-active" : "",
        onClick: toggleAddMode,
      },
      {
        id: "delete-beat",
        icon: "fa-solid fa-minus",
        label: "Delete Beat",
        className: isDeleteMode ? "delete-beat-active" : "",
        onClick: toggleDeleteMode,
      },
      {
        id: "mark-unusable",
        icon: "fa-solid fa-marker",
        label: "Mark Unusable",
        className: isMarkingUnusableMode ? "mark-unusable-active" : "",
        onClick: toggleMarkUnusableMode,
      },
      {
        id: "mark-valid",
        icon: "fa-solid fa-check",
        label: "Mark Valid Beat",
        className: isMarkValidMode ? "mark-valid-active" : "",
        onClick: toggleMarkValidMode,
      },
      {
        id: "remove-edit",
        icon: "fa-solid fa-times",
        label: "Remove Edit",
        className: isRemoveEditMode ? "remove-edit-active" : "",
        onClick: toggleRemoveEditMode,
      },
    ],
    [
      isAddMode,
      isDeleteMode,
      isMarkingUnusableMode,
      isMarkValidMode,
      isRemoveEditMode,
      toggleAddMode,
      toggleDeleteMode,
      toggleMarkUnusableMode,
      toggleMarkValidMode,
      toggleRemoveEditMode,
    ],
  );

  return (
    <div className="relative">
      <div className="flex w-full mb-4 justify-between items-center">
        <div className="flex flex-wrap gap-2 mb-5 items-center">
          <select
            className="p-0 w-[50px] h-[36px] text-center text-[16px] border border-[#3d4951] rounded-md focus:outline-none"
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

          {/* Divider separating segment navigation from editing tools */}
          <ToolbarDivider />

          {Object.values(buttonObjParams).map((param) => {
            return (
              <button
                key={param.id}
                className={param.className}
                onClick={param.onClick}
              >
                <i className={param.icon}></i>
                {param.label}
              </button>
            );
          })}

          {/* Divider separating editing tools from file actions */}
          <ToolbarDivider />

          <SaveButton fileName={fileName} allEdits={allUserEdits} />
          <ImportEditsButton onImportSuccess={onRefresh} />
          <ExportButton fileName={fileName} allEdits={allUserEdits} />
        </div>
        <div className="flex flex-row ml-auto mb-5 gap-2">
          <LabelToggle isLabelOn={isLabelOn} setIsLabelOn={setIsLabelOn} />
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
