import {
  EDIT_TYPE_ADD,
  EDIT_TYPE_DELETE,
  EDIT_TYPE_UNUSABLE,
  EDIT_TYPE_VALID,
} from "../constants/constants";
import {
  ChartClickEvent,
  ChartOptions,
  SavedBeat,
  SegmentObj,
} from "../types/types";

const createChartOptions = ({
  xAxisData,
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
  removeEdit,
  handleChartClick,
  dataTypeX,
}: ChartOptions): Highcharts.Options => {
  const addModeCoordinates: SavedBeat[] = allUserEdits.filter(
    (edit) => edit.editType === EDIT_TYPE_ADD,
  );
  const deleteModeCoordinates: SavedBeat[] = allUserEdits.filter(
    (edit) => edit.editType === EDIT_TYPE_DELETE,
  );
  const unusableSegments: SegmentObj[] = allUserEdits.filter(
    (edit): edit is SegmentObj => edit.editType === EDIT_TYPE_UNUSABLE,
  );
  const validModeCoordinates: SavedBeat[] = allUserEdits.filter(
    (edit) => edit.editType === EDIT_TYPE_VALID,
  );

  return {
    chart: {
      type: "line",
      zooming: {
        mouseWheel: {
          enabled: true,
          sensitivity: 1.3,
          type: "x",
        },
        type: isMarkingUnusableMode ? undefined : "x",
      },
      panning: {
        enabled: !isMarkingUnusableMode, // Disable panning in Mark Unusable mode
      },
      panKey: "shift",
      events: {
        click: function (event) {
          if ((isAddMode || isDeleteMode) && !event.shiftKey) {
            const chartClickEvent: ChartClickEvent = {
              point: {
                x: this.xAxis[0].toValue(event.chartX),
                y: this.yAxis[0].toValue(event.chartY),
              },
              xAxis: [{ value: this.xAxis[0].toValue(event.chartX) }],
              yAxis: [{ value: this.yAxis[0].toValue(event.chartY) }],
              target: event.target,
            };
            handleChartClick(chartClickEvent);
          }
        },
      },
      style: {
        fontFamily: "'Poppins', sans-serif",
        fontSize: "20px",
      },
      animation: false,
    },
    title: {
      text: "",
    },
    xAxis: {
      title: {
        text: dataTypeX,
      },
      labels: {
        formatter: function () {
          if (dataTypeX === "Timestamp") {
            const date = new Date(this.value);
            return date.toUTCString().split(" ")[4];
          } else {
            return String(this.value);
          }
        },
        style: {
          fontSize: "13px",
        },
      },
      minPadding: 0,
      maxPadding: 0,
      allowDecimals: true,
      plotBands: unusableSegments.map((segment) => ({
        from: segment.from,
        to: segment.to,
        color: "rgba(255, 0, 0, 0.2)",
        events: {
          click: function () {
            if (isRemoveEditMode) {
              removeEdit(segment);
            }
          },
        },
      })),
      min: xAxisData[0],
      max: xAxisData[xAxisData.length - 1],
    },
    yAxis: {
      title: {
        text: "Signal",
      },
      labels: {
        style: {
          fontSize: "0.7em",
        },
      },
      allowDecimals: true,
    },
    tooltip: {
      enabled: isLabelOn,
      formatter: function () {
        const date = new Date(this.x || 0);
        const dataType = dataTypeX === "Timestamp" ? "Time" : "Sample";
        const value =
          dataType === "Time" ? date.toUTCString().split(" ")[4] : this.x;
        return `<b>${
          this.series.name
        }</b><br/>${dataType}: ${value} <br/>Amplitude: ${this.y?.toFixed(
          3,
        )} mV`;
      },
    },
    series: [
      {
        name: "Signal",
        data: initCardiacData,
        type: "line",
        color: "#3562BD",
        turboThreshold: 0,
        states: {
          hover: {
            enabled: false,
          },
          inactive: {
            enabled: false,
          },
        },
        point: {
          events: {
            click: function (event) {
              if (isAddMode || isDeleteMode || isMarkValidMode) {
                const chartClickEvent: ChartClickEvent = {
                  point: { x: this.x, y: this.y as number },
                  xAxis: [{ value: this.x }],
                  yAxis: [{ value: this.y as number }],
                  target: event.target,
                };
                handleChartClick(chartClickEvent);
              }
            },
          },
        },
      },
      {
        name: "Beat",
        data: initBeats,
        type: "scatter",
        color: "#F9C669",
        marker: {
          symbol: "circle",
        },
        turboThreshold: 0,
        states: {
          hover: {
            enabled: false,
          },
          inactive: {
            enabled: false,
          },
        },
        point: {
          events: {
            click: function (event) {
              if (isAddMode || isDeleteMode || isMarkValidMode) {
                const chartClickEvent: ChartClickEvent = {
                  point: { x: this.x, y: this.y as number },
                  xAxis: [{ value: this.x }],
                  yAxis: [{ value: this.y as number }],
                  target: event.target,
                };
                handleChartClick(chartClickEvent);
              }
            },
          },
        },
      },
      {
        name: "Potential Artifact",
        data: initArtifacts,
        type: "scatter",
        color: "red",
        marker: {
          symbol: "circle",
        },
        visible: initArtifacts.length > 0,
        showInLegend: initArtifacts.length > 0,
        turboThreshold: 0,
        states: {
          hover: {
            enabled: false,
          },
          inactive: {
            enabled: false,
          },
        },
        point: {
          events: {
            click: function (event) {
              if (isAddMode || isDeleteMode || isMarkValidMode) {
                const chartClickEvent: ChartClickEvent = {
                  point: { x: this.x, y: this.y as number },
                  xAxis: [{ value: this.x }],
                  yAxis: [{ value: this.y as number }],
                  target: event.target,
                };
                handleChartClick(chartClickEvent);
              }
            },
          },
        },
      },
      {
        name: "Added Beats",
        data: addModeCoordinates.filter((o) => o.segment === selectedSegment),
        type: "scatter",
        color: "#02E337",
        marker: {
          symbol: "circle",
        },
        visible: addModeCoordinates.some((o) => o.segment === selectedSegment),
        showInLegend: addModeCoordinates.some(
          (o) => o.segment === selectedSegment,
        ),
        turboThreshold: 0,
        states: {
          hover: {
            enabled: false,
          },
          inactive: {
            enabled: false,
          },
        },
        point: {
          events: {
            click: function (event) {
              if (isRemoveEditMode) {
                const clickedEdit: SavedBeat = {
                  x: this.x,
                  y: this.y as number,
                  segment: selectedSegment,
                  editType: EDIT_TYPE_ADD,
                };
                handleChartClick(clickedEdit);
                return;
              }
              const chartClickEvent: ChartClickEvent = {
                point: { x: this.x, y: this.y as number },
                xAxis: [{ value: this.x }],
                yAxis: [{ value: this.y as number }],
                target: event.target,
              };
              handleChartClick(chartClickEvent);
            },
          },
        },
      },
      {
        name: "Deleted Beats",
        data: deleteModeCoordinates.filter(
          (o) => o.segment === selectedSegment,
        ),
        type: "scatter",
        color: "red",
        marker: {
          symbol: "cross",
          lineColor: undefined,
          lineWidth: 2,
        },
        visible: deleteModeCoordinates.some(
          (o) => o.segment === selectedSegment,
        ),
        showInLegend: deleteModeCoordinates.some(
          (o) => o.segment === selectedSegment,
        ),
        turboThreshold: 0,
        states: {
          hover: {
            enabled: false,
          },
          inactive: {
            enabled: false,
          },
        },
        point: {
          events: {
            click: function (event) {
              if (isRemoveEditMode) {
                const clickedEdit: SavedBeat = {
                  x: this.x,
                  y: this.y as number,
                  segment: selectedSegment,
                  editType: EDIT_TYPE_DELETE,
                };
                handleChartClick(clickedEdit);
                return;
              }

              const chartClickEvent: ChartClickEvent = {
                point: { x: this.x, y: this.y as number },
                xAxis: [{ value: this.x }],
                yAxis: [{ value: this.y as number }],
                target: event.target,
              };
              handleChartClick(chartClickEvent);
            },
          },
        },
      },
      {
        name: "Validated Beat",
        data: validModeCoordinates.filter((o) => o.segment === selectedSegment),
        type: "scatter",
        color: "#F9C669",
        marker: {
          symbol: "circle",
          lineColor: "#02E337",
          lineWidth: 2,
        },
        visible: validModeCoordinates.some((o) => o.segment === selectedSegment),
        showInLegend: validModeCoordinates.some(
          (o) => o.segment === selectedSegment,
        ),
        turboThreshold: 0,
        states: {
          hover: {
            enabled: false,
          },
          inactive: {
            enabled: false,
          },
        },
        point: {
          events: {
            click: function (event) {
              if (isRemoveEditMode) {
                const clickedEdit: SavedBeat = {
                  x: this.x,
                  y: this.y as number,
                  segment: selectedSegment,
                  editType: EDIT_TYPE_VALID,
                };
                handleChartClick(clickedEdit);
                return;
              }

              const chartClickEvent: ChartClickEvent = {
                point: { x: this.x, y: this.y as number },
                xAxis: [{ value: this.x }],
                yAxis: [{ value: this.y as number }],
                target: event.target,
              };
              handleChartClick(chartClickEvent);
            },
          },
        },
      },
    ],
  };
};

export default createChartOptions;
