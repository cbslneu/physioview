import { ChartClickEvent, ChartOptions } from "../types/types";

const createChartOptions = ({
    xAxisData,
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
  }: ChartOptions): Highcharts.Options => {
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
          enabled: !isMarkingUnusableMode, // Enable panning
        },
        panKey: "shift",
        events: {
          click: function (event) {
            if (
              (isAddMode || isDeleteMode || isMarkingUnusableMode) &&
              !event.shiftKey
            ) {
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
        plotBands: unusableSegments.map(segment => ({
          from: segment.from,
          to: segment.to,
          color: 'rgba(255, 0, 0, 0.2)',
        })),
        min: xAxisData[0],
        max: xAxisData[xAxisData.length - 1],
      },
      yAxis: {
        title: {
          text: "Signal",
        },
        allowDecimals: true,
      },
      tooltip: {
        formatter: function () {
          const date = new Date(this.x || 0);
          const dataType = dataTypeX === "Timestamp" ? "Time" : "Sample";
          const value = dataType === "Time" ? date.toUTCString().split(" ")[4] : this.x;
          return `<b>${
            this.series.name
          }</b><br/>${dataType}: ${value} <br/>Amplitude: ${this.y?.toFixed(
            3
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
                if (isAddMode || isDeleteMode || isMarkingUnusableMode) {
                  const chartClickEvent: ChartClickEvent = {
                    point: { x: this.x, y: this.y as number },
                    xAxis: [{ value: this.x }],
                    yAxis: [{ value: this.y as number }],
                    target: event.target
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
                if (isAddMode || isDeleteMode) {
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
          name: "Artifact",
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
                if (isAddMode || isDeleteMode) {
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
          visible: addModeCoordinates.some(
            (o) => o.segment === selectedSegment
          ),
          showInLegend: addModeCoordinates.some(
            (o) => o.segment === selectedSegment
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
            (o) => o.segment === selectedSegment
          ),
          type: "scatter",
          color: "red",
          marker: {
            symbol: "cross",
            lineColor: undefined,
            lineWidth: 2,
          },
          visible: deleteModeCoordinates.some(
            (o) => o.segment === selectedSegment
          ),
          showInLegend: deleteModeCoordinates.some(
            (o) => o.segment === selectedSegment
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
  