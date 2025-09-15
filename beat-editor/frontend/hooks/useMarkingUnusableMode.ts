import { useEffect } from "react";
import { EDIT_TYPE_UNUSABLE } from "../constants/constants";
import { ChartCoordinates, SegmentObj } from "../types/types";
import { isNull } from "lodash";
import HighchartsReact from "highcharts-react-official";

interface SegmentBoundary {
  from: ChartCoordinates;
  to: ChartCoordinates;
}

interface UseMarkingUnusableModeProps {
  isMarkingUnusableMode: boolean;
  chartRef: React.RefObject<HighchartsReact.RefObject | null>;
  setUnusableSegments: React.Dispatch<React.SetStateAction<SegmentObj[]>>;
  selectedSegment: string;
  dragStartRef: React.RefObject<number | null>;
  isDraggingRef: React.RefObject<boolean>;
  dragPlotBandRef: React.RefObject<Highcharts.XAxisPlotBandsOptions | null>;
  lastValidDragEnd: React.RefObject<number | null>;
  segmentBoundaries: SegmentBoundary;
};

const useMarkingUnusableMode = (
  {isMarkingUnusableMode,
  chartRef,
  setUnusableSegments,
  selectedSegment,
  dragStartRef,
  isDraggingRef,
  dragPlotBandRef,
  lastValidDragEnd,
  segmentBoundaries
}: UseMarkingUnusableModeProps) => {
  useEffect(() => {
    if (chartRef.current && chartRef.current.chart && isMarkingUnusableMode) {
      const chart = chartRef.current.chart;

      const getChartPosition = (event: MouseEvent) => {
        const chartContainer = chart.container;
        const rect = chartContainer.getBoundingClientRect();
        return {
          x: event.clientX - rect.left,
          y: event.clientY - rect.top
        }
      }

      const handleMouseDown = (event: MouseEvent) => {
        if (isMarkingUnusableMode && !event.shiftKey) {
          const target = event.target as HTMLElement;
          if (target && target.textContent?.includes("Reset zoom")) {
            return;
          }

          event.preventDefault();

          const chartPosition = getChartPosition(event);
          let dragStart = chart.xAxis[0].toValue(chartPosition.x);

          if (!isNaN(dragStart)) {
            dragStartRef.current = dragStart;
            isDraggingRef.current = true;
          } 
        }
      };

      const preventContextMenu = (event: Event) => event.preventDefault();

      document.addEventListener("contextmenu", preventContextMenu);

      const handleMouseMove = (event: MouseEvent) => {
        if (isMarkingUnusableMode && isDraggingRef.current && !event.shiftKey) {
          const chartPosition = getChartPosition(event);
          let dragEnd = chart.xAxis[0].toValue(chartPosition.x);

          if (!isNaN(dragEnd)) {
            // Clamp drag end within segment boundaries
            dragEnd = Math.max(
              segmentBoundaries.from.x,
              Math.min(segmentBoundaries.to.x, dragEnd)
            );

            // Remove previous temporary plot band
            if (dragPlotBandRef.current) {
              chart.xAxis[0].removePlotBand("draggingPlotBand");
            }

            // Add new temporary plot band while dragging
            lastValidDragEnd.current = dragEnd;
            dragPlotBandRef.current = {
              id: "draggingPlotBand",
              from: Math.min(dragStartRef.current ?? 0, dragEnd),
              to: Math.max(dragStartRef.current ?? 0, dragEnd),
              color: "rgba(255, 0, 0, 0.2)",
            };
            chart.xAxis[0].addPlotBand(dragPlotBandRef.current);
          } 
        }
      };

      const handleMouseUp = () => {
        if (isMarkingUnusableMode && isDraggingRef.current) {
          let dragEnd =
            lastValidDragEnd.current !== null && lastValidDragEnd.current > segmentBoundaries.to.x
              ? segmentBoundaries.to.x
              : lastValidDragEnd.current;

          // Clamp drag end within segment boundaries
          dragEnd = Math.max(
            segmentBoundaries.from.x,
            Math.min(segmentBoundaries.to.x, dragEnd ?? 0)
          );

          if (!isNull(dragStartRef.current) && !isNaN(dragEnd)) {
            // Remove temporary plot band
            if (dragPlotBandRef.current) {
              chart.xAxis[0].removePlotBand("draggingPlotBand");
              dragPlotBandRef.current = null;
            }

            // Add final unusable segment
            const newSegment = {
              segment: selectedSegment,
              from: Math.min(dragStartRef.current, dragEnd),
              to: Math.max(dragStartRef.current, dragEnd),
              editType: EDIT_TYPE_UNUSABLE,
              color: "rgba(255, 0, 0, 0.3)",
              x: Math.min(dragStartRef.current, dragEnd),
              y: 0,
            };

            setUnusableSegments((prevSegments) => [
              ...prevSegments,
              newSegment,
            ]);
          }

          // Reset dragging state
          dragStartRef.current = null;
          isDraggingRef.current = false;
        }
      };

      // Attach event listeners to the chart container
      chart.container.addEventListener("mousedown", handleMouseDown);
      chart.container.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleMouseUp);

      return () => {
        // Cleanup temporary plot bands if any
        if (dragPlotBandRef.current) {
          chart.xAxis[0].removePlotBand("draggingPlotBand");
          dragPlotBandRef.current = null;
        }
        
        // Cleanup event listeners
        document.removeEventListener("contextmenu", preventContextMenu);
        chart.container.removeEventListener("mousedown", handleMouseDown);
        chart.container.removeEventListener("mousemove", handleMouseMove);
        window.removeEventListener("mouseup", handleMouseUp);

        dragStartRef.current = null;
        isDraggingRef.current = false;
        lastValidDragEnd.current = null;
      };
    }
  }, [
    isMarkingUnusableMode,
    chartRef,
    setUnusableSegments,
    selectedSegment,
    dragStartRef,
    isDraggingRef,
    dragPlotBandRef,
    lastValidDragEnd,
    segmentBoundaries,
  ]);
};

export default useMarkingUnusableMode;
