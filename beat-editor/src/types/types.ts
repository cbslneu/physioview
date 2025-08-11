export interface Beat {
  Artifact: number | null;
  Beat: number | null;
  Corrected?: number | null;
  Sample?: number;
  Segment: string;
  Signal: number;
  Timestamp?: number;
}

export interface SavedBeat {
  editType: string;
  segment: string
  x: number;
  y: number;
}

export interface SegmentObj extends SavedBeat {
  color: string;
  from: number;
  to: number;
}

export interface ChartCoordinates {
  x: number;
  y: number;
}

export interface ChartClickEvent {
  point: {
    x: number;
    y: number;
  };
  xAxis: Array<{
    value: number;
  }>;
  yAxis: Array<{
    value: number;
  }>;
  target: EventTarget | null;
}

export interface ChartOptions {
  xAxisData: (number | undefined)[];
  initCardiacData: ChartCoordinates[];
  initBeats: ChartCoordinates[];
  initArtifacts: ChartCoordinates[];
  addModeCoordinates: SavedBeat[];
  deleteModeCoordinates: SavedBeat[];
  selectedSegment: string;
  unusableSegments: SegmentObj[];
  isAddMode: boolean;
  isDeleteMode: boolean;
  isMarkingUnusableMode: boolean;
  handleChartClick: (event: ChartClickEvent) => void;
  dataTypeX: string | undefined;
}
