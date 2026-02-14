import { toast } from "react-toastify";
import { SavedBeat } from "../../types/types";

interface ExportButtonParams {
  fileName: string;
  addModeCoordinates: SavedBeat[];
  deleteModeCoordinates: SavedBeat[];
  unusableSegments: SavedBeat[];
}

const ExportButton = (props: ExportButtonParams) => {
  const {
    fileName,
    addModeCoordinates,
    deleteModeCoordinates,
    unusableSegments,
  } = props;

  const exportJSON = () => {
    // Just like the save function, instead we download the json
    const data = [
      ...addModeCoordinates,
      ...deleteModeCoordinates,
      ...unusableSegments,
    ];

    const jsonData = JSON.stringify(data, null, 2);

    const jsonString = `data:text/json;chatset=utf-8,${encodeURIComponent(jsonData)}`;
    const link = document.createElement("a");

    link.href = jsonString;
    link.download = `${fileName}_edited.json`;
    link.click();

    toast.success(`${fileName}_edited.json has been exported`);
  };
  return (
    <button className="export-button" onClick={exportJSON}>
      <i className="fa-solid fa-file-export"></i>Export
    </button>
  );
};

export default ExportButton;
