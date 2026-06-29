import { toast } from "react-toastify";
import { SavedBeat, SegmentObj } from "../../types/types";

interface ExportButtonParams {
  fileName: string;
  allEdits: (SavedBeat | SegmentObj)[];
}

const ExportButton = ({ fileName, allEdits }: ExportButtonParams) => {
  const exportJSON = () => {
    // Just like the save function, instead we download the json
    const data = [...allEdits];

    const jsonData = JSON.stringify(data, null, 2);

    const jsonString = `data:text/json;charset=utf-8,${encodeURIComponent(jsonData)}`;
    const link = document.createElement("a");

    link.href = jsonString;
    link.download = `${fileName}_edited.json`;
    link.click();

    toast.success(`${fileName}_edited.json has been exported`, {
      className: "custom-toast",
    });
  };
  return (
    <button className="export-button" onClick={exportJSON}>
      <i className="fa-solid fa-file-export"></i>Export
    </button>
  );
};

export default ExportButton;
