import axios from "axios";
import { toast } from "react-toastify";
import "@fortawesome/fontawesome-free/css/all.min.css";

import { SavedBeat } from "../../types/types";

interface SaveButtonParams {
  fileName: string;
  addModeCoordinates: SavedBeat[];
  deleteModeCoordinates: SavedBeat[];
  unusableSegments: SavedBeat[];
}

const SaveButton = ({
  fileName,
  addModeCoordinates,
  deleteModeCoordinates,
  unusableSegments,
}: SaveButtonParams) => {
  const saveJSON = async () => {
    try {
      await axios.post("http://localhost:3001/saved", {
        fileName,
        data: { addModeCoordinates, deleteModeCoordinates, unusableSegments },
      });
      if (
        addModeCoordinates.length !== 0 ||
        deleteModeCoordinates.length !== 0 ||
        unusableSegments.length !== 0
      ) {
        toast.success(`${fileName}_edited.json has been saved`, {
          className: "custom-toast",
        });
      }
    } catch (err) {
      toast.error(`Error saving file: ${fileName}_edited.json`, {
        className: "custom-toast",
      });
    }
  };

  return (
    <button className="save-button" onClick={saveJSON}>
      <i className="fa-solid fa-save fa-md"></i>Save
    </button>
  );
};

export default SaveButton;
