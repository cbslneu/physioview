import axios from "axios";
import { toast } from "react-toastify";

import { SavedBeat, SegmentObj } from "../../types/types";

interface SaveButtonParams {
  fileName: string;
  allEdits: (SavedBeat | SegmentObj)[];
}

const SaveButton = ({ fileName, allEdits }: SaveButtonParams) => {
  const saveJSON = async () => {
    try {
      console.log("Saving edits:", allEdits);
      await axios.post("http://localhost:3001/saved", {
        fileName,
        data: allEdits,
      });
      if (allEdits.length !== 0) {
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
