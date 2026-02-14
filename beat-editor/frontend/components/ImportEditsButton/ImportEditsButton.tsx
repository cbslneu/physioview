import axios from "axios";
import { toast } from "react-toastify";

interface ImportEditsButtonProps {
  onImportSuccess: () => void;
}

const ImportEditsButton = ({ onImportSuccess }: ImportEditsButtonProps) => {
  const handleImport = () => {
    // Open file dialog
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "json";
    input.onchange = async (e: any) => {
      const file = e?.target.files[0];
      if (!file) return;

      // Validate file is JSON
      if (file.type !== "application/json") {
        toast.error("Please select a valid JSON file.");
        return;
      }

      // Validate the file name ends with _edited.json
      if (!file.name.endsWith("_edited.json")) {
        toast.error(
          "Please upload the exported JSON file with the correct naming convention: [filename]_edited.json",
        );
        return;
      }

      const formData = new FormData();
      formData.append("file", file);

      try {
        await axios.post("http://localhost:3001/import-edits", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });
        toast.success(`${file.name} has been imported`);
        onImportSuccess();
      } catch (err) {
        toast.error(`Error importing file: ${file.name}`);
      }
    };
    input.click();
  };

  return (
    <button className="import-button" onClick={handleImport}>
      <i className="fa-solid fa-file-import"></i>Import Edits
    </button>
  );
};

export default ImportEditsButton;
