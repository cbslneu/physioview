import { useEffect } from "react";

interface useKeyboardShortcutsProps {
  toggleAddMode: () => void;
  toggleDeleteMode: () => void;
  toggleMarkUnusableMode: () => void;
  toggleRemoveEditMode: () => void;
}

const useKeyboardShortcuts = ({
  toggleAddMode,
  toggleDeleteMode,
  toggleMarkUnusableMode,
  toggleRemoveEditMode,
}: useKeyboardShortcutsProps) => {
  const handleKeyDown = (event: KeyboardEvent) => {
    if (event.key === "A" || event.key === "a") {
      toggleAddMode();
    } else if (event.key === "D" || event.key === "d") {
      toggleDeleteMode();
    } else if (event.key === "U" || event.key === "u") {
      toggleMarkUnusableMode();
    } else if (event.key === "R" || event.key === "r") {
      toggleRemoveEditMode();
    }
  };

  useEffect(() => {
    // Checks for key presses
    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  });
};

export default useKeyboardShortcuts;
