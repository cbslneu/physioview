import { useEffect } from "react";

interface useKeyboardShortcutsProps {
    toggleAddMode: () => void;
    toggleDeleteMode: () => void;
    toggleMarkUnusableMode: () => void;
    undoLastCoordinate: () => void;
}

const useKeyboardShortcuts = ({
    toggleAddMode,
    toggleDeleteMode,
    toggleMarkUnusableMode,
    undoLastCoordinate
}: useKeyboardShortcutsProps) => {
      const handleKeyDown = (event: KeyboardEvent) => {
        if (event.key === "A" || event.key === "a") {
          toggleAddMode();
        } else if (event.key === "D" || event.key === "d") {
          toggleDeleteMode();
        } else if (event.key === "U" || event.key === "u") {
          toggleMarkUnusableMode();
        } else if (
          (event.ctrlKey && event.key === "z") ||
          (event.metaKey && event.key === "z")
        ) {
          undoLastCoordinate();
        }
      };
    
      useEffect(() => {
        // Checks for key presses
        window.addEventListener("keydown", handleKeyDown);
    
        return () => {
          window.removeEventListener("keydown", handleKeyDown);
        };
      });
}

export default useKeyboardShortcuts;
