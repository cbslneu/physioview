import { useState, useEffect, useRef } from "react";
import {
  COMMON_CLASSNAMES,
  TOP_ROW_SHORTCUTS,
  MIDDLE_ROW_SHORTCUTS,
  BOTTOM_ROW_SHORTCUTS,
} from "./constants";

const KeyboardShortcuts = () => {
  const [showKeyboardShortcut, setShowKeyboardShortcut] = useState(false);
  const keyboardShortcutsRef = useRef<HTMLDivElement>(null);
  const keyboardButtonRef = useRef<HTMLButtonElement>(null);

  const toggleKeyboardShortcut = () => {
    setShowKeyboardShortcut(!showKeyboardShortcut);
  };

  const {
    shortcutItem,
    shortcutKeys,
    shortcutLabel,
    markUnusableOption,
    keybind,
  } = COMMON_CLASSNAMES;

  const {
    uIcon,
    plusIcon,
    rightClickIcon,
    rightClickKeybind,
    markUnusableLabel,
  } = MIDDLE_ROW_SHORTCUTS[0];

  const { shiftLabel, pointerIcon, dragLabel, panLabel } =
    BOTTOM_ROW_SHORTCUTS[0];

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        showKeyboardShortcut &&
        keyboardShortcutsRef.current &&
        !keyboardShortcutsRef.current.contains(event.target as Node) &&
        keyboardButtonRef.current &&
        !keyboardButtonRef.current.contains(event.target as Node)
      ) {
        setShowKeyboardShortcut(false);
      }
    };

    if (showKeyboardShortcut) {
      document.addEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [showKeyboardShortcut]);

  /**
.pan-section {
  border-top: 1px solid #ddd;
  padding-top: 10px;
  display: flex;
  justify-content: center;
  margin-top: 10px;
}
   */

  return (
    <div className="relative">
      <div>
        <button
          className="flex justify-center items-center w-[50px] h-[40px] bg-[#47555e]"
          onClick={toggleKeyboardShortcut}
          ref={keyboardButtonRef}
        >
          <i className="fa-solid fa-keyboard fa-xl m-1 text-[1.2em]"></i>
        </button>
      </div>
      {showKeyboardShortcut && (
        <div
          className="absolute w-[420px] shadow-[0_4px_12px_rgba(0,0,0,0.15)] p-4 top-[calc(100%+15px)] left-[-160px] transform -translate-x-1/2 border border-[#47555e] rounded-lg bg-white z-[1000] text-[#47555e]"
          ref={keyboardShortcutsRef}
        >
          <div className="absolute right-[5px] top-[-11px] z-[1] h-0 w-0 -translate-x-1/2 border-x-[10px] border-x-transparent border-b-[10px] border-b-[#47555e] after:content-[''] after:absolute after:left-[-10px] after:top-[2px] after:h-0 after:w-0 after:border-x-[10px] after:border-x-transparent after:border-b-[10px] after:border-b-white" />
          <h2 className="mb-4 text-lg font-bold text-center">
            Keyboard Shortcuts
          </h2>

          <div className="grid grid-cols-4 gap-4 mb-4">
            {TOP_ROW_SHORTCUTS.map((shortcut) => {
              const { icon, label } = shortcut;
              return (
                <div className={shortcutItem}>
                  <div className={shortcutKeys}>
                    <i className={icon} />
                  </div>
                  <div className={shortcutLabel}>{label}</div>
                </div>
              );
            })}
          </div>

          <div className="flex justify-around items-center border-t border-[#ddd] pt-2">
            <div className={markUnusableOption}>
              <div className={shortcutKeys}>
                <i className={uIcon} />
                <i className={plusIcon} />
                <img
                  src={rightClickIcon}
                  className={rightClickKeybind}
                  width={35}
                  height={35}
                />
              </div>
              <div className={shortcutLabel}>{markUnusableLabel}</div>
            </div>
          </div>

          <div className="flex justify-center mt-2 pt-2 border-t border-[#ddd]">
            <div className={markUnusableOption}>
              <div className={shortcutKeys}>
                <div className={keybind}>{shiftLabel}</div>
                <i className={plusIcon}></i>
                <i className={pointerIcon}></i>
                <i className={plusIcon}></i>
                <span className={dragLabel}>{dragLabel}</span>
              </div>
              <div className={shortcutLabel}>{panLabel}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default KeyboardShortcuts;
