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

  return (
    <div className="keyboard-shortcuts-wrapper">
      <div>
        <button
          className="shortcut-button"
          onClick={toggleKeyboardShortcut}
          ref={keyboardButtonRef}
        >
          <i className="fa-solid fa-keyboard fa-xl"></i>
        </button>
      </div>
      {showKeyboardShortcut && (
        <div className="keyboard-shortcuts-popover" ref={keyboardShortcutsRef}>
          <div className="popover-arrow"></div>
          <h2 className="popover-title">Keyboard Shortcuts</h2>

          <div className="shortcuts-grid">
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

          <div className="mark-unusable-section">
            <div className={markUnusableOption}>
              <div className={shortcutKeys}>
                <i className={uIcon} />
                <i className={plusIcon} />
                <img
                  src={rightClickIcon}
                  className={rightClickKeybind}
                  width={22}
                  height={22}
                />
              </div>
              <div className={shortcutLabel}>{markUnusableLabel}</div>
            </div>
          </div>

          <div className="pan-section">
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
