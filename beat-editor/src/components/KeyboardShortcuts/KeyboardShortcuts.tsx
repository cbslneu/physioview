import { useState, useEffect, useRef } from 'react';

const KeyboardShortcuts = () => {
  const [showKeyboardShortcut, setShowKeyboardShortcut] = useState(false);
  const keyboardShortcutsRef = useRef<HTMLDivElement>(null);
  const keyboardButtonRef = useRef<HTMLButtonElement>(null);

  const toggleKeyboardShortcut = () => {
    setShowKeyboardShortcut(!showKeyboardShortcut);
  };

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
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showKeyboardShortcut]);

  return (
    <div className="keyboard-shortcuts-wrapper">
      <div>
        <button className="shortcut-button" onClick={toggleKeyboardShortcut} ref={keyboardButtonRef}>
          <i className="fa-solid fa-keyboard fa-xl"></i>
        </button>
      </div>
      {showKeyboardShortcut && (
            <div className="keyboard-shortcuts-popover" ref={keyboardShortcutsRef}>
              <div className="popover-arrow"></div>
              <h2 className="popover-title">Keyboard Shortcuts</h2>
              
              <div className="shortcuts-grid">
                <div className="shortcut-item">
                  <div className="shortcut-keys">
                    <i className="fa-solid fa-a keybind"></i>
                  </div>
                  <div className="shortcut-label">Add Beat</div>
                </div>
                
                <div className="shortcut-item">
                  <div className="shortcut-keys">
                    <i className="fa-solid fa-u keybind"></i>
                  </div>
                  <div className="shortcut-label">Mark Unusable</div>
                </div>
                
                <div className="shortcut-item">
                  <div className="shortcut-keys">
                    <i className="fa-solid fa-d keybind"></i>
                  </div>
                  <div className="shortcut-label">Delete Beat</div>
                </div>
              </div>
              
              <div className="undo-section">
                <div className="undo-option">
                  <div className="shortcut-keys">
                    <div className="keybind">CTRL</div>
                    <i className="fa-solid fa-plus plus"></i>
                    <i className="fa-solid fa-z keybind"></i>
                  </div>
                  <div className="shortcut-label">Undo</div>
                </div>
                
                <div className="or-divider">OR</div>
                
                <div className="undo-option">
                  <div className="shortcut-keys">
                    <div className="command-keybind">⌘</div>
                    <i className="fa-solid fa-plus plus"></i>
                    <i className="fa-solid fa-z keybind"></i>
                  </div>
                  <div className="shortcut-label">Undo</div>
                </div>
              </div>
              
              <div className="pan-section">
                <div className="undo-option">
                  <div className="shortcut-keys">
                    <div className="keybind">SHIFT</div>
                    <i className="fa-solid fa-plus plus"></i>
                    <i className="fa-solid fa-arrow-pointer keybind"></i>
                    <i className="fa-solid fa-plus plus"></i>
                    <span>DRAG</span>
                  </div>
                  <div className="shortcut-label">PAN</div>
                </div>
              </div>
            </div>
          )}
    </div>
  );
};

export default KeyboardShortcuts;
