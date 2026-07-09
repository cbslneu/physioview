// Auto-focus the Beat Editor iframe when its modal opens, so the editor's
// keyboard shortcuts work immediately without the user clicking inside it
window.dash_clientside = window.dash_clientside || {};
window.dash_clientside.clientside = window.dash_clientside.clientside || {};

window.dash_clientside.clientside.focusBeatEditor = function (is_open) {
    if (!is_open) {
        return window.dash_clientside.no_update;
    }

    // Move focus into the iframe (and its content window) once its content
    // has loaded, so keystrokes route to the Beat Editor's own window where
    // the shortcut listener lives
    const focusEditor = function () {
        const ifr = document.getElementById('beat-editor-iframe');
        if (!ifr) {
            return false;
        }
        const grab = function () {
            try { ifr.contentWindow.focus(); } catch (e) {}
            ifr.focus();
        };
        ifr.addEventListener('load', function () {
            setTimeout(grab, 100);
        }, { once: true });
        setTimeout(grab, 300);  // fallback if the iframe already loaded
        return true;
    };

    // The iframe is usually already present when the modal opens, so focus it
    // now; otherwise wait for it (MutationObserver only sees future changes)
    if (!focusEditor()) {
        const observer = new MutationObserver(function (_mutations, obs) {
            if (focusEditor()) {
                obs.disconnect();
            }
        });
        observer.observe(document.body, { childList: true, subtree: true });
        setTimeout(function () { observer.disconnect(); }, 5000);
    }

    return window.dash_clientside.no_update;
};
