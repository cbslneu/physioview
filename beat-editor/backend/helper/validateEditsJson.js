const EDIT_TYPES = ["ADD", "DELETE", "UNUSABLE"];

/**
 *  validateEditsJson
 *
 *  Validates the structure of the imported _edits.json file.
 *
 *  @param {Array} data - The parsed JSON data to validate.
 *  @returns {Object} An object with 'ok' boolean and 'error' message if not valid.
 */

function validateEditsJson(data) {
  if (!Array.isArray(data)) {
    return { ok: false, error: "Expected an array" };
  }

  for (const [key, value] of data.entries()) {
    if (!value || typeof value !== "object") {
      return { ok: false, error: `Item ${key} is not an object` };
    }

    const { editType, segment, x, y } = value;

    if (!EDIT_TYPES.includes(editType))
      return { ok: false, error: `Item ${key} has an invalid editType` };
    if (typeof segment !== "string")
      return { ok: false, error: `Item ${key} has an invalid segment` };
    if (typeof x !== "number" || typeof y !== "number")
      return { ok: false, error: `Item ${key} has invalid x or y` };

    if (editType === "UNUSABLE") {
      if (
        !Number.isFinite(value.from) ||
        !Number.isFinite(value.to) ||
        typeof value.color !== "string"
      ) {
        return {
          ok: false,
          error: `Item ${key} has invalid from or to for UNUSABLE editType`,
        };
      }
    }
  }

  return { ok: true };
}

module.exports = validateEditsJson;
