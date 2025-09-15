.. |br| raw:: html

   <div style="margin-top: 30px"></div>

=====================
Beat Editor I/O Files
=====================

The PhysioView Beat Editor uses **JSON files** to read and write data. When
accessed through the PhysioView Dashboard, these files are generated
automatically. If you are running the Beat Editor as a standalone
application, however, you will need to create the JSON files yourself before
loading them into the editor.

Creating a Beat Editor File
===========================

Input JSON files must include specific keys to represent time-series data and beat annotations.
Additionally, filenames should end with the ``'_edit.json'`` suffix to be recognized by the editor.

You can generate this file automatically using
`physioview.write_beat_editor_file() <api.html#physioview.physioview.write_beat_editor_file>`_,
or create one manually using the following required keys:

|br|

Required Keys
-------------

- ``Timestamp`` (or alternatively ``Sample``) representing the time at which each data point occurs.
  
  - ``Timestamp``: Unix epoch time in milliseconds.
  - ``Sample`` can be used instead if the data is indexed by sample number instead of actual time.

- ``Signal``: The actual ECG/PPG values.

- ``Beat``: Annotations of 1s marking where heartbeats occur in the signal.

|br|

Optional Key
------------

- ``Artifact`` *(optional)*: Annotations of 1s marking where artifactual
  heartbeats occur in the signal. See `SQA.Cardio.identify_artifacts() <api.html#physioview.pipeline.SQA.Cardio.identify_artifacts>`_
  for artifact identification methods.

Loading Edit Data
=================
Input JSON files must be placed in the ``beat-editor/data`` subdirectory to
be recognized by the Beat Editor. When `accessing the Beat Editor from the
dashboard <beat-editor-getting-started.html#accessing-the-beat-editor>`_,
these files are automatically placed in this subdirectory.

Processing Edited Data
======================
The Beat Editor saves all edited data as a JSON file, using the same base
filename as the input file, but with the ``'_edited.json'`` suffix.

All edited files are written to the ``beat-editor/saved`` subdirectory and
may include entries with the following keys:

- ``x``: The x-coordinate (time or sample index) of the edited beat.
- ``y``: The signal value at the edited beat location.
- ``from``: The start of a segment marked as 'Unusable'.
- ``to``: The end of a segment marked as 'Unusable'.
- ``editType``: The type of edit performed. Possible values are ``ADD``, ``DELETE``, or ``UNUSABLE``.

Edited data can be processed using `physioview.process_beat_edits() <api
.html#physioview.physioview.process_beat_edits>`_ as shown in the following
example workflow:

.. code-block:: python

    import pandas as pd
    from physioview import physioview

    # Assuming you are working from the physioview/ project root
    orig_data = pd.read_json('beat-editor/data/sample_edit.json')
    edits = pd.read_json('beat-editor/saved/sample_edited.json')
    processed_data = physioview.process_beat_edits(orig_data, edits)

    # Write the processed data
    processed_data.to_csv('processed_data.csv', index = False)