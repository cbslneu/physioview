==========
Contribute
==========

Thank you for your interest in contributing to PhysioView. We welcome and
encourage contributions from the community, and your help is greatly
appreciated. Before you start contributing, please read and follow these
guidelines to ensure a smooth and collaborative process.

---------------
Getting Started
---------------

Fork the Repository
...................

1. Visit the |PhysioView GitHub repo|.
2. Click the "Fork" button in the top right corner to create your *own copy* of the repository.

Clone Your Fork
...............

In your Terminal, run the following command, replacing [your-username] with your GitHub username:

::

    $ git clone https://github.com/your-username/physioview.git

Set Up a Development Environment
................................

Navigate to the project directory and install all project dependencies:

::

    $ cd physioview
    $ virtualenv venv -p python3
    $ pip3 install -r requirements.txt


--------------
Making Changes
--------------

Branching
.........

Create a new branch for your feature or bug fix.
Please ensure your branch name is descriptive and reflects the purpose of your changes.
All code contributions should be written and documented following the
current style and docstring format of PhysioView code.

::

    $ git checkout -b feature/your-feature-name


Committing
..........

Commit your changes with a clear and concise message:

::

    $ git commit -m 'added [feature/fix]: brief description of your changes'


-------------------------
Submitting a Pull Request
-------------------------

1. Push your branch to your fork on GitHub:

::

    $ git push origin feature/your-feature-name

2. Visit the |PhysioView Github repo|.
3. Click the "New Pull Request" button.
4. Choose your branch and the ``dev`` branch.
5. Provide a clear and detailed description of your changes in the pull request.
6. Click "Create Pull Request" to submit your contribution.


--------------
Review Process
--------------

We will review your pull request and respond with any feedback or additional
necessary changes.
Once your pull request is approved, it will be merged into the main branch.


------------------
Terms of Agreement
------------------
By contributing to this project, you agree to abide by the terms of the
project's |license| and our :doc:`conduct`. Please be respectful and
considerate of others.


.. |PhysioView GitHub repo| raw:: html

    <a href="https://github.com/cbslneu/physioview" target="_blank">PhysioView GitHub repository</a>

.. |license| raw:: html

    <a href="https://github.com/cbslneu/physioview/blob/main/LICENSE" target="_blank">GPL-3.0 license</a>