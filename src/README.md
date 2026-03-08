Using DA3 with CUT3R memory

I cloned CUT3R into the src folder of the DA3 repository

For DA3, I am using the .venv in the project root

For CUT3R, I am reusing the conda virtual environment on my workstation saved under the name "ttt3r"

Approach:

1. Data: use robomimic lift_512 task. It has 2 views, separate the streams for each view into respective folders. Load them both and ensure that DA3 receives both inputs at the same time and that the reference view is fixed (DONT use saddle, use first)

2. Demo: clearly go step-by-step through the DA3 architecture, printing shapes and data metrics at every step to verify that it works properly up until the transformer tokens are received.