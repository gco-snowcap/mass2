"""Live analysis: apply a saved mass2 recipe to pulse records as they are written to an Arrow IPC stream.

The core, which a real instrument uses:
    arrow_stream.py   write, and follow while growing, Arrow IPC stream files of polars DataFrames
    states.py         follow a DASTARD experiment-state file and label records with their state
    histogram.py      per-channel, per-state energy histograms in fixed time slices
    apply_recipe.py   mass2-live-apply: the loop that applies the recipe and writes results and histograms
    fit.py            mass2-live-fit: refits one line on the summed histograms, in its own process

Built on the core, kept separate:
    viewer/           mass2-live-view: the web page and its server
    demo/             mass2-live-demo and mass2-live-export: simulator, demo datasets, launcher, shareable page
"""
