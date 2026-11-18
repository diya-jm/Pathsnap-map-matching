# PathSnap
Reconstruct directed road segments from noisy GPS observations. The classifier scores nearby segments and Viterbi selects a sequence across the trip.

The root application and saved metrics use eight features. The ten-feature Review 1 implementation, notebook and demo are under pathsnap/. The saved root results describe the eight-feature run.

## Run the development demo
Use Python 3.12 x64. From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
New-Item -ItemType Directory -Force data | Out-Null
Copy-Item -Recurse -LiteralPath pathsnap/data/raw -Destination "data/Student folder"
.\.venv\Scripts\python serve.py api
```

In a second terminal, run `py serve.py ui` and open http://127.0.0.1:8501. Select a trip and click **Reconstruct route**. The map draws the coordinates returned by the API and works without internet. On this development computer, `Start-PathSnap.ps1` can start either component using the installed project libraries.

## Two-laptop evaluation
1. Put the repo, model, dependencies and organizer-provided evaluation folder on Laptop 1. The folder must directly contain one subfolder per trip. Filenames share the trip ID. The optional reference file stays on this machine. Use the student-test folder for rehearsals only.
2. On Laptop 1 run `.\.venv\Scripts\python serve.py api --host 0.0.0.0 --trips "C:\path\to\evaluation_folder"` from the repository root. Check the Wi-Fi IPv4 address using `ipconfig`. Keep this terminal visible. Use a shared, trusted LAN and allow inbound port 8000 on the private network if Windows asks.
3. Laptop 2 needs only Python, `serve.py` and the `ui` folder. Run `py serve.py ui`, open http://127.0.0.1:8501, enter `http://LAPTOP_1_IP:8000` in **API address**, then click **Connect**. Verify ten organizer-provided options. Select any trip and reconstruct it.
4. Capture the returned route and scores on Laptop 2 and the matching successful GET and score log in Laptop 1's terminal. Save as `evidence/ui-laptop2.png` and `evidence/api-laptop1.png`. Run `python report.py` to include them in the consolidated PDF. The team should review and explain the results before publishing.

## Reproduce the experiments
With the virtual environment active and the raw development archive extracted as above:

```powershell
python prepare.py
python train.py
python maps.py
python report.py
```

`prepare.py` writes a per-trip inventory and reusable candidate features. `train.py` records fold scores, all five baselines, eight Bayesian trials, a tuned model, the decoder comparison and a serialized estimator. The report generator accepts an exported view of the generated Folium map at `evidence/folium.png`. Open `maps/student_test_00000007.html` to explore the map. Its JavaScript and CSS are bundled. OpenStreetMap tiles are unnecessary. Metrics in the PDF come from the CSVs. Exact run times depend on the machine.

## Where to look
- `pathsnap.py` contains feature extraction and the Viterbi implementation. It reads only GPS and network data.
- `reference.py` aligns the supplied ordered route to GPS times for training labels and scoring. The point labels are estimates. The server loads reference information after predictions are complete.
- `results/` contains measured values, per-trip EDA, validation folds, tuning trials and decoding results. `artifacts/` contains the fitted model and configuration. Load only trusted model files.
- `api.py`, `serve.py` and `ui/index.html` implement the two-laptop application. GET /trips lists trip IDs. GET /match/{trip_id} returns predictions, GPS, geometry, elapsed time and aggregate scores. Missing reference data disables reference-dependent scores.
- `maps.py` and `report.py` generate geographical and experimental figures. The reference layer is used only for the development report map.
- `report/PathSnap.pdf` is the consolidated technical report. Record the team's verified authorship, review its analysis and disclose assistance according to event rules.


