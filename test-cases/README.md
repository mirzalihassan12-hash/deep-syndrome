# DeepSyndrome — Test Cases

Automated, runnable test cases for the DeepSyndrome project, kept as a
dedicated folder within the repo so it can be pointed to separately from the
rest of the source code for grading.

## Contents

- `python/` — pytest suite for the Python ML pipeline (dataset near-duplicate
  detection, face cropping, image preprocessing, and ensemble prediction
  logic). 15 test cases.
- `frontend/` — Playwright end-to-end test suite for the web application
  (authentication, image analysis, Doctor Mode, Patient Records, Admin
  access control). 12 test cases.
- `fixtures/` — a sample facial image used by both suites.
- `DeepSyndrome_Frontend_Test_Report.docx` — a documented test report (steps,
  expected vs. actual results, screenshots) covering the same 12 frontend
  test cases, for a human-readable record alongside the runnable code.

## Running the Python tests

Requires the main project's dependencies (torch, torchvision, timm,
opencv-python, albumentations, numpy, pillow — see the project's
`requirements_train.txt`).

```bash
cd test-cases/python
pip install -r requirements-test.txt
pytest -v
```

`conftest.py` finds the project source (`inference.py`, `face_crop.py`,
`training/`) automatically since this folder lives at the project root. If
you've copied `test-cases/` out on its own, set `PROJECT_ROOT` to point at
the project source instead:
```bash
PROJECT_ROOT=/path/to/deep-syndrome pytest -v
```

Model checkpoint files (`.pth`, 100MB+ each) are intentionally not required —
the inference tests verify the preprocessing and ensemble-averaging *logic*
by substituting lightweight mock models with known outputs, rather than
depending on the real trained weights.

**Result:** 15/15 passed at time of submission.

## Running the frontend tests

Requires the frontend application running and reachable (locally via
`npm run dev`/`npm start` in the project's `frontend/` folder, or against a
deployed URL), with its environment variables configured
(`MONGODB_URI`, `MONGODB_DB`, `JWT_SECRET`).

```bash
cd frontend
npm install
npx playwright install chromium
BASE_URL=http://localhost:3000 npm test
```

Test report (screenshots, trace on failure) is written to
`frontend/playwright-report/` after running.

**Result:** 12/12 passed at time of submission.

## Test case summary

| Suite | # | Covers |
|---|---|---|
| Python | 15 | Near-duplicate/leakage detection, dataset folder discovery, face cropping + fallback, image preprocessing, ensemble averaging math |
| Frontend | 12 | Login/registration, session handling, wrong-password error handling, image upload + ensemble prediction, Doctor Mode ground-truth confirmation, Patient Records, Admin role-based access control, logged-out access gating |
