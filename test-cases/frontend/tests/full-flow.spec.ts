/**
 * End-to-end test cases for the DeepSyndrome frontend: authentication,
 * the ensemble prediction flow, Doctor Mode ground-truth confirmation,
 * Patient Records, and role-based access control.
 *
 * Prerequisites:
 *   - The frontend dev/production server must be running (see the main
 *     project's frontend/README.md), reachable at BASE_URL (default
 *     http://localhost:3000).
 *   - MONGODB_URI, MONGODB_DB and JWT_SECRET must be configured on that
 *     server (frontend/.env.local or the hosting platform's env vars).
 *
 * Run with:
 *   npm install
 *   npx playwright install chromium
 *   BASE_URL=http://localhost:3000 npm test
 */
import { test, expect, type Page } from "@playwright/test";
import path from "path";

const FIXTURE_IMAGE = path.join(__dirname, "..", "..", "fixtures", "sample_face.jpg");
const testEmail = `pw_test_${Date.now()}@example.com`;
const testPassword = "password123";

test.describe.serial("DeepSyndrome frontend — full flow", () => {
  let page: Page;

  test.beforeAll(async ({ browser }) => {
    page = await browser.newPage();
  });

  test.afterAll(async () => {
    await page.close();
  });

  test("TC-01: home page shows Log In / Sign Up when logged out", async () => {
    await page.goto("/");
    await expect(page.locator("header").getByRole("link", { name: "Log In" })).toBeVisible();
    await expect(page.locator("header").getByRole("link", { name: "Sign Up" })).toBeVisible();
  });

  test("TC-02: Sign Up navigates to the registration form", async () => {
    await page.locator("header").getByRole("link", { name: "Sign Up" }).click();
    await expect(page).toHaveURL(/\/register$/);
    await expect(page.getByRole("heading", { name: /Doctor Sign Up/i })).toBeVisible();
  });

  test("TC-03: registering a doctor auto-logs-in and redirects home", async () => {
    await page.fill('input[type="text"]', "Playwright Test Doctor");
    await page.fill('input[type="email"]', testEmail);
    await page.fill('input[type="password"]', testPassword);
    await page.getByRole("button", { name: "Create Account" }).click();

    await expect(page).toHaveURL("/", { timeout: 45_000 });
    await expect(page.locator("header")).toContainText("Dr. Playwright Test Doctor");
    await expect(page.locator("header").getByRole("button", { name: "Log out" })).toBeVisible();
  });

  test("TC-04: logout clears the session and reverts the header", async () => {
    await page.locator("header").getByRole("button", { name: "Log out" }).click();
    await expect(page.locator("header").getByRole("link", { name: "Log In" })).toBeVisible();
  });

  test("TC-05: login with the wrong password shows a generic error", async () => {
    await page.locator("header").getByRole("link", { name: "Log In" }).click();
    await expect(page).toHaveURL(/\/login$/);

    await page.fill('input[type="email"]', testEmail);
    await page.fill('input[type="password"]', "wrong-password");
    await page.getByRole("button", { name: "Log In" }).click();

    await expect(page.getByText("Invalid email or password")).toBeVisible();
    // Must remain logged out.
    await expect(page.locator("header").getByRole("link", { name: "Log In" })).toBeVisible();
  });

  test("TC-06: login with the correct password succeeds", async () => {
    await page.fill('input[type="password"]', testPassword);
    await page.getByRole("button", { name: "Log In" }).click();

    await expect(page).toHaveURL("/", { timeout: 45_000 });
    await expect(page.locator("header")).toContainText("Dr. Playwright Test Doctor");
  });

  test("TC-07: uploading a facial image shows a preview and enables Analyse", async () => {
    await page.setInputFiles('input[type="file"]', FIXTURE_IMAGE);
    await expect(page.getByRole("button", { name: /Analyse Image/i })).toBeEnabled();
  });

  test("TC-08: analysing the image returns an ensemble verdict with per-model votes", async () => {
    await page.getByRole("button", { name: /Analyse Image/i }).click();
    await expect(page.getByText("ENSEMBLE VERDICT")).toBeVisible({ timeout: 90_000 });
    await expect(page.getByText("Individual Model Votes")).toBeVisible();
    // All three models must report a confidence vote (each renders "XX.X% confident").
    await expect(page.getByText(/% confident/)).toHaveCount(3);
  });

  test("TC-09: doctor can confirm the ground-truth label for retraining", async () => {
    const confirmButton = page.getByRole("button", { name: /Confirm: (Control|Down Syndrome)/ }).first();
    await confirmButton.click();
    await expect(page.getByText("Saved for future training").first()).toBeVisible({ timeout: 15_000 });
  });

  test("TC-10: Patients page lets a logged-in doctor add and list a patient", async () => {
    await page.locator("header").getByRole("link", { name: "Patients" }).click();
    await expect(page).toHaveURL(/\/patients$/);

    await page.fill('input[placeholder="Full name"]', "Ali Raza");
    await page.fill('input[placeholder="Notes (optional)"]', "Referred for screening");
    await page.getByRole("button", { name: "Add" }).click();

    await expect(page.getByText("Ali Raza")).toBeVisible({ timeout: 15_000 });
  });

  test("TC-11: Admin dashboard blocks a non-admin doctor", async () => {
    await page.goto("/admin");
    await expect(page.getByText("doesn't have admin access")).toBeVisible();
  });

  test("TC-12: protected pages gate access once logged out", async () => {
    await page.locator("header").getByRole("button", { name: "Log out" }).click();
    await page.goto("/patients");
    await expect(page.getByText("You need to log in as a doctor first")).toBeVisible();
    // No patient data should ever be exposed to a logged-out visitor.
    await expect(page.getByText("Ali Raza")).not.toBeVisible();
  });
});
