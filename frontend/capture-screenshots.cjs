// One-off script: drives the running dev app with headless Chromium and
// saves screenshots of key screens. Run with:
//   node capture-screenshots.js
// Requires both dev servers already running (frontend :5173, backend :8000).
const { chromium } = require("playwright");
const path = require("path");

const BASE = "http://localhost:5173";
const OUT = path.resolve(__dirname, "..", "screenshots");

async function shot(page, name) {
  await page.screenshot({ path: path.join(OUT, `${name}.png`), fullPage: true });
  console.log("saved", name);
}

async function login(page, email, password) {
  await page.goto(`${BASE}/login`);
  await page.fill('input[type="email"]', email);
  await page.fill('input[type="password"]', password);
  await page.click('button[type="submit"]');
  await page.waitForURL(`${BASE}/dashboard`, { timeout: 10000 });
}

async function logout(page) {
  await page.getByText("Log out").click();
  await page.waitForURL(`${BASE}/login`, { timeout: 10000 });
}

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

  // Landing page (public)
  await page.goto(`${BASE}/`);
  await page.waitForTimeout(300);
  await shot(page, "01-landing");

  // Login page
  await page.goto(`${BASE}/login`);
  await page.waitForTimeout(200);
  await shot(page, "02-login");

  // --- Staff ---
  await login(page, "staff@demo.hospital", "password123");
  await page.waitForTimeout(400);
  await shot(page, "03-staff-dashboard");

  await page.goto(`${BASE}/patients`);
  await page.waitForTimeout(400);
  await shot(page, "04-patients-list");

  await page.goto(`${BASE}/diagnose`);
  await page.waitForTimeout(400);
  await shot(page, "05-new-diagnosis-select-patient");

  // pick first patient in the list if present
  const firstPatient = page.locator("ul li button").first();
  if (await firstPatient.count()) {
    await firstPatient.click();
    await page.waitForTimeout(400);
    await shot(page, "06-new-diagnosis-choose-type");

    await page.getByText("Run Questionnaire").click();
    await page.waitForTimeout(400);
    await shot(page, "07-new-diagnosis-personal-info");
  }

  await logout(page);

  // --- Hospital Admin ---
  await login(page, "admin@demo.hospital", "password123");
  await page.waitForTimeout(400);
  await shot(page, "08-hospital-admin-dashboard");

  await page.goto(`${BASE}/admin/api-access`);
  await page.waitForTimeout(400);
  await shot(page, "09-api-access");

  await logout(page);

  // --- Doctor ---
  await login(page, "doctor@demo.hospital", "password123");
  await page.waitForTimeout(400);
  await shot(page, "10-doctor-dashboard");

  await page.goto(`${BASE}/cases`);
  await page.waitForTimeout(400);
  await shot(page, "11-my-cases");

  await logout(page);

  // --- Platform Admin ---
  await login(page, "platform@autism.dev", "password123");
  await page.waitForTimeout(400);
  await shot(page, "12-platform-admin-dashboard");

  await page.goto(`${BASE}/platform/hospitals`);
  await page.waitForTimeout(400);
  await shot(page, "13-platform-hospitals");

  await page.goto(`${BASE}/platform/questions`);
  await page.waitForTimeout(400);
  await shot(page, "14-question-bank");

  await page.goto(`${BASE}/platform/usage`);
  await page.waitForTimeout(400);
  await shot(page, "15-platform-usage");

  await page.goto(`${BASE}/platform/audit-log`);
  await page.waitForTimeout(400);
  await shot(page, "16-audit-log");

  await browser.close();
  console.log("done");
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
