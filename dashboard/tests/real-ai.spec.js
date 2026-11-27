import { test, expect } from '@playwright/test'
import { randomUUID } from 'node:crypto'
const api = 'http://127.0.0.1:18001'
async function account(request) {
  const response = await request.post(`${api}/patient/register-secure`, { data: { name: `Fictional AI ${randomUUID()}`, password: process.env.WEB_TEST_PASSWORD } })
  const user = await response.json()
  const login = await request.post(`${api}/auth/login`, { form: { username: 'DRTEST', password: process.env.WEB_TEST_PASSWORD } })
  const clinician = await login.json()
  await request.post(`${api}/api/prescriptions`, { headers: { Authorization: `Bearer ${clinician.access_token}` }, data: { patient_id: user.patient.id, exercise: 'shoulder_abduction', target_reps: 2 } })
  return user
}
async function enter(page, user) {
  await page.goto('/patient')
  await page.evaluate(token => sessionStorage.setItem('patient_token', token), user.access_token)
  await page.goto('/patient/exercise-real')
  await expect(page.getByRole('heading', { name: 'Camera exercise analysis' })).toBeVisible()
}

test('real worker processes synthetic camera and discards without saving', async ({ page, request }) => {
  test.skip(!process.env.AI_PYTHON, 'Requires isolated AI runtime')
  test.setTimeout(90000)
  const user = await account(request)
  await enter(page, user)
  await expect(page.getByRole('button', { name: 'Start real analysis' })).toBeDisabled()
  await page.getByRole('checkbox').check()
  await page.getByRole('button', { name: 'Start real analysis' }).click()
  await expect(page.getByText('Tracking: unavailable', { exact: false })).toBeVisible({ timeout: 45000 })
  await expect(page.getByRole('button', { name: 'Finish and save results' })).toBeDisabled()
  await page.getByRole('button', { name: 'Stop and discard / Reset' }).click()
  await expect(page.getByRole('button', { name: 'Start real analysis' })).toBeEnabled()
  const history = await request.get(`${api}/patient/me/last-session`, { headers: { Authorization: `Bearer ${user.access_token}` } })
  expect((await history.json()).date).toBeNull()
})

test('completion and save UI contract fixture releases camera', async ({ page, request }) => {
  const user = await account(request)
  await enter(page, user)
  let saves = 0
  // UI fixture only, not evidence of motion inference or database persistence.
  await page.route('**/patient/me/ai-sessions', route => route.fulfill({ json: { session_id: 'ui-fixture', prescription: { exercise: 'shoulder_abduction', target_reps: 2 } } }))
  await page.route('**/patient/me/ai-sessions/ui-fixture/frame', route => route.fulfill({ json: { phase: 'COMPLETE', rep_count: 2, joint_angle: 30, tracking: true, state: 'RESTING', angle_source: 'raw', feedback: 'Target reached.' } }))
  await page.route('**/patient/me/ai-sessions/ui-fixture/finish', route => { saves++; return route.fulfill({ json: { saved: true, session_id: 1, total_reps: 2 } }) })
  await page.route('**/patient/me/ai-sessions/ui-fixture', route => route.fulfill({ json: { discarded: true } }))
  await page.getByRole('checkbox').check()
  await page.getByRole('button', { name: 'Start real analysis' }).click()
  await expect(page.getByText('COMPLETE', { exact: true })).toBeVisible()
  expect(await page.locator('video').evaluate(video => video.srcObject)).toBeNull()
  await page.getByRole('button', { name: 'Finish and save results' }).click()
  await expect(page.getByText('Saved 2 repetitions.', { exact: false })).toBeVisible()
  expect(saves).toBe(1)
})

test('late AI startup is discarded after navigation', async ({ page, request }) => {
  const user = await account(request)
  await enter(page, user)
  let releaseStart, discarded = false
  const wait = new Promise(resolve => { releaseStart = resolve })
  await page.route('**/patient/me/ai-sessions', async route => { await wait; await route.fulfill({ json: { session_id: 'late-fixture', prescription: { target_reps: 2 } } }) })
  await page.route('**/patient/me/ai-sessions/late-fixture', route => { discarded = true; return route.fulfill({ json: { discarded: true } }) })
  await page.getByRole('checkbox').check()
  const started = page.waitForRequest(r => r.url().endsWith('/patient/me/ai-sessions'))
  await page.getByRole('button', { name: 'Start real analysis' }).click()
  await started
  await page.getByRole('link', { name: 'Patient dashboard' }).click()
  releaseStart()
  await expect.poll(() => discarded).toBeTruthy()
})
