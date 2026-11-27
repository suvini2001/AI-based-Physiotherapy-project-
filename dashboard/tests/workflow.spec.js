import { test, expect } from '@playwright/test'
import { randomUUID } from 'node:crypto'

const apiURL = 'http://127.0.0.1:18001'
const password = () => process.env.WEB_TEST_PASSWORD

async function register(request) {
  const response = await request.post(`${apiURL}/patient/register-secure`, { data: { name: `Fictional ${randomUUID()}`, password: password() } })
  expect(response.status()).toBe(201)
  return response.json()
}
async function prescribe(request, patient, reps = 2) {
  const login = await request.post(`${apiURL}/auth/login`, { form: { username: 'DRTEST', password: password() } })
  const clinician = await login.json()
  const response = await request.post(`${apiURL}/api/prescriptions`, {
    headers: { Authorization: `Bearer ${clinician.access_token}` },
    data: { patient_id: patient.id, exercise: 'shoulder_abduction', target_reps: reps },
  })
  expect(response.ok()).toBeTruthy()
}
async function loginPatient(page, patient) {
  await page.goto('/patient')
  await page.getByPlaceholder('Example: PT439699').fill(patient.patient_id)
  await page.getByPlaceholder('Enter password', { exact: true }).fill(password())
  await page.getByRole('button', { name: 'Login Securely' }).click()
  await expect(page.getByRole('button', { name: 'Open Browser Exercise Demo' })).toBeVisible()
}
async function recordTracks(page) {
  await page.addInitScript(() => {
    window.testTracks = []
    const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices)
    navigator.mediaDevices.getUserMedia = async (...args) => {
      const stream = await original(...args)
      window.testTracks.push(...stream.getTracks())
      return stream
    }
  })
}

test('register, clinician prescribes, patient verifies latest prescription and logs out', async ({ page, context }) => {
  const name = `Fictional Browser ${randomUUID()}`
  await page.goto('/patient')
  await page.getByRole('button', { name: 'Register', exact: true }).click()
  await page.getByPlaceholder('Enter your full name').fill(name)
  await page.getByPlaceholder('Enter password', { exact: true }).fill(password())
  await page.getByPlaceholder('Confirm your password').fill(password())
  await page.getByRole('button', { name: 'Create Account' }).click()
  await expect(page.getByText('Registration Successful!')).toBeVisible()
  await page.getByRole('button', { name: 'Continue to Patient Dashboard' }).click()
  await expect(page.getByText('Awaiting clinician prescription')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Open Browser Exercise Demo' })).toBeDisabled()

  const clinicianPage = await context.newPage()
  await clinicianPage.goto('/login')
  await clinicianPage.getByPlaceholder('DR001').fill('DRTEST')
  await clinicianPage.locator('input[type=password]').fill(password())
  await clinicianPage.getByRole('button', { name: 'Sign in' }).click()
  await clinicianPage.getByText(name, { exact: true }).click()
  await clinicianPage.getByRole('button', { name: 'prescription', exact: true }).click()
  await clinicianPage.locator('select').selectOption('shoulder_abduction')
  await clinicianPage.locator('input[type=number]').first().fill('3')
  await clinicianPage.getByRole('button', { name: 'Save prescription' }).click()
  await expect(clinicianPage.getByRole('button', { name: 'Saved', exact: true })).toBeVisible()
  await page.reload()
  await expect(page.getByText('Target: 3 repetitions')).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Shoulder Abduction' })).toBeVisible()
  await page.getByRole('button', { name: 'Sign Out' }).click()
  await page.goto('/patient/exercise-demo')
  await expect(page).toHaveURL(/\/patient$/)
})

test('camera, simulation stop/reset/completion, no upload, navigation and logout cleanup', async ({ page, request }) => {
  const account = await register(request)
  await prescribe(request, account.patient)
  await recordTracks(page)
  await loginPatient(page, account.patient)
  await page.getByRole('button', { name: 'Open Browser Exercise Demo' }).click()
  await expect(page.getByRole('heading', { name: 'Shoulder Abduction' })).toBeVisible()
  const uploads = []
  page.on('request', r => { if (r.url().includes('/api/sessions') && r.method() === 'POST') uploads.push(r.url()) })
  await page.getByRole('button', { name: 'Start Camera', exact: true }).click()
  await expect(page.getByText('Camera status: running')).toBeVisible()
  await page.getByRole('button', { name: 'Start Simulation', exact: true }).click()
  await page.getByRole('button', { name: 'Stop Simulation', exact: true }).click()
  await expect(page.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '0')
  await page.getByRole('button', { name: 'Start Simulation', exact: true }).click()
  await expect(page.getByText('Simulation Complete!', { exact: true })).toBeVisible({ timeout: 15000 })
  await expect(page.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '2')
  await page.getByRole('button', { name: 'Reset Simulation' }).click()
  await expect(page.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '0')
  await page.getByRole('button', { name: 'Stop Camera' }).click()
  expect(await page.evaluate(() => window.testTracks.every(t => t.readyState === 'ended'))).toBeTruthy()
  await page.getByRole('button', { name: 'Start Camera', exact: true }).click()
  await expect(page.getByText('Camera status: running')).toBeVisible()
  await page.getByRole('link', { name: 'Patient dashboard' }).click()
  expect(await page.evaluate(() => window.testTracks.every(t => t.readyState === 'ended'))).toBeTruthy()
  await page.getByRole('button', { name: 'Open Browser Exercise Demo' }).click()
  await page.getByRole('button', { name: 'Start Camera', exact: true }).click()
  await expect(page.getByText('Camera status: running')).toBeVisible()
  await page.getByRole('button', { name: 'Start Simulation', exact: true }).click()
  await page.getByRole('button', { name: 'Sign Out' }).click()
  await expect(page).toHaveURL(/\/patient$/)
  expect(await page.evaluate(() => window.testTracks.every(t => t.readyState === 'ended'))).toBeTruthy()
  expect(await page.evaluate(() => sessionStorage.getItem('patient_token'))).toBeNull()
  expect(uploads).toEqual([])
  const history = await request.get(`${apiURL}/patient/me/last-session`, { headers: { Authorization: `Bearer ${account.access_token}` } })
  expect((await history.json()).date).toBeNull()
})

test('forged browser profile and invalid token cannot enter patient routes', async ({ page }) => {
  await page.goto('/patient')
  await page.evaluate(() => {
    localStorage.setItem('patient', JSON.stringify({ id: 1, patient_id: 'PTFAKE', target_reps: 50 }))
    sessionStorage.setItem('patient_token', 'forged.token.invalid')
  })
  await page.goto('/patient/exercise-demo')
  await expect(page).toHaveURL(/\/patient$/)
  await expect(page.getByRole('button', { name: 'Start Simulation' })).toHaveCount(0)
  expect(await page.evaluate(() => localStorage.getItem('patient'))).toBeNull()
})

test('camera denial is visible and a pending camera resolves safely after navigation', async ({ page, request }) => {
  const account = await register(request)
  await prescribe(request, account.patient)
  await loginPatient(page, account.patient)
  await page.getByRole('button', { name: 'Open Browser Exercise Demo' }).click()
  await page.evaluate(() => {
    navigator.mediaDevices.getUserMedia = async () => { throw new DOMException('Camera denied', 'NotAllowedError') }
  })
  await page.getByRole('button', { name: 'Start Camera', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('Camera denied')
  await page.evaluate(() => {
    window.pendingTracks = []
    navigator.mediaDevices.getUserMedia = () => new Promise(resolve => {
      window.resolveCamera = () => {
        const stream = document.createElement('canvas').captureStream()
        window.pendingTracks = stream.getTracks()
        resolve(stream)
      }
    })
  })
  await page.getByRole('button', { name: 'Start Camera', exact: true }).click()
  await page.getByRole('link', { name: 'Patient dashboard' }).click()
  await page.evaluate(() => window.resolveCamera())
  await expect.poll(() => page.evaluate(() => window.pendingTracks.length > 0 && window.pendingTracks.every(t => t.readyState === 'ended'))).toBeTruthy()
})

test('token expiration leaves exercise and releases the camera', async ({ page, request }) => {
  const account = await register(request)
  await prescribe(request, account.patient)
  await recordTracks(page)
  await loginPatient(page, account.patient)
  await page.clock.install()
  await page.getByRole('button', { name: 'Open Browser Exercise Demo' }).click()
  await page.getByRole('button', { name: 'Start Camera', exact: true }).click()
  await expect(page.getByText('Camera status: running')).toBeVisible()
  await page.clock.fastForward('01:01:00')
  await expect(page).toHaveURL(/\/patient$/)
  await expect.poll(() => page.evaluate(() => window.testTracks.every(t => t.readyState === 'ended'))).toBeTruthy()
  expect(await page.evaluate(() => sessionStorage.getItem('patient_token'))).toBeNull()
})
