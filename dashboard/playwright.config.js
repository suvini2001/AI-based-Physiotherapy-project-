import { defineConfig } from '@playwright/test'
import { randomUUID } from 'node:crypto'

process.env.WEB_TEST_PASSWORD ||= randomUUID()
export default defineConfig({
  testDir: './tests',
  timeout: 45000,
  workers: 1,
  use: {
    baseURL: 'http://localhost:5189',
    launchOptions: {
      executablePath: process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
      args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'],
    },
  },
  webServer: [
    { command: '../.venv-web/bin/python ../backend/tests/browser_server.py', url: 'http://127.0.0.1:18001/docs', reuseExistingServer: false },
    { command: 'npm run dev -- --host localhost --port 5189 --strictPort', url: 'http://localhost:5189', env: { VITE_API_URL: 'http://127.0.0.1:18001' }, reuseExistingServer: false },
  ],
})
