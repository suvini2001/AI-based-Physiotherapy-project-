
/**
 * PhysioTrack - Mock AI Service
 *
 * Generates SIMULATED exercise measurements for UI testing.
 *
 * IMPORTANT:
 * - Does not analyze camera frames.
 * - Does not use the real AI pipeline.
 * - Does not communicate with the backend.
 * - Does not save patient session results.
 */

const MOVEMENT_SEQUENCE = [
  {
    jointAngle: 160,
    movementState: 'EXTENDED',
  },
  {
    jointAngle: 130,
    movementState: 'FLEXING',
  },
  {
    jointAngle: 95,
    movementState: 'FLEXING',
  },
  {
    jointAngle: 55,
    movementState: 'FLEXED',
  },
  {
    jointAngle: 100,
    movementState: 'EXTENDING',
  },
  {
    jointAngle: 140,
    movementState: 'EXTENDING',
  },
]

export function createMockAIService(onResult, targetReps = 10) {
  let intervalId = null
  let stepIndex = 0
  let repCount = 0
  let completed = false

  const validTarget =
    Number.isInteger(Number(targetReps)) &&
    Number(targetReps) > 0
      ? Number(targetReps)
      : 10

  function stopInterval() {
    if (intervalId !== null) {
      clearInterval(intervalId)
      intervalId = null
    }
  }

  function emitResult() {
    if (completed) return

    const currentStep = MOVEMENT_SEQUENCE[stepIndex]

    const lastStep =
      stepIndex === MOVEMENT_SEQUENCE.length - 1

    // One full simulated movement cycle counts as one rep.
    if (lastStep) {
      repCount = Math.min(repCount + 1, validTarget)
    }

    completed = repCount >= validTarget

    const result = {
      jointAngle: currentStep.jointAngle,
      movementState: currentStep.movementState,
      repCount,
      targetReps: validTarget,
      completed,
      feedback: completed
        ? 'SIMULATION COMPLETE: Target reached.'
        : 'SIMULATION: No real movement analysis',
      isMock: true,
    }

    // Send simulated results to the UI.
    onResult(result)

    if (completed) {
      stopInterval()
      return
    }

    stepIndex =
      (stepIndex + 1) % MOVEMENT_SEQUENCE.length
  }

  return {
    start() {
      if (intervalId !== null || completed) return

      // Create the interval before emitting the first result.
      intervalId = setInterval(emitResult, 700)
      emitResult()
    },

    stop() {
      stopInterval()
    },

    reset() {
      stopInterval()
      stepIndex = 0
      repCount = 0
      completed = false
    },

    getStatus() {
      return {
        repCount,
        targetReps: validTarget,
        completed,
        running: intervalId !== null,
      }
    },
  }
}
