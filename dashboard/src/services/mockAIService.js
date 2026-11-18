
/**
 * PhysioTrack - Mock AI Service
 *
 * Generates fake exercise measurements for UI testing.
 * Does NOT process video frames or perform real AI analysis.
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

export function createMockAIService(onResult) {
  let intervalId = null
  let stepIndex = 0
  let repCount = 0

  function emitResult() {
    const currentStep = MOVEMENT_SEQUENCE[stepIndex]

    // Count one repetition after a full simulated movement.
    if (stepIndex === MOVEMENT_SEQUENCE.length - 1) {
      repCount += 1
    }

    const result = {
      jointAngle: currentStep.jointAngle,
      movementState: currentStep.movementState,
      repCount: repCount,
      feedback: 'SIMULATION: No real movement analysis',
      isMock: true,
    }

    // Send the mock result to the React component.
    onResult(result)

    // Move to the next simulated frame.
    stepIndex = (stepIndex + 1) % MOVEMENT_SEQUENCE.length
  }

  return {
    start() {
      if (intervalId !== null) {
        return
      }

      emitResult()
      intervalId = setInterval(emitResult, 700)
    },

    stop() {
      if (intervalId !== null) {
        clearInterval(intervalId)
        intervalId = null
      }
    },

    reset() {
      this.stop()
      stepIndex = 0
      repCount = 0
    },
  }
}
