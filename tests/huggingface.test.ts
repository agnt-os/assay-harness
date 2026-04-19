import { describe, expect, it } from 'vitest'

import {
  createHuggingFaceRunner,
  type HFChatCompletionParams,
  type HFClientLike,
} from '../src/runners/huggingface.js'
import type { Scenario } from '../src/types.js'

function makeStubClient(
  output: string,
  captured: { lastCall?: HFChatCompletionParams } = {},
): HFClientLike {
  return {
    async chatCompletion(params) {
      captured.lastCall = params
      return {
        id: 'hfchat-test-123',
        model: `${params.model}@sha256:abcdef`,
        choices: [
          { message: { role: 'assistant', content: output }, finish_reason: 'stop' },
        ],
        usage: { prompt_tokens: 33, completion_tokens: 19, total_tokens: 52 },
        system_fingerprint: 'hf_fp_test',
      }
    },
  }
}

const adtechScenario: Scenario = {
  id: 'adtech-003',
  axes: ['creative-qa'],
  input: {
    messages: [
      { role: 'system', content: 'You are a creative-QA reviewer.' },
      { role: 'user', content: 'Flag any policy violations in creative C-42.' },
    ],
  },
  rubric: { kind: 'programmatic', checker: 'non-empty' },
}

describe('huggingface runner', () => {
  it('maps scenario messages to chatCompletion params and returns a ModelResponse', async () => {
    const captured: { lastCall?: HFChatCompletionParams } = {}
    const client = makeStubClient('policy cleared', captured)
    const runner = createHuggingFaceRunner('Qwen/Qwen3-4B-Instruct-2507', { client })

    const response = await runner.run(adtechScenario, { temperature: 0, seed: 11 })

    expect(captured.lastCall?.model).toBe('Qwen/Qwen3-4B-Instruct-2507')
    expect(captured.lastCall?.messages).toHaveLength(2)
    expect(captured.lastCall?.messages[0]).toEqual({
      role: 'system',
      content: 'You are a creative-QA reviewer.',
    })
    expect(captured.lastCall?.messages[1]?.role).toBe('user')
    expect(captured.lastCall?.temperature).toBe(0)
    expect(captured.lastCall?.seed).toBe(11)

    expect(response.runnerId).toBe('hf:Qwen/Qwen3-4B-Instruct-2507')
    expect(response.scenarioId).toBe('adtech-003')
    expect(response.output).toBe('policy cleared')
    expect(response.meta.provider).toBe('huggingface')
    expect(response.meta.model).toBe('Qwen/Qwen3-4B-Instruct-2507')
    expect(response.meta.version).toBe('Qwen/Qwen3-4B-Instruct-2507@sha256:abcdef')
    expect(response.meta.latencyMs).toBeGreaterThanOrEqual(0)
    expect(response.meta.temperature).toBe(0)
    expect(response.meta.seed).toBe(11)
    expect(typeof response.meta.accessedAt).toBe('string')
    expect(response.meta.extra?.finishReason).toBe('stop')
    expect(response.meta.extra?.promptTokens).toBe(33)
    expect(response.meta.extra?.completionTokens).toBe(19)
  })

  it('prepends runner systemPrompt when the scenario has no system message', async () => {
    const captured: { lastCall?: HFChatCompletionParams } = {}
    const client = makeStubClient('ok', captured)
    const runner = createHuggingFaceRunner('google/gemma-4-E4B', { client })

    const scenario: Scenario = {
      id: 'no-sys',
      axes: ['a'],
      input: { messages: [{ role: 'user', content: 'go' }] },
      rubric: { kind: 'programmatic', checker: 'non-empty' },
    }

    await runner.run(scenario, { systemPrompt: 'You are helpful.' })
    expect(captured.lastCall?.messages[0]).toEqual({
      role: 'system',
      content: 'You are helpful.',
    })
    expect(captured.lastCall?.messages[1]).toEqual({ role: 'user', content: 'go' })
  })

  it('applies scenario-level maxTokens when present', async () => {
    const captured: { lastCall?: HFChatCompletionParams } = {}
    const client = makeStubClient('', captured)
    const runner = createHuggingFaceRunner('Qwen/Qwen3-4B-Instruct-2507', { client })
    const scenario: Scenario = {
      ...adtechScenario,
      input: { ...adtechScenario.input, meta: { maxTokens: 192 } },
    }
    await runner.run(scenario)
    expect(captured.lastCall?.max_tokens).toBe(192)
  })

  it('omits max_tokens when neither scenario nor runner provides it', async () => {
    const captured: { lastCall?: HFChatCompletionParams } = {}
    const client = makeStubClient('', captured)
    const runner = createHuggingFaceRunner('Qwen/Qwen3-4B-Instruct-2507', { client })
    await runner.run(adtechScenario)
    expect(captured.lastCall?.max_tokens).toBeUndefined()
  })

  it('throws a contextual error when the API call fails', async () => {
    const client: HFClientLike = {
      async chatCompletion() {
        throw new Error('service_unavailable')
      },
    }
    const runner = createHuggingFaceRunner('Qwen/Qwen3-4B-Instruct-2507', { client })
    await expect(runner.run(adtechScenario)).rejects.toThrow(
      /hf:Qwen\/Qwen3-4B-Instruct-2507.*adtech-003.*service_unavailable/,
    )
  })

  it('constructs without HF_TOKEN and surfaces no-token failures through the runner error path', async () => {
    const originalKey = process.env['HF_TOKEN']
    delete process.env['HF_TOKEN']
    try {
      // A no-token call against a gated repo returns 401. Simulate with a
      // stub that throws, mirroring how the real SDK surfaces the failure.
      const client: HFClientLike = {
        async chatCompletion() {
          throw new Error('401 Unauthorized: gated model requires a token')
        },
      }
      const runner = createHuggingFaceRunner('Qwen/Qwen3-4B-Instruct-2507', { client })
      await expect(runner.run(adtechScenario)).rejects.toThrow(
        /hf:Qwen\/Qwen3-4B-Instruct-2507.*adtech-003.*401 Unauthorized/,
      )
    } finally {
      if (originalKey !== undefined) process.env['HF_TOKEN'] = originalKey
    }
  })
})
