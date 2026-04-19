import { describe, expect, it } from 'vitest'

import type {
  ChatCompletionCreateParams,
  OpenAIClientLike,
} from '../src/runners/openai.js'
import { createVllmRunner } from '../src/runners/vllm.js'
import type { Scenario } from '../src/types.js'

function makeStubClient(
  output: string,
  captured: { lastCall?: ChatCompletionCreateParams } = {},
): OpenAIClientLike {
  return {
    chat: {
      completions: {
        async create(params) {
          captured.lastCall = params
          return {
            id: 'chatcmpl-vllm-test-1',
            model: `${params.model}@vllm`,
            choices: [{ message: { content: output }, finish_reason: 'stop' }],
            usage: { prompt_tokens: 24, completion_tokens: 8, total_tokens: 32 },
            system_fingerprint: 'fp_vllm_local',
          }
        },
      },
    },
  }
}

const adtechScenario: Scenario = {
  id: 'adtech-007',
  axes: ['bid-shading'],
  input: {
    messages: [
      { role: 'system', content: 'You are a bid-shading auditor.' },
      { role: 'user', content: 'Rate the shade applied to request REQ-42.' },
    ],
  },
  rubric: { kind: 'programmatic', checker: 'non-empty' },
}

describe('vllm runner', () => {
  it('forwards messages, uses the resolved baseURL, and returns a ModelResponse', async () => {
    const captured: { lastCall?: ChatCompletionCreateParams } = {}
    const client = makeStubClient('shaded cleanly', captured)
    const runner = createVllmRunner('Qwen/Qwen3-4B-Instruct-2507', {
      client,
      baseUrl: 'http://gpu-host:8000/v1',
    })

    const response = await runner.run(adtechScenario, { temperature: 0, seed: 11 })

    expect(captured.lastCall?.model).toBe('Qwen/Qwen3-4B-Instruct-2507')
    expect(captured.lastCall?.messages).toHaveLength(2)
    expect(captured.lastCall?.messages[0]).toEqual({
      role: 'system',
      content: 'You are a bid-shading auditor.',
    })
    expect(captured.lastCall?.messages[1]?.role).toBe('user')
    expect(captured.lastCall?.temperature).toBe(0)
    expect(captured.lastCall?.seed).toBe(11)

    expect(response.runnerId).toBe('vllm:Qwen/Qwen3-4B-Instruct-2507')
    expect(response.scenarioId).toBe('adtech-007')
    expect(response.output).toBe('shaded cleanly')
    expect(response.meta.provider).toBe('vllm')
    expect(response.meta.model).toBe('Qwen/Qwen3-4B-Instruct-2507')
    expect(response.meta.version).toBe('Qwen/Qwen3-4B-Instruct-2507@vllm')
    expect(response.meta.latencyMs).toBeGreaterThanOrEqual(0)
    expect(response.meta.temperature).toBe(0)
    expect(response.meta.seed).toBe(11)
    expect(typeof response.meta.accessedAt).toBe('string')
    expect(response.meta.extra?.finishReason).toBe('stop')
    expect(response.meta.extra?.systemFingerprint).toBe('fp_vllm_local')
    expect(response.meta.extra?.promptTokens).toBe(24)
    expect(response.meta.extra?.completionTokens).toBe(8)
    expect(response.meta.extra?.responseId).toBe('chatcmpl-vllm-test-1')
    expect(response.meta.extra?.baseUrl).toBe('http://gpu-host:8000/v1')
  })

  it('falls back to the default baseURL when neither opts nor env are set', async () => {
    const originalEnv = process.env['VLLM_BASE_URL']
    delete process.env['VLLM_BASE_URL']
    try {
      const client = makeStubClient('ok')
      const runner = createVllmRunner('local-model', { client })
      const response = await runner.run(adtechScenario)
      expect(response.meta.extra?.baseUrl).toBe('http://localhost:8000/v1')
    } finally {
      if (originalEnv !== undefined) process.env['VLLM_BASE_URL'] = originalEnv
    }
  })

  it('appends /v1 defensively when the provided baseURL lacks it', async () => {
    const client = makeStubClient('ok')
    const runner = createVllmRunner('local-model', {
      client,
      baseUrl: 'http://gpu-host:8000',
    })
    const response = await runner.run(adtechScenario)
    expect(response.meta.extra?.baseUrl).toBe('http://gpu-host:8000/v1')
  })

  it('opts.baseUrl takes precedence over VLLM_BASE_URL env', async () => {
    const originalEnv = process.env['VLLM_BASE_URL']
    process.env['VLLM_BASE_URL'] = 'http://env-host:9000/v1'
    try {
      const client = makeStubClient('ok')
      const runner = createVllmRunner('local-model', {
        client,
        baseUrl: 'http://opts-host:8000/v1',
      })
      const response = await runner.run(adtechScenario)
      expect(response.meta.extra?.baseUrl).toBe('http://opts-host:8000/v1')
    } finally {
      if (originalEnv === undefined) delete process.env['VLLM_BASE_URL']
      else process.env['VLLM_BASE_URL'] = originalEnv
    }
  })

  it('applies scenario-level maxTokens when present', async () => {
    const captured: { lastCall?: ChatCompletionCreateParams } = {}
    const client = makeStubClient('', captured)
    const runner = createVllmRunner('local-model', { client })
    const scenario: Scenario = {
      ...adtechScenario,
      input: { ...adtechScenario.input, meta: { maxTokens: 256 } },
    }
    const response = await runner.run(scenario)
    expect(captured.lastCall?.max_tokens).toBe(256)
    expect(response.meta.extra?.maxTokens).toBe(256)
  })

  it('adds a "Confirm vLLM is running" hint on a connection-refused error', async () => {
    const client: OpenAIClientLike = {
      chat: {
        completions: {
          async create() {
            throw new Error('connect ECONNREFUSED 127.0.0.1:8000')
          },
        },
      },
    }
    const runner = createVllmRunner('local-model', {
      client,
      baseUrl: 'http://localhost:8000/v1',
    })
    await expect(runner.run(adtechScenario)).rejects.toThrow(
      /vllm:local-model.*adtech-007.*ECONNREFUSED.*Confirm vLLM is running at http:\/\/localhost:8000\/v1/,
    )
  })
})
