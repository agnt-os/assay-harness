import type OpenAI from 'openai'

import type { Message, ModelResponse, Runner, RunnerOptions, Scenario } from '../types.js'
import type {
  ChatCompletionCreateParams,
  ChatCompletionMessage,
  ChatCompletionResponse,
  OpenAIClientLike,
} from './openai.js'

export interface VllmRunnerOptions {
  /**
   * Inject a client for tests. If omitted, a real openai SDK client is
   * constructed lazily on first run, pointed at the local vLLM endpoint.
   */
  client?: OpenAIClientLike
  /**
   * Override the vLLM base URL. Takes precedence over VLLM_BASE_URL. If
   * omitted, falls back to process.env.VLLM_BASE_URL, then to
   * 'http://localhost:8000/v1'. A base without the '/v1' suffix has it
   * appended defensively so callers need not memorise the exact shape.
   */
  baseUrl?: string
  /**
   * Override max_tokens. Default undefined (let the server decide).
   * Scenarios can still set input.meta.maxTokens to cap output per-scenario.
   */
  defaultMaxTokens?: number
}

const DEFAULT_BASE_URL = 'http://localhost:8000/v1'

/**
 * Local vLLM runner.
 *
 * vLLM (https://vllm.ai) exposes an OpenAI-compatible
 * /v1/chat/completions endpoint, so this runner reuses the `openai` SDK with
 * a custom baseURL. It is intended for users who serve open-weights models
 * on their own hardware and want to score them against the same scenarios
 * the Agentsia Labs releases use.
 *
 * The model id is taken verbatim as the suffix of the runner id; it must
 * match whatever the local vLLM server exposes, e.g.
 * 'vllm:Qwen/Qwen3-4B-Instruct-2507' hits the model 'Qwen/Qwen3-4B-Instruct-2507'.
 *
 * VLLM_API_KEY is honoured when set (some deployments sit behind a proxy
 * that requires a key); otherwise the SDK is given the placeholder
 * 'not-needed' because the SDK itself requires a non-empty string.
 *
 * The resolved base URL is recorded in ModelResponse.meta.extra.baseUrl so
 * that reproducers can verify which endpoint was called.
 */
export function createVllmRunner(model: string, opts: VllmRunnerOptions = {}): Runner {
  const id = `vllm:${model}`
  const baseUrl = resolveBaseUrl(opts.baseUrl)
  let cached: OpenAIClientLike | null = opts.client ?? null

  async function getClient(): Promise<OpenAIClientLike> {
    if (cached) return cached
    const mod = (await import('openai')) as unknown as {
      default: new (init?: { apiKey?: string; baseURL?: string }) => OpenAI
    }
    const apiKey = process.env['VLLM_API_KEY'] ?? 'not-needed'
    cached = new mod.default({ apiKey, baseURL: baseUrl }) as unknown as OpenAIClientLike
    return cached
  }

  return {
    id,
    provider: 'vllm',
    model,
    async run(scenario: Scenario, runOpts: RunnerOptions = {}): Promise<ModelResponse> {
      const client = await getClient()
      const messages = prepareMessages(scenario.input.messages, runOpts.systemPrompt)
      const maxTokens = readMaxTokens(scenario, opts.defaultMaxTokens)
      const started = Date.now()

      const params: ChatCompletionCreateParams = { model, messages }
      if (runOpts.temperature !== undefined) params.temperature = runOpts.temperature
      if (runOpts.seed !== undefined) params.seed = runOpts.seed
      if (maxTokens !== undefined) params.max_tokens = maxTokens

      let apiResponse: ChatCompletionResponse
      try {
        apiResponse = await client.chat.completions.create(params)
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        const hint = isConnectionError(message)
          ? ` Confirm vLLM is running at ${baseUrl}.`
          : ''
        throw new Error(
          `[${id}] vllm chat.completions.create failed for scenario "${scenario.id}": ${message}${hint}`,
        )
      }

      const latencyMs = Date.now() - started
      const output = extractContent(apiResponse)

      const response: ModelResponse = {
        runnerId: id,
        scenarioId: scenario.id,
        output,
        meta: {
          provider: 'vllm',
          model,
          version: apiResponse.model,
          accessedAt: new Date().toISOString(),
          latencyMs,
          extra: {
            finishReason: apiResponse.choices[0]?.finish_reason ?? null,
            systemFingerprint: apiResponse.system_fingerprint ?? null,
            promptTokens: apiResponse.usage?.prompt_tokens,
            completionTokens: apiResponse.usage?.completion_tokens,
            responseId: apiResponse.id ?? null,
            baseUrl,
            ...(maxTokens !== undefined ? { maxTokens } : {}),
          },
        },
      }
      if (runOpts.temperature !== undefined) response.meta.temperature = runOpts.temperature
      if (runOpts.seed !== undefined) response.meta.seed = runOpts.seed

      return response
    },
  }
}

/**
 * Resolve the vLLM base URL with the documented precedence:
 *   opts.baseUrl -> process.env.VLLM_BASE_URL -> DEFAULT_BASE_URL.
 *
 * The OpenAI-compatible vLLM route is mounted at /v1. If the caller provides
 * a base that does not already end with '/v1' (with or without a trailing
 * slash), append it so the SDK emits the correct path.
 */
function resolveBaseUrl(fromOpts?: string): string {
  const raw = fromOpts ?? process.env['VLLM_BASE_URL'] ?? DEFAULT_BASE_URL
  const trimmed = raw.replace(/\/+$/, '')
  if (/\/v1$/.test(trimmed)) return trimmed
  return `${trimmed}/v1`
}

/**
 * Heuristic: does this error message look like the vLLM server is not
 * reachable at all (refused connection, unresolved host)? If so, the caller
 * is almost certainly running the harness without starting vLLM first, and
 * the generic SDK error is not helpful on its own.
 */
function isConnectionError(message: string): boolean {
  return (
    /ECONNREFUSED/i.test(message) ||
    /ENOTFOUND/i.test(message) ||
    /connection refused/i.test(message) ||
    /fetch failed/i.test(message)
  )
}

/**
 * vLLM follows OpenAI Chat Completions semantics: role-tagged messages,
 * system included. If the scenario has no system message and the runner
 * was given a fallback systemPrompt, prepend it as a system role.
 */
function prepareMessages(input: Message[], fallbackSystem?: string): ChatCompletionMessage[] {
  const hasSystem = input.some((m) => m.role === 'system')
  const mapped: ChatCompletionMessage[] = input.map((m) => ({ role: m.role, content: m.content }))
  if (!hasSystem && fallbackSystem) {
    mapped.unshift({ role: 'system', content: fallbackSystem })
  }
  const nonSystem = mapped.filter((m) => m.role !== 'system')
  if (nonSystem.length === 0) {
    throw new Error('vllm: scenario must include at least one user or assistant message')
  }
  return mapped
}

function readMaxTokens(scenario: Scenario, fallback?: number): number | undefined {
  const fromScenario = scenario.input.meta?.['maxTokens']
  if (typeof fromScenario === 'number' && Number.isFinite(fromScenario) && fromScenario > 0) {
    return Math.floor(fromScenario)
  }
  return fallback
}

function extractContent(response: ChatCompletionResponse): string {
  const content = response.choices[0]?.message?.content
  return typeof content === 'string' ? content.trim() : ''
}
