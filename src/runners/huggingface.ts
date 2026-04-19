import type { InferenceClient } from '@huggingface/inference'

import type { Message, ModelResponse, Runner, RunnerOptions, Scenario } from '../types.js'

/**
 * Minimal Hugging Face client contract the runner depends on. Kept narrow so
 * unit tests can inject a stub without pulling the full SDK surface.
 *
 * HF's `chatCompletion` method takes an OpenAI-shaped payload (role-tagged
 * messages, temperature, seed, max_tokens) and returns an OpenAI-shaped
 * response (`choices[].message.content`, `usage`, `model`, `finish_reason`).
 */
export interface HFClientLike {
  chatCompletion(params: HFChatCompletionParams): Promise<HFChatCompletionResponse>
}

export interface HFChatCompletionMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
}

export interface HFChatCompletionParams {
  model: string
  messages: HFChatCompletionMessage[]
  temperature?: number
  seed?: number
  max_tokens?: number
}

export interface HFChatCompletionResponse {
  id?: string
  model?: string
  choices: Array<{
    message: { role?: string; content?: string | null }
    finish_reason?: string | null
  }>
  usage?: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number }
  system_fingerprint?: string
}

export interface HuggingFaceRunnerOptions {
  /**
   * Inject a client for tests. If omitted, a real @huggingface/inference
   * client is constructed lazily on first run. Reads HF_TOKEN from env if
   * present; unset is permitted (public models work, gated models return 401).
   */
  client?: HFClientLike
  /**
   * Override max_tokens. Default undefined (let the model decide). Scenarios
   * can still set input.meta.maxTokens to cap output per-scenario.
   */
  defaultMaxTokens?: number
}

/**
 * Hugging Face Inference runner.
 *
 * The runner id is `hf:<org>/<repo>`, e.g. `hf:Qwen/Qwen3-4B-Instruct-2507`.
 * The suffix after `hf:` is the Hugging Face repo id (which may contain a
 * `/`) and is passed verbatim as the `model` field to `chatCompletion`.
 *
 * HF_TOKEN is read from env when present but never required. Public-model
 * access works without a token; gated or rate-limited models return 401,
 * which is surfaced through the standard runner error path.
 *
 * Per Hugging Face's model licensing, benchmark publication is permitted
 * under each model's upstream licence (Apache-2.0, Llama, Gemma, Qwen, etc.)
 * provided version and access date are disclosed. Those fields travel in
 * ModelResponse.meta.
 */
export function createHuggingFaceRunner(
  repoId: string,
  opts: HuggingFaceRunnerOptions = {},
): Runner {
  const id = `hf:${repoId}`
  let cached: HFClientLike | null = opts.client ?? null

  async function getClient(): Promise<HFClientLike> {
    if (cached) return cached
    const mod = (await import('@huggingface/inference')) as unknown as {
      InferenceClient: new (accessToken?: string) => InferenceClient
    }
    const token = process.env['HF_TOKEN']
    cached = new mod.InferenceClient(token) as unknown as HFClientLike
    return cached
  }

  return {
    id,
    provider: 'huggingface',
    model: repoId,
    async run(scenario: Scenario, runOpts: RunnerOptions = {}): Promise<ModelResponse> {
      const client = await getClient()
      const messages = prepareMessages(scenario.input.messages, runOpts.systemPrompt)
      const maxTokens = readMaxTokens(scenario, opts.defaultMaxTokens)
      const started = Date.now()

      const params: HFChatCompletionParams = { model: repoId, messages }
      if (runOpts.temperature !== undefined) params.temperature = runOpts.temperature
      if (runOpts.seed !== undefined) params.seed = runOpts.seed
      if (maxTokens !== undefined) params.max_tokens = maxTokens

      let apiResponse: HFChatCompletionResponse
      try {
        apiResponse = await client.chatCompletion(params)
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        throw new Error(
          `[${id}] chatCompletion failed for scenario "${scenario.id}": ${message}`,
        )
      }

      const latencyMs = Date.now() - started
      const output = extractContent(apiResponse)

      const extra: Record<string, unknown> = {
        finishReason: apiResponse.choices[0]?.finish_reason ?? null,
        promptTokens: apiResponse.usage?.prompt_tokens,
        completionTokens: apiResponse.usage?.completion_tokens,
      }
      if (maxTokens !== undefined) extra['maxTokens'] = maxTokens

      const response: ModelResponse = {
        runnerId: id,
        scenarioId: scenario.id,
        output,
        meta: {
          provider: 'huggingface',
          model: repoId,
          accessedAt: new Date().toISOString(),
          latencyMs,
          extra,
        },
      }
      if (apiResponse.model) response.meta.version = apiResponse.model
      if (runOpts.temperature !== undefined) response.meta.temperature = runOpts.temperature
      if (runOpts.seed !== undefined) response.meta.seed = runOpts.seed

      return response
    },
  }
}

/**
 * HF's chat endpoint uses role-tagged messages including system, matching
 * OpenAI shape. If the scenario has no system message and the runner was
 * given a fallback systemPrompt, prepend it as a system role.
 */
function prepareMessages(
  input: Message[],
  fallbackSystem?: string,
): HFChatCompletionMessage[] {
  const hasSystem = input.some((m) => m.role === 'system')
  const mapped: HFChatCompletionMessage[] = input.map((m) => ({
    role: m.role,
    content: m.content,
  }))
  if (!hasSystem && fallbackSystem) {
    mapped.unshift({ role: 'system', content: fallbackSystem })
  }
  const nonSystem = mapped.filter((m) => m.role !== 'system')
  if (nonSystem.length === 0) {
    throw new Error('huggingface: scenario must include at least one user or assistant message')
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

function extractContent(response: HFChatCompletionResponse): string {
  const content = response.choices[0]?.message?.content
  return typeof content === 'string' ? content.trim() : ''
}
