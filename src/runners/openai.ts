import type OpenAI from 'openai'

import type { Message, ModelResponse, Runner, RunnerOptions, Scenario } from '../types.js'

/**
 * Minimal OpenAI client contract the runner depends on. Kept narrow so
 * unit tests can inject a stub without pulling the full SDK surface.
 */
export interface OpenAIClientLike {
  chat: {
    completions: {
      create(params: ChatCompletionCreateParams): Promise<ChatCompletionResponse>
    }
  }
}

export interface ChatCompletionMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
}

export interface ChatCompletionCreateParams {
  model: string
  messages: ChatCompletionMessage[]
  temperature?: number
  seed?: number
  max_tokens?: number
}

export interface ChatCompletionResponse {
  id?: string
  model: string
  choices: Array<{
    message: { content?: string | null }
    finish_reason?: string | null
  }>
  usage?: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number }
  system_fingerprint?: string
}

export interface OpenAIRunnerOptions {
  /**
   * Inject a client for tests. If omitted, a real openai SDK client is
   * constructed lazily on first run. Requires OPENAI_API_KEY in env.
   */
  client?: OpenAIClientLike
  /**
   * Override max_tokens. Default undefined (let the model decide). Scenarios
   * can still set input.meta.maxTokens to cap output per-scenario.
   */
  defaultMaxTokens?: number
}

/**
 * OpenAI Chat Completions runner.
 *
 * Reads OPENAI_API_KEY from env, unless an explicit client is provided via
 * OpenAIRunnerOptions. The model id is taken verbatim as the suffix of the
 * runner id, e.g. 'openai:gpt-6' resolves to the API model 'gpt-6'.
 *
 * Per the OpenAI Sharing & Publication Policy, benchmark publication is
 * permitted with disclosure of model version and access date per run.
 * ModelResponse.meta.version carries the server-reported model id (which
 * pins the dated snapshot even when the runner id uses an alias).
 * system_fingerprint travels in meta.extra for reproducibility analysis.
 */
export function createOpenAIRunner(model: string, opts: OpenAIRunnerOptions = {}): Runner {
  const id = `openai:${model}`
  let cached: OpenAIClientLike | null = opts.client ?? null

  async function getClient(): Promise<OpenAIClientLike> {
    if (cached) return cached
    const mod = (await import('openai')) as unknown as {
      default: new (init?: { apiKey?: string }) => OpenAI
    }
    const apiKey = process.env['OPENAI_API_KEY']
    if (!apiKey) {
      throw new Error(
        `[${id}] OPENAI_API_KEY is not set. ` +
          `Set it in the environment or pass a client via createOpenAIRunner's opts.`,
      )
    }
    cached = new mod.default({ apiKey }) as unknown as OpenAIClientLike
    return cached
  }

  return {
    id,
    provider: 'openai',
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
        throw new Error(
          `[${id}] chat.completions.create failed for scenario "${scenario.id}": ${message}`,
        )
      }

      const latencyMs = Date.now() - started
      const output = extractContent(apiResponse)

      const response: ModelResponse = {
        runnerId: id,
        scenarioId: scenario.id,
        output,
        meta: {
          provider: 'openai',
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
 * OpenAI takes role-tagged messages including system. If the scenario has
 * no system message and the runner was given a fallback systemPrompt,
 * prepend it as a system role.
 */
function prepareMessages(input: Message[], fallbackSystem?: string): ChatCompletionMessage[] {
  const hasSystem = input.some((m) => m.role === 'system')
  const mapped: ChatCompletionMessage[] = input.map((m) => ({ role: m.role, content: m.content }))
  if (!hasSystem && fallbackSystem) {
    mapped.unshift({ role: 'system', content: fallbackSystem })
  }
  const nonSystem = mapped.filter((m) => m.role !== 'system')
  if (nonSystem.length === 0) {
    throw new Error('openai: scenario must include at least one user or assistant message')
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
