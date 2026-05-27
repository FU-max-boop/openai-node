import * as Errors from './error';
import { OpenAI } from './client';
import type { ApiKeySetter, ClientOptions } from './client';
import { isObj, readEnv } from './internal/utils';
import type { RequestOptions } from './internal/request-options';
import type { ResponseStreamParams } from './lib/responses/ResponseStream';
import { addOutputText } from './lib/ResponsesParser';
import * as API from './resources/index';
import * as ResponsesAPI from './resources/responses/responses';

const BEDROCK_RESPONSES_ONLY_ERROR =
  'Amazon Bedrock support in the OpenAI SDK only supports the Responses API. Use `client.responses` with `BedrockOpenAI` instead.';

const BEDROCK_CAPABILITIES = Object.freeze({
  responsesWebsocket: false,
  unsupportedToolTypes: new Set([
    'apply_patch',
    'code_interpreter',
    'computer',
    'computer_use',
    'computer_use_preview',
    'file_search',
    'image_generation',
    'local_shell',
    'mcp',
    'shell',
    'web_search',
    'web_search_2025_08_26',
    'web_search_preview',
    'web_search_preview_2025_03_11',
  ]),
  unsupportedInputItemTypes: new Set([
    'apply_patch_call',
    'apply_patch_call_output',
    'code_interpreter_call',
    'computer_call',
    'computer_call_output',
    'file_search_call',
    'image_generation_call',
    'mcp_approval_request',
    'mcp_approval_response',
    'mcp_call',
    'mcp_list_tools',
    'shell_call',
    'shell_call_output',
    'web_search_call',
  ]),
  unsupportedIncludePrefixes: [
    'code_interpreter_call.',
    'computer_call_output.',
    'file_search_call.',
    'web_search_call.',
  ],
});

export interface BedrockClientOptions extends Omit<
  ClientOptions,
  'apiKey' | 'adminAPIKey' | 'baseURL' | 'workloadIdentity'
> {
  /**
   * Bedrock bearer token used for authentication.
   *
   * Defaults to process.env['AWS_BEARER_TOKEN_BEDROCK'].
   */
  apiKey?: string | null | undefined;

  /**
   * Bedrock API root.
   *
   * Defaults to process.env['AWS_BEDROCK_BASE_URL'], or derives
   * `https://bedrock-mantle.<region>.api.aws/openai/v1` from `awsRegion`,
   * process.env['AWS_REGION'], or process.env['AWS_DEFAULT_REGION'].
   */
  baseURL?: string | null | undefined;

  /**
   * BedrockOpenAI only supports Bedrock bearer token authentication.
   */
  adminAPIKey?: never;

  /**
   * BedrockOpenAI only supports Bedrock bearer token authentication.
   */
  workloadIdentity?: never;

  /**
   * AWS region used to derive the default Bedrock Mantle endpoint.
   *
   * Defaults to process.env['AWS_REGION'] or process.env['AWS_DEFAULT_REGION'].
   */
  awsRegion?: string | undefined;

  /**
   * A function that returns a Bedrock bearer token and is invoked before each request.
   */
  bedrockTokenProvider?: ApiKeySetter | undefined;
}

function unsupportedResource(resourceName: string): never {
  throw new Errors.OpenAIError(`${BEDROCK_RESPONSES_ONLY_ERROR} \`${resourceName}\` is not supported.`);
}

function unsupportedFeature(featureName: string): never {
  throw new Errors.OpenAIError(`Amazon Bedrock does not support \`${featureName}\` through this SDK client.`);
}

function deriveBedrockBaseURL(awsRegion: string | undefined): string {
  const region = awsRegion?.trim();
  if (!region) {
    throw new Errors.OpenAIError(
      'Must provide one of the `baseURL` or `awsRegion` arguments, or set the `AWS_BEDROCK_BASE_URL`, `AWS_REGION`, or `AWS_DEFAULT_REGION` environment variable.',
    );
  }

  return `https://bedrock-mantle.${region}.api.aws/openai/v1`;
}

function normalizeBedrockBaseURL(baseURL: string): string {
  const url = new URL(baseURL);
  const responsesMatch = url.pathname.match(/\/responses(?:\/.*)?$/);
  if (responsesMatch?.index !== undefined) {
    url.pathname = url.pathname.slice(0, responsesMatch.index) || '/';
  }

  return url.toString().replace(/\/$/, '');
}

function recordType(value: unknown): string | undefined {
  if (!isObj(value)) return undefined;

  const type = value['type'];
  return typeof type === 'string' ? type : undefined;
}

function objectValues(value: unknown): unknown[] {
  if (Array.isArray(value)) return value;
  if (isObj(value)) return Object.values(value);

  return [];
}

function validateToolLike(value: unknown): void {
  const type = recordType(value);
  if (type && BEDROCK_CAPABILITIES.unsupportedToolTypes.has(type)) {
    unsupportedFeature(`responses tools of type \`${type}\``);
  }

  if (type === 'allowed_tools' && isObj(value)) {
    for (const tool of objectValues(value['tools'])) {
      validateToolLike(tool);
    }
  }
}

function validateInputValue(value: unknown): void {
  const type = recordType(value);
  if (type && BEDROCK_CAPABILITIES.unsupportedInputItemTypes.has(type)) {
    unsupportedFeature(`responses input items of type \`${type}\``);
  }

  for (const nestedValue of objectValues(value)) {
    validateInputValue(nestedValue);
  }
}

function validateInclude(include: unknown): void {
  for (const item of objectValues(include)) {
    if (typeof item !== 'string') continue;
    if (BEDROCK_CAPABILITIES.unsupportedIncludePrefixes.some((prefix) => item.startsWith(prefix))) {
      unsupportedFeature(`responses include value \`${item}\``);
    }
  }
}

type BedrockResponseBody = {
  include?: unknown;
  input?: unknown;
  tool_choice?: unknown;
  tools?: unknown;
};

function validateResponseCreateBody(body: BedrockResponseBody): void {
  for (const tool of objectValues(body.tools)) {
    validateToolLike(tool);
  }

  if (body.tool_choice != null) {
    validateToolLike(body.tool_choice);
  }

  if (body.include != null) {
    validateInclude(body.include);
  }

  if (body.input != null) {
    validateInputValue(body.input);
  }
}

function addBedrockOutputText<ResponseT extends ResponsesAPI.Response>(response: ResponseT): ResponseT {
  if (!Object.getOwnPropertyDescriptor(response, 'output_text')) {
    addOutputText(response);
  }

  return response;
}

function guardBedrockResponses(responses: API.Responses): API.Responses {
  const create = responses.create.bind(responses);
  const compact = responses.compact.bind(responses);
  const retrieve = responses.retrieve.bind(responses);
  const stream = responses.stream.bind(responses);
  const inputItems = responses.inputItems;
  const inputItemsList = inputItems.list.bind(inputItems);
  const inputTokens = responses.inputTokens;
  const inputTokensCount = inputTokens.count.bind(inputTokens);

  responses.create = ((body: ResponsesAPI.ResponseCreateParams, options?: RequestOptions) => {
    validateResponseCreateBody(body);
    return create(body, options);
  }) as API.Responses['create'];

  responses.compact = ((body: ResponsesAPI.ResponseCompactParams, options?: RequestOptions) => {
    validateResponseCreateBody(body);
    return compact(body, options);
  }) as API.Responses['compact'];

  responses.retrieve = ((
    responseID: string,
    query?: ResponsesAPI.ResponseRetrieveParams,
    options?: RequestOptions,
  ) => {
    validateInclude(query?.include);
    return retrieve(responseID, query, options);
  }) as API.Responses['retrieve'];

  responses.stream = ((body: ResponseStreamParams, options?: RequestOptions) => {
    if (!('response_id' in body)) {
      validateResponseCreateBody(body);
    }

    const responseStream = stream(body, options);
    const finalResponse = responseStream.finalResponse.bind(responseStream);
    responseStream.finalResponse = async () => addBedrockOutputText(await finalResponse());

    return responseStream;
  }) as API.Responses['stream'];

  inputItems.list = ((responseID, query, options) => {
    validateInclude(query?.include);
    return inputItemsList(responseID, query, options);
  }) as typeof inputItems.list;

  inputTokens.count = ((body, options) => {
    validateResponseCreateBody(body ?? {});
    return inputTokensCount(body, options);
  }) as typeof inputTokens.count;

  return responses;
}

function defineUnsupportedResource(client: BedrockOpenAI, resourceName: string): void {
  Object.defineProperty(client, resourceName, {
    configurable: true,
    enumerable: true,
    get: () => unsupportedResource(resourceName),
  });
}

/** API Client for interfacing with Amazon Bedrock's OpenAI-compatible Responses API. */
export class BedrockOpenAI extends OpenAI {
  private readonly bedrockTokenProvider: ApiKeySetter | undefined;

  /**
   * API Client for interfacing with Amazon Bedrock's OpenAI-compatible Responses API.
   *
   * @param {string | null | undefined} [opts.apiKey=process.env['AWS_BEARER_TOKEN_BEDROCK'] ?? null]
   * @param {string | null | undefined} [opts.baseURL=process.env['AWS_BEDROCK_BASE_URL'] ?? derived from opts.awsRegion or AWS_REGION/AWS_DEFAULT_REGION]
   * @param {string | undefined} [opts.awsRegion=process.env['AWS_REGION'] ?? process.env['AWS_DEFAULT_REGION'] ?? undefined]
   * @param {ApiKeySetter | undefined} opts.bedrockTokenProvider - A function that returns a Bedrock bearer token and is invoked before each request.
   */
  constructor({
    baseURL = readEnv('AWS_BEDROCK_BASE_URL'),
    apiKey,
    awsRegion = readEnv('AWS_REGION') ?? readEnv('AWS_DEFAULT_REGION'),
    bedrockTokenProvider,
    adminAPIKey,
    workloadIdentity,
    ...opts
  }: BedrockClientOptions = {}) {
    if (adminAPIKey || workloadIdentity) {
      throw new Errors.OpenAIError('BedrockOpenAI only supports Bedrock bearer token authentication.');
    }

    if (apiKey === undefined && !bedrockTokenProvider) {
      apiKey = readEnv('AWS_BEARER_TOKEN_BEDROCK') ?? null;
    }

    if (typeof (apiKey as unknown) === 'function') {
      throw new Errors.OpenAIError(
        'Pass refreshable Bedrock credentials via `bedrockTokenProvider`, not `apiKey`.',
      );
    }

    if (apiKey && bedrockTokenProvider) {
      throw new Errors.OpenAIError(
        'The `apiKey` and `bedrockTokenProvider` arguments are mutually exclusive; only one can be passed at a time.',
      );
    }

    if (!apiKey && !bedrockTokenProvider) {
      throw new Errors.OpenAIError(
        'Missing credentials. Please pass an `apiKey` or `bedrockTokenProvider`, or set the `AWS_BEARER_TOKEN_BEDROCK` environment variable.',
      );
    }

    const configuredBaseURL = baseURL?.trim() ? baseURL : deriveBedrockBaseURL(awsRegion);

    super({
      apiKey: bedrockTokenProvider ?? apiKey,
      adminAPIKey: null,
      baseURL: normalizeBedrockBaseURL(configuredBaseURL),
      ...opts,
    });

    this.bedrockTokenProvider = bedrockTokenProvider;
    this.responses = guardBedrockResponses(new API.Responses(this));

    for (const resourceName of [
      'admin',
      'audio',
      'batches',
      'beta',
      'chat',
      'completions',
      'containers',
      'conversations',
      'embeddings',
      'evals',
      'files',
      'fineTuning',
      'graders',
      'images',
      'models',
      'moderations',
      'realtime',
      'skills',
      'uploads',
      'vectorStores',
      'videos',
      'webhooks',
    ]) {
      defineUnsupportedResource(this, resourceName);
    }
  }

  override withOptions(options: Partial<BedrockClientOptions>): this {
    const bedrockTokenProvider =
      options.apiKey !== undefined ? undefined : (options.bedrockTokenProvider ?? this.bedrockTokenProvider);

    return super.withOptions({
      ...options,
      ...(bedrockTokenProvider ? { apiKey: undefined, bedrockTokenProvider } : {}),
    } as Partial<ClientOptions>);
  }
}

export function isBedrockOpenAI(client: OpenAI): client is BedrockOpenAI {
  return client instanceof BedrockOpenAI;
}

export function supportsBedrockResponsesWebsocket(): boolean {
  return BEDROCK_CAPABILITIES.responsesWebsocket;
}
