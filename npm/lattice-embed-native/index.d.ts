export type LatticeEmbedErrorCode =
  | 'FL_EMBED_BAD_OPTIONS'
  | 'FL_EMBED_BAD_MODEL'
  | 'FL_EMBED_EMPTY_INPUT'
  | 'FL_EMBED_BAD_BATCH'
  | 'FL_EMBED_UNSUPPORTED_PLATFORM'
  | 'FL_EMBED_NATIVE_LOAD_FAILED'

export interface LatticeEmbedError extends Error {
  code?: LatticeEmbedErrorCode
}

export interface LoadModelOptions {
  /**
   * Local directory containing a BERT-family model: `model.safetensors`,
   * `config.json`, and a tokenizer (`vocab.txt` or `tokenizer.json`). This
   * v0 binding only supports loading from an already-local directory --
   * there is no remote-fetch tier here (contrast the `lattice-embed-wasm`
   * package's model registry + resolver).
   */
  modelPath: string

  /**
   * Optional model-family override (pooling strategy + expected dimension).
   * If omitted, the family is inferred from modelPath's final path
   * component, which works automatically for a directory named after its
   * canonical slug (e.g. `all-minilm-l6-v2`, `bge-small-en-v1.5`).
   */
  modelId?: string

  /**
   * Whether returned vectors are L2-normalized. Only `true` (or omitted) is
   * supported in this v0 binding: the underlying engine
   * (`BertModel::encode`/`encode_batch`) always L2-normalizes and has no
   * public non-normalizing path, so `normalize: false` is rejected with
   * `FL_EMBED_BAD_OPTIONS` rather than silently ignored.
   */
  normalize?: true
}

export interface EmbeddingBatch {
  readonly data: Float32Array
  readonly rows: number
  readonly dimensions: number
  readonly normalized: boolean

  /**
   * Return a zero-copy view into the flat batch buffer.
   */
  vector(index: number): Float32Array
}

export declare class EmbeddingModel {
  private constructor()

  readonly dimension: number
  readonly normalized: boolean

  embed(text: string): Promise<Float32Array>
  embedSync(text: string): Float32Array

  embedBatch(texts: readonly string[]): Promise<EmbeddingBatch>
  embedBatchSync(texts: readonly string[]): EmbeddingBatch

  /**
   * Convenience API. Each returned vector is a subarray view into the same flat
   * batch buffer, not an independent copy.
   */
  embedBatchList(texts: readonly string[]): Promise<Float32Array[]>
  embedBatchListSync(texts: readonly string[]): Float32Array[]
}

export declare function loadModel(options: LoadModelOptions): Promise<EmbeddingModel>
export declare function loadModelSync(options: LoadModelOptions): EmbeddingModel

export declare function splitBatch(batch: EmbeddingBatch): Float32Array[]
