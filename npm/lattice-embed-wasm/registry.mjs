// Model registry: static metadata for every embedding model this package
// knows how to load. Each entry describes what pooling strategy to apply,
// the output dimensionality, the three source files the wasm core needs
// (see wasm/lattice_embed.d.ts: `LatticeEmbedder` takes model, config, and
// tokenizer bytes), and where to fetch those files from if they are not
// already cached locally.
//
// Weight hosting: GitHub release assets on this repository, pinned by
// content hash so a compromised or edited release asset is rejected before
// use (see resolve.mjs). `WEIGHTS_RELEASE_TAG` is publish-time config: it is
// set once the model weights are actually uploaded as release assets, and
// every consumer of this package picks it up automatically once it lands
// here (no code change needed downstream).

export const WEIGHTS_RELEASE_TAG = 'embed-weights-v1';

const RELEASE_ASSET_BASE =
  'https://github.com/ohdearquant/lattice/releases/download';

function releaseUrl(assetName) {
  return `${RELEASE_ASSET_BASE}/${WEIGHTS_RELEASE_TAG}/${assetName}`;
}

export const DEFAULT_MODEL = 'minilm';

// name -> { pooling, dimensions, localDir, files: {model, config, tokenizer},
//           sha256: {<filename>: <hex>}, releaseAssets: {<filename>: <url>} }
export const MODEL_REGISTRY = {
  minilm: {
    pooling: 'mean',
    dimensions: 384,
    localDir: 'all-minilm-l6-v2',
    files: {
      model: 'model.safetensors',
      config: 'config.json',
      tokenizer: 'tokenizer.json',
    },
    sha256: {
      'model.safetensors':
        '53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db',
      'config.json':
        '953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41',
      'tokenizer.json':
        'be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037',
    },
    releaseAssets: {
      'model.safetensors': releaseUrl('all-minilm-l6-v2-model.safetensors'),
      'config.json': releaseUrl('all-minilm-l6-v2-config.json'),
      'tokenizer.json': releaseUrl('all-minilm-l6-v2-tokenizer.json'),
    },
  },
  bge: {
    pooling: 'cls',
    dimensions: 384,
    localDir: 'bge-small-en-v1.5',
    files: {
      model: 'model.safetensors',
      config: 'config.json',
      tokenizer: 'tokenizer.json',
    },
    sha256: {
      'model.safetensors':
        '3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad',
      'config.json':
        '094f8e891b932f2000c92cfc663bac4c62069f5d8af5b5278c4306aef3084750',
      'tokenizer.json':
        'd241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66',
    },
    releaseAssets: {
      'model.safetensors': releaseUrl('bge-small-en-v1.5-model.safetensors'),
      'config.json': releaseUrl('bge-small-en-v1.5-config.json'),
      'tokenizer.json': releaseUrl('bge-small-en-v1.5-tokenizer.json'),
    },
  },
  'paraphrase-minilm': {
    pooling: 'mean',
    dimensions: 384,
    localDir: 'paraphrase-multilingual-minilm-l12-v2',
    files: {
      model: 'model.safetensors',
      config: 'config.json',
      tokenizer: 'tokenizer.json',
    },
    sha256: {
      'model.safetensors':
        'eaa086f0ffee582aeb45b36e34cdd1fe2d6de2bef61f8a559a1bbc9bd955917b',
      'config.json':
        '6300193cb75e01cf80c96decef7187dfb33094d97cc1490b7ead6ff134476e4e',
      'tokenizer.json':
        '2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8',
    },
    releaseAssets: {
      'model.safetensors': releaseUrl(
        'paraphrase-multilingual-minilm-l12-v2-model.safetensors',
      ),
      'config.json': releaseUrl(
        'paraphrase-multilingual-minilm-l12-v2-config.json',
      ),
      'tokenizer.json': releaseUrl(
        'paraphrase-multilingual-minilm-l12-v2-tokenizer.json',
      ),
    },
  },
};

// Models this package is aware of but does not serve over the wasm channel,
// along with why. The wasm core wraps a BERT encoder (see
// crates/embed/src/wasm.rs); qwen3-0.6b is a decoder-style embedding model
// and belongs to the native lattice-embed binding instead. Listing it here
// (rather than treating an unknown name as an error) lets callers degrade
// deliberately: see `embed()` in index.mjs, which returns null for any name
// in this set instead of throwing.
export const UNSUPPORTED_MODELS = {
  'qwen3-0.6b':
    'decoder-style embedding model, not a BERT encoder; unsupported in the wasm channel, use the native lattice-embed binding',
};
