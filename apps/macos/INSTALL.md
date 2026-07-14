# Installing Lattice Studio

Lattice Studio is a native macOS app for running language models on your own Mac.
It does chat, embeddings, quantization, and LoRA fine-tuning, all fully offline.
The app bundles its own inference engine, so once it is installed there is nothing
else to set up: no Python, no Rust, no command line.

This guide is for people who want to install and use the app. If you want to build
the distributable bundle for other people (signing, notarization, icons), see
[`DISTRIBUTION.md`](DISTRIBUTION.md). For the engine, the command-line tools, and
the HTTP server, see the [main README](../../README.md).

---

## Requirements

- **macOS 14 (Sonoma) or later.**
- **An Apple Silicon Mac (M1 or later) for chat.** The chat engine runs on the
  Metal GPU, which is built for Apple Silicon. Intel Macs can run the app but fall
  back to a slower CPU path that supports fewer model formats.
- **Disk space for models.** A small chat model is about 1.5 GB; embedding models
  range from 90 MB to 1.3 GB. Models are stored under `~/.lattice/models`.

---

## 1. Get the app

The app is self-contained: the `.app` bundle carries the engine binaries inside it,
so a Mac that installs the bundle needs no developer tools. Today there are two ways
to get that bundle.

### Build it from source (the path that works today)

From a checkout of the repository, with Xcode (Swift 6.3+) and a Rust toolchain
installed, run:

```bash
./apps/macos/scripts/package-app.sh
```

This compiles the Swift app and all ten engine binaries in release mode and assembles
`apps/macos/dist/Lattice.app`, along with `Lattice.dmg` and `Lattice.zip`. Any of these
can be copied to another Mac and will run there without a toolchain.

### Install a `.dmg` someone built

If a developer hands you a `Lattice.dmg` (or you built one with the command
above), you do not need any developer tools to install it. Open the `.dmg` and go
to step 2.

> A signed, downloadable build hosted on the GitHub releases page is planned. Until
> that lands, the two paths above are how you get the app. Hosting a public download
> involves code-signing and notarization decisions that are still open.

---

## 2. Install and open it

1. Drag **Lattice** from the `.dmg` into your **Applications** folder.
2. The app is signed with a free ad-hoc certificate, not a paid Apple Developer ID,
   so the first launch needs one extra step. **Right-click** (or Control-click)
   **Lattice** in Applications, choose **Open**, then click **Open** in the
   dialog. macOS remembers this, so every launch after the first is a normal
   double-click.

   The terminal equivalent of that one-time step is:

   ```bash
   xattr -dr com.apple.quarantine /Applications/Lattice.app
   ```

---

## 3. Get your first model

The app opens on the **Models** screen. To add a model, open **Get Models**. There
are three ways to add one.

### Download an embedding model (one click)

Embedding models power search and similarity. In the download section, pick one (for
example `bge-small-en-v1.5`, about 130 MB) and click **Download**. The app fetches it,
verifies its checksum, and shows it as installed.

### Import a chat model from HuggingFace

The generative chat models (the Qwen family) are added by importing a download from
HuggingFace:

1. In the import section, click **Copy HF URL** next to a model. Start with
   **qwen3.5-0.8b**, the smallest and fastest.
2. Download the model files. The app copied a URL like
   `https://huggingface.co/Qwen/Qwen3.5-0.8B`; the repository id is the part after
   `huggingface.co/`. Download it with HuggingFace's CLI:

   ```bash
   pip install -U "huggingface_hub[cli]"
   huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir qwen3.5-0.8b
   ```

   or clone it directly (needs `git-lfs`):

   ```bash
   git clone https://huggingface.co/Qwen/Qwen3.5-0.8B
   ```

   Use whichever repository the app gave you. The **Copy HF URL** button is the
   source of truth.
3. Back in the app, choose **Import from Disk**, then **Import Model**, and select the
   folder you downloaded. A model folder must contain `config.json` and at least one
   `.safetensors` file. The app copies it into `~/.lattice/models`.

### Import any model already on disk

If you already have a model folder (a 16-bit model, a Q4-quantized model, or an
embedding model), use **Import from Disk** directly. The same `config.json` plus
`.safetensors` rule applies.

---

## 4. Start chatting

1. Open the **Chat** screen.
2. Pick your model from the selector. Embedding models do not appear here because
   they do not generate text. On Apple Silicon the GPU path runs both 16-bit and Q4
   models; the CPU path runs 16-bit models only.
3. Type a message and send it. The first message loads the model into memory. After
   that the app keeps the model warm, so follow-up messages skip the reload. The
   header shows the model and whether it is `warm` or `ready`.
4. Use the reasoning toggle to turn the model's thinking trace on or off. With it on,
   the thinking and the final answer are shown separately.

Everything runs on your Mac. Nothing is sent to a server.

---

## Troubleshooting

- **"LatticeStudio can't be opened because Apple cannot check it for malicious
  software."** This is the ad-hoc-signing prompt from step 2. Right-click the app,
  choose **Open**, then **Open** again. You only do this once.
- **A model does not appear in the Chat selector.** Embedding models are excluded
  from chat; import a generative (Qwen) model instead. On an Intel Mac, Q4 models are
  also excluded, so import a 16-bit model.
- **A model fails to import or load.** The folder must contain `config.json` and at
  least one `.safetensors` file. The app shows the error verbatim, and it usually
  names the missing file.
- **Where are my models?** Under `~/.lattice/models`. Delete a folder there to remove
  a model.
