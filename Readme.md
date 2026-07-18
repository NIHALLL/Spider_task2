# SPML Task 2 — Project Overview

Three separate pieces that ended up living under one task: a custom ResNet for image classification, a hand-built LSTM for time-series forecasting, and a RAG system that answers medical questions grounded in real clinical sources. They don't share code, but they share a philosophy: build the core mechanism by hand instead of importing it, so the point is understanding *why* it works, not just getting a number out at the end.

This doc is meant to be read as "what is this, how do you actually use it, why does it look the way it does, and what would I genuinely change if I kept working on it." Not a wishlist of every possible ML best practice, just the changes that would actually matter.

---

## 1. Medical RAG System

### What it does, and how you'd actually use it

You run the script, type a medical question at the prompt, and it retrieves relevant passages from four knowledge sources (MedQuAD Q&A pairs, WHO PDFs, CDC PDFs, NICE clinical guidelines), feeds them to a local LLM through Ollama, and returns an answer that's supposed to stick strictly to what's in those sources, no outside knowledge, no diagnosis, no dosage recommendations. There's a confidence label attached (HIGH/MEDIUM/LOW) based on how good the retrieval match was, and a safety layer that intercepts anything that looks like a medical emergency or a request for self-harm-adjacent information before it ever reaches the LLM.

The first time you run it, it has to walk every source folder, chunk everything, and build the embeddings into a local Chroma vector store, this part is slow, and it's supposed to be slow once, not every time. After that, it just loads the existing store and answers instantly.

### Why it's built this way

**Local-first (Ollama, not an API).** Medical queries, even hypothetical or coursework-driven ones, are the kind of thing you don't necessarily want going to a third-party API by default. Running the embedding model and the LLM locally means nothing leaves the machine, which also happens to make this reproducible without needing an API key.

**RAG instead of just asking the LLM directly.** `llama3.2` alone doesn't know what NICE's actual current hypertension targets are, and even if it did, you couldn't trust it without a source to check. Grounding every answer in retrieved passages, and citing them, means the system's answers are only as good as its sources, which is a much easier thing to audit than "trust the model's training data."

**The safety layer runs before retrieval, not after.** Emergency and unsafe patterns get caught by a simple substring check on the raw query, before any embedding search or LLM call happens. That's deliberate: it's the cheapest, fastest, most auditable place to put a hard stop. You don't want a system that retrieves clinical guideline passages about chest pain and then generates a thoughtful RAG answer to "I think I'm having a heart attack", you want it to immediately say "call emergency services," full stop.

**Four sources instead of one.** MedQuAD gives broad Q&A coverage, WHO/CDC give public-health-level guidance, NICE gives actual clinical practice guidelines. No single source is authoritative on its own, and the `source_bonus` in `estimate_confidence` explicitly rewards answers backed by more than one source agreeing, which is a reasonable proxy for "this isn't just one document's idiosyncratic phrasing."

### Where this would actually get better

- **The confidence score is a rough proxy, not a calibrated one.** Right now it's just the best similarity score plus a flat bonus per extra source. It'd be worth actually checking, on a handful of real questions, whether "HIGH confidence" answers are reliably better than "MEDIUM" ones, or whether the thresholds (0.7 / 0.4) were picked without ever being validated against real output quality. That's a half-day of manual spot-checking, not a research project, and it's the difference between a confidence label that means something and one that's decorative.
- **No re-ranking after retrieval.** Right now the top-5 chunks by embedding similarity go straight into the prompt. A cheap improvement: after retrieval, do a quick relevance filter (even just checking if the chunk actually mentions terms from the query) before handing it to the LLM, since embedding similarity alone sometimes pulls in a chunk that's topically adjacent but not actually useful for the specific question asked.
- **The safety patterns are a hardcoded list.** `"heart attack"`, `"can't breathe"`, etc. is a start, but it's brittle, someone typing "my chest has been hurting really bad for an hour" doesn't hit any of these strings. Worth either expanding the pattern list meaningfully (not exhaustively, just the obvious paraphrases) or, longer term, running a cheap separate classifier pass on the query before the main pipeline.
- **No conversation memory.** Every question is answered cold, with no memory of what was asked before. Fine for a single-question demo, but if this were used as an actual tool, a natural follow-up ("what about for someone with kidney disease?") has no way to inherit context from the previous question.
- **Chroma rebuild is all-or-nothing.** Right now, adding one new PDF means either manually deleting `chroma_db` and re-embedding everything from scratch, or writing a bespoke script to add just the new file. A small `--rebuild` flag or an incremental add path would save a lot of re-embedding time as the knowledge sources grow.

---

## 2. Custom ResNet — CIFAR-10 Classification

### What it does, and how you'd actually use it

Run the script, it downloads CIFAR-10 automatically, trains a from-scratch ResNet (no `torchvision.models.resnet18` import, every residual block hand-built) for 50 epochs, and produces training/validation curves, a classification report, a confusion matrix, and the best checkpoint by validation accuracy. Point someone at `best_resnet.pth` and they can load the trained weights without retraining.

Final result: ~94% validation accuracy, with training accuracy nearly saturating (~99.9%) by the same point, more on what that gap actually means below.

### Why it's built this way

The core idea worth understanding, not just implementing, is that a residual block doesn't ask a layer to learn some arbitrary new transformation, it asks it to learn a *correction* on top of the identity (`output = F(x) + x`). If the ideal transformation at some depth is close to "just pass this through unchanged," that's a much easier thing for the optimizer to converge to than a from-scratch mapping through fresh weights. And because the shortcut path is a direct, largely unimpeded connection, gradients have a route back to early layers that doesn't get shrunk by every weight matrix along the way, that's the actual fix for vanishing gradients in deep nets, not a side benefit.

Everything else is a deliberate fit to the dataset, not a default copied from ImageNet-scale ResNets:

- **3 stages, not 4.** CIFAR-10 images are 32x32. A 4th downsampling stage would shrink the feature map past the point of being useful at this resolution.
- **A single 3x3 conv stem, not the usual 7x7 + max-pool.** That aggressive early downsample makes sense when your input is 224x224 and you have detail to spare, at 32x32 it would throw away information the residual stages haven't even had a chance to use yet.
- **SGD + momentum + cosine annealing over Adam.** Adam converges faster in the short term, but SGD with a decaying schedule is the better-established choice for generalizing well on CNN image benchmarks at this scale, the tradeoff is slower early training for a better final result.

### Where this would actually get better

- **The train/val accuracy gap (99.9% vs 94%) is the actual thing to fix next, not just report.** This is overfitting, plainly, and there are cheap, well-understood levers for it that weren't used here: mixup or cutmix augmentation, label smoothing on the cross-entropy loss, or just adding dropout before the final FC layer. Any one of these would likely close some of that gap without touching the architecture itself.
- **No test-time augmentation.** Averaging predictions over a few augmented views of each test image (horizontal flip, small crops) is a nearly-free accuracy bump at inference time, and it's a one-function addition, not a retrain.
- **Per-class weaknesses are visible but unaddressed.** The classification report shows `cat`, `bird`, and `deer` dragging down the average while `automobile` and `ship` do fine. That's a real, specific finding, worth digging into with a few misclassified examples pulled up visually (which images is it actually getting wrong, and do they look genuinely ambiguous even to a human) rather than just noting the numbers and moving on.
- **No learning rate finder or warmup.** The LR (0.1) and schedule were chosen by convention, not tuned for this exact setup. A short LR range test before committing to the full 50-epoch run would validate whether 0.1 is actually a good starting point or just a reasonable guess that happened to work.
- **Single run, no seed variation.** One run at ~94% doesn't tell you how stable that number is. Running the same config with two or three different random seeds and reporting a mean ± spread would be a more honest number to put in a final report than a single run's result.

---

## 3. Custom LSTM — Jena Climate Forecasting

### What it does, and how you'd actually use it

Downloads the Jena Climate dataset automatically, resamples it to hourly readings, and trains a from-scratch LSTM (manually implemented gates, no `nn.LSTM`) to predict the next 12 hours of temperature from the previous 72 hours of full weather data. Produces loss curves, a forecast plot, and Huber/MAE/MSE metrics, with the best-validation checkpoint saved to `best_lstm.pth`.

### Why it's built this way

The point of hand-writing the LSTM cell is to make the memory mechanism visible instead of hidden inside a library call. The cell state update, `c_t = (f_t * c_prev) + (i_t * g_t)`, is elementwise multiplication and addition, not a matrix multiply, and that's exactly why it solves the vanishing gradient problem that plain RNNs have: the gradient flowing back through this operation is just the forget gate's value, not that value further shrunk by a weight matrix. When the forget gate stays close to 1, information (and gradient) can survive across many timesteps largely intact.

The four gates each do a specific job: forget decides what old memory to keep, input decides what new information actually gets written in (not everything that arrives), candidate proposes the new content itself, and output decides how much of the current memory actually gets exposed as the hidden state versus held in reserve. Two of these stacked cells give the network more representational depth per timestep than one, at the cost of twice the sequential computation.

Gradient clipping and Huber loss (instead of MSE) are both there for the same underlying reason: 72 sequential timesteps of backprop-through-time is enough for gradients to drift, and an occasional real-world temperature outlier shouldn't be allowed to produce an outsized loss spike partway through training.

### Where this would actually get better

- **The window size doesn't match the Transformer counterpart.** This uses 72 hours in / 12 hours out; the brief's Level 3 spec calls for 720/24. If these two models are ever meant to be compared on the same test split, one of them has to change. This isn't a nice-to-have, it's the single most important fix before any LSTM-vs-Transformer comparison is meaningful.
- **No teacher forcing or autoregressive option to compare against.** Right now the model predicts all 12 hours in one shot from the final hidden state. That's a legitimate design choice, but it'd be a genuinely interesting addition to also try an autoregressive variant (predict hour 1, feed it back in, predict hour 2, ...) and see whether conditioning later predictions on earlier ones actually helps, or whether it just compounds error. Right now that question is unanswered, not just unaddressed.
- **No per-horizon error breakdown.** MAE/MSE are currently reported as one aggregate number across all 12 hours. Splitting it out by hour-ahead (is hour 1 much more accurate than hour 12?) would directly answer one of the "sequence length considerations" questions instead of leaving it as a theoretical point.
- **Missing hours from `dropna()` aren't tracked.** If resampling produced any gaps, some "72 consecutive hours" windows might quietly span a small time discontinuity. A one-line check (`df.index.to_series().diff().value_counts()`) would confirm whether this is a real issue or a non-issue, right now it's just unknown.
- **No saved test-set predictions.** The script computes everything fresh every run. Saving `predictions`/`targets` to disk (a `.npz`, nothing fancy) after evaluation means a later comparison script doesn't have to reload the model and rerun inference just to build a shared plot against the Transformer.

---

## The common thread across all three

Every one of these has the same shape of remaining work: the *core mechanism* is solid and hand-built correctly, and the *evaluation rigor around it* is where the honest gaps are. None of the improvements above are "redo the architecture", they're "measure this more carefully," "save this artifact so it's reusable," or "check this assumption you're currently taking on faith." That's a good place to be. The hard part, understanding and implementing the thing by hand, is done. What's left is the part that turns a working model into a result you can actually stand behind.
