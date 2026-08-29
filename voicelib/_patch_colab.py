import re

COLAB = r'voicelib\colab_cell3.py'

with open(COLAB, 'r', encoding='utf-8') as f:
    content = f.read()

print("File loaded, length:", len(content))

# Patch 1 via regex - replace split_sentences function
pattern1 = re.compile(
    r'def split_sentences\(text: str, min_chars: int = 40\) -> list\[str\]:.*?return out or \[text\]',
    re.DOTALL
)
m = pattern1.search(content)
if m:
    print("Found split_sentences at char", m.start())
    new_func = (
        'def split_sentences(text, target_chars=120, max_chars=220):\n'
        '    """Chunk text for long-form synthesis. Handles 5000 chars. target_chars=120 = 4-8s per T4 pass."""\n'
        '    raw = re.split(r"(?<=[.!?;])\\s+|\\n+", text.strip())\n'
        '    sentences = [s.strip() for s in raw if s.strip()]\n'
        '    merged, cur = [], ""\n'
        '    for s in sentences:\n'
        '        if not cur:\n'
        '            cur = s\n'
        '        elif len(cur) + 1 + len(s) <= target_chars:\n'
        '            cur = cur + " " + s\n'
        '        else:\n'
        '            merged.append(cur); cur = s\n'
        '    if cur: merged.append(cur)\n'
        '    final = []\n'
        '    for chunk in merged:\n'
        '        if len(chunk) <= max_chars:\n'
        '            final.append(chunk)\n'
        '        else:\n'
        '            parts = re.split(r"(?<=,)\\s+", chunk)\n'
        '            sub, c2 = [], ""\n'
        '            for p in parts:\n'
        '                if not c2: c2 = p\n'
        '                elif len(c2) + 1 + len(p) <= max_chars: c2 = c2 + " " + p\n'
        '                else: sub.append(c2); c2 = p\n'
        '            if c2: sub.append(c2)\n'
        '            final.extend(sub if sub else [chunk])\n'
        '    return final or [text]'
    )
    content = pattern1.sub(new_func, content, count=1)
    print("Patch 1 applied")
else:
    print("Patch 1 - pattern not found, trying looser match")
    if 'def split_sentences' in content:
        print("Function exists but signature changed")

# Patch 2 - synthesis loop head
old_head_marker = 'sentences = split_sentences(clean_text, min_chars=35)'
if old_head_marker in content:
    content = content.replace(
        '            # ── Neural synthesis ─────────────────────────────────────────────\n'
        '            try:\n'
        '                if len(clean_text) > 80 or re.search(r"[,;]", clean_text):\n'
        '                    # Multi-sentence path \xe2\x80\x94 stitch with natural pauses\n'
        '                    sentences = split_sentences(clean_text, min_chars=35)\n'
        '                    chunks: list[np.ndarray] = []\n'
        '                    fade = int(gen_sr * 0.005)\n\n'
        '                    for i, sent in enumerate(sentences):',
        '            # ── Neural synthesis \xe2\x80\x94 handles up to 5000 characters \xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\n'
        '            try:\n'
        '                sentences = split_sentences(clean_text)\n'
        '                total = len(sentences)\n'
        '                print(f"Synthesising {len(clean_text)} chars in {total} chunks...")\n'
        '                chunks, fade = [], int(gen_sr * 0.005)\n'
        '                chunk_t0 = time.perf_counter()\n\n'
        '                for i, sent in enumerate(sentences):'
    )
    print("Patch 2 head applied")
else:
    print("Patch 2 head - already patched or not found")
    if 'sentences = split_sentences(clean_text)' in content:
        print("  Already uses new call - OK")

# Patch 2b - fix len(sentences) ref to total + add GPU flush
old_pause = 'if i < len(sentences) - 1:'
new_pause = 'if i < total - 1:'
count = content.count(old_pause)
print(f"Found {count} occurrences of old pause check")
content = content.replace(old_pause, new_pause)

# Add GPU cache flush after chunks.append if not already there
if 'empty_cache' not in content:
    old_append_block = (
        '                        chunks.append(chunk)\n\n'
        '                        # Natural inter-sentence pause\n'
        '                        if i < total - 1:\n'
        '                            pause_ms = 220.0 if s.endswith((".", "!")) else 160.0\n'
        '                            chunks.append(np.zeros(int(gen_sr * pause_ms / 1000), dtype=np.float32))\n\n'
        '                    gen_np = np.concatenate(chunks).astype(np.float32)'
    )
    new_append_block = (
        '                        chunks.append(chunk)\n\n'
        '                        # Natural inter-sentence pause\n'
        '                        if i < total - 1:\n'
        '                            pause_ms = 220.0 if s.endswith((".", "!")) else 150.0\n'
        '                            chunks.append(np.zeros(int(gen_sr * pause_ms / 1000), dtype=np.float32))\n\n'
        '                        # Free VRAM every 10 chunks (prevents OOM on 5000-char texts)\n'
        '                        if (i + 1) % 10 == 0:\n'
        '                            if torch.cuda.is_available(): torch.cuda.empty_cache()\n'
        '                            elapsed = time.perf_counter() - chunk_t0\n'
        '                            eta = (elapsed / (i + 1)) * (total - i - 1)\n'
        '                            print(f"  chunk {i+1}/{total} | {elapsed:.0f}s | ~{eta:.0f}s left")\n\n'
        '                gen_np = np.concatenate(chunks).astype(np.float32)\n'
        '                print(f"All {total} chunks done in {time.perf_counter()-chunk_t0:.1f}s")'
    )
    if old_append_block in content:
        content = content.replace(old_append_block, new_append_block)
        print("Patch 2b GPU flush applied")
    else:
        print("Patch 2b - block not found exactly")

# Remove single-pass else branch if still present
old_else = (
    '                else:\n'
    '                    # Single-pass synthesis\n'
    '                    with torch.inference_mode():\n'
    '                        wav_t = model.generate(\n'
    '                            clean_text,\n'
    '                            audio_prompt_path=ref_path,\n'
    '                            cfg_weight=active_cfg,\n'
    '                            exaggeration=active_exag,\n'
    '                        )\n'
    '                    gen_np = wav_t.squeeze().detach().cpu().numpy().astype(np.float32)'
)
if old_else in content:
    content = content.replace(old_else, '')
    print("Single-pass else branch removed")

with open(COLAB, 'w', encoding='utf-8') as f:
    f.write(content)
print("File saved.")
